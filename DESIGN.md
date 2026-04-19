# VoxAgent v1 — Design Document

> A text-based customer support agent with a rigorous evaluation layer that detects hallucinations, retries bad responses, and logs everything for observability.

---

## 1. Motivation

VoxAgent is a portfolio project built to demonstrate the skills most relevant to production AI agent work: LLM integration, real-time request handling, evaluation and hallucination detection, and backend system design.

The project is deliberately scoped around a single differentiating idea: **an evaluator that runs on every response, grounds its judgments in retrieved context, and drives a retry/fallback loop.** Everything else — the FastAPI wiring, the Postgres logging, the customer-support domain — is scaffolding in service of that idea.

Non-goals for v1: voice, streaming, multi-tenant auth, horizontal scaling. These are captured in the production roadmap at the end of this document.

---

## 2. System overview

```
  CLI / curl / HTTP client
          │
          ▼
  FastAPI app
  ├── POST /chat { session_id, message }
  └── GET  /healthz
          │
          ▼
  Agent orchestrator
   ├── Memory (in-memory dict: session_id → [messages])
   ├── Retriever (Phase 2 — pgvector + Voyage embeddings)
   ├── Prompt Builder (system + retrieved context + history + user turn)
   ├── LLM Client (Anthropic SDK — Claude Sonnet)
   └── Evaluator
        ├── Heuristic checks (deterministic, free, fast)
        └── LLM-as-judge (Claude Haiku, structured JSON output)
          │
          ▼
  Postgres (asyncpg + raw SQL)
  ├── conversations
  ├── turns
  ├── evaluations
  └── documents (Phase 2)
```

### Request flow

1. Client sends `POST /chat` with a session ID and a message.
2. Agent loads conversation history from memory.
3. (Phase 2) Retriever fetches relevant policy chunks from pgvector.
4. Prompt Builder assembles the full prompt.
5. LLM client calls Claude Sonnet; response returned.
6. Evaluator runs — heuristics first, then Haiku judge — and produces a verdict.
7. If verdict is `retry`, agent re-prompts Sonnet with a stricter instruction. Max 1 retry.
8. If verdict is `escalate` (after retry or on first pass for severe cases), fallback message is returned.
9. Final turn + evaluation + metadata are logged to Postgres.
10. Response returned to client, including the verdict in response headers for observability.

---

## 3. Key design decisions

### 3.1 Model tiering: Sonnet generator, Haiku judge

The generator uses Claude Sonnet because generation is the harder, more open-ended task. It requires tone, reasoning, context-tracking, and handling ambiguity — all places where capability matters.

The judge uses Claude Haiku because judging against a structured rubric (especially with retrieved context, in Phase 2) is a narrower task. Haiku handles it reliably and keeps eval cheap enough to run on every turn in the request path rather than sampling in the background.

**Using a different (smaller) model for judging is a deliberate choice, not a compromise**, for two reasons:

1. **Reduced correlated errors.** A judge that shares the generator's blind spots won't catch errors those blind spots produce. Different model = independent second opinion.
2. **Cost/latency shape.** Every turn is 1 generator call + 1 judge call (+ maybe 1 retry). If both were Sonnet, eval would double per-turn cost. Haiku as judge makes eval a small marginal add.

This decision is validated empirically in the Phase 3 meta-eval harness (§8).

Model names are configurable via `.env` so swapping (or upgrading the judge to Sonnet) is a one-line change.

### 3.2 Evaluator as two-layer defense

The evaluator has two layers, cheapest first:

**Layer 1 — Heuristics (deterministic, free, fast):**

- **Length sanity** — not empty, not suspiciously long
- **Refusal detection** — did the agent decline to answer a reasonable support question?
- **Hedge-word density** — "I think", "probably", "might be" are flagged when the question expects specifics
- **Grounding check (Phase 2)** — scan response for specific claims (numbers, named policies, dates) and flag any not appearing in retrieved chunks

**Layer 2 — LLM-as-judge (Claude Haiku, structured JSON):**
Returns:

```json
{
  "relevance": 0-5,
  "groundedness": 0-5,
  "hallucination_risk": "low" | "medium" | "high",
  "reasoning": "one-sentence explanation"
}
```

**Combiner → verdict:**

- `good` → return to user
- `retry` → re-prompt with explicit correction ("Your previous response lacked X. Be more specific about Y and stay within the provided policy documents.")
- `escalate` → safe fallback ("Let me connect you with a human agent.")

Heuristics run first and can short-circuit the judge call (e.g., on a clearly empty response). This keeps median latency low.

### 3.3 Retry loop: bounded and explicit

- **Max 1 retry per turn.** Unbounded retries are a cost and latency risk.
- **Retry prompt is specific**, not generic. It names the failure ("groundedness was low — your response mentioned a 14-day return window, but the policy says 30 days").
- **Escalation is a real fallback**, not another LLM call. Hard-coded copy: "Let me connect you with a human agent. Your reference number is {turn_id}."

### 3.4 Web framework: FastAPI

FastAPI was chosen over Flask and Django for three reasons specific to this workload:

1. **Async-native.** Every dependency in this system — Claude (generator), Claude (judge), Voyage (embeddings in Phase 2), Postgres — is I/O-bound. FastAPI's `async`/`await` model lets a single worker handle many concurrent in-flight LLM requests while they wait on network I/O. Flask is sync-first (async is bolted on); Django's async story is incomplete (large parts of the ORM still block).
2. **Pydantic-native validation.** Request/response schemas are defined as Pydantic models and validated automatically. The same Pydantic models are reused to parse and validate the judge's structured JSON output — one validation layer, two uses. Auto-generated OpenAPI docs at `/docs` are a free byproduct.
3. **Right-sized and idiomatic for AI/ML APIs.** The project has three endpoints and no HTML rendering, admin, or multi-user auth — Django's batteries are all dead weight here. FastAPI has also become the de facto framework for serving Python ML/LLM workloads (Hugging Face, vLLM, LangServe, most agent framework reference implementations), which makes it the choice with the least explaining to do in review.

### 3.5 Data layer: asyncpg + raw SQL

Over SQLAlchemy/SQLModel because:

- Schema is small (3 tables + 1 for Phase 2) and stable
- Raw SQL is transparent and reviewable — a reviewer reads `db.py` in 30 seconds
- `asyncpg` is fast, async-native, and has a small API surface
- Forces (and demonstrates) SQL literacy
- Migrations as numbered `.sql` files — no migration framework needed at this scale

### 3.6 RAG with pgvector + Voyage (Phase 2)

- **pgvector** keeps everything in Postgres — no new service to run
- **Voyage AI embeddings** chosen over OpenAI for: coherence with an Anthropic-centric stack (Voyage is Anthropic's recommended embedding partner), generous free tier (200M tokens), and upgrade path to their reranker if retrieval quality becomes a bottleneck
- Chunk size: ~300 tokens with 50-token overlap
- Retrieval: top-3 cosine similarity
- **Both generator and judge see the retrieved chunks**, so the judge can verify grounding rather than guess at factuality

### 3.7 In-memory conversation store

A plain dict keyed by `session_id`. Not persistent across restarts. This is v1-appropriate for three reasons:

1. Conversations are ephemeral by design in the demo
2. Full trace is logged to Postgres — memory loss doesn't lose audit data
3. Interface is abstracted so a Postgres-backed or Redis-backed store can swap in

### 3.8 LLM provider abstraction

vox-agent separates _agent logic_ from _LLM vendor specifics_ via an `LLMProvider` protocol. Agent code (`agent.py`, `evaluator.py`) never imports Anthropic, OpenAI, or Gemini SDKs directly — it receives an `LLMProvider` instance and calls `provider.generate(...)` or `provider.judge(...)`.

**v1 ships only the `AnthropicProvider` implementation.** The `OpenAIProvider` and `GeminiProvider` exist as placeholder stubs in the `providers/` folder — each contains class skeletons that raise `NotImplementedError` with a clear message pointing to the roadmap. This signals architectural intent without promising undelivered functionality.

**Interface (sketch):**

```python
class LLMProvider(Protocol):
    async def generate(
        self, messages: list[Message], system: str, model: str,
        temperature: float, max_tokens: int
    ) -> LLMResponse: ...

    async def judge(
        self, user_message: str, agent_response: str,
        system: str, model: str, temperature: float,
        response_schema: type[BaseModel]
    ) -> BaseModel | None: ...
```

**Rationale for this design choice:**

1. **Avoids vendor lock-in.** Real production systems migrate between LLM providers for cost, capability, outage resilience, and pricing negotiation. An agent tied to one SDK is a liability.
2. **Makes testing simpler.** Agent-layer tests mock the `LLMProvider` interface, not a specific SDK. Adding new providers doesn't touch existing tests.
3. **Isolates provider quirks.** Token-counting fields, system-prompt placement, error types, retry semantics, and structured-output mechanisms all differ between providers. Each quirk lives inside one provider class, not spread across the codebase.
4. **Honest roadmap.** Stubs document the intent ("this will support OpenAI and Gemini") without shipping untested code that claims to work.

**Provider selection via config:**

```bash
VOXAGENT_LLM_PROVIDER=anthropic   # "anthropic" | "openai" | "gemini"
```

A factory in `config.py` reads this and returns the correct provider instance. v1 only resolves `anthropic`; the others raise a clear "not yet implemented, see roadmap" error.

**Provider-specific deps are optional extras** in `pyproject.toml`:

```toml
[project.optional-dependencies]
openai-provider = ["openai>=1.50.0"]
gemini-provider = ["google-genai>=0.3.0"]
```

So users only install SDKs they actually use. `uv sync` installs Anthropic by default (it's in core deps).

---

## 4. Data model

Three tables in v1, plus one for Phase 2.

```sql
-- conversations: one row per session
CREATE TABLE conversations (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- turns: one row per message (user or assistant)
CREATE TABLE turns (
    id               BIGSERIAL PRIMARY KEY,
    conversation_id  BIGINT NOT NULL REFERENCES conversations(id),
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    latency_ms       INTEGER,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    retry_count      INTEGER NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_turns_conversation ON turns(conversation_id);

-- evaluations: one row per assistant turn
CREATE TABLE evaluations (
    id                 BIGSERIAL PRIMARY KEY,
    turn_id            BIGINT NOT NULL REFERENCES turns(id),
    relevance          SMALLINT,
    groundedness       SMALLINT,
    hallucination_risk TEXT CHECK (hallucination_risk IN ('low', 'medium', 'high')),
    verdict            TEXT NOT NULL CHECK (verdict IN ('good', 'retry', 'escalate')),
    heuristic_flags    JSONB NOT NULL DEFAULT '{}'::jsonb,
    judge_reasoning    TEXT,
    judge_model        TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_evaluations_turn ON evaluations(turn_id);

-- documents: Phase 2 only
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE documents (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,         -- e.g. 'returns.md'
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(1024) NOT NULL, -- voyage-3 dimension
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_documents_embedding ON documents
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);
```

---

## 5. API contract

### `POST /chat`

**Request:**

```json
{
  "session_id": "sess_abc123",
  "message": "My order hasn't arrived in 5 days, what should I do?"
}
```

**Response (200):**

```json
{
  "session_id": "sess_abc123",
  "reply": "I'm sorry to hear about the delay. Standard shipping typically takes 3-5 business days...",
  "turn_id": 42
}
```

**Response headers (observability):**

- `X-Verdict: good|retry|escalate`
- `X-Retry-Count: 0|1`
- `X-Latency-Ms: 1243`
- `X-Eval-Relevance: 5`
- `X-Eval-Groundedness: 4`

### `GET /healthz`

Returns 200 `{"status": "ok"}` if the app is up and Postgres is reachable.

---

## 6. Project layout

```
voxagent/
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── README.md
├── DESIGN.md                    # this document
├── corpus/                      # Phase 2
│   ├── returns.md
│   ├── shipping.md
│   ├── accounts.md
│   └── order_tracking.md
├── migrations/
│   ├── 001_init.sql
│   ├── 002_evaluations.sql
│   └── 003_pgvector.sql         # Phase 2
├── src/voxagent/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app
│   ├── config.py                # env-driven settings + provider factory
│   ├── schemas.py               # Pydantic request/response models
│   ├── agent.py                 # orchestration — takes LLMProvider
│   ├── llm.py                   # LLMProvider protocol + LLMResponse
│   ├── memory.py                # conversation store
│   ├── evaluator.py             # heuristics + judge (uses LLMProvider)
│   ├── prompts.py               # system prompts, judge rubric
│   ├── retriever.py             # Phase 2
│   ├── db.py                    # asyncpg pool + queries
│   └── providers/
│       ├── __init__.py
│       ├── anthropic_provider.py    # implemented in v1
│       ├── openai_provider.py       # stub — raises NotImplementedError
│       └── gemini_provider.py       # stub — raises NotImplementedError
├── scripts/
│   ├── init_db.py               # runs migrations
│   └── ingest.py                # Phase 2 — embed corpus
├── evals/                       # Phase 3
│   ├── cases.jsonl              # canned Q/A with expected verdicts
│   └── run_eval.py              # meta-eval harness
└── tests/
    ├── conftest.py
    ├── test_evaluator.py
    ├── test_agent.py
    └── test_retriever.py        # Phase 2
```

---

## 7. Build phases

### Phase 1 — Core eval loop (text only)

| Step | Deliverable                                                                                                                                        |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.1  | Scaffold: `uv init`, deps, layout, `.env.example`, `.gitignore`, README stub                                                                       |
| 1.2  | FastAPI skeleton: `/chat` (hardcoded) + `/healthz`                                                                                                 |
| 1.3  | `LLMProvider` protocol + `AnthropicProvider` implementation (generator real, judge stub) + `OpenAIProvider` and `GeminiProvider` placeholder stubs |
| 1.4  | In-memory conversation store with swappable interface                                                                                              |
| 1.5  | Evaluator heuristics (length, refusal, hedging)                                                                                                    |
| 1.6  | LLM-as-judge (Haiku) with structured JSON rubric                                                                                                   |
| 1.7  | Retry loop + fallback copy                                                                                                                         |
| 1.8  | Postgres logging (3 tables, asyncpg pool)                                                                                                          |
| 1.9  | Tests for evaluator + integration test for full loop                                                                                               |

**Phase 1 exit criteria:** `curl` hits `/chat`, gets a real response, verdict in headers, full trace in Postgres.

### Phase 2 — RAG

| Step | Deliverable                                              |
| ---- | -------------------------------------------------------- |
| 2.1  | Seed corpus: 15–20 fake policy markdown files            |
| 2.2  | pgvector migration + `documents` table                   |
| 2.3  | Ingest script: chunk → embed (Voyage) → insert           |
| 2.4  | Retriever module: query → top-3 chunks                   |
| 2.5  | Wire retrieved chunks into generator and judge prompts   |
| 2.6  | Grounding heuristic: flag claims not in retrieved chunks |

**Phase 2 exit criteria:** agent answers policy questions from corpus content; evaluator flags fabricated details.

### Phase 3 — Demo polish

| Step | Deliverable                                                                  |
| ---- | ---------------------------------------------------------------------------- |
| 3.1  | Meta-eval harness (20–30 cases, Sonnet-judge vs Haiku-judge agreement)       |
| 3.2  | Example traces in README: good, hallucination caught, retry, escalation      |
| 3.3  | Architecture diagram (Mermaid)                                               |
| 3.4  | README: what it is, why, design decisions, how to run, eval results, roadmap |
| 3.5  | Optional: `vhs` demo GIF                                                     |

**Phase 3 exit criteria:** repo is portfolio-ready; a reviewer can understand what's interesting in 5 minutes.

**Estimated effort:** ~1 week focused work, ~2 weeks calendar time.

---

## 8. Meta-evaluation (Phase 3)

A small eval harness that tests the _evaluator itself_. 20–30 canned cases:

```jsonl
{"user": "...", "response": "...", "retrieved": [...], "expected_verdict": "good"}
{"user": "...", "response": "...", "retrieved": [...], "expected_verdict": "retry"}
{"user": "...", "response": "...", "retrieved": [...], "expected_verdict": "escalate"}
```

Cases include:

- Clearly correct responses (should be `good`)
- Subtle hallucinations — invented numbers, fabricated policy details
- Off-topic / refusal-when-shouldn't
- Ambiguous — close to threshold

Runner computes:

- Accuracy (verdict matches expected)
- Haiku-judge vs Sonnet-judge agreement rate
- Heuristic-only vs full evaluator accuracy (shows judge's marginal value)

Results published in README. This is what elevates the project from "I used an LLM judge" to "I tested and tuned my evaluator."

---

## 9. Configuration

All via `.env` (see `.env.example`):

```bash
# Required
POSTGRES_DSN=postgresql://voxagent:voxagent@localhost:5432/voxagent

# LLM provider selection
VOXAGENT_LLM_PROVIDER=anthropic   # "anthropic" (v1) | "openai" (roadmap) | "gemini" (roadmap)

# Anthropic (required when provider=anthropic)
ANTHROPIC_API_KEY=sk-ant-...
VOXAGENT_ANTHROPIC_GENERATOR=claude-sonnet-4-6
VOXAGENT_ANTHROPIC_JUDGE=claude-haiku-4-5-20251001

# OpenAI (roadmap — only needed if provider=openai)
OPENAI_API_KEY=sk-...
VOXAGENT_OPENAI_GENERATOR=gpt-4o
VOXAGENT_OPENAI_JUDGE=gpt-4o-mini

# Gemini (roadmap — only needed if provider=gemini)
GEMINI_API_KEY=...
VOXAGENT_GEMINI_GENERATOR=gemini-2.5-pro
VOXAGENT_GEMINI_JUDGE=gemini-2.5-flash

# Tuning (applies to whichever provider is active)
VOXAGENT_MAX_RETRIES=1
VOXAGENT_JUDGE_TEMPERATURE=0.0
VOXAGENT_GENERATOR_TEMPERATURE=0.3

# Phase 2
VOYAGE_API_KEY=pa-...
VOXAGENT_EMBEDDING_MODEL=voyage-3
VOXAGENT_RETRIEVAL_TOP_K=3
```

---

## 10. Local development

### Postgres via Docker

```bash
docker run --name voxagent-pg \
  -e POSTGRES_USER=voxagent \
  -e POSTGRES_PASSWORD=voxagent \
  -e POSTGRES_DB=voxagent \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

This image has `pgvector` preinstalled — no separate extension install needed when we reach Phase 2.

### Python environment

```bash
uv sync                 # installs deps from pyproject.toml
uv run python scripts/init_db.py   # runs migrations
uv run voxagent         # unified logging via programmatic uvicorn entry point
```

### Smoke test

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "message": "How long do I have to return an item?"}'
```

---

## 11. Production roadmap (explicitly out of scope for v1)

Items called out as v2+ work — not missing features but deliberately deferred:

- **OpenAI provider implementation** — wire up the `OpenAIProvider` stub to use `openai` SDK with structured outputs (`response_format={"type": "json_schema", ...}`) for the judge. Provider interface is already defined in v1; only the concrete implementation is pending.
- **Gemini provider implementation** — wire up the `GeminiProvider` stub to use `google-genai` SDK with `response_schema` in `generationConfig` for the judge. Same pattern as OpenAI — interface defined, concrete implementation pending.
- **Redis** for conversation memory across workers and for job queues
- **Docker Compose** for reproducible multi-service local dev
- **AWS deploy** (Fargate + RDS + Secrets Manager) with CI/CD
- **Streaming responses** via SSE or WebSocket — required for perceived real-time UX
- **Voice support** (separate project) — Deepgram STT streaming + ElevenLabs TTS streaming, with the evaluator moved out of the request path
- **Auth & rate limiting** — JWT or API key, per-session and per-IP
- **Multi-tenant corpus isolation** — per-customer knowledge bases
- **Reranking layer** — Voyage `rerank-2` on top of top-k retrieval for higher precision
- **Eval-driven prompt iteration** — nightly runs of the meta-eval harness to detect regressions when prompts change
- **PII scrubbing** in the logging layer — sanitize user inputs before Postgres writes

---

## 12. Open questions / explicit unknowns

Things we've decided to not over-think at design time:

- Exact heuristic thresholds (hedge-word count, claim-specificity rules) — will tune empirically during Phase 1.9 tests.
- Judge rubric wording — will iterate once we see real Haiku outputs.
- Retry prompt phrasing — will iterate based on observed failure modes.
- Whether to ship a CLI client — probably yes, as `scripts/chat.py`, if time permits in Phase 3.

---

_Document owner: Tushar. Last updated at project design._
