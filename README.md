# vox-agent

> Customer-support agent with inline evaluation, hallucination detection, and retry/fallback logic. Claude · FastAPI · pgvector.

**Status:** 🚧 Phase 1 — core eval loop in progress. See [`DESIGN.md`](./DESIGN.md) for the full plan.

---

## What it does

vox-agent answers customer support questions (orders, returns, shipping, accounts) using an LLM. **Every response is evaluated on-the-fly by a separate smaller model** against a structured rubric. If the response looks low-quality or ungrounded, the agent retries with a stricter prompt or escalates to a safe fallback. Every turn and every evaluation is logged to Postgres for observability.

## Why it exists

Production agents fail in two ways that matter: they hallucinate facts, and they fail silently. vox-agent demonstrates one approach to both — an inline evaluator that grounds its judgments in retrieved context (Phase 2) and emits structured verdicts for every turn.

See [`DESIGN.md`](./DESIGN.md) for the full architecture, design decisions, and build plan.

---

## Quickstart

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Docker (for Postgres)
- An API key for your chosen LLM provider (Anthropic in v1)

### 1. Start Postgres

```bash
docker run --name vox-agent-pg \
  -e POSTGRES_USER=voxagent \
  -e POSTGRES_PASSWORD=voxagent \
  -e POSTGRES_DB=voxagent \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

> The `pgvector/pgvector:pg16` image is Postgres 16 with the pgvector extension preinstalled — no separate install needed when we reach Phase 2.

### 2. Install dependencies

```bash
# Phase 1 only (Anthropic provider + dev tools)
uv sync

# Phase 2 — adds RAG
uv sync --extra rag
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — at minimum, set ANTHROPIC_API_KEY
# VOXAGENT_LLM_PROVIDER=anthropic is the default
```

### 4. Run migrations

```bash
uv run python scripts/init_db.py
```

### 5. Run the server

```bash
uv run voxagent
```

### 6. Smoke test

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "message": "How long do I have to return an item?"}'
```

Check the response headers for evaluation metadata:

```
X-Verdict: good
X-Retry-Count: 0
X-Latency-Ms: 1243
X-Eval-Relevance: 5
X-Eval-Groundedness: 4
```

---

## Architecture

```
Client → FastAPI → Agent
                   ├── Memory (in-memory for v1)
                   ├── Retriever (Phase 2 — pgvector + Voyage)
                   ├── LLMProvider (interface)
                   │     ├── AnthropicProvider (v1)
                   │     ├── OpenAIProvider (stub — roadmap)
                   │     └── GeminiProvider (stub — roadmap)
                   └── Evaluator
                        ├── Heuristics
                        └── Judge (via LLMProvider)
                   ↓
                   Postgres (logs + evals + embeddings)
```

Full diagram and component responsibilities in [`DESIGN.md`](./DESIGN.md#2-system-overview).

### Key design decisions

- **Generator: Sonnet. Judge: Haiku.** Different-model judging reduces correlated errors and keeps eval cheap enough to run on every turn. Detailed rationale in [`DESIGN.md §3.1`](./DESIGN.md#31-model-tiering-sonnet-generator-haiku-judge).
- **Pluggable LLM providers.** Agent logic never imports provider SDKs directly — it uses an `LLMProvider` protocol. v1 ships with `AnthropicProvider` implemented; `OpenAIProvider` and `GeminiProvider` are placeholder stubs (see below). Detailed rationale in [`DESIGN.md §3.8`](./DESIGN.md#38-llm-provider-abstraction).
- **FastAPI + asyncpg + raw SQL.** Async-native for I/O-bound LLM calls; raw SQL for a transparent small-schema data layer.
- **pgvector for RAG (Phase 2).** Everything in Postgres — no new services.
- **Meta-eval in Phase 3.** A small harness that tests the evaluator itself, comparing Sonnet-judge vs Haiku-judge agreement on canned cases.

---

## LLM provider support

vox-agent is designed to be **LLM-agnostic**. Agent logic depends on an `LLMProvider` protocol, not on any specific SDK. Provider-specific code lives in `src/voxagent/providers/` — one file per provider.

### Current status

| Provider           | Status                | Notes                                                                                                                                                  |
| ------------------ | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Anthropic (Claude) | ✅ **Implemented**    | Default in v1. Sonnet for generation, Haiku for judging.                                                                                               |
| OpenAI (GPT)       | 🚧 **Stub — roadmap** | `OpenAIProvider` class exists and implements the interface, but every method raises `NotImplementedError`. Implementation pending — see roadmap below. |
| Google Gemini      | 🚧 **Stub — roadmap** | `GeminiProvider` class exists with the same stub pattern. Implementation pending.                                                                      |

The stubs are deliberate. They document architectural intent — "this system is designed to support OpenAI and Gemini" — without shipping untested code that claims to work. Wiring up a real implementation is an afternoon of work per provider once someone is ready to build it.

### Selecting a provider

```bash
# In .env
VOXAGENT_LLM_PROVIDER=anthropic   # "anthropic" | "openai" | "gemini"
```

In v1, only `anthropic` resolves to a working instance. Selecting `openai` or `gemini` raises a clear error pointing to the roadmap.

### Installing provider SDKs

Provider-specific SDKs are optional extras — you only install what you use:

```bash
# Anthropic is in core deps — installed by default
uv sync

# When OpenAI provider is implemented
uv sync --extra openai-provider

# When Gemini provider is implemented
uv sync --extra gemini-provider
```

### Why this design

Three reasons:

1. **No vendor lock-in.** Real production systems migrate between providers for cost, capability, outage resilience, and pricing leverage. Building against one SDK is a liability.
2. **Provider quirks are isolated.** Token-counting fields, system-prompt placement, error types, and structured-output mechanisms all differ between providers. Each quirk lives in one class, not scattered through the codebase.
3. **Testing stays simple.** Unit tests mock the `LLMProvider` protocol. Adding new providers doesn't require new test infrastructure.

---

## Project layout

```
vox-agent/
├── pyproject.toml          # uv-managed deps, provider SDKs as optional extras
├── .env.example
├── DESIGN.md               # full design doc
├── README.md               # this file
├── corpus/                 # Phase 2: fake policy docs
├── migrations/             # numbered .sql files
├── src/voxagent/
│   ├── main.py             # FastAPI app
│   ├── config.py           # settings + provider factory
│   ├── agent.py            # orchestration
│   ├── llm.py              # LLMProvider protocol
│   ├── evaluator.py        # heuristics + judge
│   ├── memory.py           # conversation store
│   ├── db.py               # asyncpg pool + queries
│   └── providers/
│       ├── anthropic_provider.py    # implemented
│       ├── openai_provider.py       # stub — roadmap
│       └── gemini_provider.py       # stub — roadmap
├── scripts/                # init_db, ingest
├── evals/                  # Phase 3: meta-eval harness
└── tests/
```

---

## Development

### Running tests

```bash
uv run pytest
```

### Linting

```bash
uv run ruff check .
uv run ruff format .
```

---

## Roadmap (v2+)

Explicitly deferred for v1 — see [`DESIGN.md §11`](./DESIGN.md#11-production-roadmap-explicitly-out-of-scope-for-v1):

- **OpenAI provider implementation** — wire up `OpenAIProvider` using `openai` SDK with structured outputs for the judge
- **Gemini provider implementation** — wire up `GeminiProvider` using `google-genai` SDK with response schemas for the judge
- Redis for multi-worker conversation memory and job queues
- Docker Compose for reproducible multi-service dev
- AWS deploy (Fargate + RDS + Secrets Manager) with CI/CD
- Streaming responses (SSE / WebSocket) for perceived real-time UX
- Voice support (separate project — Deepgram STT + ElevenLabs TTS, streaming)
- Auth & rate limiting
- Reranking (Voyage `rerank-2`) on top of retrieval
- PII scrubbing in the logging layer

---

## License

MIT
