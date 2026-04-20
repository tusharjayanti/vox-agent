# vox-agent

A customer-support AI agent with **inline response evaluation**, hallucination detection, and automatic retry/fallback. Built with Claude Sonnet + Claude Haiku (as judge), FastAPI, and Postgres.

---

## What it does

vox-agent is a FastAPI chat service where every agent response is evaluated inline by a second model before reaching the user. Claude Sonnet generates the response; Claude Haiku scores it against a rubric covering relevance, groundedness, and hallucination risk. Responses that fail get one targeted retry with a failure-specific correction; unrecoverable failures return a safe fallback message with a lookup ID. Every turn and every evaluation is persisted to Postgres for historical analysis.

## Why it exists

Most LLM demos skip the hardest part of running agents in production: **knowing when the agent is wrong**. Post-hoc eval runs on sampled traffic miss live failures; system-prompt engineering only goes so far. vox-agent treats evaluation as a first-class request-time concern, with the cost shape (Haiku as judge) that makes evaluating every turn affordable in real-world deployments.

## Architecture

```text
┌──────────────┐
│ POST /chat   │
└──────┬───────┘
       ▼
┌────────────────────────────────────────┐
│ agent.chat()                           │
│  1. Load history (memory.py)           │
│  2. Generate (Claude Sonnet)           │
│  3. Evaluate:                          │
│     ├─ Heuristics (sync, free)         │
│     └─ LLM judge (Claude Haiku, JSON)  │
│  4. If retry: targeted correction → re-gen + re-eval
│  5. If escalate: hardcoded fallback    │
│  6. Persist turn + evaluation          │
└──────┬─────────────────────────────────┘
       ▼
┌──────────────┐
│   Postgres   │  turns + evaluations (asyncpg)
└──────────────┘
```

**Two-layer evaluator.** Heuristics (length, refusal detection, hedge density, off-topic vocabulary) run first — deterministic, free, short-circuit the judge when a response is obviously bad. The LLM judge handles subtler failures: hallucinated specifics, poor relevance, ungrounded claims. A combiner merges both signals into a single `good | retry | escalate` verdict.

**Different model for judging.** Sonnet generates; Haiku judges. Using a different model reduces correlated blind spots (a hallucination Sonnet produced is exactly the kind Sonnet-as-judge might miss), and Haiku's cost shape keeps per-request overhead around 10-15% — affordable on every turn, not just sampled traffic.

## Quick start

**Prerequisites:** Python 3.12+, [uv](https://github.com/astral-sh/uv), Docker, Anthropic API key.

```bash
# 1. Clone + install
git clone <repo-url> vox-agent
cd vox-agent
uv sync

# 2. Configure
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY

# 3. Start Postgres
docker run --name vox-agent-pg \
  -e POSTGRES_USER=voxagent \
  -e POSTGRES_PASSWORD=voxagent \
  -e POSTGRES_DB=voxagent \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16

# 4. Run migrations
uv run scripts/init_db.py

# 5. Start the server
uv run voxagent
```

## Example request

```bash
curl -si -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"How long do I have to return an item?"}'
```

Response:

```
HTTP/1.1 200 OK
X-Verdict: escalate
X-Retry-Count: 1
X-Eval-Relevance: 5
X-Eval-Groundedness: 2
X-Hallucination-Risk: high

{
  "session_id": "demo",
  "reply": "I apologize, but I wasn't able to provide a helpful response to your question. Please try rephrasing, or contact our support team at support@acmestore.com. (Reference ID: 7)",
  "turn_id": 7
}
```

The headers surface the evaluator's judgment live. Here, Sonnet scored 5/5 on relevance (directly addressed the question) but 2/5 on groundedness (invented specifics not in the policy) → verdict `escalate` → user received the safe fallback with a lookup ID.

## Observability

Every response carries evaluation metadata as HTTP headers:

| Header                 | Values                          |
| ---------------------- | ------------------------------- |
| `X-Verdict`            | `good` \| `retry` \| `escalate` |
| `X-Retry-Count`        | `0` or `1`                      |
| `X-Eval-Relevance`     | `0`-`5`                         |
| `X-Eval-Groundedness`  | `0`-`5`                         |
| `X-Hallucination-Risk` | `low` \| `medium` \| `high`     |

Postgres stores the same data for historical analysis:

```sql
SELECT verdict, judge_hallucination_risk, COUNT(*)
FROM evaluations
GROUP BY verdict, judge_hallucination_risk;
```

## LLM provider support

| Provider               | generate | judge | Status                          |
| ---------------------- | -------- | ----- | ------------------------------- |
| **Anthropic** (Claude) | ✅       | ✅    | Fully supported                 |
| OpenAI                 | stub     | stub  | Placeholder — see DESIGN.md §11 |
| Google Gemini          | stub     | stub  | Placeholder — see DESIGN.md §11 |

Switching providers is a config change:

```bash
VOXAGENT_LLM_PROVIDER=anthropic  # or openai / gemini
```

The `LLMProvider` Protocol in `llm.py` defines `generate()` and `judge()`. All agent code depends on the Protocol, never on a specific SDK. Adding a provider means adding a class in `providers/` that implements both methods — the factory in `config.py` routes on `VOXAGENT_LLM_PROVIDER`.

## Tests

```bash
uv run pytest        # 81 tests, all mocked, ~1 second
uv run pytest -v     # verbose
```

No tests hit real APIs. The `LLMProvider` Protocol is mocked at the fixture level; database calls are patched via `monkeypatch`. Test categories:

- **Smoke** (`test_smoke.py`) — config, endpoints, provider factory
- **Memory** (`test_memory.py`) — session isolation, mutation protection
- **Heuristics** (`test_evaluator_heuristics.py`) — each check + combiner
- **Judge** (`test_evaluator_judge.py`) — JSON parsing, error handling, combiner priority
- **Anthropic provider** (`test_anthropic_provider_judge.py`) — real provider behaviour with mocked SDK
- **Agent** (`test_agent.py`) — retry paths, escalation, token accounting, memory integration

## Project structure

```text
src/voxagent/
├── agent.py              # Orchestrator (generate → eval → retry → fallback)
├── cli.py                # Console entry point (uv run voxagent)
├── config.py             # Pydantic Settings + provider factory
├── db.py                 # asyncpg pool + query functions
├── evaluator.py          # Heuristics + judge + combiner
├── llm.py                # LLMProvider Protocol + Message + LLMResponse
├── logging_config.py     # Unified logging via uvicorn log_config
├── main.py               # FastAPI app, lifespan, /chat endpoint
├── memory.py             # In-memory conversation store
├── prompts.py            # Generator + judge system prompts
└── providers/
    ├── anthropic_provider.py    # Real implementation
    ├── openai_provider.py       # Stub
    └── gemini_provider.py       # Stub
migrations/           # Numbered SQL migration files
scripts/init_db.py    # Idempotent migration runner
tests/                # 81 tests, fully mocked
```

## Design deep dive

See [DESIGN.md](DESIGN.md) for the full design document — motivation, alternatives considered, data model, configuration, and production roadmap.

## Known limitations & future work

Phase 1 scope is deliberately narrow. Known gaps:

- **Conversation memory is in-process.** Sessions don't survive server restarts. Production would swap in a Postgres- or Redis-backed store behind the same `ConversationMemory` interface.
- **No RAG yet.** The agent's knowledge is in a placeholder system prompt in `prompts.py`. Phase 2 adds pgvector + Voyage AI embeddings for retrieval against a policy corpus, which will substantially reduce the groundedness-failure mode demonstrated in the example above.
- **No database integration tests.** The `db.py` layer is covered by manual verification and type-checked query functions. Integration tests with a test database fixture are planned for Phase 2.
- **Evaluator thresholds are defensible defaults, not tuned.** A meta-eval harness (Phase 3) will run the evaluator against canned scenarios to measure agreement with human judgment and guide tuning of thresholds and the judge prompt.
- **Only Anthropic provider is implemented.** OpenAI and Gemini provider stubs raise `NotImplementedError` with roadmap pointers.
- **No streaming responses.** Phase 3.- **Single-server deployment only.** No horizontal scaling considerations (shared memory store, sticky sessions, pool sizing under load) yet.

See DESIGN.md §11 for the full production roadmap.

## License

MIT — see [LICENSE](LICENSE).
