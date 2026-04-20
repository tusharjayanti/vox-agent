-- Phase 1.8 migration 002: evaluations table
-- One row per assistant turn, recording heuristic flags and judge scores.

CREATE TABLE IF NOT EXISTS evaluations (
    id BIGSERIAL PRIMARY KEY,
    turn_id BIGINT NOT NULL REFERENCES turns(id),
    -- Verdict from the combiner
    verdict TEXT NOT NULL CHECK (verdict IN ('good', 'retry', 'escalate')),
    -- Heuristic layer
    heuristic_flags JSONB NOT NULL DEFAULT '[]'::jsonb,
    heuristic_verdict TEXT,
    -- Judge layer (nullable — judge may have been skipped or failed)
    judge_relevance SMALLINT,
    judge_groundedness SMALLINT,
    judge_hallucination_risk TEXT
        CHECK (judge_hallucination_risk IN ('low', 'medium', 'high')),
    judge_reasoning TEXT,
    judge_model TEXT,
    -- Provenance — which provider generated this
    provider TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evaluations_turn_id
    ON evaluations(turn_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_verdict
    ON evaluations(verdict);
CREATE INDEX IF NOT EXISTS idx_evaluations_hallucination_risk
    ON evaluations(judge_hallucination_risk)
    WHERE judge_hallucination_risk IS NOT NULL;
