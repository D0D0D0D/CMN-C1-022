"""AgentCore Platform v1.0 — CMN-C1-022 main node."""

from __future__ import annotations

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.services.service import LLMScorer, ScoringService


class ScoreAndRankNode(FunctionNode):
    """Score every candidate, rank them, and attach a rationale.

    Deterministic by default (numeric / boolean criteria). An optional
    ``LLMScorer`` is injected only when ``llm``-typed criteria are used.
    """

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, llm_scorer: LLMScorer | None = None) -> None:
        super().__init__()
        self._service = ScoringService(llm_scorer=llm_scorer)

    def execute(self, state: dict) -> dict:
        # F-04 — ERROR passthrough: pre_process → main is unconditional, so a
        # validation/gate failure upstream must skip business logic here.
        if _is_error(state.get("status")):
            return {}

        candidates = state.get("candidates", [])
        criteria = state.get("criteria", [])
        validated = state.get("validated_input", {}) or {}
        tie_break = validated.get("tie_break")

        try:
            scored = self._service.score_all(candidates, criteria)
        except ValueError as exc:
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"ScoreAndRankNode: {exc}"],
            }

        ranked = self._rank(scored, criteria, tie_break)

        emit_trace_event(
            "candidates_ranked",
            {"ranked_count": len(ranked), "top_id": ranked[0]["id"] if ranked else None},
            state,
        )

        return {
            "scored_candidates": scored,
            "ranked_results": ranked,
            "status": AgentStatus.SUCCESS.value,
        }

    # ── helpers ──────────────────────────────────────────────────────────────

    def _rank(self, scored: list[dict], criteria: list[dict], tie_break) -> list[dict]:
        # Stable sort: primary = total desc, secondary = tie_break criterion desc,
        # final = id asc (deterministic). Python sort is stable, so apply keys
        # from least to most significant.
        ordered = sorted(scored, key=lambda s: str(s.get("id", "")))
        if tie_break:
            ordered.sort(key=lambda s: s.get("scores", {}).get(tie_break, 0.0), reverse=True)
        ordered.sort(key=lambda s: s.get("total", 0.0), reverse=True)

        ranked: list[dict] = []
        for i, s in enumerate(ordered, start=1):
            ranked.append(
                {
                    "rank": i,
                    "id": s.get("id"),
                    "score": s.get("total", 0.0),
                    "rationale": self._service.rationale_for(s, criteria),
                }
            )
        return ranked


def _is_error(status) -> bool:
    return status == AgentStatus.ERROR or status == AgentStatus.ERROR.value
