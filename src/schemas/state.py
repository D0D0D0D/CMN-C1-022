"""AgentCore Platform v1.0 — CMN-C1-022 ScoringRankingAgent state."""

# ADR-005: State must be a flat TypedDict — never a rich model object.
# LangGraph checkpoints use msgpack serialization; rich model objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or rich model objects.

from __future__ import annotations

from framework.schemas.agent_state import AgentState


class State(AgentState):
    """ScoringRankingAgent state.

    Shared fields (user_input, status, session_id, node_history, error_log,
    caller_trust_level, validated_input, formatted_output, …) are inherited
    from AgentState.  Only the scoring-specific fields are added below.
    All values are plain JSON-serializable primitives / list / dict
    (msgpack-safe — no rich model objects or credentials).
    """

    # Parsed pool of entities to evaluate.
    #   each item: {"id": str, "attributes": dict}
    candidates: list[dict]

    # Scoring criteria (rubric).
    #   each item: {"name": str, "type": "numeric"|"boolean"|"llm",
    #               "weight": float, "direction"?: "higher_better"|"lower_better",
    #               "rubric"?: str}
    criteria: list[dict]

    # Per-candidate criterion scores produced by the main node.
    #   each item: {"id": str, "scores": {criterion: float},
    #               "rationales": {criterion: str},  # llm criteria only
    #               "total": float}
    scored_candidates: list[dict]

    # Final ranked list produced by the main node.
    #   each item: {"rank": int, "id": str, "score": float, "rationale": str}
    ranked_results: list[dict]
