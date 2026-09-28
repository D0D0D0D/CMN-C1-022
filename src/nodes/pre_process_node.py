"""AgentCore Platform v1.0 — CMN-C1-022 pre_process node."""

# Node contract (agents_layer_design.md §1):
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings [A1]
#  - Never import from mediator/, api/, or other agents

from __future__ import annotations

import json

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.services.service import VALID_CRITERION_TYPES

# S-2 input-size guards (DoS / cost ceiling). Tunable via config/config.yaml in
# deriving templates; conservative defaults here.
MAX_INPUT_CHARS = 200_000
MAX_CANDIDATES = 1_000
MAX_CRITERIA = 50
# Cost ceiling for `llm`-typed criteria: one model call per candidate per llm
# criterion. Without this a single accepted request could reach
# MAX_CANDIDATES * MAX_CRITERIA = 50,000 paid calls. AgentBaseGraph has no
# budget_usd equivalent, so the ceiling has to be enforced at admission.
MAX_LLM_SCORE_CALLS = 200


class ValidateScoringRequestNode(FunctionNode):
    """Parse and validate the scoring request before ranking.

    Accepts the request as a JSON string in ``user_input`` (falls back to
    ``input_context``). Populates ``candidates`` / ``criteria`` /
    ``validated_input`` on success, or returns ERROR on any structural problem.
    """

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def _extra_security_gate_input(self, state: AgentState) -> AgentState:
        """S-2 — reject oversized payloads before parsing (DoS guard)."""
        user_input = state.get("user_input", "") or ""
        if len(user_input) > MAX_INPUT_CHARS:
            # AgentStatus is a str-enum, so this still satisfies the framework's
            # `state["status"] == AgentStatus.ERROR.value` short-circuit check
            # while keeping the status field a single, consistent type.
            state["status"] = AgentStatus.ERROR.value
            errors = list(state.get("error_log", []))
            errors.append(
                f"ValidateScoringRequestNode: input exceeds {MAX_INPUT_CHARS} chars (S-2)"
            )
            state["error_log"] = errors
        return state

    def execute(self, state: dict) -> dict:
        request = self._parse_request(state)
        if request is None:
            return self._error("request is not a valid JSON object")

        candidates = request.get("candidates")
        criteria = request.get("criteria")

        error = self._validate(candidates, criteria)
        if error:
            return self._error(error)

        emit_trace_event(
            "scoring_request_validated",
            {"candidate_count": len(candidates), "criteria_count": len(criteria)},
            state,
        )

        return {
            "candidates": candidates,
            "criteria": criteria,
            "validated_input": {
                "top_k": request.get("top_k"),
                "tie_break": request.get("tie_break"),
                "candidate_count": len(candidates),
                "criteria_count": len(criteria),
            },
            "status": AgentStatus.SUCCESS.value,
        }

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_request(state: dict) -> dict | None:
        raw = state.get("user_input")
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                return None
            return parsed if isinstance(parsed, dict) else None
        ctx = state.get("input_context")
        return ctx if isinstance(ctx, dict) else None

    def _validate(self, candidates, criteria) -> str | None:
        if not isinstance(candidates, list) or not candidates:
            return "candidates must be a non-empty list"
        if not isinstance(criteria, list) or not criteria:
            return "criteria must be a non-empty list"
        if len(candidates) > MAX_CANDIDATES:
            return f"too many candidates (> {MAX_CANDIDATES})"
        if len(criteria) > MAX_CRITERIA:
            return f"too many criteria (> {MAX_CRITERIA})"
        for i, cand in enumerate(candidates):
            if not isinstance(cand, dict) or "id" not in cand:
                return f"candidate[{i}] must be an object with an 'id'"
            if not isinstance(cand.get("attributes", {}), dict):
                return f"candidate[{i}].attributes must be an object"
        weight_sum = 0.0
        llm_criteria = 0
        for i, crit in enumerate(criteria):
            if not isinstance(crit, dict) or "name" not in crit:
                return f"criterion[{i}] must be an object with a 'name'"
            ctype = crit.get("type")
            if ctype not in VALID_CRITERION_TYPES:
                return (
                    f"criterion[{i}].type must be one of "
                    f"{VALID_CRITERION_TYPES}, got {ctype!r}"
                )
            if ctype == "llm":
                llm_criteria += 1
            try:
                weight_sum += float(crit.get("weight", 0.0))
            except (TypeError, ValueError):
                return f"criterion[{i}].weight must be numeric"
        if weight_sum <= 0:
            return "sum of criterion weights must be > 0"
        llm_calls = len(candidates) * llm_criteria
        if llm_calls > MAX_LLM_SCORE_CALLS:
            return (
                f"request would need {llm_calls} LLM scoring calls "
                f"(> {MAX_LLM_SCORE_CALLS}); reduce candidates or llm criteria"
            )
        return None

    @staticmethod
    def _error(message: str) -> dict:
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [f"ValidateScoringRequestNode: {message}"],
        }
