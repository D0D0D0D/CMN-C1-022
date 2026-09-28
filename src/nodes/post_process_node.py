"""AgentCore Platform v1.0 — CMN-C1-022 post_process node."""

from __future__ import annotations

import re

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

# S-3 — defense-in-depth: redact any credential-like token that slipped into a
# rationale string (e.g. via an LLM-typed criterion's output).
_CREDENTIAL_RE = re.compile(
    r"\b(sk-[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{12,}|eyJ[A-Za-z0-9_-]{10,})\b"
)
_REDACTED = "[REDACTED]"


class FormatRankingNode(FunctionNode):
    """Shape the ranked results into the response payload (with top_k)."""

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def _extra_security_gate_output(self, result: dict) -> dict:
        """S-3 — scrub credential-like tokens from rationale strings."""
        output = result.get("formatted_output")
        if isinstance(output, dict):
            for row in output.get("ranking", []):
                rationale = row.get("rationale")
                if isinstance(rationale, str):
                    row["rationale"] = _CREDENTIAL_RE.sub(_REDACTED, rationale)
        return result

    def execute(self, state: dict) -> dict:
        # F-04 — ERROR passthrough.
        if _is_error(state.get("status")):
            return {}

        ranked = state.get("ranked_results", [])
        criteria = state.get("criteria", [])
        validated = state.get("validated_input", {}) or {}
        top_k = validated.get("top_k")

        if isinstance(top_k, int) and top_k > 0:
            ranked = ranked[:top_k]

        formatted_output = {
            "ranking": ranked,
            "criteria_used": [c.get("name") for c in criteria],
            "returned": len(ranked),
            "total_candidates": validated.get("candidate_count", len(state.get("candidates", []))),
        }

        emit_trace_event(
            "ranking_formatted",
            {"returned": len(ranked), "top_k": top_k},
            state,
        )

        return {
            "formatted_output": formatted_output,
            "status": AgentStatus.SUCCESS.value,
        }


def _is_error(status) -> bool:
    return status == AgentStatus.ERROR or status == AgentStatus.ERROR.value
