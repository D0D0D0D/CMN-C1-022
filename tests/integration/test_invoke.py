# CMN-C1-022 — Integration Tests: full invoke chain (S-1 trust + end-to-end ranking)

import json

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from src.graph.graph import ScoringRankingGraph


def _agent():
    agent = ScoringRankingGraph()
    agent.compile()
    return agent


def _request():
    return json.dumps(
        {
            "candidates": [
                {"id": "lead-1", "attributes": {"budget": 100, "qualified": True}},
                {"id": "lead-2", "attributes": {"budget": 50, "qualified": True}},
                {"id": "lead-3", "attributes": {"budget": 0, "qualified": False}},
            ],
            "criteria": [
                {"name": "budget", "type": "numeric", "weight": 0.7, "direction": "higher_better"},
                {"name": "qualified", "type": "boolean", "weight": 0.3},
            ],
            "top_k": 2,
        }
    )


def _is_error(status) -> bool:
    return status == AgentStatus.ERROR or status == AgentStatus.ERROR.value


def test_end_to_end_ranking_verified_external():
    """Full chain with sufficient trust returns a ranked, top_k-limited result."""
    ctx = InvocationContext(session_id="s1", caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
    out = _agent().invoke(_request(), ctx=ctx)
    assert out["status"] == AgentStatus.SUCCESS
    payload = out["output"]
    ranking = payload["ranking"]
    assert len(ranking) == 2  # top_k
    assert ranking[0]["id"] == "lead-1"  # highest budget + qualified
    assert ranking[0]["rank"] == 1
    assert payload["total_candidates"] == 3


def test_s1_trust_gate_denies_anonymous():
    """PB: ANONYMOUS caller is denied by the S-1 trust gate (no ranking).

    The same request succeeds under VERIFIED_EXTERNAL, so error status + empty
    output here isolates the trust gate as the cause.
    """
    ctx = InvocationContext(session_id="s2", caller_trust_level=TrustLevel.ANONYMOUS)
    out = _agent().invoke(_request(), ctx=ctx)
    assert _is_error(out["status"])
    assert not out.get("output")
    # the denied node is recorded in the trace
    assert "ValidateScoringRequestNode" in out.get("node_history", [])


def test_invalid_request_routes_to_error():
    """Zero candidates → pre_process ERROR → main/post passthrough → no ranking."""
    bad = json.dumps({"candidates": [], "criteria": [{"name": "v", "type": "numeric", "weight": 1.0}]})
    ctx = InvocationContext(session_id="s3", caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
    out = _agent().invoke(bad, ctx=ctx)
    assert _is_error(out["status"])
    assert not out.get("output")
