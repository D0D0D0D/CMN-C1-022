# CMN-C1-022 — Unit Tests: FormatRankingNode (top_k, passthrough, S-3)

from framework.schemas.agent_status import AgentStatus
from src.nodes.post_process_node import FormatRankingNode


def _state(ranked, **vi):
    return {
        "status": AgentStatus.SUCCESS,
        "ranked_results": ranked,
        "criteria": [{"name": "v"}],
        "candidates": [{"id": r["id"]} for r in ranked],
        "validated_input": {"candidate_count": len(ranked), **vi},
        "node_history": [],
        "error_log": [],
    }


class TestFormatRankingNode:
    def setup_method(self):
        self.node = FormatRankingNode()

    def test_top_k_truncation_bl(self):
        """BL: top_k=1 returns only the top-ranked candidate."""
        ranked = [
            {"rank": 1, "id": "a", "score": 0.9, "rationale": "r"},
            {"rank": 2, "id": "b", "score": 0.5, "rationale": "r"},
        ]
        result = self.node.execute(_state(ranked, top_k=1))
        out = result["formatted_output"]
        assert out["returned"] == 1
        assert out["ranking"][0]["id"] == "a"
        assert out["total_candidates"] == 2

    def test_no_top_k_returns_all(self):
        ranked = [{"rank": 1, "id": "a", "score": 0.9, "rationale": "r"}]
        result = self.node.execute(_state(ranked))
        assert result["formatted_output"]["returned"] == 1

    def test_error_passthrough(self):
        state = _state([])
        state["status"] = AgentStatus.ERROR
        assert self.node.execute(state) == {}

    def test_s3_redacts_credential_like_rationale(self):
        """S-3: credential-like tokens in rationale are redacted."""
        result = {
            "formatted_output": {
                "ranking": [{"rank": 1, "id": "a", "score": 1.0, "rationale": "leak sk-ABCDEFGHIJ1234567890XYZ here"}]
            }
        }
        scrubbed = self.node._extra_security_gate_output(result)
        assert "[REDACTED]" in scrubbed["formatted_output"]["ranking"][0]["rationale"]
        assert "sk-ABCDEFGHIJ" not in scrubbed["formatted_output"]["ranking"][0]["rationale"]
