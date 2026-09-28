# CMN-C1-022 — Unit Tests: ScoreAndRankNode (BL: ranking, tie-break, passthrough)

import inspect

from framework.schemas.agent_status import AgentStatus
from src.nodes.main_node import ScoreAndRankNode


def _state(candidates, criteria, **vi):
    return {
        "status": AgentStatus.SUCCESS,
        "candidates": candidates,
        "criteria": criteria,
        "validated_input": {"candidate_count": len(candidates), **vi},
        "node_history": [],
        "error_log": [],
    }


class TestScoreAndRankNode:
    def setup_method(self):
        self.node = ScoreAndRankNode()

    def test_ranking_order(self):
        """BL: candidates are ranked by weighted total descending."""
        state = _state(
            [
                {"id": "low", "attributes": {"v": 0}},
                {"id": "high", "attributes": {"v": 10}},
                {"id": "mid", "attributes": {"v": 5}},
            ],
            [{"name": "v", "type": "numeric", "weight": 1.0}],
        )
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS
        ids = [r["id"] for r in result["ranked_results"]]
        assert ids == ["high", "mid", "low"]
        assert [r["rank"] for r in result["ranked_results"]] == [1, 2, 3]

    def test_tie_break_by_id_asc(self):
        """BL-02: equal totals break by candidate id ascending (deterministic)."""
        state = _state(
            [
                {"id": "b", "attributes": {"v": 5}},
                {"id": "a", "attributes": {"v": 5}},
            ],
            [{"name": "v", "type": "numeric", "weight": 1.0}],
        )
        result = self.node.execute(state)
        ids = [r["id"] for r in result["ranked_results"]]
        assert ids == ["a", "b"]

    def test_tie_break_by_secondary_criterion(self):
        """BL: tie_break criterion decides order when totals collide."""
        state = _state(
            [
                {"id": "a", "attributes": {"primary": 5, "freshness": 0}},
                {"id": "b", "attributes": {"primary": 5, "freshness": 10}},
            ],
            [
                {"name": "primary", "type": "numeric", "weight": 1.0},
                {"name": "freshness", "type": "numeric", "weight": 0.0},
            ],
            tie_break="freshness",
        )
        result = self.node.execute(state)
        assert [r["id"] for r in result["ranked_results"]] == ["b", "a"]

    def test_error_passthrough(self):
        """F-04: ERROR status upstream → main skips and returns empty dict."""
        state = _state([], [{"name": "v", "type": "numeric", "weight": 1.0}])
        state["status"] = AgentStatus.ERROR
        result = self.node.execute(state)
        assert result == {}

    def test_rationale_present(self):
        state = _state(
            [{"id": "a", "attributes": {"v": 1}}],
            [{"name": "v", "type": "numeric", "weight": 1.0}],
        )
        result = self.node.execute(state)
        assert "rationale" in result["ranked_results"][0]
        assert result["ranked_results"][0]["rationale"]

    def test_execute_method_signature(self):
        """node contract: execute(self, state) and no _invoke_impl."""
        assert hasattr(ScoreAndRankNode, "execute")
        params = list(inspect.signature(ScoreAndRankNode.execute).parameters.keys())
        assert params[1] == "state"
        assert "_invoke_impl" not in ScoreAndRankNode.__dict__


class TestScoreAndRankNodeLLMFailure:
    """The LLM failure path is what reaches the caller's screen — see below.

    A deployed entry adapter republishes the agent's own error_log to the end
    user. A provider exception message can contain the endpoint, a token or the
    request echoed back, so only the exception TYPE may travel.
    """

    def test_provider_message_never_reaches_error_log(self):
        class LeakyScorer:
            def score(self, rubric, attributes):
                raise RuntimeError("POST https://api.example.invalid key=sk-LEAKED123 failed")

        node = ScoreAndRankNode(llm_scorer=LeakyScorer())
        state = _state(
            [{"id": "a", "attributes": {"note": "x"}}],
            [{"name": "fit", "type": "llm", "weight": 1.0, "rubric": "rate fit"}],
        )
        result = node.execute(state)
        assert result["status"] == AgentStatus.ERROR.value
        logged = " ".join(result["error_log"])
        assert "RuntimeError" in logged
        assert "sk-LEAKED123" not in logged
        assert "api.example.invalid" not in logged
