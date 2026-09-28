# CMN-C1-022 — Unit Tests: ValidateScoringRequestNode (validation + S-2)

import json

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.nodes.pre_process_node import (
    MAX_INPUT_CHARS,
    ValidateScoringRequestNode,
)


def _req(payload):
    return {"user_input": json.dumps(payload), "node_history": [], "error_log": []}


class TestValidateScoringRequestNode:
    def setup_method(self):
        self.node = ValidateScoringRequestNode()

    def test_valid_request(self):
        state = _req(
            {
                "candidates": [{"id": "a", "attributes": {"v": 1}}],
                "criteria": [{"name": "v", "type": "numeric", "weight": 1.0}],
                "top_k": 5,
            }
        )
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS
        assert result["candidates"][0]["id"] == "a"
        assert result["validated_input"]["top_k"] == 5

    def test_empty_candidates_error(self):
        """BL: zero candidates → ERROR."""
        state = _req({"candidates": [], "criteria": [{"name": "v", "type": "numeric", "weight": 1.0}]})
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.ERROR

    def test_invalid_criterion_type_error(self):
        state = _req(
            {
                "candidates": [{"id": "a", "attributes": {}}],
                "criteria": [{"name": "v", "type": "magic", "weight": 1.0}],
            }
        )
        assert self.node.execute(state)["status"] == AgentStatus.ERROR

    def test_zero_weight_sum_error(self):
        state = _req(
            {
                "candidates": [{"id": "a", "attributes": {}}],
                "criteria": [{"name": "v", "type": "numeric", "weight": 0.0}],
            }
        )
        assert self.node.execute(state)["status"] == AgentStatus.ERROR

    def test_non_json_input_error(self):
        state = {"user_input": "not json", "node_history": [], "error_log": []}
        assert self.node.execute(state)["status"] == AgentStatus.ERROR

    def test_s2_oversized_input_rejected(self):
        """S-2: _extra_security_gate_input flags oversized payloads."""
        state = {"user_input": "x" * (MAX_INPUT_CHARS + 1), "error_log": []}
        out = self.node._extra_security_gate_input(state)
        assert out["status"] == AgentStatus.ERROR.value
        assert any("S-2" in e for e in out["error_log"])

    def test_s2_normal_input_passes(self):
        state = {"user_input": "x" * 100, "error_log": []}
        out = self.node._extra_security_gate_input(state)
        assert out.get("status") != AgentStatus.ERROR.value

    def test_s2_via_call_chain_skips_execute(self):
        """F-03: oversized input is rejected through the full __call__ chain.

        With sufficient trust (S-1 passes), the S-2 gate must short-circuit so
        execute() never parses the payload — no candidates appear in the result.
        """
        state = {
            "user_input": "x" * (MAX_INPUT_CHARS + 1),
            "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL,
            "node_history": [],
            "error_log": [],
            "execution_time": {},
        }
        result = self.node(state)  # BaseNode.__call__: S-1 → S-2 → execute → S-3
        assert _is_error(result.get("status"))
        assert "candidates" not in result  # execute() was skipped by S-2


def _is_error(status):
    return status == AgentStatus.ERROR or status == AgentStatus.ERROR.value


class TestLLMCallCeiling:
    """An `llm` criterion costs one model call per candidate.

    MAX_CANDIDATES * MAX_CRITERIA is 50,000 calls, and AgentBaseGraph has no
    budget_usd equivalent, so the ceiling is enforced when the request is admitted.
    """

    def _request(self, candidate_count, llm_criteria):
        criteria = [{"name": "v", "type": "numeric", "weight": 1.0}]
        criteria += [{"name": f"j{i}", "type": "llm", "weight": 1.0, "rubric": "judge"} for i in range(llm_criteria)]
        return {
            "candidates": [{"id": str(i), "attributes": {"v": i}} for i in range(candidate_count)],
            "criteria": criteria,
        }

    def _run(self, request):
        node = ValidateScoringRequestNode()
        return node.execute(
            {
                "user_input": json.dumps(request),
                "status": AgentStatus.SUCCESS,
                "node_history": [],
                "error_log": [],
            }
        )

    def test_request_at_the_ceiling_is_accepted(self):
        # 100 candidates x 2 llm criteria = 200 calls, exactly the ceiling.
        result = self._run(self._request(100, 2))
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_request_above_the_ceiling_is_rejected(self):
        # 101 candidates x 2 llm criteria = 202 calls.
        result = self._run(self._request(101, 2))
        assert result["status"] == AgentStatus.ERROR.value
        assert "LLM scoring calls" in result["error_log"][0]

    def test_deterministic_request_is_never_capped(self):
        """No llm criterion means no model calls, so candidate count is unaffected."""
        result = self._run(self._request(1000, 0))
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_one_call_over_the_ceiling_is_rejected(self):
        """Boundary: the ceiling is 200, so 201 calls must not be admitted."""
        request = self._request(201, 1)  # 201 candidates x 1 llm criterion
        result = self._run(request)
        assert result["status"] == AgentStatus.ERROR.value
        assert "201 LLM scoring calls" in result["error_log"][0]

    def test_exactly_the_ceiling_with_one_criterion_is_admitted(self):
        result = self._run(self._request(200, 1))
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_zero_weight_llm_criteria_still_count(self):
        """A weight of 0.0 does not make a criterion free.

        ScoringService scores every criterion regardless of weight, so a
        zero-weight llm criterion still costs one model call per candidate.
        """
        request = self._request(1, 0)
        request["criteria"].append({"name": "free", "type": "llm", "weight": 0.0, "rubric": "judge"})
        request["candidates"] = [{"id": str(i), "attributes": {"v": i}} for i in range(300)]
        result = self._run(request)
        assert result["status"] == AgentStatus.ERROR.value
        assert "300 LLM scoring calls" in result["error_log"][0]
