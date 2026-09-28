# CMN-C1-022 — Integration Tests: the production construction path reaches the LLM
#
# These tests use the SAME construction call a real caller makes —
# ScoringRankingGraph(config=...) — and never touch a private attribute. A test
# that assigns agent._nodes["main"]._service._llm_scorer would stay green while the
# deployed build has no LLM at all; that is the defect this file exists to prevent.

import inspect
import json

import pytest
from framework.errors import ConfigError
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from shared.services.llm.base_llm import BaseLLM

from src.graph.graph import ScoringRankingGraph


class RecordingLLM(BaseLLM):
    def __init__(self, content='{"score": 0.9, "rationale": "fits the rubric"}'):
        self.content = content
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        return {"content": self.content, "tool_calls": [], "model": "test-model", "usage": {}}

    def stream(self, messages):  # pragma: no cover
        yield ""

    def bind_tools(self, tools):  # pragma: no cover
        return self


def _request_with_llm_criterion():
    return json.dumps(
        {
            "candidates": [
                {"id": "v-1", "attributes": {"price": 100, "profile": "10 years of aerospace machining"}},
                {"id": "v-2", "attributes": {"price": 300, "profile": "new entrant, consumer goods"}},
            ],
            "criteria": [
                {"name": "price", "type": "numeric", "weight": 0.4, "direction": "lower_better"},
                {"name": "fit", "type": "llm", "weight": 0.6, "rubric": "Fit for aerospace machining?"},
            ],
        }
    )


def _ctx(session_id):
    return InvocationContext(session_id=session_id, caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)


def _agent(config=None):
    agent = ScoringRankingGraph(config=config) if config is not None else ScoringRankingGraph()
    agent.compile()
    return agent


def test_graph_takes_no_bespoke_llm_keyword():
    """A keyword no framework caller passes is permanently unwired.

    BaseGraph.__init__(self, config=None) is the only construction contract, so the
    signature must expose nothing else.
    """
    params = inspect.signature(ScoringRankingGraph.__init__).parameters
    assert set(params) == {"self", "config"}


def test_llm_criterion_is_scored_through_config_llm():
    llm = RecordingLLM()
    out = _agent({"llm": llm}).invoke(_request_with_llm_criterion(), ctx=_ctx("w1"))

    assert out["status"] == AgentStatus.SUCCESS
    assert len(llm.calls) == 2  # one call per candidate
    ranking = out["output"]["ranking"]
    assert ranking[0]["id"] == "v-1"
    # price: v-1 normalises to 1.0 (lower_better), fit is 0.9 from the LLM.
    #   0.4 * 1.0 + 0.6 * 0.9 = 0.94
    assert ranking[0]["score"] == pytest.approx(0.94)
    # The model's own wording is adopted for the llm criterion.
    assert "fits the rubric" in ranking[0]["rationale"]


def test_llm_criterion_without_injection_is_a_configuration_error():
    """Unchanged behaviour when nothing is injected: no invented score."""
    out = _agent().invoke(_request_with_llm_criterion(), ctx=_ctx("w2"))
    assert out["status"] == AgentStatus.ERROR.value
    assert not out.get("output")


def test_deterministic_request_never_calls_the_llm():
    """A numeric/boolean-only request stays fully deterministic even when injected."""
    llm = RecordingLLM()
    request = json.dumps(
        {
            "candidates": [
                {"id": "a", "attributes": {"budget": 100}},
                {"id": "b", "attributes": {"budget": 10}},
            ],
            "criteria": [{"name": "budget", "type": "numeric", "weight": 1.0}],
        }
    )
    out = _agent({"llm": llm}).invoke(request, ctx=_ctx("w3"))
    assert out["status"] == AgentStatus.SUCCESS
    assert llm.calls == []
    assert out["output"]["ranking"][0]["id"] == "a"


def test_failing_llm_is_reported_as_error_not_degraded():
    """There is no deterministic substitute for a rubric judgement."""

    class BrokenLLM(RecordingLLM):
        def complete(self, messages):
            raise RuntimeError("connect to api.example.invalid failed: token=abcdef")

    out = _agent({"llm": BrokenLLM()}).invoke(_request_with_llm_criterion(), ctx=_ctx("w4"))
    assert out["status"] == AgentStatus.ERROR.value
    assert not out.get("output")


def test_unusable_llm_injection_fails_at_compile():
    agent = ScoringRankingGraph(config={"llm": "not-a-client"})
    with pytest.raises(ConfigError):
        agent.compile()


def test_compiling_twice_keeps_the_injected_llm():
    """register_nodes() must read config["llm"], never consume it.

    compile() can legitimately run more than once (a re-compile with a
    checkpointer, a rebuilt graph), and a consuming read would leave the second
    build silently unwired.
    """
    llm = RecordingLLM()
    config = {"llm": llm}
    agent = ScoringRankingGraph(config=config)
    agent.compile()
    agent.compile()
    assert config["llm"] is llm
    out = agent.invoke(_request_with_llm_criterion(), ctx=_ctx("w5"))
    assert out["status"] == AgentStatus.SUCCESS
    assert len(llm.calls) == 2
