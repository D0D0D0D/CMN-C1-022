"""AGENTIC STAR Marketplace entry adapter boundary tests."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.base import MissingSecret, NullProvider
from framework.secrets.context import bound_secrets

import cli
from src.graph.graph import ScoringRankingGraph


def _set_required_environment(monkeypatch, suffix: str = "one") -> dict[str, str]:
    values = {key: f"test-{suffix}-{key.lower()}" for key in cli._REQUIRED_SECRET_KEYS}
    # The startup gate requires a bare Azure resource endpoint, so
    # this value cannot be a placeholder like the other secrets.
    values["AZURE_OPENAI_ENDPOINT"] = f"https://{suffix}.services.ai.azure.example"
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return values


def test_marketplace_rebuilds_context_without_conversation_history(monkeypatch):
    values = _set_required_environment(monkeypatch)
    captured = {}

    def parent_invoke(self, user_input, session_id="", ctx=None, input_context=None):
        captured.update(ctx=ctx, input_context=input_context)
        return {"status": "success"}

    monkeypatch.setattr(ScoringRankingGraph, "invoke", parent_invoke)
    original_ctx = InvocationContext(
        caller_id="marketplace-user",
        caller_trust_level=TrustLevel.VERIFIED_EXTERNAL,
        hitl_allowed=True,
    )
    result = cli.MarketplaceGraph().invoke(
        "{}",
        ctx=original_ctx,
        input_context={
            "conversation_history": [{"role": "user", "content": "untrusted"}],
            "production_company_id": "attacker-company",
        },
    )

    assert result == {"status": "success"}
    assert captured["input_context"] == {
        "production_company_id": values["STG_DEFAULT_COMPANY_ID"],
        "project_id": values["STG_DEFAULT_PROJECT_ID"],
    }
    assert "conversation_history" not in captured["input_context"]
    assert captured["ctx"] is not original_ctx
    assert captured["ctx"].caller_trust_level is TrustLevel.VERIFIED_EXTERNAL
    assert captured["ctx"].hitl_allowed is False


@pytest.mark.parametrize(
    "invalid_scope",
    ["", "   ", "company id", " company-id", "company-id\n"],
)
def test_invalid_scope_fails_during_startup(monkeypatch, invalid_scope):
    _set_required_environment(monkeypatch)
    monkeypatch.setenv("STG_DEFAULT_COMPANY_ID", invalid_scope)

    with pytest.raises((MissingSecret, ValueError), match="STG_DEFAULT_COMPANY_ID"):
        cli.MarketplaceGraph().provision_secrets(NullProvider())


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview",
        "https://example.services.ai.azure.example/openai",
        "https://example.services.ai.azure.example/v1",
        "https://example.services.ai.azure.example?api-version=2024-10-21",
        "example.services.ai.azure.example",
    ],
    ids=["api-path", "openai-path", "base-path", "query-only", "missing-scheme"],
)
def test_unusable_azure_endpoint_shape_fails_during_startup(monkeypatch, endpoint):
    """AzureOpenAIClient appends its own path, so anything but a bare endpoint breaks.

    The resulting error surfaces inside complete(); a node that swallows it
    degrades silently while the invocation still reports status: success, so the
    adapter has to reject the shape at startup instead.
    """

    _set_required_environment(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", endpoint)

    with pytest.raises(ValueError, match="bare"):
        cli.MarketplaceGraph().provision_secrets(NullProvider())


@pytest.mark.parametrize("missing_key", cli._REQUIRED_SECRET_KEYS)
def test_missing_required_secret_fails_during_startup(monkeypatch, missing_key):
    _set_required_environment(monkeypatch)
    monkeypatch.delenv(missing_key)

    with pytest.raises(MissingSecret, match=missing_key):
        cli.MarketplaceGraph().provision_secrets(NullProvider())


def test_azure_llm_resolves_fresh_credentials_for_every_call(monkeypatch):
    configs = []

    class FakeAzureOpenAIClient:
        def __init__(self, config):
            configs.append(dict(config))

        def complete(self, messages):
            return {"content": "ok", "messages": messages}

        def stream(self, messages):
            yield "ok"

        def bind_tools(self, tools):
            return self

    azure_module = ModuleType("shared.services.llm.azure_openai_client")
    azure_module.AzureOpenAIClient = FakeAzureOpenAIClient
    monkeypatch.setitem(sys.modules, azure_module.__name__, azure_module)
    provider = cli._AllowlistedEnvironmentProvider()
    llm = cli.AzureLLM()
    assert vars(llm) == {}

    with bound_secrets(provider):
        first = _set_required_environment(monkeypatch, "first")
        llm.complete([{"role": "user", "content": "first"}])
        assert vars(llm) == {}

        second = _set_required_environment(monkeypatch, "second")
        llm.complete([{"role": "user", "content": "second"}])
        assert vars(llm) == {}

    assert configs == [
        {
            "api_key": first["AZURE_OPENAI_API_KEY"],
            "azure_endpoint": first["AZURE_OPENAI_ENDPOINT"],
            "azure_deployment": first["AZURE_OPENAI_DEPLOYMENT"],
        },
        {
            "api_key": second["AZURE_OPENAI_API_KEY"],
            "azure_endpoint": second["AZURE_OPENAI_ENDPOINT"],
            "azure_deployment": second["AZURE_OPENAI_DEPLOYMENT"],
        },
    ]


def test_marketplace_graph_injects_azure_before_compile():
    agent = cli.MarketplaceGraph({"max_retry": 3})

    assert isinstance(agent.config["llm"], cli.AzureLLM)
    agent.compile()
    assert agent._nodes["main"]._service._llm_scorer._llm is agent.config["llm"]


def test_provisioned_provider_is_environment_only_and_allowlisted(monkeypatch):
    values = _set_required_environment(monkeypatch)
    monkeypatch.setenv("UNRELATED_PLATFORM_SECRET", "must-not-be-visible")
    agent = cli.MarketplaceGraph()

    agent.provision_secrets(NullProvider())

    provider = agent._secrets_provider
    assert {key: provider.require(key) for key in cli._REQUIRED_SECRET_KEYS} == values
    assert provider.get("UNRELATED_PLATFORM_SECRET") is None


def test_main_uses_marketplace_contract_and_repo_config(monkeypatch):
    called = {}

    def fake_runner(agent_cls, **kwargs):
        called["agent_cls"] = agent_cls
        called.update(kwargs)

    monkeypatch.setattr(cli, "run_agent_marketplace", fake_runner)
    cli.main()

    assert called["agent_cls"] is cli.MarketplaceGraph
    assert called["agent_name"] == "CMN-C1-022"
    assert called["namespace"] == "cmn"
    assert called["config"] == {"max_retry": 3, "timeout_seconds": 30}
