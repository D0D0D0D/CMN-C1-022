"""AGENTIC STAR Marketplace container entry adapter for CMN-C1-022."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from framework.schemas.invocation_context import InvocationContext
from framework.secrets.base import SecretProvider
from framework.secrets.context import current_secrets

try:
    from framework.utils.config_loader import load_agent_config
except ImportError:  # AgentCore 1.0.0 / 1.0.1 compatibility
    from framework.utils.config_loader import load_config

    def load_agent_config(agent_dir: Path) -> dict[str, Any]:
        config_path = agent_dir / "config" / "config.yaml"
        return load_config(str(config_path)) if config_path.exists() else {}


try:
    from shared.bootstrap import run_agent_marketplace
except ModuleNotFoundError as exc:  # pragma: no cover - old local wheel has no Marketplace extra
    if exc.name != "shared.bootstrap":
        raise

    def run_agent_marketplace(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("The installed AgentCore wheel does not include Marketplace support")


from shared.services.llm.base_llm import BaseLLM

from src.entry_adapter import (
    STG_DEFAULT_ORG_ENV,
    STG_DEFAULT_TENANT_ENV,
    VERIFIED_ORG_STATE_ATTR,
    VERIFIED_TENANT_STATE_ATTR,
)
from src.graph.graph import ScoringRankingGraph

_NAMESPACE = "cmn"
_AGENT_NAME = "CMN-C1-022"
_PROVIDER_AGENT_NAME = "cmn_c1_022"

_AZURE_OPENAI_SECRET_KEYS = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
)

# entry-adapter v0.2 exposes the two domain key names separately rather than as
# the newer SCOPE_KEYS tuple. Compose that declaration only from the part's
# existing constants, then keep all scope construction data-driven.
SCOPE_KEYS: tuple[str, ...] = (
    VERIFIED_TENANT_STATE_ATTR,
    VERIFIED_ORG_STATE_ATTR,
)
_SCOPE_ENV_BY_KEY = dict(zip(SCOPE_KEYS, (STG_DEFAULT_TENANT_ENV, STG_DEFAULT_ORG_ENV), strict=True))
_SCOPE_ENV_KEYS = tuple(_SCOPE_ENV_BY_KEY[key] for key in SCOPE_KEYS)
_REQUIRED_SECRET_KEYS = (*_AZURE_OPENAI_SECRET_KEYS, *_SCOPE_ENV_KEYS)
_ALLOWED_SECRET_KEYS = frozenset(_REQUIRED_SECRET_KEYS)


class _AllowlistedEnvironmentProvider(SecretProvider):
    """Expose only this single-agent container's declared environment values."""

    def __init__(self) -> None:
        self._namespace = _NAMESPACE
        self._agent_name = _PROVIDER_AGENT_NAME

    def get(self, key: str, default: str | None = None) -> str | None:
        if key not in _ALLOWED_SECRET_KEYS:
            return default
        value = os.environ.get(key)
        return value if value else default


def _scope_from_environment(provider: SecretProvider) -> dict[str, str]:
    """Build the declared scope without trimming or repairing invalid values."""

    scope: dict[str, str] = {}
    for scope_key in SCOPE_KEYS:
        env_name = _SCOPE_ENV_BY_KEY[scope_key]
        value = provider.require(env_name)
        if not value or any(character.isspace() for character in value):
            raise ValueError(
                f"Marketplace scope environment variable {env_name!r} " "must be a non-empty string without whitespace"
            )
        scope[scope_key] = value
    return scope


class AzureLLM(BaseLLM):  # type: ignore[misc]
    """Resolve Azure credentials and construct a fresh client for every call."""

    def __init__(self, *, tools: tuple[Any, ...] = ()) -> None:
        if tools:
            self._tools = tools

    def _client(self) -> BaseLLM:
        # The registry-only wheel does not include this optional integration.
        # All framework/security imports remain eager because Marketplace calls
        # invoke() through asyncio.to_thread and late framework imports can hang.
        from shared.services.llm.azure_openai_client import AzureOpenAIClient

        secrets = current_secrets()
        client = AzureOpenAIClient(
            {
                "api_key": secrets.require("AZURE_OPENAI_API_KEY"),
                "azure_endpoint": secrets.require("AZURE_OPENAI_ENDPOINT"),
                "azure_deployment": secrets.require("AZURE_OPENAI_DEPLOYMENT"),
            }
        )
        tools = getattr(self, "_tools", ())
        return client.bind_tools(list(tools)) if tools else client

    def complete(self, messages: list[Any]) -> dict[str, Any]:
        return cast(dict[str, Any], self._client().complete(messages))

    def stream(self, messages: list[Any]) -> Iterator[str]:
        yield from self._client().stream(messages)

    def bind_tools(self, tools: list[Any]) -> "AzureLLM":
        return type(self)(tools=tuple(tools))


def _require_azure_endpoint(provider: SecretProvider) -> str:
    """Reject ``AZURE_OPENAI_ENDPOINT`` values that are not a bare Azure resource endpoint.

    ``AzureOpenAIClient`` appends its own ``/openai/v1/`` path, so anything
    beyond a scheme and host silently produces a broken request URL, and the
    failure surfaces inside ``complete()``. A node that swallows that exception
    degrades quietly while the invocation still reports ``status: success``,
    which is why this has to be a startup check. The value is reported here,
    never repaired.
    """

    endpoint = provider.require("AZURE_OPENAI_ENDPOINT")
    parts = urlsplit(endpoint)
    try:
        port = parts.port
    except ValueError:
        port = -1
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or port == -1
        or parts.path.strip("/")
        or parts.query
        or parts.fragment
        or any(character.isspace() for character in endpoint)
    ):
        raise ValueError(
            "Marketplace environment variable 'AZURE_OPENAI_ENDPOINT' must be a bare Azure "
            "resource endpoint -- scheme and host only, with no path (such as "
            "'/openai'), query or fragment"
        )
    return endpoint


class MarketplaceGraph(ScoringRankingGraph):
    """Marketplace-only context hardening and Azure dependency wiring."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # The Marketplace runner may compile before provision_secrets(). Install
        # the credential-free facade now; it resolves connection values only
        # while an invocation's provider is bound.
        marketplace_config = dict(config or {})
        marketplace_config["llm"] = AzureLLM()
        super().__init__(config=marketplace_config)

    def invoke(
        self,
        user_input: str,
        session_id: str = "",
        ctx: InvocationContext | None = None,
        input_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Runner conversation history is untrusted free text outside this
        # agent's S-2 scan. Rebuild context exclusively from adapter-owned data.
        del input_context
        scope = _scope_from_environment(_AllowlistedEnvironmentProvider())

        # Marketplace has no resume channel, so never allow a node introduced
        # by a derived graph/configuration to suspend in AWAITING_HUMAN.
        marketplace_ctx = replace(ctx, hitl_allowed=False) if ctx is not None else None
        return cast(
            dict[str, Any],
            super().invoke(
                user_input,
                session_id=session_id,
                ctx=marketplace_ctx,
                input_context=scope,
            ),
        )

    def provision_secrets(self, provider: SecretProvider) -> None:
        # Ignore the runner's broader provider (including dotenv fallback).
        # A Marketplace Pod is allowed to see only its declared process env.
        del provider
        environment_provider = _AllowlistedEnvironmentProvider()
        for key in _REQUIRED_SECRET_KEYS:
            environment_provider.require(key)
        _scope_from_environment(environment_provider)
        _require_azure_endpoint(environment_provider)
        super().provision_secrets(environment_provider)


def main() -> None:
    run_agent_marketplace(
        MarketplaceGraph,
        agent_name=_AGENT_NAME,
        namespace=_NAMESPACE,
        config=load_agent_config(Path(__file__).resolve().parent),
    )


if __name__ == "__main__":
    main()
