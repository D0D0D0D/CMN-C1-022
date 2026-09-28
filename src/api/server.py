"""AgentCore Platform v1.0 — CMN-C1-022 HTTP entry point.

Entry points are adapters only — no business logic here. For platform-level
routing, AgentGateway calls agent.invoke() directly.

Entry authentication is enforced by entry_adapter.auth:
  - a verified middleware trust (request.state.trust_level) is used as-is;
  - otherwise INVOKE_AUTH_TOKEN Bearer auth promotes ANONYMOUS →
    VERIFIED_EXTERNAL (_resolve_trust raises 401 on token mismatch);
  - an unauthenticated caller (no verified middleware trust AND no token
    promotion) is rejected with 401. The old fallback that defaulted such
    callers to ANONYMOUS and let the request through was fail-open (S-1 relied
    on the node trust gate alone) and is removed.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from framework.utils.config_loader import load_config
from shared.secrets import factory as secrets_factory

from src.entry_adapter.auth import _require_verified_trust, _resolve_trust
from src.graph.graph import ScoringRankingGraph

app = FastAPI(title="CMN-C1-022 ScoringRankingAgent")

# Runtime parameters come from config/config.yaml and are passed through
# unchanged. The previous `ScoringRankingGraph()` construction discarded the file
# entirely, so every tunable in it (max_retry, timeout_seconds) silently fell back
# to a framework default.
#
# This standalone adapter does NOT supply config["llm"]: no model identifier and
# no concrete client belongs in a template. A real LLM is injected by whatever
# entry adapter deploys the agent (the platform entry point fills config["llm"]),
# or by a caller that constructs the graph with one.
_CONFIG_YAML = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
_config: dict = load_config(str(_CONFIG_YAML)) if _CONFIG_YAML.exists() else {}

agent = ScoringRankingGraph(config=_config)
agent.compile()
agent.provision_secrets(secrets_factory(namespace="cmn", agent_name="cmn_c1_022"))


def _ctx(request: Request, session_id: str) -> InvocationContext:
    """Build the InvocationContext AFTER entry authentication.

    Scope note: the entry_adapter part also exposes a tenant/org scope contract,
    but this template has NO scope model. Scoring/ranking is a stateless, purely
    deterministic Cat 1 capability — every candidate/criterion arrives in the
    request body and no code path reads request tenant/org scope. Enforcing
    authenticate_scope() would gate on a scope nothing consumes (STG-default
    theater; a "default" fallback would silently match everything), so only the
    auth core (_resolve_trust / _require_verified_trust) is wired.
    """
    trust, token_authenticated = _resolve_trust(request)
    # Unauthenticated = trust is still ANONYMOUS after resolution and no token
    # promotion. Test the trust VALUE, not attribute presence — a middleware
    # that explicitly stamps ANONYMOUS must still be treated as unauthenticated
    # (→ 401), not silently accepted as "verified because the attr exists".
    if trust is TrustLevel.ANONYMOUS and not token_authenticated:
        raise HTTPException(status_code=401, detail="Authentication is required.")
    _require_verified_trust(trust)
    return InvocationContext(
        session_id=session_id or str(uuid4()),
        caller_trust_level=trust,
        caller_id=getattr(request.state, "caller_id", ""),
    )


class InvokeRequest(BaseModel):
    # `input` is a JSON string: {"candidates": [...], "criteria": [...], "top_k"?, "tie_break"?}
    input: str
    session_id: str = ""


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request):
    ctx = _ctx(request, req.session_id)
    with bound_secrets(agent._secrets_provider):
        return agent.invoke(req.input, ctx=ctx)


@app.get("/health")
def health():
    return {"status": "ok", "agent": "cmn_c1_022"}
