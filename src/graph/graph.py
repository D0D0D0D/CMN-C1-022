"""AgentCore Platform v1.0 — CMN-C1-022 ScoringRankingAgent graph.

Cat 1 — single technical capability (candidate scoring & ranking),
use-case-agnostic. Inherits the L1 framework backbone AgentBaseGraph and fills
the three domain slots; the fixed 5-node pipeline and routing are owned by the
framework and must NOT be overridden.

    START → initialize → pre_process → main → {route} → post_process → finalize → END
                                          ↓ (RETRY, max 3)
                                       pre_process

Rules:
  ✅ Inherit AgentBaseGraph
  ✅ Call super().register_nodes() — injects initialize + finalize
  ✅ Fill pre_process / main / post_process
  ❌ Do NOT override add_edges() or route()
  ❌ No business logic here — nodes own execution logic
"""

from __future__ import annotations

from framework.graph.agent_base_graph import AgentBaseGraph

from src.nodes.main_node import ScoreAndRankNode
from src.nodes.post_process_node import FormatRankingNode
from src.nodes.pre_process_node import ValidateScoringRequestNode
from src.schemas.state import State
from src.services.llm_client import resolve_llm_scorer


class ScoringRankingGraph(AgentBaseGraph):
    """Fixed-pipeline graph for CMN-C1-022 ScoringRankingAgent.

    The LLM is received through ``config["llm"]`` and is used only when the
    request contains ``llm``-typed criteria. With nothing injected the agent runs
    fully deterministically on numeric / boolean criteria, and an ``llm``-typed
    criterion is reported as a configuration error.

    The constructor takes no bespoke keyword: ``BaseGraph.__init__(self,
    config=None)`` is the framework's only construction contract, and a keyword no
    real caller passes leaves the branch behind it permanently unwired.
    See ``src/services/llm_client.py``.
    """

    @property
    def name(self) -> str:
        return "cmn_c1_022_scoring_ranking"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        super().register_nodes()  # injects InitializeNode + FinalizeNode

        # config["llm"] is read here, not in __init__: the framework builds nodes
        # in register_nodes(), which compile() runs after the config is in place.
        llm_scorer = resolve_llm_scorer(self.config.get("llm"))

        self._nodes["pre_process"] = ValidateScoringRequestNode()
        self._nodes["main"] = ScoreAndRankNode(llm_scorer=llm_scorer)
        self._nodes["post_process"] = FormatRankingNode()


# Backwards-compatible alias for the scaffold entry-point name.
Graph = ScoringRankingGraph
