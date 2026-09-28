"""Shared fixtures for live-check harnesses."""

from __future__ import annotations

import pytest


@pytest.fixture
def agent_config() -> dict:
    """No extra construction values.

    The harness layers config/config.yaml underneath and fills ``config["llm"]``
    itself, which is the only LLM seam this agent has.
    """
    return {}


@pytest.fixture
def case_payload() -> dict:
    """A scoring request that exercises all three criterion types.

    The `llm` criterion is what makes this the agent's real workload rather than a
    smoke payload: `numeric` and `boolean` are pure arithmetic and reach no model,
    so a request without an `llm` criterion cannot tell a wired build from an
    unwired one. `rubric` here is a genuine judgement — it cannot be answered by
    matching an attribute value.
    """
    return {
        "candidates": [
            {
                "id": "sup-A",
                "attributes": {
                    "unit_price_jpy": 1200,
                    "iso9001": True,
                    "profile": "Machining subcontractor, 12 years supplying aerospace "
                    "actuator housings. Two production lines, in-house CMM "
                    "inspection, has passed customer audits for flight parts.",
                },
            },
            {
                "id": "sup-B",
                "attributes": {
                    "unit_price_jpy": 800,
                    "iso9001": False,
                    "profile": "Consumer plastics moulder, founded last year. No metal "
                    "machining capability and no inspection equipment.",
                },
            },
            {
                "id": "sup-C",
                "attributes": {
                    "unit_price_jpy": 1500,
                    "iso9001": True,
                    "profile": "General precision machining shop. Some medical device "
                    "work, no aerospace history, but holds ISO 9001 and runs "
                    "a documented first-article inspection process.",
                },
            },
        ],
        "criteria": [
            {"name": "unit_price_jpy", "type": "numeric", "weight": 0.2, "direction": "lower_better"},
            {"name": "iso9001", "type": "boolean", "weight": 0.2},
            {
                "name": "profile",
                "type": "llm",
                "weight": 0.6,
                "rubric": "Judge how well this supplier can machine flight-critical "
                "aerospace actuator housings. Weigh demonstrated aerospace "
                "experience and inspection capability above general capacity.",
            },
        ],
        "top_k": 2,
    }


@pytest.fixture
def live_input_context() -> dict:
    """The invoke path takes the whole request through ``user_input``."""
    return {}
