# CMN-C1-022 — Unit Tests: ScoringService (BL: weight math, normalization)

import json

import pytest

from src.services.service import ScoringService


def test_weight_normalization_bl01():
    """BL-01: weighted total = Σ(weight × normalized) / Σ(weight)."""
    candidates = [
        {"id": "a", "attributes": {"budget": 100, "qualified": True}},
        {"id": "b", "attributes": {"budget": 0, "qualified": False}},
    ]
    criteria = [
        {"name": "budget", "type": "numeric", "weight": 0.6, "direction": "higher_better"},
        {"name": "qualified", "type": "boolean", "weight": 0.4},
    ]
    scored = ScoringService().score_all(candidates, criteria)
    by_id = {s["id"]: s for s in scored}
    # a: budget normalized to 1.0 (max), qualified 1.0 → (0.6*1 + 0.4*1)/1.0 = 1.0
    assert by_id["a"]["total"] == pytest.approx(1.0)
    # b: budget 0.0 (min), qualified 0.0 → 0.0
    assert by_id["b"]["total"] == pytest.approx(0.0)


def test_numeric_lower_better():
    """Numeric direction=lower_better inverts the normalization."""
    candidates = [
        {"id": "a", "attributes": {"risk": 10}},
        {"id": "b", "attributes": {"risk": 0}},
    ]
    criteria = [{"name": "risk", "type": "numeric", "weight": 1.0, "direction": "lower_better"}]
    scored = {s["id"]: s for s in ScoringService().score_all(candidates, criteria)}
    assert scored["b"]["total"] == pytest.approx(1.0)  # lowest risk → best
    assert scored["a"]["total"] == pytest.approx(0.0)


def test_numeric_zero_spread_returns_full():
    """When all candidates share a value, the criterion does not differentiate."""
    candidates = [
        {"id": "a", "attributes": {"x": 5}},
        {"id": "b", "attributes": {"x": 5}},
    ]
    criteria = [{"name": "x", "type": "numeric", "weight": 1.0}]
    scored = {s["id"]: s for s in ScoringService().score_all(candidates, criteria)}
    assert scored["a"]["total"] == pytest.approx(1.0)
    assert scored["b"]["total"] == pytest.approx(1.0)


def test_llm_criterion_without_scorer_raises():
    """An llm-typed criterion with no injected scorer is a configuration error."""
    candidates = [{"id": "a", "attributes": {}}]
    criteria = [{"name": "fit", "type": "llm", "weight": 1.0, "rubric": "rate fit"}]
    with pytest.raises(ValueError):
        ScoringService().score_all(candidates, criteria)


def test_llm_scorer_injection():
    """Injected LLMScorer is used for llm-typed criteria."""

    class StubScorer:
        def score(self, rubric, attributes):
            return 0.75, "stub rationale"

    candidates = [{"id": "a", "attributes": {"note": "x"}}]
    criteria = [{"name": "fit", "type": "llm", "weight": 1.0, "rubric": "rate fit"}]
    service = ScoringService(llm_scorer=StubScorer())
    scored = service.score_all(candidates, criteria)
    assert scored[0]["scores"]["fit"] == pytest.approx(0.75)
    assert scored[0]["total"] == pytest.approx(0.75)
    # F-02: the LLM-supplied rationale text is captured and adopted.
    assert scored[0]["rationales"]["fit"] == "stub rationale"
    assert "stub rationale" in service.rationale_for(scored[0], criteria)


def test_deterministic_rationale_is_rule_based():
    """numeric/boolean criteria yield a rule-based rationale (no rationales)."""
    candidates = [{"id": "a", "attributes": {"v": 1}}]
    criteria = [{"name": "v", "type": "numeric", "weight": 1.0}]
    service = ScoringService()
    scored = service.score_all(candidates, criteria)
    assert scored[0]["rationales"] == {}
    assert "v=" in service.rationale_for(scored[0], criteria)


def test_nan_attribute_does_not_flatten_the_ranking():
    """json.loads accepts the NaN literal, and NaN survives min()/max() clamping.

    Before this guard a single NaN attribute normalised EVERY candidate to 1.0 and
    the ranking collapsed, silently, on the deterministic core path.
    """
    candidates = json.loads('[{"id":"bad","attributes":{"v":NaN}},{"id":"good","attributes":{"v":10}}]')
    criteria = [{"name": "v", "type": "numeric", "weight": 1.0}]
    scored = {s["id"]: s for s in ScoringService().score_all(candidates, criteria)}
    assert scored["good"]["total"] == pytest.approx(1.0)
    assert scored["bad"]["total"] == pytest.approx(0.0)


def test_infinite_attribute_is_treated_as_missing():
    candidates = json.loads('[{"id":"inf","attributes":{"v":Infinity}},{"id":"n","attributes":{"v":4}}]')
    criteria = [{"name": "v", "type": "numeric", "weight": 1.0}]
    scored = {s["id"]: s for s in ScoringService().score_all(candidates, criteria)}
    assert scored["n"]["total"] == pytest.approx(1.0)
    assert scored["inf"]["total"] == pytest.approx(0.0)


def test_foreign_scorer_exception_is_reduced_to_its_type():
    """An injected scorer is foreign code; its message must not reach error_log."""

    class LeakyScorer:
        def score(self, rubric, attributes):
            raise ConnectionError("https://api.example.invalid token=sk-LEAKED12345")

    criteria = [{"name": "fit", "type": "llm", "weight": 1.0, "rubric": "r"}]
    with pytest.raises(ValueError) as excinfo:
        ScoringService(llm_scorer=LeakyScorer()).score_all([{"id": "a", "attributes": {}}], criteria)
    message = str(excinfo.value)
    assert "ConnectionError" in message
    assert "sk-LEAKED12345" not in message
    assert "api.example.invalid" not in message


def test_scorer_returning_a_nan_score_is_treated_as_zero():
    class NanScorer:
        def score(self, rubric, attributes):
            return float("nan"), "n"

    criteria = [{"name": "fit", "type": "llm", "weight": 1.0, "rubric": "r"}]
    scored = ScoringService(llm_scorer=NanScorer()).score_all([{"id": "a", "attributes": {}}], criteria)
    assert scored[0]["scores"]["fit"] == pytest.approx(0.0)


def test_missing_numeric_attribute_is_still_treated_as_zero():
    """Pre-existing contract: an absent attribute scores 0.0, it does not raise."""
    candidates = [{"id": "a", "attributes": {}}, {"id": "b", "attributes": {"v": 4}}]
    criteria = [{"name": "v", "type": "numeric", "weight": 1.0}]
    scored = {s["id"]: s for s in ScoringService().score_all(candidates, criteria)}
    assert scored["a"]["total"] == pytest.approx(0.0)
    assert scored["b"]["total"] == pytest.approx(1.0)


def test_unusable_numeric_value_behaves_exactly_like_a_missing_one():
    """NaN, a non-numeric string and an absent attribute must be indistinguishable.

    That equivalence — not a particular rank — is the contract. Under
    `lower_better` all three land at the favourable end, because the established
    policy maps an unusable value to 0.0 and 0.0 is then the best price. That
    direction-blindness predates this change and applies to a missing attribute on
    `develop` today; the point of this test is that NaN did not invent a fourth
    behaviour of its own.
    """
    criteria = [{"name": "v", "type": "numeric", "weight": 1.0, "direction": "lower_better"}]
    candidates = json.loads(
        '[{"id":"nan","attributes":{"v":NaN}},'
        ' {"id":"missing","attributes":{}},'
        ' {"id":"text","attributes":{"v":"n/a"}},'
        ' {"id":"real","attributes":{"v":50}}]'
    )
    scored = {s["id"]: s["total"] for s in ScoringService().score_all(candidates, criteria)}
    assert scored["nan"] == scored["missing"] == scored["text"]
    assert scored["real"] < scored["nan"]
