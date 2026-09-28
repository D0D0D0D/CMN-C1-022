"""AgentCore Platform v1.0 — CMN-C1-022 scoring engine.

Service layer: pure scoring computation. No business routing, no credentials,
no framework/state coupling. The main node owns orchestration; this owns the
deterministic math (and the optional LLM rubric seam).
"""

from __future__ import annotations

import math
from typing import Any, Protocol


VALID_CRITERION_TYPES = ("numeric", "boolean", "llm")


class LLMScoringError(ValueError):
    """An ``llm``-typed criterion could not be scored.

    Subclasses ``ValueError`` so it travels the path the service layer already
    uses for a misconfigured ``llm`` criterion and the main node reports it as
    ERROR. There is no deterministic substitute for a rubric judgement, so a
    failed model call must never be degraded into an invented score.

    Messages of this type are authored by this template and are safe to surface;
    every other exception raised by an injected scorer is reduced to its type
    name before it can reach ``error_log`` (see ``_score_one``).
    """


class LLMScorer(Protocol):
    """Injection seam for ``llm``-typed criteria (opt-in).

    A deriving Cat 2 template supplies a concrete implementation. The default
    Cat 1 template ships none, so ``numeric``/``boolean`` criteria work with no
    LLM dependency at all.
    """

    def score(self, rubric: str, attributes: dict[str, Any]) -> tuple[float, str]:
        """Return (score in 0.0–1.0, short rationale) for one candidate."""
        ...


class ScoringService:
    """Compute weighted, normalized scores for a pool of candidates."""

    def __init__(self, llm_scorer: LLMScorer | None = None) -> None:
        # Optional — required only when a criterion has type == "llm".
        self._llm_scorer = llm_scorer

    # ── public API ──────────────────────────────────────────────────────────

    def score_all(self, candidates: list[dict], criteria: list[dict]) -> list[dict]:
        """Score every candidate against every criterion.

        Returns a list of ``{"id", "scores": {criterion: float}, "total": float}``
        in the same order as ``candidates``. ``total`` is the weight-normalized
        sum of per-criterion scores (each already in 0.0–1.0).

        Raises ``ValueError`` if an ``llm`` criterion is present but no LLM
        scorer was injected (surfaced as ERROR by the calling node).
        """
        weight_sum = sum(float(c.get("weight", 0.0)) for c in criteria) or 1.0

        # Pre-compute per-criterion normalization ranges for numeric criteria.
        numeric_ranges = {c["name"]: self._numeric_range(candidates, c) for c in criteria if c.get("type") == "numeric"}

        scored: list[dict] = []
        for cand in candidates:
            attrs = cand.get("attributes", {})
            per_criterion: dict[str, float] = {}
            rationales: dict[str, str] = {}
            weighted = 0.0
            for crit in criteria:
                name = crit["name"]
                weight = float(crit.get("weight", 0.0))
                s, rationale = self._score_one(crit, attrs, numeric_ranges.get(name))
                per_criterion[name] = round(s, 4)
                if rationale:
                    rationales[name] = rationale
                weighted += weight * s
            scored.append(
                {
                    "id": cand.get("id"),
                    "scores": per_criterion,
                    "rationales": rationales,  # populated only for llm criteria
                    "total": round(weighted / weight_sum, 4),
                }
            )
        return scored

    def rationale_for(self, scored: dict, criteria: list[dict], top_n: int = 2) -> str:
        """Build a short rationale citing the highest-contributing criteria.

        For ``llm`` criteria the LLM-supplied rationale text is adopted (per
        design); ``numeric``/``boolean`` criteria use a rule-based summary.
        """
        weights = {c["name"]: float(c.get("weight", 0.0)) for c in criteria}
        contributions = sorted(
            scored["scores"].items(),
            key=lambda kv: weights.get(kv[0], 0.0) * kv[1],
            reverse=True,
        )
        top = contributions[: max(1, top_n)]
        rationales = scored.get("rationales", {})
        parts = []
        for name, score in top:
            if name in rationales:
                parts.append(f"{name}: {rationales[name]}")  # adopt LLM rationale
            else:
                parts.append(f"{name}={score:.2f}")
        return "Top criteria: " + ", ".join(parts)

    # ── internals ───────────────────────────────────────────────────────────

    def _score_one(
        self, crit: dict, attrs: dict, numeric_range: tuple[float, float] | None
    ) -> tuple[float, str | None]:
        """Return (score in 0.0–1.0, optional rationale text).

        Rationale text is only produced by ``llm`` criteria; deterministic
        criteria return None and the caller falls back to a rule-based summary.
        """
        ctype = crit.get("type")
        name = crit["name"]
        if ctype == "numeric":
            raw = _as_float(attrs.get(name))
            lo, hi = numeric_range if numeric_range else (0.0, 0.0)
            if hi == lo:
                # No spread across candidates → criterion does not differentiate.
                return 1.0, None
            norm = (raw - lo) / (hi - lo)
            if crit.get("direction", "higher_better") == "lower_better":
                norm = 1.0 - norm
            return _clamp01(norm), None
        if ctype == "boolean":
            return (1.0 if attrs.get(name) else 0.0), None
        if ctype == "llm":
            if self._llm_scorer is None:
                raise ValueError(
                    f"criterion '{name}' is type 'llm' but no LLM scorer is "
                    f"configured (inject one or use numeric/boolean criteria)"
                )
            try:
                score, rationale = self._llm_scorer.score(crit.get("rubric", ""), attrs)
                clamped = _clamp01(_as_float(score))
                text = "" if rationale is None else str(rationale)
            except LLMScoringError:
                # Only a scorer that declares _sanitizes_errors asserts its message
                # is free of provider text. LLMScoringError is exported, so an
                # injected scorer can raise it too; class alone is not a warrant.
                #
                # This is default-deny against ACCIDENTAL disclosure by a scorer
                # that never considered the question — not a boundary against
                # hostile in-process code, which could equally return a secret as
                # the rationale. Whoever injects a scorer already trusts it.
                #
                # The lookup itself is guarded: on a scorer with __getattr__ or a
                # property, getattr() runs foreign code, and an exception raised
                # there would escape these except clauses into the framework's own
                # error path with its message intact.
                try:
                    warranted = getattr(self._llm_scorer, "_sanitizes_errors", False) is True
                except Exception:  # noqa: BLE001 — a failed warrant is no warrant
                    warranted = False
                if warranted:
                    raise
                raise ValueError(f"criterion '{name}': LLM scorer failed (LLMScoringError)") from None
            except Exception as exc:
                # An injected scorer is foreign code. Its exception message can
                # carry an endpoint, a token, or the request echoed back, and the
                # entry adapter surfaces error_log to the caller, so only the
                # exception TYPE may travel.
                raise ValueError(f"criterion '{name}': LLM scorer failed ({type(exc).__name__})") from None
            return clamped, text
        raise ValueError(f"unknown criterion type: {ctype!r}")

    @staticmethod
    def _numeric_range(candidates: list[dict], crit: dict) -> tuple[float, float]:
        values = [_as_float(c.get("attributes", {}).get(crit["name"])) for c in candidates]
        return (min(values), max(values)) if values else (0.0, 0.0)


def _as_float(value: Any) -> float:
    """Coerce to float, mapping anything unusable to 0.0.

    NaN and infinity are unusable in the same sense as a missing attribute and
    are rejected here rather than left to ``_clamp01()``: ``min()``/``max()``
    pass NaN through untouched, so a single NaN attribute would normalise every
    candidate to a perfect 1.0 and flatten the ranking. ``json.loads`` accepts
    the literals ``NaN``, ``Infinity`` and ``-Infinity``, so this is reachable
    from a plain request body.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
