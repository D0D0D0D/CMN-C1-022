# CMN-C1-022 — ScoringRankingAgent

> **Category**: Cat 1 (single technical capability)
> **Industry**: CMN

## Overview

ScoringRankingAgent evaluates a collection of candidates against configurable weighted criteria
and returns a ranked list with a score and a concise rationale for each candidate. Criteria come
in three types — `numeric`, `boolean` and `llm` — and the default path uses no language model at
all: numeric and boolean criteria are scored deterministically, so the same input always produces
the same ranking.

It suits work where a set of things has to be compared under one explicit rule: prioritising leads
by budget and qualification, comparing proposals against a defined weighted rubric, or narrowing a
long candidate list down to a consistent top few before a person looks at it.

The division between what is computed and what is judged is the point of the design:

- **Numeric criteria are min-max normalised within the submitted candidate set.** A criterion on
  which every candidate carries the same value does not differentiate, so all candidates score
  `1.0` on it rather than being ranked by noise. Scores are relative to the batch, not absolute.
- **`llm`-typed criteria are opt-in and fail loudly.** No model client is bundled. A rubric that
  declares an `llm` criterion while no scorer is injected ends the invocation with an error naming
  the criterion — it does not silently fall back to a neutral score, which would produce a ranking
  that looks complete and is not.
- **Rationale text produced by a model is redacted before it leaves the agent.** Credential-shaped
  spans are replaced on the way out, and the output gate scans the ranking before it is returned.
- **The request is bounded before it is parsed.** Oversized payloads are refused by the input gate
  rather than being expanded into memory first.

The agent does not learn or tune weights from outcomes, does not decide whether a rubric is
appropriate for the decision at hand, and does not replace human judgment about the ranking it
produces.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Integration

Provide:

- an AgentCore runtime;
- caller authentication in front of the entry adapter;
- **only if the rubric uses `llm` criteria** — a scorer injected through `config["llm"]`. The
  constructor takes no bespoke keyword: `BaseGraph.__init__(self, config=None)` is the framework's
  only construction contract, so a keyword no caller passes would leave the branch behind it
  permanently unwired. Numeric and boolean rubrics need nothing here.

Runtime parameters read from `config/config.yaml` (defaults in parentheses):

| Key | Meaning |
|---|---|
| `max_retry` (`3`) | Maximum retry cycles for the graph backbone |
| `timeout_seconds` (`30`) | Execution timeout |

Fixed request bounds, enforced in code rather than configuration: at most **200,000** characters
of input, **1,000** candidates, **50** criteria, and **200** model scoring calls per invocation
(candidates × `llm` criteria) — the last one exists because an unbounded rubric could otherwise
turn one request into fifty thousand paid calls.

The template does not fetch candidates, maintain a knowledge base, persist rankings between
invocations, or hold any tenant/organisation scope model — every candidate and criterion arrives
in the request body.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent runs inside the platform runtime. Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Supplied by the platform's own package registry at build time. It is not published on public PyPI, so `pip install -e .` alone does not fetch it. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run on AGENTIC STAR, and the agent does not fall back to a degraded
mode when it is absent. Without the framework package installed, `src/api/server.py` cannot be
imported at all — it imports `framework.*` at module level, so start-up ends in `ImportError`
rather than in a partially working server. With the framework present but no model client
injected, the server does start: deterministic rubrics work normally, and a rubric containing an
`llm` criterion returns an error naming that criterion. The test suite runs without a platform
connection.

## Input

`POST /invoke` takes a JSON body with `input` and `session_id`. `input` is a **JSON string**
carrying `candidates`, `criteria` and the optional `top_k` and `tie_break`.

Callers must be authenticated. A request whose trust level is not stamped by verified upstream
middleware is rejected with **401** unless `INVOKE_AUTH_TOKEN` is set on the server environment
and the request carries the matching `Authorization: Bearer` header, which promotes the caller to
`VERIFIED_EXTERNAL`; a caller below verified trust is rejected with **403**. There is no anonymous
path into the graph — the entry node requires `VERIFIED_EXTERNAL` regardless.

```bash
curl -X POST http://127.0.0.1:8000/invoke \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${INVOKE_AUTH_TOKEN}" \
  -d '{
    "session_id": "ranking-demo-001",
    "input": "{\"candidates\":[{\"id\":\"lead-1\",\"attributes\":{\"budget\":100,\"qualified\":true}},{\"id\":\"lead-2\",\"attributes\":{\"budget\":50,\"qualified\":true}},{\"id\":\"lead-3\",\"attributes\":{\"budget\":0,\"qualified\":false}}],\"criteria\":[{\"name\":\"budget\",\"type\":\"numeric\",\"weight\":0.7,\"direction\":\"higher_better\"},{\"name\":\"qualified\",\"type\":\"boolean\",\"weight\":0.3}],\"top_k\":2}"
  }'
```

Input fails closed: a body that is not a JSON object, a missing or malformed `candidates` /
`criteria` list, or a payload past any of the bounds above ends the invocation as `status=error`
with the reason recorded in `error_log`. `GET /health` reports liveness only.

## Output

The standard AgentCore envelope is preserved. `output` carries the ranking:

```json
{
  "status": "success",
  "output": {
    "ranking": [
      {"rank": 1, "id": "lead-1", "score": 1.0,
       "rationale": "Top criteria: budget=1.00, qualified=1.00"},
      {"rank": 2, "id": "lead-2", "score": 0.65,
       "rationale": "Top criteria: budget=0.50, qualified=1.00"}
    ],
    "criteria_used": ["budget", "qualified"],
    "returned": 2,
    "total_candidates": 3
  }
}
```

`returned` is the number of rows after `top_k` is applied; `total_candidates` is how many were
submitted, so a truncated ranking is visible as such rather than looking like the whole set.
Values above are illustrative — real output depends on the candidates and rubric supplied.

## Security and Limitations

The entry node requires verified-external trust, so an anonymous caller is stopped before any node
body runs. Oversized input is refused at the input gate, before parsing. On the way out, the
output gate scans the ranking — including rationale text, which is the one field a model can
write — and credential-shaped spans in model output are redacted by the scorer before they reach
the gate.

Failure is never dressed up as success. An unparsable request, an out-of-bounds payload, and an
`llm` criterion with no scorer configured all end as `status=error`; none of them return a partial
ranking.

Known limits of the ranking itself:

- Scores are **relative to the submitted batch**. The same candidate can score differently in a
  different set, and a score is not a measure of absolute quality.
- Weights are taken as given. The agent does not validate that a rubric is well-posed, that its
  weights sum to anything in particular, or that the criteria are independent.
- `llm`-typed criteria inherit the judgment of whatever model is injected, including its
  inconsistency between runs. The deterministic types do not.

Output is machine-generated and assistive. It does not substitute for expert review, and any
decision that matters should be made by a person against the underlying evidence.

## Release

- Version: 1.0.0
- Category: single technical capability (Cat 1)
- Generation mode: `llm` (optional — the default rubric path is deterministic)
- HITL: not enabled
- Persistence: none — no checkpointer is configured
- Secrets: none required by the manifest
- External API clients: deployment-injected; none bundled

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself needs the framework package:
the standalone server in `src/api/server.py` loads `config/config.yaml`, passes it through
unchanged, and deliberately does **not** supply `config["llm"]` — a real model is supplied by the
platform entry adapter, or by a caller that constructs the graph with one.

## Documentation

`docs/02_design.md` carries the node flow, the state schema and the security design.
`docs/03_test_spec.md` records the test specification and the verification results.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and test specifications
```

See `docs/02_design.md` for the design and `docs/03_test_spec.md` for the test specification.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the knowledge sources and sample data with your own.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
