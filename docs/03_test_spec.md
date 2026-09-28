# Test Specification — CMN-C1-022 ScoringRankingAgent

## Test Strategy
- Coverage target: ノード・サービスの分岐を網羅（unit）＋ full invoke chain（integration）
- Test types: Unit（ノード/サービス）/ Integration（invoke 経由）/ Proof-of-Boundary
- 実行: `python -m pytest tests/`（ローカル 27 件 PASS 済み）

## Framework Compliance Tests (Mandatory)

| TC-ID | Test | Expected Result | Result |
|-------|------|----------------|--------|
| TC-01 | State contract: flat TypedDict | PB-2 AST scan pass, no Pydantic/dataclass | ✅ |
| TC-02 | 不正入力で ERROR を返す（例外送出ではなく status） | ERROR status surfaced | ✅ |
| TC-03 | No JWT/Credential in State | CI `gate-credential-scan`: 0 violations | ✅ |
| TC-04 | InvocationContext via configurable only | State に持たない | ✅ |
| TC-05 | S-4: no duplicate lifecycle events in `execute()` | `node_start`/`node_complete`/`node_error` を execute で出さない | ✅ |
| TC-06 | S-2: `_security_gate_input()` not overridden | `_extra_security_gate_input()` のみ使用 | ✅ |
| TC-07 | S-3: `_security_gate_output()` not overridden | `_extra_security_gate_output()` のみ使用 | ✅ |
| TC-08 | `required_trust_level` declared and enforced — dedicated S-1 boundary coverage MUST call `node(state)` (via `__call__`) for every privileged node, or use equivalent parameterized coverage | Insufficient trust level → execution refused *before* `execute()`; test must prove `execute()` was not called. Direct `node.execute(state)` is permitted only for isolated business-logic unit tests: it does not satisfy TC-08, but is not a finding when TC-08 and PB-6 coverage pass. | ✅ |
| TC-09 | S-2: `_extra_security_gate_input()` non-trivial | 入力サイズ上限チェック（test_pre_process_node） | ✅ |
| TC-10 | S-3: `_extra_security_gate_output()` non-trivial | rationale の credential 再スキャン（test_post_process_node） | ✅ |
| TC-11 | S-4: ≥1 domain `emit_trace_event()` per execute | pre/main/post 各 execute で発火 | ✅ |

## Proof-of-Boundary Tests (Mandatory)

| PB-ID | Boundary | Test | Expected Result | Result |
|-------|----------|------|----------------|--------|
| PB-1 | BaseNode → EventEmitter | `emit_trace_event()` fires on every invocation path | No silent failures | ✅ |
| PB-2 | State serialization | Post-invoke State is primitives only | No Pydantic/dataclass（test_state_safety） | ✅ |
| PB-3 | External service | **本テンプレ既定（numeric/boolean）は外部サービス非依存** = 接続テスト N/A。`llm` 基準を使う派生時のみ、注入された LLMScorer 実体への接続確認を行う（その派生テンプレで実施） | 分岐明記（F-03） | ✅ (N/A for default) |
| PB-4 | Import isolation | No Level 0 imports | AST scan: 0 violations（test_import_isolation） | ✅ |
| PB-5 | Checkpoint safety | No JWT/Pydantic in checkpoint | Inspection pass | ✅ |
| PB-6 | Invoke execution order | `__call__()`: S-1 → S-4 node_start → S-2 → execute → S-3 → S-4 node_complete | framework 保証（ANONYMOUS 拒否で S-1 先行を確認） | ✅ |

> **PB-3 補足（F-03 対応）**: ScoringRankingAgent の既定パス（`numeric`/`boolean` 基準）は外部サービスへ
> 一切接続しない完全決定的処理。「外部サービス接続」テストは `llm` 基準を有効化した派生テンプレートに
> 限り発生し、その場合は注入された `LLMScorer` プロバイダへの接続確認として実施する。

## Business Logic Tests

| BL-ID | Test | Input | Expected Result | Result |
|-------|------|-------|----------------|--------|
| BL-01 | 重み正規化計算 | candidates=[a(budget100,qualified),b(budget0,not)], criteria=[budget w0.6 higher, qualified w0.4] | a.total=1.0 / b.total=0.0（Σ(w×norm)/Σw） | ✅ test_scoring_service::test_weight_normalization_bl01 |
| BL-02 | タイブレーク（同点→id 昇順） | 同 total の候補 b,a | ranking=[a,b] | ✅ test_main_node::test_tie_break_by_id_asc |
| BL-03 | top_k 切り詰め | ranked 2件 + top_k=1 | returned=1, top のみ, total_candidates=2 | ✅ test_post_process_node::test_top_k_truncation_bl |
| BL-04 | 候補ゼロ → ERROR | candidates=[] | status ERROR（ranking なし） | ✅ test_pre_process_node::test_empty_candidates_error + integration |
| BL-05 | numeric lower_better 反転 | risk: a=10,b=0 lower_better | b.total=1.0 / a.total=0.0 | ✅ test_scoring_service::test_numeric_lower_better |
| BL-06 | 二次基準タイブレーク | total 同点 + tie_break=freshness | freshness 高い方が上位 | ✅ test_main_node::test_tie_break_by_secondary_criterion |
| BL-07 | llm 基準・scorer 未注入 → ERROR | criterion type=llm, scorer なし | ValueError → ERROR | ✅ test_scoring_service::test_llm_criterion_without_scorer_raises |

## Test Execution Summary
- Execution date: 2026-06-23
- Total tests: 27（unit 24 + integration 3）
- Pass: 27 / Fail: 0 / Skip: 0
- Notes: ローカル venv（Python 3.11 + agenticstar-agentcore==1.0.0）で全 PASS。CI（run-tests）でも同等を確認。

<!-- LIVE-LLM:BEGIN (generated — 手書きはこのブロックの外へ) -->
## 実 LLM 1 回転検証

モックだけで固めた実装は実 LLM を繋いだ瞬間に動かないことがあるため、注入シームへ実 LLM クライアントを刺して 1 回 invoke し、**中身のある出力が出ること**を確認している。

| 項目 | 内容 |
|---|---|
| 実施日 | 2026-08-29 |
| 使用 LLM | Azure OpenAI 上の GPT モデルを `BaseLLM` 実装として注入 |
| 注入シーム | `config[llm]` |
| 実 LLM 呼び出し | 3 回 |
| 出力 | ['ranking', 'criteria_used', 'returned', 'total_candidates'] が非空。縮退理由なし |

本番の LLM は `AnthropicClient` 等で、これは検証用シーム。テンプレの `src/` は変更していない（一時テストを生成→実行→削除する方式）。CI には入れていない（外部 LLM の鍵が要るため）。
<!-- LIVE-LLM:END -->

### 業務入力での確認（2026-08-29）

汎用ペイロードは `numeric`/`boolean` だけで成立してしまい、LLM に一度も到達しないため
**配線済みビルドと未配線ビルドを区別できない**。そこで `tests/conftest.py` の
`case_payload` を、この AG の本来の入力 — 航空宇宙アクチュエータ筐体の機械加工サプライヤ
3 社の絞り込み（`unit_price_jpy` 数値・`iso9001` 真偽・`profile` を `llm` ルーブリックで判断）
— に置き換えて回した。`llm` 基準は attribute の一致では答えられない判断であり、
実 LLM でなければ埋まらない。

結果（候補 3 社 = LLM 呼び出し 3 回・`top_k=2`）:

| rank | id | total | `profile`（LLM）の判断 |
|---|---|---|---|
| 1 | sup-A | 0.8497 | 0.94 — 航空宇宙アクチュエータ筐体の実績・自社 CMM 検査・フライト部品の顧客監査通過 |
| 2 | sup-C | 0.3680 | 0.28 — ISO 9001 と初品検査はあるが航空宇宙実績なし |
| 圏外 | sup-B | — | 0.00 — 金属加工能力・検査設備・認証のいずれも無し |

`sup-B`（消費財の樹脂成形）が単価最安（800 円）にもかかわらず圏外に落ちること、
`sup-C` が認証を持ちながら 2 位に留まることは、**ルーブリック判断が実際に順位を
決めている**ことを示す（決定論部分だけなら `sup-B` が単価で優位になる）。
入力は `tests/conftest.py` に残してあるので同じ検証を再実行できる。

