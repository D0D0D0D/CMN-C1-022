# テンプレート設計仕様 — CMN-C1-022 ScoringRankingAgent

## AgentCore アーキテクチャ上の位置づけ

- **Agent Class**: `ScoringRankingGraph`（`src/graph/graph.py`）
- **Marketplace entry adapter**: `cli.py` の `MarketplaceGraph`（信頼済み scope の再構築、HITL 抑止、呼び出し単位の Azure LLM 接続）
- **L1 Base**: AgentBaseGraph（Cat 1・固定 3 スロットパイプラインを直接継承）
- **三層分離**:
  - State: flat TypedDict 合成（Pydantic 不可 — msgpack 非互換）
  - Node: L1 継承（Template Method: `execute(self, state: dict) -> dict` のみ override）
  - Graph: 合成（`register_nodes()` でノード差し替え）

> 本テンプレートは「候補集合を重み付き基準で評価しランク付けする」**単一の汎用能力**を
> 提供する Cat 1 ユニット。業種知識はコードに持たず、ルーブリック・重みはすべて
> `config/config.yaml` ＋実行時入力で注入する。

## アーキテクチャ概要

### ノード構成

| Node | 責務 | Input State | Output State | Inherits/Overrides |
|------|---------------|-------------|--------------|-------------------|
| initialize | schema_version / session_id / caller_trust_level 設定 | — | （framework 既定） | InitializeNode (default) |
| pre_process | リクエスト解析（候補・基準・重み）と検証。空・不正・重み合計不正を弾く | `user_input`（JSON）/ `input_context` | `validated_input`(dict), `candidates`, `criteria`, `status` | PreProcessNode → `ValidateScoringRequestNode` |
| main | 各候補を基準ごとにスコアリング→重み付き合計→降順ソート（タイブレーク）→根拠付与 | `validated_input`, `candidates`, `criteria` | `scored_candidates`, `ranked_results`, `status` | MainNode → `ScoreAndRankNode` |
| post_process | 出力スキーマへ整形（rank/id/score/rationale）、`top_k` 切り詰め、根拠スクラブ | `ranked_results` | `formatted_output`, `status` | PostProcessNode → `FormatRankingNode` |
| finalize | response_metadata / total_time_ms 構築 | — | （framework 既定） | FinalizeNode (default) |

### データフロー

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                       ↓ (retry, max 3)
                                    pre_process
```

- `pre_process → main` は無条件辺。`route()` は **main の後**に `state["status"]` を読む。
  各ノードは `status` に **`AgentStatus` enum メンバー**（`AgentStatus.SUCCESS` / `AgentStatus.RETRY` /
  `AgentStatus.ERROR`、値は小文字 `success`/`retry`/`error`）を返し、`route()` が `AgentStatus(value)` で
  解決して分岐する（SUCCESS→post_process / RETRY→pre_process / ERROR・その他→finalize）。
  生の文字列は返さない。`add_edges()`/`route()` は override しない。
- 入力検証エラー（候補ゼロ等）は pre_process が `AgentStatus.ERROR` を返し、main/post はそれをパススルー → finalize。

**ERROR パススルー契約（実装責任・必須）:**
`pre_process → main` は無条件辺のため、入力検証エラーでも main は実行される。各ドメインノード
（`ScoreAndRankNode` / `FormatRankingNode`）は **execute 冒頭で `state.get("status") == AgentStatus.ERROR`
を検知したら、処理をスキップして変更なしの空 dict（または `{"status": AgentStatus.ERROR}`）を返す
early-return を実装する**。これにより空 `candidates`/`criteria` での誤動作を防ぎ、`route()` が
ERROR を finalize へ送る。pre_process 自身は `AgentStatus.RETRY` を返すと backbone が pre_process を
再実行する（最大 max_retry=3）。

### State 定義

`State(AgentState)` に以下を追加（共有フィールド user_input / status / session_id /
node_history / error_log 等は AgentState から継承）:

| Field | Type | Purpose | Required |
|-------|------|---------|----------|
| `candidates` | `list[dict]` | 評価対象の候補集合。各要素 `{ "id": str, "attributes": dict }` | ✅（pre_process が設定） |
| `criteria` | `list[dict]` | スコアリング基準。各要素 `{ "name": str, "type": "numeric|boolean|llm", "weight": float, ... }` | ✅（pre_process が設定） |
| `validated_input` | `dict` | 解析・検証済みリクエスト（`top_k`, `tie_break` 等のオプション含む） | ✅ |
| `scored_candidates` | `list[dict]` | 候補ごとの基準別スコア `{ "id", "scores": {criterion: float}, "total": float }` | main が設定 |
| `ranked_results` | `list[dict]` | 降順ソート済み `{ "rank", "id", "score", "rationale" }` | main が設定 |
| `formatted_output` | `dict` | 最終整形出力（`{ "ranking": [...], "criteria_used": [...] }`） | post_process が設定 |

**State 制約（必須）:**
- flat TypedDict のみ（primitive ＋ JSON-serializable）。候補/基準は dict/list の素の構造で保持。
- JWT・API キー・credential を State に置かない（checkpoint DB 漏洩防止）。
- InvocationContext は `config["configurable"]` 経由のみ（State に入れない）。
- Pydantic / dataclass / 任意 Python オブジェクト不可（msgpack 非互換）。

### スコアリングエンジン（ScoringService）

`src/services/service.py` の `ScoringService` が基準別スコアの算出を担う（business logic は
main ノードが所有、Service は純粋な算出ユニット）。基準 `type` を 3 種サポート:

| type | 算出方法 | LLM 依存 |
|------|---------|---------|
| `numeric` | 候補 attribute の数値を min-max 正規化（0.0–1.0）。`direction: higher_better/lower_better` 対応。数値化できない値（欠損・非数値文字列・`NaN`／`Infinity`）は既存方針どおり `0.0` として扱う（`json.loads` は `NaN` リテラルを受理するため、正規化前に落とさないと全候補が 1.0 に潰れる） | なし（決定的） |
| `boolean` | 候補 attribute の真偽を 1.0 / 0.0 に写像 | なし（決定的） |
| `llm` | ルーブリック文＋候補 attribute を LLM に渡し 0.0–1.0 のスコア＋短い根拠を得る | あり（注入シーム経由） |

- **総合スコア** = Σ(criterion.weight × normalized_score) / Σ(weight)（重み正規化）。
- **タイブレーク**: 同点時は `tie_break` 指定基準のスコア、なければ候補 `id` 昇順で安定ソート。
- **根拠（rationale）**: 各候補について上位寄与基準を要約（`numeric/boolean` は規則ベース、
  `llm` は LLM 出力を採用）。Cat 1 既定は LLM 非依存（`numeric/boolean` のみ）で完全決定的に動作し、
  `llm` 基準を含む場合のみ LLM クライアントを注入する（デフォルトは未設定でも numeric/boolean で成立）。

#### LLM 注入シーム（`config["llm"]`）

LLM は **`config["llm"]` 経由でのみ**受け取る。framework の構築契約は
`BaseGraph.__init__(self, config=None)` だけであり、`register_nodes()` がそこから読んで
main ノードへ渡す。

注入されるのは `BaseLLM`（`complete(messages) -> dict`）で、本 AG のノードが呼ぶのは
`LLMScorer.score(rubric, attributes) -> (float, str)` である。この差は
`src/services/llm_client.py` の `BaseLLMScorer` が吸収する（**`src/` に置く** —
テスト側にだけ置くとテストだけが緑になり本番にアダプタが無い）。`config["llm"]` に
`score()` を持つ実体（Cat 2 派生の独自スコアラ）が入っていればそのまま採用する。

| 状況 | 挙動 |
|---|---|
| 未注入（`config["llm"]` なし） | `numeric`/`boolean` は完全決定的に成立。`llm` 基準は設定エラー（ERROR） |
| 注入あり・`llm` 基準なし | LLM は 1 度も呼ばれない |
| 注入あり・`llm` 基準あり | 候補 × `llm` 基準の回数だけ `complete()` を呼ぶ |
| LLM 呼び出しが失敗 | **縮退せず ERROR**。ルーブリック判断に決定論的代替は無く、捏造したスコアで順位を出す方が悪い |
| `config["llm"]` が LLM でもスコアラでもない値 | `compile()` 時点で `ConfigError`。黙って捨てると「シームはあるのに何も届かない」状態に戻る |

- **失敗理由の既定は例外の型名のみ**。プロバイダの例外文はエンドポイント・トークン・
  依頼文の反射を含みうるため State に入れない。例外は下記「注入スコアラの例外契約」で
  秘匿済みを宣言したスコアラの `LLMScoringError` だけで、これはテンプレート所有の文言
  （`LLM response is not valid JSON` 等）に限られる。
- **モデル ID も具体クライアントもテンプレートに書かない**。model / temperature / max_tokens は
  注入されたクライアントが保持する。
- **コスト上限**: `llm` 基準は候補 1 件につき 1 回の課金呼び出しになる。`AgentBaseGraph` に
  `budget_usd` 相当が無いため、pre_process が受理時点で
  `候補数 × llm基準数 ≤ MAX_LLM_SCORE_CALLS` を強制する。

##### S-3 の姿勢（IMP-143）— rationale に資格情報が混じった場合

wheel 1.0.1 の S-3 は result の全 string leaf を**再帰的に**走査し、一致すると
`RuntimeError` で**打ち切る**（fail-closed）。しかもこれは `_extra_security_gate_output()`
より**前**に走る。したがって post_process の秘匿フックは、main ノードが資格情報入りの
rationale を返した実行には**到達しない**。本テンプレートの姿勢:

| 経路 | 姿勢 |
|---|---|
| `config["llm"]` に `BaseLLM` を注入（AGENTIC STAR 既定） | **produce 側で秘匿**。`BaseLLMScorer._parse_score()` が `framework.security.detect_credentials` で span を置換してから返すので、S-3 は綺麗な出力を見て通る＝**秘匿済み SUCCESS** |
| `config["llm"]` に派生テンプレ独自の `score()` 実体を注入 | **BLOCK**。その実体が自分で秘匿しない限り S-3 が run 全体を ERROR にする。秘匿責任は実体の所有者にある |

post_process の `_extra_security_gate_output()` は多層防御として残すが、
**端から端まで到達する経路ではない**（`docs/03` の主張もこの範囲に留める）。

##### 注入スコアラの例外契約

`LLMScoringError` は「送出者がメッセージを秘匿済みだと宣言する」型であり、クラスだけでは
保証にならない（export されているので誰でも送出できる）。`ScoringService` は
`_sanitizes_errors = True` を宣言したスコアラの `LLMScoringError` だけを本文ごと公開し、
それ以外の例外は**型名だけ**に落とす。`BaseLLMScorer` はこのフラグを宣言している。

## フレームワーク活用

### 利用する共有コンポーネント
- [x] InvocationContext（correlation_id, session_id, permissions, credential handle）
- [x] ConnectionPolicy（LLM 基準利用時の retry/timeout）
- [x] SecurityViolationError
- [x] **S-2**: `_extra_security_gate_input()` — `ValidateScoringRequestNode` で
      入力サイズ上限を強制（候補数 ≤ `max_candidates`、基準数 ≤ `max_criteria`）。
      既定 PII スキャン後に走る追加チェック。**`_security_gate_input()` は override しない**（`@final`）。
- [x] **S-3**: `_extra_security_gate_output()` — `FormatRankingNode` で
      rationale 文字列の credential 再スキャン（既定 credential スキャン後の追加チェック）。
      **`_security_gate_output()` は override しない**（`@final`）。
- [x] **S-4**: `emit_trace_event()` — 各 `execute()` 内で最低 1 件のドメインイベントを発火
      （pre: `scoring_request_validated` / main: `candidates_ranked` / post: `ranking_formatted`）。
      `node_start`/`node_complete`/`node_error` は `BaseNode.__call__()` が自動発火するため出さない。

> **S-2/S-3 のノード種別による挙動（ADR-017）:**
> - 本テンプレートのドメインノードは `FunctionNode` サブクラス → framework `@final` ゲートが
>   常に自動実行され、`_extra_security_gate_input()` / `_extra_security_gate_output()` で拡張する。

### 合成パターン

- **Pattern**: Standalone（単一 AgentBaseGraph・サブグラフ/リモート合成なし）
- **Composition target**: なし
- **Error propagation strategy**: propagate（pre_process の `status="error"` を main/post がパススルー → finalize）

## Import Isolation 確認
- [x] Level 0 SDK（agenticstar-platform）を import しない
- [x] import 先は `framework/` と `shared/`（＋ローカル `src.`）のみ。`agents/base/` 継承は不要

## 設計判断記録（Design Decision Record）

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| L1 base type | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | 候補評価→ランク付けは固定段（検証→採点→整形）で完結し、自律ループ不要。Cat 1 の標準 3 スロット backbone が最適 |
| 合成パターン | Standalone | GraphNode subgraph | **Standalone** | 単一能力で内部に複数ドメインワークフローを持たない。Cat 2 派生が config 差し替えで構成 |
| スコアリング既定 | 決定的（numeric/boolean） | LLM 必須 | **決定的を既定・LLM は任意** | Cat 1＝ゼロドメイン仮定・テスト容易・再現性。`llm` 基準は `config["llm"]` 注入シームで opt-in |
| LLM の受け口 | 外側 Graph の独自 kwarg | `config["llm"]` | **`config["llm"]`** | framework の構築契約は `BaseGraph.__init__(config=None)` のみ。独自 kwarg は呼び出し側が渡さず永久に未配線になる |
| LLM 失敗時 | 縮退して success | ERROR | **ERROR** | ルーブリック判断に決定論的代替が無い。捏造スコアで順位を出す方が有害 |
| 入力チャネル | `user_input`（JSON文字列） | 専用 schema | **`user_input` JSON ＋ `input_context`** | backbone 契約に整合。pre_process で parse/validate |

## 既知の割り切り（MVP スコープ）

- 既定実装は `numeric`/`boolean` 基準による完全決定的スコアリング。`llm` 基準は
  `config["llm"]` に注入された `BaseLLM` で成立する。テンプレートは LLM プロバイダ実体も
  モデル ID も同梱せず、`BaseLLM` → `LLMScorer` のアダプタだけを持つ（`src/services/llm_client.py`）。
- 学習型（過去実績からの重み自動調整）は対象外 — 重みは設定で与える静的値。
- ランキング説明は寄与基準ベースの簡潔な根拠。詳細な反実仮想説明（why-not）は対象外。
