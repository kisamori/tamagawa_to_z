# Scripts Documentation

このディレクトリには、地名データから考古学的な遺跡候補地を発見・評価するための一連の分析を実行するスクリプトが格納されています。

## ワークフロー概要

このプロジェクトは、大きく分けて以下の4つのステップで構成されています。各スクリプトは、それぞれのステップで特定の役割を担う専門家のように機能します。

1.  **準備・辞書づくり (Preparation & Dictionary Building)**
    - 分析の基礎となるデータや、地名を解釈するための辞書を作成します。
2.  **候補地を探す (Candidate Exploration)**
    - 地理空間データと地名を照合し、遺跡の可能性がある場所を探索します。
3.  **候補地を評価・分析する (Candidate Evaluation & Analysis)**
    - AIや機械学習を用いて、見つかった候補地をスコアリングし、有望な順にランク付けします。
4.  **全体を効率化・自動化する (Workflow Automation & Optimization)**
    - 一連の分析フローを統括したり、最適な分析条件を自動で探索したりします。

---

## 1. 準備・辞書づくり (Preparation & Dictionary Building)

### `run_root_extraction.py`

#### 役割
**【言葉の辞書を作る人】**
地名の中から「川」「水辺」など、特定のカテゴリ（水、地形など）に関連する言葉（語根）を見つけ出し、分析の基礎となる単語リストを作成・更新します。

#### 主な機能
- OSM (OpenStreetMap) データから地名を収集します。
- AI (LLM) を使って収集した地名を分析し、新しい語根の候補を発見します。
- 既存の語根辞書 (`data/dict/*.csv`) に新しい語根を自動または手動で追加・更新します。
- カテゴリ別の語根辞書を `all_roots.csv` に統合する機能も持ちます。

#### 使用方法
- **基本実行（Acre地域の水関連語根を抽出）**
  ```bash
  python scripts/run_root_extraction.py --region acre --visualize
  ```
- **カテゴリ別CSVを統合して `all_roots.csv` を作成**
  ```bash
  python scripts/run_root_extraction.py --create-all-roots
  ```
- **主な引数**
  - `--region`: 対象地域 (`acre`, `marajo`など) を指定します。
  - `--bbox`: 対象のBBox (座標範囲) を手動で指定します。
  - `--pbf-path`: 使用するOSMのPBFファイルパスを指定します。
  - `--output-dir`: 結果を出力するディレクトリを指定します。
  - `--sample-size`: LLM分析にかける地名のサンプル数を指定し、コストを削減します。
  - `--visualize`: 処理過程で生成される地図などの可視化画像を保存します。
  - `--create-all-roots`: 他の処理は行わず、カテゴリ別CSVを `all_roots.csv` にマージします。

### `run_split.py`

#### 役割
**【データ整理の専門家】**
既知の遺跡データを、機械学習モデルが公平かつ正確に学習・評価できるよう、「訓練用」「検証用」「テスト用」といったグループに分割します。

#### 主な機能
- 遺跡データを、発見年や地理的な位置に基づいて分割します。
  - **Train:** モデルの学習用データ
  - **Validation:** 学習中のモデルの性能評価用データ
  - **Test-time:** 時間的に新しいデータ（未来のデータに対する性能評価用）
  - **Test-region:** 地理的に異なる地域のデータ（未知の地域に対する汎化性能評価用）
- 分割したデータを、それぞれ個別のGISファイル (`.gpkg`) として出力します。

#### 使用方法
- **基本実行**
  ```bash
  python scripts/run_split.py --config configs/dataset_split.yaml --sites data/known/known_acre.kmz --output data/known/split
  ```
- **主な引数**
  - `--config`: 分割のルールを定義した設定ファイル (`.yaml`) のパス。
  - `--sites`: 分割対象となる遺跡データ (`.kmz`, `.csv`, `.gpkg`) のパス。
  - `--output`: 分割後のファイルを出力するディレクトリのパス。
  - `--dry-run`: 実際にはファイルを出力せず、分割結果の統計情報のみを表示します。

---

## 2. 候補地を探す (Candidate Exploration)

### `run_site_identification.py`

#### 役割
**【候補地探しのメイン担当】**
「**今は川がないのに、名前に『川』と付く場所**」を探し出します。このような場所は、昔の川の跡（古河道）である可能性が高く、遺跡候補地となります。

#### 主な機能
- 地名データ、現在の河川データ (HydroRIVERS)、過去の水域存在確率データ (GSW) を統合的に分析します。
- 川からの距離や水域の頻度に基づいて候補地をスコアリングし、リストアップします。
- Optunaによる最適化の対象となる、コアな分析パイプラインです。

#### 使用方法
- **基本実行（Acre地域）**
  ```bash
  python scripts/run_site_identification.py --region acre --visualize
  ```
- **パラメータを指定して実行**
  ```bash
  python scripts/run_site_identification.py --region acre --dist-threshold 2.5 --occ-threshold 15.0
  ```
- **主な引数**
  - `--region`: 対象地域 (`acre`, `marajo`など) を指定します。
  - `--rivers-path`, `--gsw-path`, `--pbf-path`: データパスを個別に指定します。
  - `--output-path`: 最終的な候補地リスト (`.csv`) の出力先を指定します。
  - `--dist-threshold`: 「川からこれ以上離れている」という距離のしきい値 (km)。
  - `--occ-threshold`: 「水があった確率がこれ以下」という水域頻度のしきい値 (%)。
  - `--visualize`: 候補地の分布図などを画像として保存します。

### `run_analyze_site.py`

#### 役割
**【既知遺跡の分析官】**
すでに見つかっている遺跡の周辺では、地名がどのようなパターン（距離や方角）で分布しているかを詳しく分析します。これは、候補地のパターンが「遺跡らしいか」を判断するための基準となります。

#### 主な機能
- 既知の遺跡周辺の地名をOSMから抽出します。
- 抽出した地名を、遺跡を中心とした極座標（距離と角度）に変換します。
- 地名と最寄りの川との距離を計算します。
- 分析結果をCSVファイルに出力し、オプションで可視化も行います。

#### 使用方法
- **基本実行**
  ```bash
  python scripts/run_analyze_site.py --region acre --radius 5.0 --visualize
  ```
- **主な引数**
  - `--region`: 分析対象の地域 (`acre`, `marajo`など) を指定します。
  - `--radius`: 各遺跡から地名を検索する半径 (km) を指定します。
  - `--visualize`: 分析結果（地名分布図、極座標プロットなど）を画像として保存します。
  - `--similarity-analysis`: 抽出したデータを使って、続けて類似度分析を実行します。

---

## 3. 候補地を評価・分析する (Candidate Evaluation & Analysis)

### `run_similarity_analysis.py`

#### 役割
**【AI鑑定士（スコア付け）】**
機械学習を使い、見つけた候補地が「既知の遺跡の地名パターンとどれくらい似ているか」を**点数（スコア）付け**し、有望な順にランキングを作成します。さらにAIが「なぜ似ているのか」という理由も解説してくれます。

#### 主な機能
- 既知遺跡の地名分布から特徴量（距離、密度、方角など）を生成します。
- 機械学習モデル (kNN, クラスタリング等) を構築し、候補地と既知遺跡との類似度をスコアリングします。
- 候補地を類似度の高い順にランキングし、結果をCSVやKMZ形式で出力します。
- OpenAIのLLMを使い、各候補地の類似性の根拠を文章で自動生成します。

#### 使用方法
- **基本実行（候補地のランキングを作成）**
  ```bash
  python scripts/run_similarity_analysis.py --region acre --mode candidate_ranking
  ```
- **主な引数**
  - `--region`: 分析対象の地域 (`acre`, `marajo`など) を指定します。
  - `--mode`: `candidate_ranking` (候補地を評価) または `similarity_only` (既知遺跡間の類似度のみ評価) を選択します。
  - `--output-dir`: レポートやランキング結果を出力するディレクトリを指定します。
  - `--config`: 地域ごとのデータパスを定義した設定ファイル (`.yaml`) を指定します。

### `run_inspector.py`

#### 役割
**【AI監査官】**
分析結果全体をチェックし、「この分析はうまくいっているか？」「もっと精度を上げるには、どのパラメータを調整すればいいか？」といった**改善案を提案**します。

#### 主な機能
- 候補地データと既知の遺跡データを比較し、Recall（再現率）やmAP（平均適合率）などの評価指標を計算します。
- 分析結果に基づいて、パラメータ調整などの改善提案を生成します。
- 分析レポート (Markdown形式) と改善計画 (YAML形式) を出力します。

#### 使用方法
- **基本実行**
  ```bash
  python scripts/run_inspector.py --candidates data/output/candidates/paleochannel_candidates.csv --known data/known/known_acre.kmz
  ```
- **主な引数**
  - `--candidates`: 評価対象の候補地データ (`.csv`) のパス。
  - `--known`: 比較基準となる既知の遺跡データ (`.kmz`, `.gpkg`など) のパス。
  - `--output`: レポートや計画ファイルを出力するディレクトリを指定します。

### `run_researcher.py`

#### 役割
**【AI博士】**
監査官のレポートをさらに深掘りし、「どの改善案が一番効果的か？」「次に何を試すべきか？」という、より具体的な**研究計画を立ててくれる**コンサルタントです。

#### 主な機能
- `run_inspector.py` の出力（分析レポートと改善計画）を入力として受け取ります。
- AI (LLM) を用いて、より掘り下げた考察や、複数の改善案の優先順位付けを行います。
- 研究レポートと、より詳細な改善計画を出力します。

#### 使用方法
- **基本実行（最新のInspectorレポートを自動で読み込む）**
  ```bash
  python scripts/run_researcher.py
  ```
- **主な引数**
  - `--artefacts`: Inspectorの出力が含まれるディレクトリのパス。指定しない場合、最新のものが自動で選択されます。
  - `--output`: 研究レポートなどを出力するディレクトリを指定します。

---

## 4. 全体を効率化・自動化する (Workflow Automation & Optimization)

### `run_optuna.py`

#### 役割
**【最適化の専門家】**
「川から何km離れた場所を探すか？」といった分析の条件（パラメータ）を、コンピューターが自動で何百回も試行錯誤し、**最も良い結果が出る「黄金のパラメータ」**を見つけ出してくれます。

#### 主な機能
- `run_site_identification.py` の分析パイプラインを、パラメータを変えながら繰り返し実行します。
- 最適化ライブラリOptunaを使い、最も良いスコアが得られるパラメータの組み合わせを探索します。
- 見つかった最適なパラメータはJSONファイルとして保存され、`run_best_params.py` で利用されます。

#### 使用方法
- **基本実行**
  ```bash
  python scripts/run_optuna.py --region acre --trials 50 --sites data/known/split/val.gpkg
  ```
- **主な引数**
  - `--region`: 最適化の対象となる地域 (`acre`, `marajo`など) を指定します。
  - `--trials`: 最適化の試行回数を指定します。
  - `--sites`: 評価基準となる検証用サイトデータ (`.gpkg`など) のパス。
  - `--timeout`: 1試行あたりの最大実行時間 (秒) を指定します。
  - `--resume`: 中断した最適化を再開します。

### `run_best_params.py`

#### 役割
**【最終実行者】**
最適化の専門家 (`run_optuna.py`) が見つけた「黄金のパラメータ」を使って、**最高の条件で最終的な分析を実行**し、結果を評価・可視化します。

#### 主な機能
- Optunaが出力した最適なパラメータ (JSONファイル) を読み込みます。
- そのパラメータを使って、遺跡候補地の特定とスコアリングを行います。
- 評価データセット（Validation/Test）に対してスコアを算出し、性能を評価します。
- オプションで、分析の各ステップや最終的な候補地の分布を詳細に可視化します。

#### 使用方法
- **基本実行**
  ```bash
  python scripts/run_best_params.py --region acre --params data/output/optuna/best_params.json --sites data/known/known_acre.kmz --visualize
  ```
- **主な引数**
  - `--region`: 対象地域 (`acre`, `marajo`など) を指定します。
  - `--params`: Optunaが出力した最良パラメータのJSONファイルパス。
  - `--sites`: 評価に使用する遺跡データのパス。
  - `--output`: 評価結果などを出力するディレクトリを指定します。
  - `--visualize`: 最終結果や分析過程を詳細に可視化します。

### `run_harmonizer.py`

#### 役割
**【現場監督】**
これまで説明した複数のスクリプト（辞書作りや候補地探しなど）を、**適切な順番で実行してくれる**まとめ役です。プロジェクト全体のワークフローを統括します。

#### 主な機能
- `run_root_extraction.py` と `run_site_identification.py` を内部で呼び出します。
- `--mode` オプションによって、両方のタスクを実行するか、どちらか一方だけを実行するかを選択できます。

#### 使用方法
- **両方のタスクを実行**
  ```bash
  python scripts/run_harmonizer.py --mode both
  ```
- **辞書管理のみ実行**
  ```bash
  python scripts/run_harmonizer.py --mode root-extraction
  ```
- **サイト特定のみ実行**
  ```bash
  python scripts/run_harmonizer.py --mode site-identification
  ```
- **主な引数**
  - `--mode`: `both`, `root-extraction`, `site-identification` から実行モードを選択します。