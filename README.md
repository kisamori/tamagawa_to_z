# tamagawa-to-z

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-1.4.0+-blue.svg)](https://python-poetry.org/)

アマゾン古河道・集落探索に向けた多言語トポニム × データ同化シミュレーション × マルチ‑エージェントフレームワーク

## 概要

tamagawa-to-z は、アマゾン流域における古河道や集落跡の探索を支援するためのフレームワークです。以下の3つの主要コンポーネントを統合しています：

1. **多言語トポニム解析（System-A）**: 地名データから水関連の地名を抽出・分類し、水関連確率マップを生成します。
2. **データ同化シミュレーション（System-B）**: 水理学データ同化と地形変化データ同化を組み合わせて、古河道の存在確率を推定します。
3. **マルチエージェントシステム**: CrewAIを用いたマルチエージェントシステムにより、上記のコンポーネントを統合し、効率的な探索を実現します。

## クイックスタート

### セットアップ

```bash
# 1. リポジトリのクローン
git clone https://github.com/username/tamagawa-to-z.git
cd tamagawa-to-z

# 2. Poetry を使用してインストール
poetry install

# 3. 必要なデータを配置
# HydroRIVERS_SA.shp と GSW_occurrence.tif を data/raw/ に配置
```

### アクレ州パイプラインの実行

```bash
# スクリプトを使用して実行
bash scripts/run_harmonizer.sh

# または直接コマンドを実行
poetry run python -m tamagawa_to_z acre-pipeline
```

## インストール

### Poetry を使用する場合（推奨）

```bash
# リポジトリのクローン
git clone https://github.com/username/tamagawa-to-z.git
cd tamagawa-to-z

# Poetry を使用してインストール
poetry install
```

### pip を使用する場合

```bash
# リポジトリのクローン
git clone https://github.com/username/tamagawa-to-z.git
cd tamagawa-to-z

# pip を使用してインストール
pip install -e .
```

### Kaggle で使用する場合

```bash
# requirements.txt を使用してインストール
pip install -r requirements.txt
```

## 使い方

### アクレ州マデイラ川上流西部のS-1〜S-5パイプライン

このパイプラインは、アクレ州マデイラ川上流西部の水場系トポニムを抽出し、古河道候補地点を特定します。

```bash
# パイプラインの実行
poetry run python -m tamagawa_to_z acre-pipeline --out data/interim/acre_candidates.parquet

# オプションパラメータ
# --rivers: HydroRIVERSファイルのパス（デフォルト: data/raw/HydroRIVERS_SA.shp）
# --gsw: GSW occurrenceファイルのパス（デフォルト: data/raw/GSW_occurrence.tif）
# --log-level: ログレベル（デフォルト: INFO）
```

### コマンドラインインターフェース

tamagawa-to-z は、以下のサブコマンドを提供するコマンドラインインターフェースを備えています：

#### 多言語トポニム解析

```bash
# 多言語トポニム解析の実行
tamagawa-to-z harmonizer --input data/toponyms.csv --output output/water_prob.tif
```

#### 水理学データ同化

```bash
# 水理学データ同化の実行
tamagawa-to-z hydro_da --dem data/dem.tif --output output/hydro_da
```

#### 地形変化データ同化

```bash
# 地形変化データ同化の実行
tamagawa-to-z morph_da --dem data/dem.tif --water-history data/water_history.tif --toponym-prob output/water_prob.tif --output output/morph_da
```

#### マルチエージェント

```bash
# マルチエージェントの実行
tamagawa-to-z agents --output output/agents --workflow output/workflow.md
```

### Python API

tamagawa-to-z は、Python API も提供しています：

```python
import tamagawa_to_z as tmgw

# 多言語トポニム解析
harmonizer = tmgw.harmonizer.Harmonizer()
processed = harmonizer.process(toponyms)
prob_map = harmonizer.create_probability_map(processed)

# 水理学データ同化
hydraulics_da = tmgw.hydro_da.HydraulicsDA()
parameters, results = hydraulics_da.run_enkf()

# 地形変化データ同化
morph_da = tmgw.morph_da.MorphDA()
past_dem, paleo_channel_prob = morph_da.run_smoother(dem_data, water_history, toponym_prob)

# マルチエージェント
agent_manager = tmgw.agents.AgentManager()
agent_manager.create_agents()
agent_manager.create_tasks()
agent_manager.create_crew()
results = agent_manager.run()
```

## プロジェクト構成

```
tamagawa-to-z/
├── README.md           # このファイル
├── LICENSE             # MITライセンス
├── pyproject.toml      # Poetry設定
├── requirements.txt    # Kaggle用依存関係
├── .gitignore          # Git除外設定
│
├── notebooks/          # 実験・可視化・デモ
│   ├── 01_harmonizer.ipynb
│   ├── 02_hydro_da.ipynb
│   └── 99_kaggle_demo.ipynb
│
├── src/                # パッケージ本体
│   ├── tamagawa_to_z/
│   │   ├── __init__.py
│   │   ├── config/            # 設定
│   │   ├── data/              # 軽量リソース
│   │   ├── harmonizer/        # System-A
│   │   │   ├── preprocess.py  # S-1, S-2, S-3
│   │   │   ├── distance.py    # S-4
│   │   │   ├── watermask.py   # S-5 (GSW)
│   │   │   └── agent.py       # S-5 (LLM)
│   │   ├── hydro_da/          # System-B-A
│   │   ├── morph_da/          # System-B-B
│   │   ├── agents/            # CrewAI wrappers
│   │   └── utils/             # ユーティリティ
│   └── cli.py                 # CLI
│
├── scripts/            # スクリプト
│   ├── run_harmonizer.sh
│   └── batch_enkf.slurm
│
├── tests/              # テスト
│   ├── test_harmonizer.py
│   └── test_da.py
│
├── data/               # データ（Gitに含めない）
│   ├── raw/            # 入力データ
│   │   ├── HydroRIVERS_SA.shp  # 南米河川ネットワーク
│   │   └── GSW_occurrence.tif  # 水域頻度
│   └── interim/        # 中間データ
│       └── acre_candidates.parquet  # 候補地点
│
├── models/             # モデル（Git-LFS or DVC）
│
├── dags/               # ワークフロー定義
│
└── docs/               # ドキュメント
    ├── report.md
    └── figures/
```

## データソース

| カテゴリ            | データセット                                                                           | 主用途             |
| --------------- | -------------------------------------------------------------------------------- | --------------- |
| 地名              | **BNGB** (IBGE API) ([ibge.gov.br][1])                                           | 正式ガゼティア         |
|                 | **OpenStreetMap / Overpass API** ([wiki.openstreetmap.org][2])                   | 俗称・小水系補完        |
| 河川網             | **HydroRIVERS (HydroSHEDS)** ([hydrosheds.org][3])                               | 現況河道ベクター        |
| 水域履歴            | **JRC Global Surface Water (1984‑2021)** ([global-surface-water.appspot.com][4]) | 過去水出現確率         |
| 地形              | **SRTM 1″ DEM** (USGS) ([usgs.gov][5])                                           | 微高地抽出           |
| 土地被覆            | **MapBiomas Collection 9** (1985‑2023) ([brasil.mapbiomas.org][6])               | 農地・裸地マスク        |
| 森林変化            | **Hansen Global Forest Change v1.12** ([developers.google.com][7])               | 伐採年判定           |
| Hydrodynamics   | **Delft3D‑FM** + **OpenDA** coupling ([content.oss.deltares.nl][8])              | EnKF / Smoother |
| Embedding Model | **distiluse‑base‑multilingual‑v2** ([huggingface.co][9])                         | 512‑d 多言語埋込     |
| LLM Agent FW    | **CrewAI** ([github.com][10]), **LangGraph** ([langchain-ai.github.io][11])      | マルチ‑エージェント      |

[1]: https://www.ibge.gov.br/
[2]: https://wiki.openstreetmap.org/wiki/Overpass_API
[3]: https://www.hydrosheds.org/
[4]: https://global-surface-water.appspot.com/
[5]: https://www.usgs.gov/
[6]: https://brasil.mapbiomas.org/
[7]: https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2021_v1_9
[8]: https://content.oss.deltares.nl/
[9]: https://huggingface.co/sentence-transformers/distiluse-base-multilingual-v2
[10]: https://github.com/joaomdmoura/crewAI
[11]: https://langchain-ai.github.io/langgraph/

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 貢献

貢献は歓迎します！バグ報告、機能リクエスト、プルリクエストなど、どんな形でも構いません。

## 引用

このプロジェクトを引用する場合は、以下の形式を使用してください：

```
@software{tamagawa-to-z,
  author = {tamagawa-to-z Contributors},
  title = {tamagawa-to-z: アマゾン古河道・集落探索フレームワーク},
  year = {2025},
  url = {https://github.com/username/tamagawa-to-z}
}
```
