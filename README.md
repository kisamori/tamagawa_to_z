# tamagawa_to_z

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-1.4.0+-blue.svg)](https://python-poetry.org/)

アマゾン古河道・集落探索に向けた多言語トポニム × データ同化シミュレーション × マルチエージェントフレームワーク

## 概要

tamagawa_to_z は、アマゾン流域における古河道や集落跡の探索を支援するための統合フレームワークです。
多言語トポニム解析、データ同化シミュレーション、マルチエージェントシステムを組み合わせ、
地名データから水関連の地名を抽出・分類し、現河道との距離や水域頻度を分析することで、
古河道候補地点を特定します。

## クイックスタート

### セットアップ

**前提条件:**
- Python 3.10以上が必要です（3.8や3.9では動作しません）
- 地理空間ライブラリ（特にpyproj、geopandas）がPython 3.10以上を要求します

```bash
# 1. リポジトリのクローン
git clone https://github.com/username/tamagawa_to_z.git
cd tamagawa_to_z

# 2. Poetry を使用してインストール
poetry install
```

### 3. 必要なデータを配置

#### 必要なデータファイル

| ファイル名 | 内容 | 入手先 | 役割 |
|-----------|------|--------|------|
| `HydroRIVERS_SA.shp` | 南米河川ネットワーク（約95MB） | [HydroSHEDS](https://hydrosheds.org) | 現河道との距離計算 |
| `GSW_occurrence.tif` | 水面出現頻度データ（1984-2021） | [GSW ポータル](https://global-surface-water.appspot.com) | 水域頻度判定 |

**注意:** ファイルサイズが大きいため、各自でダウンロードが必要です。

### 手動ダウンロード手順

#### 1. HydroRIVERS の取得
```bash
# ダウンロード・展開
wget -O hydrorivers_sa.zip "https://data.hydrosheds.org/file/HydroRIVERS/SA_HydroRIVERS_v10_shp.zip"
unzip hydrorivers_sa.zip -d data/raw/hydrorivers_sa
ln -s data/raw/hydrorivers_sa/HydroRIVERS_SA.shp data/raw/
```

#### 2. GSW occurrence の取得
1. [GSW ポータル](https://global-surface-water.appspot.com) で 70°W~60°W / 0°~-20° のタイルをダウンロード
2. 複数タイルがある場合は結合：
```bash
gdal_merge.py -o data/raw/GSW_occurrence_raw.tif tile*.tif
gdal_translate -projwin -70.5 -8.5 -66.5 -11.5 data/raw/GSW_occurrence_raw.tif data/raw/GSW_occurrence.tif
```

#### 3. 配置確認
```bash
ls data/raw/  # HydroRIVERS_SA.shp と GSW_occurrence.tif があることを確認
```
### Jupyter Notebookでの実行

```bash
# Jupyter Notebookを起動
poetry run jupyter notebook notebooks/01_harmonizer.ipynb
```

## インストール

**注意:** このプロジェクトはPython 3.10以上が必要です。pyenvなどを使用して適切なPythonバージョンを設定してください。

### Poetry を使用する場合（推奨）

```bash
# リポジトリのクローン
git clone https://github.com/username/tamagawa_to_z.git
cd tamagawa_to_z

# Poetry を使用してインストール
poetry install
```

### Kaggle で使用する場合

**注意:** Kaggleでも Python 3.10以上のランタイムを選択してください。

```bash
# requirements.txt を使用してインストール
pip install -r requirements.txt

# または以下のようにKaggleノートブックの先頭に記述
!pip install git+https://github.com/username/tamagawa_to_z.git
```

## 使い方

### 最新の推奨ワークフロー（Hybrid-BO v2.0）

このプロジェクトは Bayesian Optimization と LLM 統合による最適化フレームワークに対応しています。以下の手順で実行してください：

#### 1. データ準備・分割

```bash
# KMZファイルを使用してデータを分割
python scripts/run_split.py --sites data/known/known_acre.kmz --train 0.7 --val 0.2 --test 0.1
```

#### 2. ハイパーパラメータ最適化

```bash
# Optuna + LLM統合による最適化を実行（50トライアル）
python scripts/run_optuna.py --trials 50 --verbose
```

#### 3. 最良パラメータでの本実行 + 自動評価

```bash
# 最良パラメータで実行し、Inspector/Researcher分析も自動実行
python scripts/run_best_params.py --params data/output/optuna/20250623_152052/best_params.json --run-analysis --verbose
```

**注意：** `--run-analysis` フラグを使用すると以下が自動実行されます：
- ベストパラメータでの候補地点予測
- Inspector Agent による評価分析  
- Researcher Agent による改善提案

#### 4. 個別分析（必要に応じて）

```bash
# Inspector Agent（評価分析）
python scripts/run_inspector.py --candidates data/output/optuna/20250623_152052/best_run_val_candidates.csv --known data/known/known_acre.kmz

# Researcher Agent（改善提案）
python scripts/run_researcher.py --verbose
```

### 出力ディレクトリ構造

最適化結果は以下の構造で保存されます：

```
data/output/optuna/
├── 20250623_152052/              # タイムスタンプディレクトリ
│   ├── best_params.json          # 最良パラメータ
│   ├── best_run_val_candidates.csv       # バリデーション候補
│   ├── best_run_test_time_candidates.csv # テスト候補
│   └── trial_results.json        # 全トライアル結果
├── 20250623_153045/              # 別実行の結果
│   └── ...
└── optuna.db                     # Optuna SQLite DB
```

### レガシー Jupyter Notebook

従来の方法でも実行可能です：

```python
# パッケージのインポート
from tamagawa_to_z.harmonizer.harmonizer import HarmonizerPipeline
from tamagawa_to_z.inspector_agent.metrics import AgentManager

# 統合パイプラインの初期化
pipeline = HarmonizerPipeline()

# S-1: 対象地域のBBox定義と基本データ収集
bbox = pipeline.make_bbox_gdf()
toponyms = pipeline.collect_toponyms(bbox)

# S-2: 候補地点のフィルタリング
candidates = pipeline.filter_candidates(toponyms)

# 結果の保存
candidates.to_csv("data/output/candidates.csv", index=False)
```

## プロジェクト構成

```
tamagawa_to_z/
├── README.md           # このファイル
├── LICENSE             # MITライセンス  
├── pyproject.toml      # Poetry設定（Python 3.10以上が必要）
├── requirements.txt    # Kaggle用依存関係
├── .gitignore          # Git除外設定
│
├── scripts/            # CLI実行スクリプト（Hybrid-BO v2.0）
│   ├── run_split.py       # データ分割（KMZ→train/val/test.gpkg）
│   ├── run_optuna.py      # ハイパーパラメータ最適化
│   ├── run_best_params.py # 最良パラメータ実行+自動評価
│   ├── run_inspector.py   # Inspector Agent（評価分析）
│   └── run_researcher.py  # Researcher Agent（改善提案）
│
├── configs/            # 設定ファイル
│   ├── optuna_space.yaml  # 最適化パラメータ空間定義
│   └── pipeline_config.yaml # パイプライン設定
│
├── notebooks/          # 実験・可視化・デモ
│   └── 99_kaggle_demo.ipynb  # Kaggleデモノートブック
│
├── src/                # パッケージ本体
│   └── tamagawa_to_z/
│       ├── __init__.py
│       ├── harmonizer/        # 多言語トポニム解析
│       │   ├── __init__.py
│       │   ├── harmonizer.py  # 主要処理
│       │   ├── distance.py    # 距離計算
│       │   ├── embed.py       # 埋め込み処理
│       │   └── cluster.py     # クラスタリング
│       ├── tuning/            # ハイパーパラメータ最適化
│       │   ├── __init__.py
│       │   ├── optuna_hybrid.py    # Optuna + LLM統合
│       │   └── pipeline_runner.py  # パラメータ実行エンジン
│       ├── inspector_agent/   # 評価・分析エージェント
│       │   ├── __init__.py
│       │   ├── inspector.py   # Inspector Agent実装
│       │   ├── researcher.py  # Researcher Agent実装
│       │   └── metrics.py     # 評価指標計算
│       └── utils/             # ユーティリティ
│           ├── __init__.py
│           ├── geo.py         # 地理空間処理
│           ├── io.py          # データ入出力
│           ├── metrics.py     # 評価指標
│           └── viz.py         # 可視化
│
├── tests/              # テスト
│   └── test_harmonizer.py
│
└── data/               # データ（Gitに含めない）
    ├── raw/            # 入力データ
    │   ├── HydroRIVERS_SA.shp      # HydroRIVERS データ
    │   └── GSW_occurrence.tif      # Global Surface Water データ
    ├── known/          # 既知サイトデータ
    │   └── known_acre.kmz          # 訓練用既知サイト（KMZ形式）
    ├── splits/         # データ分割結果
    │   ├── train.gpkg  # 訓練データ
    │   ├── val.gpkg    # バリデーションデータ
    │   └── test.gpkg   # テストデータ
    └── output/         # 出力結果
        └── optuna/     # 最適化結果
            ├── 20250623_152052/    # タイムスタンプディレクトリ
            │   ├── best_params.json
            │   ├── best_run_val_candidates.csv
            │   └── best_run_test_time_candidates.csv
            └── optuna.db           # Optuna SQLite DB
```

## データソース

| カテゴリ | データセット | 主用途 |
| --- | --- | --- |
| 地名 | **BNGB** (IBGE API) ([ibge.gov.br][1]) | 正式ガゼティア |
|  | **OpenStreetMap / Overpass API** ([wiki.openstreetmap.org][2]) | 俗称・小水系補完 |
| 河川網 | **HydroRIVERS (HydroSHEDS)** ([hydrosheds.org][3]) | 現況河道ベクター |
| 水域履歴 | **JRC Global Surface Water (1984‑2021)** ([global-surface-water.appspot.com][4]) | 過去水出現確率 |

[1]: https://www.ibge.gov.br/
[2]: https://wiki.openstreetmap.org/wiki/Overpass_API
[3]: https://www.hydrosheds.org/
[4]: https://global-surface-water.appspot.com/

## 主要な変更履歴

### v2.0 - Hybrid-BO 統合 (2025-06-23)

- **CLI スクリプトの argparse 対応**: Typer から標準 argparse に変更し、通常のコマンドライン引数で使用可能
- **タイムスタンプディレクトリ管理**: 実行結果を `data/output/optuna/{timestamp}/` 形式で整理
- **自動評価チェーン**: `--run-analysis` フラグで Inspector/Researcher 分析を自動実行
- **WKT 幾何形式対応**: CSV 出力で Shapely Point オブジェクトの代わりに WKT 文字列を使用
- **KMZ 入力対応**: `.gpkg` の代わりに `.kmz` 形式のファイルを標準入力として採用
- **train=0.0 時の自動削除**: 既存の train.gpkg ファイルを自動削除してクリーンな状態を維持

#### 主要修正ファイル
- `scripts/run_split.py`: Typer → argparse、KMZ デフォルト、train.gpkg 削除機能
- `scripts/run_optuna.py`: Typer → argparse、タイムスタンプディレクトリ対応
- `scripts/run_best_params.py`: Typer → argparse、自動評価チェーン実装
- `src/tamagawa_to_z/tuning/pipeline_runner.py`: タイムスタンプディレクトリ自動作成、WKT 出力
- `src/tamagawa_to_z/harmonizer/harmonizer.py`: 実装の non-mock 化
- `src/tamagawa_to_z/inspector_agent/metrics.py`: WKT/GeoDataFrame 入力対応

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 引用

このプロジェクトを引用する場合は、以下の形式を使用してください：

```
@software{tamagawa_to_z,
  author = {tamagawa_to_z Contributors},
  title = {tamagawa_to_z: アマゾン古河道・集落探索フレームワーク},
  year = {2025},
  url = {https://github.com/username/tamagawa_to_z}
}
```
