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

### Jupyter Notebook

`notebooks/99_kaggle_demo.ipynb` を使用して、アマゾン流域の水場系トポニムを抽出し、データ同化シミュレーションとマルチエージェント分析により古河道候補地点を特定します。

```python
# パッケージのインポート
from tamagawa_to_z.harmonizer.harmonizer import HarmonizerPipeline
from tamagawa_to_z.agents.agent_manager import AgentManager
from tamagawa_to_z.hydro_da.hydraulics_da import HydraulicsDA
from tamagawa_to_z.morph_da.morph_da import MorphologyDA

# 統合パイプラインの初期化
pipeline = HarmonizerPipeline()
agent_manager = AgentManager()
hydro_da = HydraulicsDA()
morph_da = MorphologyDA()

# S-1: 対象地域のBBox定義と基本データ収集
bbox = pipeline.make_bbox_gdf()
toponyms = pipeline.collect_toponyms(bbox)

# S-2: 多言語トポニム解析 & データ同化
harmonized_data = pipeline.harmonize_toponyms(toponyms)
hydro_results = hydro_da.simulate_flow_patterns(harmonized_data)
morph_results = morph_da.analyze_landforms(harmonized_data)

# S-3: マルチエージェント分析
candidates = agent_manager.analyze_candidates(
    harmonized_data, hydro_results, morph_results
)

# 結果の保存
candidates.to_parquet("data/interim/acre_candidates.parquet")
```

### Pythonスクリプト

CLI インターフェースを使用してコマンドラインから実行することも可能です：

```bash
# 基本実行
tamagawa_to_z --bbox -70.5 -11.5 -66.5 -8.5

# エージェント並列実行
tamagawa_to_z --bbox -70.5 -11.5 -66.5 -8.5 --agents 4

# データ同化シミュレーション有効化
tamagawa_to_z --bbox -70.5 -11.5 -66.5 -8.5 --enable-hydro-da --enable-morph-da
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
│       ├── agents/            # マルチエージェントシステム
│       │   ├── __init__.py
│       │   ├── agent_manager.py  # エージェント管理
│       │   └── agents.py         # エージェント実装
│       ├── hydro_da/          # 水理データ同化
│       │   ├── __init__.py
│       │   └── hydraulics_da.py  # 水理学的データ同化
│       ├── morph_da/          # 地形データ同化
│       │   ├── __init__.py
│       │   └── morph_da.py       # 地形学的データ同化
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
    │   ├── hydrorivers_sahydrorivers_sa/  # HydroRIVERS データ  
    │   └── GSW_occurrence/                # Global Surface Water データ
    └── interim/        # 中間データ
        └── acre_candidates.parquet  # 候補地点
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
