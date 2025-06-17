# tamagawa_to_z

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-1.4.0+-blue.svg)](https://python-poetry.org/)

アマゾン古河道・集落探索に向けた多言語トポニム解析フレームワーク

## 概要

tamagawa_to_z は、アマゾン流域における古河道や集落跡の探索を支援するためのフレームワークです。
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

`notebooks/01_harmonizer.ipynb` を使用して、アクレ州マデイラ川上流西部の水場系トポニムを抽出し、古河道候補地点を特定します。

```python
# パッケージのインポート
from tamagawa_to_z.harmonizer import (
    make_bbox_gdf, collect_names, collect_osm_names, merge_toponyms, process_toponyms,
    attach_distance, water_occurrence, filter_candidates, score_candidates
)

# S-1: 対象地域のBBox定義
bbox = make_bbox_gdf()

# S-2: 水場系トポニムの抽出
bngb_names = collect_names(bbox.geometry.iloc[0])
osm_names = collect_osm_names(bbox.geometry.iloc[0])
names = merge_toponyms(bngb_names, osm_names)

# S-3: クレンジング & タイプ付け
names = process_toponyms(names)

# S-4: 現河道との距離計算
names = attach_distance(names, "data/raw/HydroRIVERS_SA.shp")

# S-5: "川が無いのに川名が残る"ポイント抽出
names = water_occurrence(names, "data/raw/GSW_occurrence.tif")
candidates = filter_candidates(names)
candidates = score_candidates(candidates)

# 結果の保存
candidates.to_parquet("data/interim/acre_candidates.parquet")
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
│   └── 01_harmonizer.ipynb  # アクレ州パイプライン実行
│
├── src/                # パッケージ本体
│   └── tamagawa_to_z/
│       ├── __init__.py
│       ├── harmonizer/        # 多言語トポニム解析
│       │   ├── __init__.py
│       │   ├── preprocess.py  # S-1, S-2, S-3
│       │   ├── distance.py    # S-4
│       │   ├── watermask.py   # S-5 (GSW)
│       │   └── agent.py       # S-5 (閾値判定)
│       └── utils/             # ユーティリティ
│           └── __init__.py
│
├── tests/              # テスト
│   └── test_harmonizer.py
│
└── data/               # データ（Gitに含めない）
    ├── raw/            # 入力データ
    │   ├── HydroRIVERS_SA.shp  # 南米河川ネットワーク
    │   └── GSW_occurrence.tif  # 水域頻度
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
