# Tamagawa to Z: AIと共に、アマゾンに眠る失われた川と古代遺跡の謎を追う

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-1.4.0+-blue.svg)](https://python-poetry.org/)

## はじめに：失われた川の記憶を求めて

川は、文明の揺りかごだ。しかし、その流れは永遠ではない。
悠久の時の中で、川は蛇行し、その姿を大きく変えていく。かつて文明が栄えた場所から、水はとうに失われているかもしれない。

だが、もし、その記憶が「地名」として現代にまで語り継がれているとしたら？

我々の冒険は、そんなロマンあふれる仮説から始まった。土地の名前や伝承は、人々が去り、川が涸れてもなお、その場所に根付く魂のタイムカプセルなのではないか。

このプロジェクト「Tamagawa to Z」の名は、我々チームメンバーが暮らす日本の「多摩川」に由来する。かつて氾濫を繰り返したこの川は、その両岸に同じ「等々力」という地名を残した。川が分かち、時が隔てた二つの土地を、今も地名が繋いでいる。我々はこの地名が持つ物語の力に魅せられ、遠く離れたアマゾンの大地に、同じ奇跡を探す旅に出ることを決意した。

### 我々の羅針盤：LLMと多言語トポニム

アマゾンの歴史は、多様な言語の交差点だ。原住民族の言葉、渡来人の言葉。それらは混ざり合い、時を経て変化し、土地の本当の姿を複雑なベールの奥に隠してしまった。

我々の最大の武器は、この言語の壁を打ち破るLLMの力と、「トポニム（地名学）」の知見だ。

単に「水」を意味する単語を探すのではない。我々はLLMに、アマゾンに点在する無数の地名を多言語の文脈で解釈させ、その語源、意味の類似性、そして歴史的背景を総合的に判断させることで、いわば「言語の分類学」を実践した。これは、言葉の奥に眠る、正規化された「土地の記憶」を掘り起こす作業だ。

このプロセスを経て、我々は世界で一つだけの**「多言語トポニム辞書」**を創り上げた。これは、我々の長く険しい探索行における、唯一無二の羅針盤である。

### 発見への航路

我々は、まだ見ぬ発見の可能性に満ちたブラジル・アクレ州を、冒険の舞台に選んだ。森林伐採や開発により、奇しくも「新たな地表」が次々と現れているこの地は、近年、衛星画像から数々の地上絵が見つかっている、まさにフロンティアだ。

我々の航路は、以下の通りである。

1.  **地図に眠る声を聞く：** 作成した「多言語トポニム辞書」を手に、OSM（OpenStreetMap）からアクレ州の地名を拾い上げ、水に関連する可能性のある場所をリストアップした。

2.  **過去への舵を切る：** 我々の目的は「失われた川」の探索。そのため、現在の川から一定以上離れ、かつての居住地であった可能性を示唆する「湿地度が低い」土地に絞り込んだ。この絞り込みの精度を高めるため、既知の遺跡をどれだけ再現できるかを指標に、LLM（O3 Pro）自身と対話しながらパラメータを調整した。それはまるで、AIという経験豊富な老練な航海士と、未知の海図について議論を重ねるような体験だった。

3.  **星々を頼りに：** 選び出された8つの候補地。我々はそれらを、既知の遺跡との距離、人の手による破壊の可能性、衛星画像（DEM）から読み取れる地形的な優位性など、複数の観点からスコアリングした。

### 約束の地： Ramal Olho D'água

数多の分析と、AIとの対話の末、我々はついに一つの場所にたどり着いた。

| サイト名 | Ramal Olho D'água |
| :--- | :--- |
| **座標** | **-9.839247, -68.498725** |
| 複合スコア | 0.0751 |

この場所は、既知の遺跡群からは離れた、まさに“空白地帯“に位置する。
衛星データは、古代人が集落を築きやすい「微高地」と「低湿地」がモザイク状に広がる、理想的な環境であることを示唆している。

そして、決定的な証拠。
我々が「Olho D’Água（水の瞳）」という地名を手がかりに、この土地にまつわる民話をGPTに尋ねたところ、驚くべき事実が判明した。アレク州のインディオ保護団体の資料に、**まさにこの「Olho D’Água」の周辺に、かつてインディオが暮らしていた**という記述が存在したのだ。

AIによって掘り起こされた地名の記憶が、古の民話と重なった瞬間だった。

---


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
| OSMデータファイル群 | 南米北部・BR地域別OSMデータ | [Geofabrik](https://download.geofabrik.de/south-america.html) | 地名データ抽出 |

**注意:** ファイルサイズが大きいため、各自でダウンロードが必要です。

### 手動ダウンロード手順

#### 1. HydroRIVERS の取得
```bash
# ダウンロード・展開
wget -O hydrorivers_sa.zip "https://data.hydrosheds.org/file/HydroRIVERS/SA_HydroRIVERS_v10_shp.zip"
unzip hydrorivers_sa.zip -d data/raw/hydrorivers_sa
```

#### 2. GSW occurrence の取得
1. [GSW ポータル](https://global-surface-water.appspot.com) で 80°W~40°W / 0°~20°S のタイルをダウンロード
2. ダウンロードしたファイルをGSW_occurrenceディレクトリに配置：
```bash
mkdir -p data/raw/GSW_occurrence
# ダウンロードしたタイルファイルをこのディレクトリに移動
mv occurrence_*.tif data/raw/GSW_occurrence/
```

#### 3. OSM データの取得
```bash
# OSMディレクトリを作成
mkdir -p data/raw/osm

# 南米北部・ブラジル各地域のOSMデータをダウンロード
wget -O data/raw/osm/bolivia-latest.osm.pbf "https://download.geofabrik.de/south-america/bolivia-latest.osm.pbf"
wget -O data/raw/osm/peru-latest.osm.pbf "https://download.geofabrik.de/south-america/peru-latest.osm.pbf"
wget -O data/raw/osm/centro-oeste-latest.osm.pbf "https://download.geofabrik.de/south-america/brazil/centro-oeste-latest.osm.pbf"
wget -O data/raw/osm/nordeste-latest.osm.pbf "https://download.geofabrik.de/south-america/brazil/nordeste-latest.osm.pbf"
wget -O data/raw/osm/norte-latest.osm.pbf "https://download.geofabrik.de/south-america/brazil/norte-latest.osm.pbf"
wget -O data/raw/osm/sudeste-latest.osm.pbf "https://download.geofabrik.de/south-america/brazil/sudeste-latest.osm.pbf"
wget -O data/raw/osm/sul-latest.osm.pbf "https://download.geofabrik.de/south-america/brazil/sul-latest.osm.pbf"
```

#### 4. 配置確認
```bash
ls data/raw/hydrorivers_sa/   # HydroRIVERS ファイル群があることを確認
ls data/raw/GSW_occurrence/   # GSW occurrence タイル群があることを確認
ls data/raw/osm/      # 各地域のOSMファイルがあることを確認
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
!pip install git+https://github.com/kisamori/tamagawa_to_z.git
```

## 使い方

このプロジェクトは、AIと地理空間データを組み合わせてアマゾンの古代遺跡候補地を探索します。

**詳細な使用方法については、[scripts/README.md](scripts/README.md) を参照してください。**

### 基本ワークフロー

1. **データ準備** - 既知遺跡データの分割（訓練/検証/テスト）
2. **パラメータ最適化** - Bayesian Optimization + LLM による最適化
3. **候補地予測** - 最良パラメータでの候補地点抽出
4. **評価・分析** - AI エージェントによる結果評価と改善提案

### クイック実行

```bash
# 1. データ分割
python scripts/run_split.py --config configs/dataset_split.yaml

# 2. パラメータ最適化
python scripts/run_optuna.py --config configs/optuna_run.yaml --trials 50

# 3. 予測実行（評価・分析も自動実行）
python scripts/run_best_params.py --params data/output/optuna/.../best_params.json --run-analysis
```

### 分析ワークフロー

分析の詳細な流れについては、以下のJupyter Notebookを参照してください：

**📊 メインワークフロー（予定）:** `notebooks/analysis_workflow.ipynb`

このノートブックでは、以下の分析手順を段階的に解説します：
- 多言語トポニム辞書の構築
- 地名データの収集と前処理
- 機械学習による候補地点の抽出
- 評価と結果の可視化

## プロジェクト構成

```
tamagawa_to_z/
├── README.md              # このファイル
├── LICENSE                # MITライセンス  
├── pyproject.toml         # Poetry設定（Python 3.10以上が必要）
├── requirements.txt       # Kaggle用依存関係
│
├── scripts/               # CLI実行スクリプト
│   ├── README.md             # 詳細な使用方法
│   ├── run_split.py          # データ分割
│   ├── run_optuna.py         # ハイパーパラメータ最適化
│   ├── run_best_params.py    # 最良パラメータ実行
│   ├── run_inspector.py      # 評価分析
│   └── run_researcher.py     # 改善提案
│
├── configs/               # 設定ファイル
│   ├── optuna_space.yaml     # 最適化パラメータ空間定義
│   └── dataset_split.yaml    # データ分割設定
│
├── notebooks/             # 実験・可視化・デモ
│   ├── 01_harmonizer.ipynb      # メイン処理デモ
│   └── 99_kaggle_demo.ipynb     # Kaggleデモ
│
├── src/                   # パッケージ本体
│   └── tamagawa_to_z/
│       ├── harmonizer/           # 多言語トポニム解析
│       ├── tuning/               # ハイパーパラメータ最適化
│       ├── inspector_agent/      # 評価・分析エージェント
│       ├── researcher_agent/     # 改善提案エージェント
│       ├── site_analysis/        # サイト分析ツール
│       └── utils/                # ユーティリティ
│
├── tests/                 # テスト
│
└── data/               # データ（Gitに含めない）
    ├── raw/            # 入力データ
    │   ├── hydrorivers_sa/         # HydroRIVERS データ
    │   │   └── HydroRIVERS_v10_sa.shp  # 南米河川ネットワーク
    │   ├── GSW_occurrence/         # Global Surface Water データ
    │   │   └── occurrence_*.tif        # 水面出現頻度タイル群
    │   └── osm/                    # OSM データ
    │       ├── bolivia-latest.osm.pbf     # ボリビア
    │       ├── peru-latest.osm.pbf        # ペルー
    │       ├── centro-oeste-latest.osm.pbf # ブラジル中西部
    │       ├── nordeste-latest.osm.pbf    # ブラジル北東部
    │       ├── norte-latest.osm.pbf       # ブラジル北部
    │       ├── sudeste-latest.osm.pbf     # ブラジル南東部
    │       └── sul-latest.osm.pbf         # ブラジル南部
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
