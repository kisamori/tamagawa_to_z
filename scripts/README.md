# スクリプト実行ガイド

このディレクトリには、地名調和処理（ハーモナイゼーション）と古河道候補地特定のためのスクリプトが含まれています。

## 概要

処理は2つの主要タスクに分離されています：

- **Task i)** 辞書作成と語根抽出：LLMハーモナイゼーション、新語根発見
- **Task ii)** 遺跡候補地特定：地理空間分析パイプライン（S-1～S-5）

## スクリプト構成

### 1. run_harmonizer.py（統合実行スクリプト）
モード選択による統合実行が可能です。

### 2. run_root_extraction.py（辞書・語根管理専用）
Task i)を独立実行します。

### 3. run_site_identification.py（遺跡候補地特定専用）
Task ii)を独立実行します。

### 前提条件

- Pythonパッケージがインストールされていること
  ```
  cd tamagawa_to_z
  pip install -e .
  ```
  または
  ```
  cd tamagawa_to_z
  poetry install
  ```

## 使用方法

### 1. 統合実行（推奨）

**両方のタスクを順次実行:**
```bash
python scripts/run_harmonizer.py --mode both
```

**辞書管理のみ実行:**
```bash
python scripts/run_harmonizer.py --mode root-extraction --sample-size 100
```

**遺跡候補地特定のみ実行:**
```bash
python scripts/run_harmonizer.py --mode site-identification --skip-water-freq
```

### 2. 個別実行

**辞書作成・語根抽出:**
```bash
python scripts/run_root_extraction.py --sample-size 50 --visualize
```

**古河道候補地特定:**
```bash
python scripts/run_site_identification.py --rivers-path data/raw/HydroRIVERS_v10_sa.shp --visualize
```

## 共通オプション

### 地理的設定
- `--bbox LON_MIN LAT_MIN LON_MAX LAT_MAX`: 対象領域のBBOX（デフォルト: `-70.5 -11.5 -66.5 -8.5`）
- `--pbf-path PATH`: OSM PBFファイルパス（デフォルト: `data/raw/osm/norte-latest.osm.pbf`）
- `--visualize`: 処理結果を可視化する（OpenStreetMap背景付き画像ファイルとして保存）
- `--viz-output-dir PATH`: 可視化画像の出力ディレクトリ（デフォルト: `data/plots`）

### データフィルタリング設定
- `--include-water-features`: 水域タグを持つ地物も地名候補として含める（デフォルトは除外）
  
  **詳細**: デフォルトでは、OSMの水域タグ（`waterway`, `natural=water`, `water`, `wetland`, `riverbank`）を持つ地物は地名候補から除外されます。このオプションを指定すると、これらのタグを持つ地物も地名候補に含まれます。
  
  **注意**: 現在の対象地域（アマゾン北部）のOSMデータでは、水域タグを持つ名前付き地物が存在しないため、このオプションの有無による抽出数の変化は見られません。将来的に異なる地域やデータセットを使用する際に有効になる可能性があります。

### データファイル設定
- `--rivers-path PATH`: HydroRIVERSシェープファイルパス（デフォルト: `data/raw/hydrorivers_sahydrorivers_sa/HydroRIVERS_v10_sa.shp`）
- `--gsw-path PATH`: GSW occurrenceのTIFFファイルパス（デフォルト: `data/raw/GSW_occurrence/occurrence_70W_10Sv1_4_2021.tif`）

## 辞書・語根管理専用オプション（run_root_extraction.py）

### LLM・処理設定
- `--sample-size N`: LLMハーモナイゼーションのサンプルサイズ（コスト削減用）
- `--output-dir PATH`: 出力ディレクトリ（デフォルト: `data/interim`）
- `--no-merge-roots`: 新語根の自動マージを無効化する

### OSMキー設定（地名抽出範囲の制御）
- `--osm-keys-config PATH`: OSMキー設定ファイルのパス（デフォルト: `data/config/osm_keys.yaml`）
- `--osm-keys-mode MODE`: OSMキー抽出モード（デフォルト: `standard`）

**利用可能なOSMキー抽出モード:**
- `conservative`: 最小限のキー（place, natural, historic）
- `standard`: 標準設定（place, landuse, man_made, highway）
- `extended`: 包括的抽出（上記＋amenity, tourism, leisure, shop, natural, waterway, railway, historic, office, building）
- `water_focused`: 水関連地名特化（place, natural, waterway, man_made, historic, landuse）

**OSMキー設定ファイル（osm_keys.yaml）について:**

設定ファイル `data/config/osm_keys.yaml` では、OpenStreetMapから地名を抽出する際に使用するキー（データカテゴリ）を制御できます：

```yaml
# 抽出モードごとの設定
extraction_modes:
  standard:           # デフォルト設定
    - place           # 村、集落、都市など
    - landuse         # 土地利用エリア
    - man_made        # 人工構造物
    - highway         # 道路・交通インフラ
  
  extended:           # より包括的な抽出
    - place
    - landuse
    - man_made
    - highway
    - amenity         # 施設・サービス（銀行、レストラン、学校、病院）
    - tourism         # 観光施設（ホテル、博物館、観光地）
    - leisure         # レクリエーション施設（公園、スポーツセンター）
    - shop            # 商業施設（店舗、市場）
    - natural         # 自然地形（山、丘、洞窟、海岸）
    - waterway        # 水路関連（川、運河、水路）※水関連地名に特に有用
    - railway         # 鉄道インフラ（駅、路線、プラットフォーム）
    - historic        # 歴史的建造物（遺跡、記念碑、考古学的遺跡）
    - office          # オフィス・行政施設（政府、企業）
    - building        # 名前付き建物・構造物
```

**注意事項:**
- より多くのキーを使用すると包括的な抽出が可能ですが、処理時間が増加し、ノイズも増える可能性があります
- 水語彙フィルタリングは初期抽出後に適用されるため、抽出されるのは水関連語彙を含む地名のみです
- 各OSM地物は name, alt_name, old_name, loc_name フィールドのいずれかを持つ必要があります

**出力ファイル:**
- `toponym_harmonization_results.csv`: ハーモナイゼーション済み地名辞書
- `new_root_analysis.csv`: 新語根分析結果

## 遺跡候補地特定専用オプション（run_site_identification.py）

- `--output-path PATH`: 出力ファイルパス（デフォルト: `data/interim/paleochannel_candidates.csv`）
- `--skip-water-freq`: 水域頻度計算をスキップ（距離のみで候補抽出）

**出力ファイル:**
- `paleochannel_candidates.csv`: 古河道候補地リスト（スコア付き）

### 可視化画像（--visualizeオプション使用時）

**出力ディレクトリ構造:**
```
data/plots/
├── site_identification_20231220_143052/    # タイムスタンプ付きディレクトリ
│   ├── step1_bbox.png
│   ├── step2_toponyms_distribution.png
│   ├── step3_type_distribution.png
│   ├── step4_distance_distribution.png
│   └── step5_paleochannel_candidates.png
└── root_extraction_20231220_143105/        # タイムスタンプ付きディレクトリ
    ├── root_extraction_toponyms_distribution.png
    └── root_extraction_type_distribution.png
```

**各画像の内容:**
- `step1_bbox.png`: 対象領域の境界（OpenStreetMap背景付き、英語表記）
- `step2_toponyms_distribution.png`: 収集したトポニムの分布（OpenStreetMap背景付き、英語表記）
- `step3_type_distribution.png`: 水系タイプ別の件数（棒グラフ、英語表記）
- `step4_distance_distribution.png`: 現河道からの距離分布（ヒストグラム、英語表記）
- `step5_paleochannel_candidates.png`: 古河道候補地点（スコア別色分け、地図背景付き、英語表記）
- `root_extraction_toponyms_distribution.png`: 地名分布（語根抽出用、OpenStreetMap背景付き、英語表記）
- `root_extraction_type_distribution.png`: 水系タイプ分布（語根抽出用、棒グラフ、英語表記）

## 処理ステップ

### Task i) 辞書作成・語根抽出（run_root_extraction.py）

1. **地名収集**: 対象領域からの水関連地名抽出
2. **基本処理**: 正規化・タイプ推定
3. **LLMハーモナイゼーション**: 既存辞書との照合、類似度判定
4. **新語根発見**: パターン分析による未知語根候補抽出
5. **結果保存**: ハーモナイゼーション結果と新語根分析の保存

### Task ii) 遺跡候補地特定（run_site_identification.py）

1. **S-1**: 対象地域のBBox定義
2. **S-2**: 水場系トポニムの抽出（PyrosmでOSMデータ使用）
3. **S-3**: クレンジング & タイプ付け（LLMなし）
4. **S-4**: 現河道との距離計算
5. **S-5**: "川が無いのに川名が残る"ポイント抽出・スコアリング

## 環境設定

### 必須データファイル

1. **OSM PBFファイル**: `data/raw/osm/norte-latest.osm.pbf`
   - ダウンロード: [Geofabrik](https://download.geofabrik.de/south-america/brazil/norte-latest.osm.pbf)

2. **HydroRIVERS**: `data/raw/hydrorivers_sahydrorivers_sa/HydroRIVERS_v10_sa.shp`
   - ダウンロード: [HydroSHEDS](https://www.hydrosheds.org/products/hydrorivers)

3. **GSW occurrence**: `data/raw/GSW_occurrence/occurrence_70W_10Sv1_4_2021.tif`
   - ダウンロード: [Global Surface Water](https://global-surface-water.appspot.com/download)

### OpenAI API設定（辞書管理用）

```bash
# .envファイルに設定
OPENAI_API_KEY=your_api_key_here
```

## 実行例とコスト管理

### 低コスト実行（開発・テスト用）
```bash
# 辞書管理のみ、少量サンプル
python scripts/run_root_extraction.py --sample-size 20

# 遺跡候補地特定のみ、水域頻度計算スキップ
python scripts/run_site_identification.py --skip-water-freq

# 水域タグを持つ地物も含めて地名抽出
python scripts/run_root_extraction.py --include-water-features --sample-size 20

# 保守的モードで最小限のキーを使用
python scripts/run_root_extraction.py --osm-keys-mode conservative --sample-size 20

# 水関連地名特化モードを使用
python scripts/run_root_extraction.py --osm-keys-mode water_focused --sample-size 20
```

### 本格実行
```bash
# 辞書管理＋遺跡候補地特定の統合実行
python scripts/run_harmonizer.py --mode both --sample-size 200 --visualize

# 水域タグを持つ地物も含めて統合実行
python scripts/run_harmonizer.py --mode both --include-water-features --sample-size 200 --visualize

# 包括的抽出モードで辞書管理を実行
python scripts/run_root_extraction.py --osm-keys-mode extended --sample-size 100 --visualize

# カスタムOSMキー設定ファイルを使用
python scripts/run_root_extraction.py --osm-keys-config custom_osm_keys.yaml --sample-size 100
```

### ログ出力

すべてのスクリプトは実行中の進捗状況や結果をログとして出力します。ログレベルはデフォルトで`INFO`に設定されています。

### エラーハンドリング

入力データファイルが存在しない場合は警告を表示し、可能な処理のみ実行します。スクリプトは段階的に処理を進めるため、一部のステップが失敗しても後続処理を継続します。

## トラブルシューティング

1. **Pyrosmエラー**: OSM PBFファイルのパスを確認
2. **LLM実行エラー**: OpenAI APIキーの設定を確認
3. **メモリ不足**: `--sample-size`でLLM処理データを削減
4. **距離計算エラー**: HydroRIVERSファイルの存在を確認
5. **地図背景エラー**: `poetry install`でcontextilyがインストールされているか確認
