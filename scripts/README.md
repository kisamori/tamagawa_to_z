# スクリプト実行ガイド

このディレクトリには、Jupyter Notebookの処理を通常のPythonスクリプトとして実行するためのスクリプトが含まれています。

## run_harmonizer.py

`run_harmonizer.py`は、`notebooks/01_harmonizer.ipynb`の処理を通常のPythonスクリプトとして実行できるようにしたものです。

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

### 使用方法

基本的な実行:
```bash
python scripts/run_harmonizer.py
```

オプションを指定して実行:
```bash
python scripts/run_harmonizer.py --rivers_path データ/HydroRIVERS.shp --gsw_path データ/GSW_occurrence.tif --output_path 結果/candidates.parquet --visualize
```

### オプション

- `--rivers_path PATH`: HydroRIVERSのシェープファイルパス（デフォルト: `data/raw/HydroRIVERS_v10_sa.shp`）
- `--gsw_path PATH`: GSW occurrenceのTIFFファイルパス（デフォルト: `data/raw/occurrence_70W_10Sv1_4_2021.tif`）
- `--output_path PATH`: 出力ファイルパス（デフォルト: `data/interim/acre_candidates.parquet`）
- `--visualize`: 処理結果を可視化する（デフォルト: False）
- `--bbox LON_MIN LAT_MIN LON_MAX LAT_MAX`: 対象領域のBBoxを指定（デフォルト: `-70.5 -11.5 -66.5 -8.5`）

### 処理ステップ

スクリプトは以下のステップで処理を行います：

1. **S-1**: 対象地域のBBox定義
2. **S-2**: 水場系トポニムの抽出（BNGB API、OpenStreetMap）
3. **S-3**: クレンジング & タイプ付け
4. **S-4**: 現河道との距離計算
5. **S-5**: "川が無いのに川名が残る"ポイント抽出

各ステップの詳細は、スクリプト内のコメントやドキュメント文字列を参照してください。

### ログ出力

スクリプトは実行中の進捗状況や結果をログとして出力します。ログレベルはデフォルトで`INFO`に設定されています。

### エラーハンドリング

入力データファイルが存在しない場合は警告を表示し、可能な処理のみ実行します。
