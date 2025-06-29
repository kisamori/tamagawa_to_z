# 地域データ設定ファイルの使用方法

## 概要

`region_data_paths.yaml`ファイルは考古学遺跡類似度分析スクリプトで使用する地域別データパスを管理します。

## ファイル構造

```yaml
regions:
  region_id:
    name: "地域表示名"
    description: "地域の説明"
    known_sites:
      csv: "既知遺跡データのCSVファイルパス"
      description: "既知遺跡データの説明"
    candidate_sites:
      csv: "候補地データのCSVファイルパス" 
      description: "候補地データの説明"
```

## 新しい地域の追加方法

### 1. YAMLファイルにエントリを追加

```yaml
regions:
  # 既存の地域...
  
  new_region:
    name: "新地域名"
    description: "新地域の詳細説明"
    known_sites:
      csv: "path/to/new_region_known_sites.csv"
      description: "新地域の既知遺跡データ"
    candidate_sites:
      csv: "path/to/new_region_candidate_sites.csv"
      description: "新地域の候補地データ"
```

### 2. スクリプト実行

```bash
poetry run python scripts/run_similarity_analysis.py --region new_region --mode candidate_ranking
```

## CSVデータの要件

各CSVファイルは以下の列を含む必要があります：

- `site_name`: サイト名
- `toponym_name`: 地名
- `angle`: 角度
- `radius`: 半径
- `river_angle`: 河川角度  
- `river_radius`: 河川半径
- `region`: 地域名
- `culture_tag`: 文化タグ
- `toponym_lat`: 地名の緯度
- `toponym_lon`: 地名の経度

## カスタム設定ファイルの使用

独自の設定ファイルを使用する場合：

```bash
poetry run python scripts/run_similarity_analysis.py --region my_region --config my_custom_config.yaml
```

## トラブルシューティング

### エラー: "サポートされていない地域です"
- `region_data_paths.yaml`に該当地域が定義されているか確認
- 地域IDのスペルミスがないか確認

### エラー: "データファイルが見つかりません"
- CSVファイルパスが正しいか確認
- ファイルが実際に存在するか確認
- パスは`tamagawa_to_z`ディレクトリからの相対パスで指定

### エラー: "地域設定ファイルの読み込みに失敗しました"
- YAMLファイルの構文が正しいか確認
- ファイルが存在するか確認
- UTF-8エンコーディングで保存されているか確認