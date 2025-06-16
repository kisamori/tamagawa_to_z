#!/bin/bash
# アクレ州マデイラ川上流西部のS-1〜S-5パイプラインを実行するスクリプト

# スクリプトのディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# プロジェクトルートに移動
cd "$PROJECT_ROOT" || exit 1

# データディレクトリの確認
mkdir -p data/raw
mkdir -p data/interim

# 必要なデータファイルの確認
if [ ! -f "data/raw/HydroRIVERS_SA.shp" ]; then
    echo "エラー: HydroRIVERS_SA.shpファイルが見つかりません。"
    echo "data/rawディレクトリにHydroRIVERS_SA.shpファイルを配置してください。"
    exit 1
fi

if [ ! -f "data/raw/GSW_occurrence.tif" ]; then
    echo "エラー: GSW_occurrence.tifファイルが見つかりません。"
    echo "data/rawディレクトリにGSW_occurrence.tifファイルを配置してください。"
    exit 1
fi

# パイプラインの実行
echo "アクレ州マデイラ川上流西部のS-1〜S-5パイプラインを実行します..."

# Poetryを使用して実行
if command -v poetry &> /dev/null; then
    poetry run python -m tamagawa_to_z acre-pipeline --out data/interim/acre_candidates.parquet
else
    # Poetryがない場合はPythonを直接使用
    python -m tamagawa_to_z acre-pipeline --out data/interim/acre_candidates.parquet
fi

# 結果の確認
if [ -f "data/interim/acre_candidates.parquet" ]; then
    echo "パイプラインが正常に完了しました。"
    echo "結果: data/interim/acre_candidates.parquet"
    
    # 結果の表示（オプション）
    if command -v poetry &> /dev/null; then
        echo "候補地点の概要:"
        poetry run python -c "import geopandas as gpd; df = gpd.read_parquet('data/interim/acre_candidates.parquet'); print(f'候補地点数: {len(df)}'); print(df.head())"
    fi
else
    echo "エラー: パイプラインの実行中に問題が発生しました。"
    exit 1
fi
