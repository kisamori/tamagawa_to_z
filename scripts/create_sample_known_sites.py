#!/usr/bin/env python3
"""サンプル既知遺跡データ作成スクリプト

Inspector-Validator Agentのテスト用に、
候補データの座標範囲内に仮想的な既知遺跡ポイントを作成します。
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def create_sample_known_sites(candidates_csv_path: str, output_path: str, num_sites: int = 5):
    """サンプル既知遺跡データを作成する
    
    Parameters
    ----------
    candidates_csv_path : str
        候補データのCSVファイルパス
    output_path : str
        出力GeoPackageファイルパス
    num_sites : int
        作成する遺跡数
    """
    print(f"📍 候補データから座標範囲を分析: {candidates_csv_path}")
    
    # 候補データを読み込み
    candidates = pd.read_csv(candidates_csv_path)
    
    # geometry列からx,y座標を抽出
    coords = []
    for geom_str in candidates['geometry']:
        coord_str = geom_str.replace('POINT (', '').replace(')', '')
        x, y = map(float, coord_str.split())
        coords.append((x, y))
    
    coords = np.array(coords)
    
    # 座標範囲の確認
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    print(f"経度範囲: {x_min:.4f} to {x_max:.4f}")
    print(f"緯度範囲: {y_min:.4f} to {y_max:.4f}")
    
    # 範囲を少し拡張してサンプル遺跡を配置
    x_buffer = (x_max - x_min) * 0.1  # 10%のバッファ
    y_buffer = (y_max - y_min) * 0.1
    
    # ランダムシードを固定して再現可能にする
    np.random.seed(42)
    
    # サンプル遺跡の座標を生成
    sample_sites = []
    site_names = [
        "Sítio Arqueológico Timbó",
        "Aldeia Antiga Juruá", 
        "Sambaqui do Rio Acre",
        "Terra Preta Amazônica",
        "Sítio Cerâmico Purus"
    ]
    
    for i in range(min(num_sites, len(site_names))):
        # 候補範囲内外にバランスよく配置
        if i < len(coords):
            # 既存候補の近くに配置（検証用）
            base_x, base_y = coords[i]
            # 100-500m程度のオフセット
            offset_x = np.random.uniform(-0.005, 0.005)  # 約±500m
            offset_y = np.random.uniform(-0.005, 0.005)
            x = base_x + offset_x
            y = base_y + offset_y
        else:
            # ランダムに配置
            x = np.random.uniform(x_min - x_buffer, x_max + x_buffer)
            y = np.random.uniform(y_min - y_buffer, y_max + y_buffer)
        
        sample_sites.append({
            'site_id': i + 1,
            'site_name': site_names[i],
            'period': np.random.choice(['Pre-Columbian', 'Formative', 'Late Period']),
            'type': np.random.choice(['Settlement', 'Ceremonial', 'Burial', 'Workshop']),
            'confidence': np.random.uniform(0.7, 0.95),
            'discovery_year': np.random.randint(1990, 2020),
            'geometry': Point(x, y)
        })
    
    # GeoDataFrameを作成
    gdf = gpd.GeoDataFrame(sample_sites, crs="EPSG:4326")
    
    # GeoPackageとして保存
    gdf.to_file(output_path, driver="GPKG")
    
    print(f"✅ サンプル既知遺跡データを作成: {output_path}")
    print(f"   遺跡数: {len(gdf)}")
    print(f"   座標系: {gdf.crs}")
    
    # 概要を表示
    print("\n📊 作成された遺跡の概要:")
    for idx, site in gdf.iterrows():
        x, y = site.geometry.x, site.geometry.y
        print(f"  {site['site_id']}. {site['site_name']}")
        print(f"     座標: ({x:.4f}, {y:.4f})")
        print(f"     時代: {site['period']}, タイプ: {site['type']}")
    
    return gdf


def create_run_metadata(output_path: str, region: str = "Acre, Brazil"):
    """実行メタデータを作成する
    
    Parameters
    ----------
    output_path : str
        出力YAMLファイルパス
    region : str
        対象地域
    """
    import yaml
    from datetime import datetime
    import uuid
    
    metadata = {
        'run_id': str(uuid.uuid4())[:8],
        'timestamp': datetime.now().isoformat(),
        'region': region,
        'data_source': 'sample_data',
        'purpose': 'inspector_validator_test',
        'parameters': {
            'distance_threshold_km': 3.0,
            'water_occurrence_threshold': 5.0,
            'min_score': 0.3
        },
        'notes': 'テストデータを使用したInspector-Validator Agentの動作確認'
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 実行メタデータを作成: {output_path}")
    print(f"   実行ID: {metadata['run_id']}")
    print(f"   対象地域: {metadata['region']}")


def main():
    """メイン実行関数"""
    # ファイルパスの設定
    candidates_path = "data/output/candidates/acre_candidates.csv"
    known_sites_path = "data/raw/known_sites.gpkg"
    metadata_path = "config/run_meta.yaml"
    
    print("🏛️ サンプル既知遺跡データ作成スクリプト")
    print("=" * 50)
    
    # 出力ディレクトリの作成
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("config").mkdir(parents=True, exist_ok=True)
    
    try:
        # サンプル既知遺跡データの作成
        gdf = create_sample_known_sites(candidates_path, known_sites_path, num_sites=5)
        
        print()
        
        # 実行メタデータの作成
        create_run_metadata(metadata_path)
        
        print()
        print("🎯 次のステップ:")
        print("  1. Inspector-Validator Agentを実行:")
        print(f"     poetry run python scripts/run_inspector.py \\")
        print(f"       --candidates {candidates_path} \\")
        print(f"       --known {known_sites_path} \\")
        print(f"       --meta {metadata_path}")
        print()
        print("  2. 生成されたレポートとYAMLを確認")
        print("  3. 必要に応じてパラメータを調整")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()