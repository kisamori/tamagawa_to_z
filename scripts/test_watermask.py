#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_watermask.py: 水域頻度計算のテストスクリプト

このスクリプトは、水域頻度計算の問題を小さなデータセットでテストします。
"""

import os
import sys
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# プロジェクトのルートディレクトリをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from tamagawa_to_z.harmonizer.watermask import water_occurrence

def create_test_data():
    """テスト用の小さなデータセットを作成"""
    # アクレ州の範囲内でテスト用のポイントを作成
    test_points = [
        Point(-69.0, -9.0),
        Point(-69.5, -9.5),
        Point(-68.5, -10.0)
    ]
    
    gdf = gpd.GeoDataFrame({
        'name': ['Test Point 1', 'Test Point 2', 'Test Point 3'],
        'type': ['test', 'test', 'test']
    }, geometry=test_points, crs='EPSG:4326')
    
    return gdf

def main():
    """メイン処理"""
    print("水域頻度計算のテストを開始します...")
    
    # テストデータの作成
    test_gdf = create_test_data()
    print(f"テストデータ: {len(test_gdf)}件のポイント")
    
    # GSWファイルのパス
    gsw_path = PROJECT_ROOT / 'data/raw/GSW_occurrence/occurrence_70W_10Sv1_4_2021.tif'
    
    if not gsw_path.exists():
        print(f"エラー: GSWファイルが見つかりません: {gsw_path}")
        return
    
    print(f"GSWファイル: {gsw_path}")
    
    try:
        # 水域頻度計算の実行
        result = water_occurrence(test_gdf, str(gsw_path))
        
        print("テスト成功!")
        print("結果:")
        print(result[['name', 'occ_pct']])
        
    except Exception as e:
        print(f"テスト失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()