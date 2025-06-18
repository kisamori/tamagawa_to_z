"""
watermask_safe: 安全な水域マスク処理モジュール

大きなGSWラスターでセグメンテーションフォルトが発生する場合の
フォールバック実装を提供します。
"""

import numpy as np
import geopandas as gpd
from typing import Union, Optional, List, Dict, Any
import warnings

def water_occurrence_safe(gdf: gpd.GeoDataFrame, 
                         gsw_tif: str = "data/raw/GSW_occurrence.tif") -> gpd.GeoDataFrame:
    """GSW occurrenceデータから水域頻度を計算する（安全版）
    
    大きなラスターファイルでGDALがクラッシュする場合の
    フォールバック実装です。距離ベースの簡易推定を使用します。
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        地名データ
    gsw_tif : str, optional
        GSW occurrenceファイルのパス（使用されません）
        
    Returns
    -------
    gpd.GeoDataFrame
        水域頻度情報が追加された地名データ
    """
    # 地名データのコピーを作成
    result_gdf = gdf.copy()
    
    # 空のGeoDataFrameの場合は処理をスキップ
    if result_gdf.empty:
        print("警告: 水域頻度計算の入力GeoDataFrameが空です。処理をスキップします。")
        result_gdf["occ_pct"] = []
        return result_gdf
    
    print("フォールバック水域頻度計算を実行中（距離ベース推定）...")
    
    # 距離ベースの簡易推定
    # 現河道からの距離が近いほど水域頻度が高いと仮定
    if 'dist_km' in result_gdf.columns:
        # 距離を水域頻度に変換（逆相関）
        # 距離0km = 100%、距離10km以上 = 0%
        max_dist = 10.0
        result_gdf["occ_pct"] = np.maximum(
            0, 
            100 * (1 - np.minimum(result_gdf["dist_km"], max_dist) / max_dist)
        )
        
        # 水系タイプによる調整
        type_multipliers = {
            'rio': 1.0,      # 川
            'igarape': 0.8,  # 小川
            'lagoa': 0.6,    # 湖
            'parana': 0.9,   # 支流
            'porto': 0.4     # 港（陸上地名の可能性が高い）
        }
        
        for idx, row in result_gdf.iterrows():
            water_type = row.get('type', 'unknown')
            multiplier = type_multipliers.get(water_type, 0.5)
            result_gdf.loc[idx, 'occ_pct'] = result_gdf.loc[idx, 'occ_pct'] * multiplier
            
    else:
        # 距離情報がない場合は、タイプのみで推定
        print("距離情報がないため、タイプのみで水域頻度を推定します")
        type_base_occ = {
            'rio': 50,
            'igarape': 30,
            'lagoa': 40,
            'parana': 45,
            'porto': 10
        }
        
        result_gdf["occ_pct"] = result_gdf["type"].map(
            lambda t: type_base_occ.get(t, 20)
        )
    
    print(f"フォールバック水域頻度計算完了: {len(result_gdf)}件処理")
    return result_gdf


def water_occurrence_with_fallback(gdf: gpd.GeoDataFrame, 
                                  gsw_tif: str = "data/raw/GSW_occurrence.tif") -> gpd.GeoDataFrame:
    """GSW occurrenceデータから水域頻度を計算する（フォールバック付き）
    
    まず通常のrasterstatsを試行し、失敗した場合は安全版を使用します。
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        地名データ
    gsw_tif : str, optional
        GSW occurrenceファイルのパス
        
    Returns
    -------
    gpd.GeoDataFrame
        水域頻度情報が追加された地名データ
    """
    try:
        # まず通常の処理を試行
        from .watermask import water_occurrence
        print("通常の水域頻度計算を試行中...")
        return water_occurrence(gdf, gsw_tif)
        
    except Exception as e:
        print(f"通常の水域頻度計算が失敗しました: {e}")
        print("フォールバック処理に切り替えます...")
        warnings.warn(
            "GSWラスター処理が失敗したため、距離ベースの推定を使用します。", 
            UserWarning
        )
        return water_occurrence_safe(gdf, gsw_tif)