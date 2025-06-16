"""
watermask: 水域マスク処理モジュール

このモジュールは、Global Surface Water (GSW) occurrenceデータを使用して
水域頻度を計算するための機能を提供します。
S-5前半: "川が無いのに川名が残る"ポイント抽出（水域頻度判定）
"""

import numpy as np
import geopandas as gpd
from rasterstats import zonal_stats
from typing import Union, Optional, List, Dict, Any


def water_occurrence(gdf: gpd.GeoDataFrame, 
                     gsw_tif: str = "data/raw/GSW_occurrence.tif") -> gpd.GeoDataFrame:
    """GSW occurrenceデータから水域頻度を計算する
    
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
    # 地名データのコピーを作成
    result_gdf = gdf.copy()
    
    # ゾーン統計の計算
    # GSWデータでは255がNODATAなので、それを除外
    stats = zonal_stats(
        vectors=result_gdf["geometry"],
        raster=gsw_tif,
        stats=["mean"],
        nodata=255
    )
    
    # 結果の処理（NODATAの場合は0に設定）
    result_gdf["occ_pct"] = [s["mean"] or 0 for s in stats]
    
    return result_gdf


def buffer_occurrence(gdf: gpd.GeoDataFrame, 
                      gsw_tif: str = "data/raw/GSW_occurrence.tif",
                      buffer_km: float = 1.0) -> gpd.GeoDataFrame:
    """バッファ領域内の水域頻度を計算する
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        地名データ
    gsw_tif : str, optional
        GSW occurrenceファイルのパス
    buffer_km : float, optional
        バッファ半径（キロメートル）
        
    Returns
    -------
    gpd.GeoDataFrame
        バッファ領域の水域頻度情報が追加された地名データ
    """
    # 地名データのコピーを作成
    result_gdf = gdf.copy()
    
    # メルカトル投影に変換
    gdf_proj = result_gdf.to_crs(3857)
    
    # バッファの作成（メートル単位）
    buffer_m = buffer_km * 1000
    gdf_buffer = gdf_proj.copy()
    gdf_buffer["geometry"] = gdf_proj.buffer(buffer_m)
    
    # 元の座標系に戻す
    gdf_buffer = gdf_buffer.to_crs(4326)
    
    # ゾーン統計の計算
    stats = zonal_stats(
        vectors=gdf_buffer["geometry"],
        raster=gsw_tif,
        stats=["mean", "max", "min", "count"],
        nodata=255
    )
    
    # 結果の処理
    result_gdf["buffer_occ_mean"] = [s["mean"] or 0 for s in stats]
    result_gdf["buffer_occ_max"] = [s["max"] or 0 for s in stats]
    result_gdf["buffer_occ_min"] = [s["min"] or 0 for s in stats]
    result_gdf["buffer_occ_count"] = [s["count"] or 0 for s in stats]
    
    return result_gdf


def classify_by_occurrence(gdf: gpd.GeoDataFrame, 
                           threshold_pct: float = 5.0) -> gpd.GeoDataFrame:
    """水域頻度に基づいて地名を分類する
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        水域頻度情報が追加された地名データ
    threshold_pct : float, optional
        水域頻度の閾値（パーセント）
        
    Returns
    -------
    gpd.GeoDataFrame
        分類情報が追加された地名データ
    """
    # 地名データのコピーを作成
    result_gdf = gdf.copy()
    
    # 'occ_pct'カラムがない場合はエラー
    if "occ_pct" not in result_gdf.columns:
        raise ValueError("GeoDataFrame must have an 'occ_pct' column")
    
    # 水域頻度に基づく分類
    result_gdf["water_class"] = "wet"  # デフォルト
    
    # 閾値未満の水域頻度を持つ地名を「乾燥」に分類
    dry_mask = result_gdf["occ_pct"] < threshold_pct
    result_gdf.loc[dry_mask, "water_class"] = "dry"
    
    return result_gdf


def find_paleo_candidates(gdf: gpd.GeoDataFrame,
                          dist_threshold: float = 3.0,
                          occ_threshold: float = 5.0) -> gpd.GeoDataFrame:
    """古河道候補地点を抽出する
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        距離と水域頻度情報が追加された地名データ
    dist_threshold : float, optional
        距離の閾値（キロメートル）
    occ_threshold : float, optional
        水域頻度の閾値（パーセント）
        
    Returns
    -------
    gpd.GeoDataFrame
        古河道候補地点
    """
    # 必要なカラムの存在確認
    required_cols = ["dist_km", "occ_pct"]
    for col in required_cols:
        if col not in gdf.columns:
            raise ValueError(f"GeoDataFrame must have a '{col}' column")
    
    # 条件に基づくフィルタリング
    # 1. 現河道から離れている（dist_km >= dist_threshold）
    # 2. 水域頻度が低い（occ_pct < occ_threshold）
    candidates = gdf[(gdf["dist_km"] >= dist_threshold) & 
                     (gdf["occ_pct"] < occ_threshold)].copy()
    
    # 候補フラグの追加
    candidates["is_candidate"] = True
    
    return candidates
