"""
distance: 地名と現河道との距離計算モジュール

このモジュールは、地名と現河道（HydroRIVERS）との距離を計算するための機能を提供します。
S-4: 現河道との距離計算
"""

import geopandas as gpd
from pyproj import CRS
from typing import Union, Optional


def attach_distance(names_gdf: gpd.GeoDataFrame, rivers_path: str) -> gpd.GeoDataFrame:
    """地名と現河道との距離を計算する
    
    Parameters
    ----------
    names_gdf : gpd.GeoDataFrame
        地名データ
    rivers_path : str
        HydroRIVERSファイルのパス
        
    Returns
    -------
    gpd.GeoDataFrame
        距離情報が追加された地名データ
    """
    # 地名データのコピーを作成
    result_gdf = names_gdf.copy()
    
    # HydroRIVERSの読み込み
    rivers = gpd.read_file(rivers_path).to_crs(4326)
    
    # メルカトル投影に変換して距離計算（メートル単位）
    names_proj = result_gdf.to_crs(3857)
    rivers_union = rivers.to_crs(3857).unary_union
    
    # 距離計算（メートル→キロメートル変換）
    result_gdf["dist_km"] = names_proj.distance(rivers_union) / 1000
    
    return result_gdf


def find_nearest_river(names_gdf: gpd.GeoDataFrame, rivers_path: str) -> gpd.GeoDataFrame:
    """各地名に最も近い河川を特定する
    
    Parameters
    ----------
    names_gdf : gpd.GeoDataFrame
        地名データ
    rivers_path : str
        HydroRIVERSファイルのパス
        
    Returns
    -------
    gpd.GeoDataFrame
        最寄りの河川情報が追加された地名データ
    """
    # 地名データのコピーを作成
    result_gdf = names_gdf.copy()
    
    # HydroRIVERSの読み込み
    rivers = gpd.read_file(rivers_path).to_crs(4326)
    
    # 必要なカラムのみ選択
    rivers_slim = rivers[["HYRIV_ID", "ORD_STRA", "geometry"]].copy()
    
    # メルカトル投影に変換
    names_proj = result_gdf.to_crs(3857)
    rivers_proj = rivers_slim.to_crs(3857)
    
    # 最寄りの河川を検索
    nearest = gpd.sjoin_nearest(
        names_proj,
        rivers_proj,
        how="left",
        distance_col="dist_m"
    )
    
    # 結果を元の座標系に戻す
    nearest = nearest.to_crs(4326)
    
    # 距離をキロメートルに変換
    nearest["dist_km"] = nearest["dist_m"] / 1000
    
    # 不要なカラムを削除
    nearest = nearest.drop(columns=["dist_m", "index_right"])
    
    return nearest


def classify_by_distance(names_gdf: gpd.GeoDataFrame, 
                         threshold_km: float = 3.0) -> gpd.GeoDataFrame:
    """距離に基づいて地名を分類する
    
    Parameters
    ----------
    names_gdf : gpd.GeoDataFrame
        距離情報が追加された地名データ
    threshold_km : float, optional
        距離の閾値（キロメートル）
        
    Returns
    -------
    gpd.GeoDataFrame
        分類情報が追加された地名データ
    """
    # 地名データのコピーを作成
    result_gdf = names_gdf.copy()
    
    # 'dist_km'カラムがない場合はエラー
    if "dist_km" not in result_gdf.columns:
        raise ValueError("GeoDataFrame must have a 'dist_km' column")
    
    # 距離に基づく分類
    result_gdf["distance_class"] = "near"  # デフォルト
    
    # 閾値以上の距離を持つ地名を「遠い」に分類
    far_mask = result_gdf["dist_km"] >= threshold_km
    result_gdf.loc[far_mask, "distance_class"] = "far"
    
    return result_gdf
