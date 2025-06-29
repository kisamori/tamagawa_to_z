"""バッファ生成機能 - 遺跡周辺のバッファ領域を作成する."""

from __future__ import annotations

import logging
from typing import Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)


def make_buffers(
    gdf: gpd.GeoDataFrame, 
    radius_km: float,
    target_crs: str = "EPSG:4326",
    processing_crs: str = "EPSG:3857"
) -> gpd.GeoDataFrame:
    """
    地点の周囲にバッファを作成する.
    
    Args:
        gdf: 入力GeoDataFrame（Point geometry）
        radius_km: バッファ半径（km）
        target_crs: 出力座標系（デフォルト: WGS84）
        processing_crs: 計算用座標系（デフォルト: Web Mercator）
        
    Returns:
        バッファ領域を含むGeoDataFrame
        
    Raises:
        ValueError: 入力データが不正な場合
    """
    if len(gdf) == 0:
        logger.warning("Empty GeoDataFrame provided")
        return gdf.copy()
        
    # 入力検証
    if not all(isinstance(geom, Point) for geom in gdf.geometry):
        raise ValueError("All geometries must be Point type")
        
    logger.info(f"Creating buffers for {len(gdf)} points with radius {radius_km} km")
    
    # 元の座標系を保存
    original_crs = gdf.crs
    
    # メートル単位の座標系に変換（正確なバッファ計算のため）
    gdf_projected = gdf.to_crs(processing_crs)
    
    # バッファ作成（メートル単位）
    radius_m = radius_km * 1000
    buffer_geoms = gdf_projected.geometry.buffer(radius_m)
    
    # 結果GeoDataFrame作成
    result_gdf = gdf.copy()
    result_gdf.geometry = buffer_geoms
    
    # 指定された座標系に変換
    if target_crs != processing_crs:
        result_gdf = result_gdf.to_crs(target_crs)
        
    # メタデータ追加
    result_gdf['buffer_radius_km'] = radius_km
    result_gdf['buffer_area_km2'] = result_gdf.geometry.area / (1000 * 1000)
    
    logger.info(f"Created {len(result_gdf)} buffers")
    logger.info(f"Total buffer area: {result_gdf['buffer_area_km2'].sum():.2f} km²")
    
    return result_gdf


def create_negative_samples(
    positive_gdf: gpd.GeoDataFrame,
    bbox: Union[tuple, list],
    negative_ratio: float = 4.0,
    min_distance_km: float = 1.0,
    max_attempts: int = 10000,
    target_crs: str = "EPSG:4326",
    processing_crs: str = "EPSG:3857"
) -> gpd.GeoDataFrame:
    """
    正例サイトから離れた負例サンプルを生成する.
    
    Args:
        positive_gdf: 正例サイトのGeoDataFrame
        bbox: 対象領域の境界（west, south, east, north）
        negative_ratio: 負例/正例の比率
        min_distance_km: 正例からの最小距離（km）
        max_attempts: 最大試行回数
        target_crs: 出力座標系
        processing_crs: 計算用座標系
        
    Returns:
        負例サンプルのGeoDataFrame
        
    Raises:
        ValueError: 十分な負例サンプルが生成できない場合
    """
    import numpy as np
    
    if len(positive_gdf) == 0:
        raise ValueError("No positive samples provided")
        
    logger.info(f"Generating negative samples: ratio={negative_ratio}, min_distance={min_distance_km}km")
    
    # 必要な負例数を計算
    n_negative = int(len(positive_gdf) * negative_ratio)
    
    # 境界領域の定義
    west, south, east, north = bbox
    
    # 正例サイトをメートル座標系に変換
    positive_projected = positive_gdf.to_crs(processing_crs)
    min_distance_m = min_distance_km * 1000
    
    # 負例候補を生成
    negative_samples = []
    attempts = 0
    
    np.random.seed(42)  # 再現性のため
    
    while len(negative_samples) < n_negative and attempts < max_attempts:
        # ランダムな座標を生成
        lon = np.random.uniform(west, east)
        lat = np.random.uniform(south, north)
        
        # Point作成
        candidate_point = Point(lon, lat)
        candidate_gdf = gpd.GeoDataFrame(
            geometry=[candidate_point], 
            crs=target_crs
        ).to_crs(processing_crs)
        
        # 正例サイトからの最小距離を計算
        distances = positive_projected.geometry.distance(candidate_gdf.geometry.iloc[0])
        min_dist = distances.min()
        
        # 距離条件を満たす場合に採用
        if min_dist >= min_distance_m:
            negative_samples.append({
                'lon': lon,
                'lat': lat,
                'site_name': f'negative_{len(negative_samples):04d}',
                'culture_tag': 'negative',
                'discovery_year': -1,  # 負例マーカー
                'min_distance_to_positive_km': min_dist / 1000
            })
            
        attempts += 1
        
        # 進捗ログ
        if attempts % 1000 == 0:
            logger.debug(f"Generated {len(negative_samples)}/{n_negative} negative samples (attempts: {attempts})")
            
    if len(negative_samples) < n_negative * 0.8:  # 80%未満の場合は警告
        logger.warning(f"Only generated {len(negative_samples)}/{n_negative} negative samples")
        
    # GeoDataFrame作成
    negative_df = pd.DataFrame(negative_samples)
    negative_gdf = gpd.GeoDataFrame(
        negative_df,
        geometry=gpd.points_from_xy(negative_df.lon, negative_df.lat),
        crs=target_crs
    )
    
    logger.info(f"Generated {len(negative_gdf)} negative samples")
    
    return negative_gdf


def merge_positive_negative(
    positive_gdf: gpd.GeoDataFrame,
    negative_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    正例と負例を統合する.
    
    Args:
        positive_gdf: 正例GeoDataFrame
        negative_gdf: 負例GeoDataFrame
        
    Returns:
        統合されたGeoDataFrame（is_positive列追加）
    """
    # ラベル追加
    positive_labeled = positive_gdf.copy()
    positive_labeled['is_positive'] = True
    
    negative_labeled = negative_gdf.copy()
    negative_labeled['is_positive'] = False
    
    # 統合
    merged_gdf = pd.concat([positive_labeled, negative_labeled], ignore_index=True)
    
    # 共通カラムで整理
    common_cols = ['site_name', 'lat', 'lon', 'culture_tag', 'discovery_year', 'is_positive', 'geometry']
    available_cols = [col for col in common_cols if col in merged_gdf.columns]
    
    result_gdf = gpd.GeoDataFrame(merged_gdf[available_cols], crs=positive_gdf.crs)
    
    logger.info(f"Merged dataset: {len(positive_gdf)} positive + {len(negative_gdf)} negative = {len(result_gdf)} total")
    
    return result_gdf


if __name__ == "__main__":
    # テスト実行
    import numpy as np
    
    # テスト用サンプルデータ作成
    np.random.seed(42)
    test_points = [
        Point(-68.0, -9.0),   # Acre region
        Point(-68.2, -9.2),
        Point(-63.0, -17.8),  # Casarabe region
        Point(-62.8, -17.6)
    ]
    
    test_gdf = gpd.GeoDataFrame({
        'site_name': ['Site_A', 'Site_B', 'Site_C', 'Site_D'],
        'culture_tag': ['acre', 'acre', 'casarabe', 'casarabe'],
        'discovery_year': [2015, 2018, 2019, 2020]
    }, geometry=test_points, crs="EPSG:4326")
    
    print("=== Buffer Test ===")
    buffers = make_buffers(test_gdf, radius_km=0.5)
    print(f"Original points: {len(test_gdf)}")
    print(f"Buffer areas: {buffers['buffer_area_km2'].tolist()}")
    
    print("\n=== Negative Sampling Test ===")
    bbox = (-70.0, -20.0, -60.0, -5.0)  # Amazon region
    negative_gdf = create_negative_samples(
        test_gdf, 
        bbox, 
        negative_ratio=2.0, 
        min_distance_km=1.0,
        max_attempts=1000
    )
    print(f"Generated {len(negative_gdf)} negative samples")
    
    print("\n=== Merge Test ===")
    merged = merge_positive_negative(test_gdf, negative_gdf)
    print(f"Merged dataset: {len(merged)} total")
    print(f"Positive: {merged['is_positive'].sum()}")
    print(f"Negative: {(~merged['is_positive']).sum()}")