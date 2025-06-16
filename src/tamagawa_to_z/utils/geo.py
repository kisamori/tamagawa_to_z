# DEMO FILE: 地理空間ユーティリティ

"""
geo: 地理空間ユーティリティモジュール

このモジュールは、地理空間データの処理に関するユーティリティ関数を提供します。
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box
from shapely.ops import unary_union
from typing import Dict, List, Tuple, Union, Optional, Any


def reproject_vector(gdf: gpd.GeoDataFrame, 
                    target_crs: Union[str, Dict[str, Any]]) -> gpd.GeoDataFrame:
    """ベクターデータの座標参照系を変換する
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        ベクターデータ
    target_crs : Union[str, Dict[str, Any]]
        変換先の座標参照系
        
    Returns
    -------
    gpd.GeoDataFrame
        変換後のベクターデータ
    """
    # 座標参照系の変換
    return gdf.to_crs(target_crs)


def reproject_raster(src_path: str, 
                    dst_path: str,
                    target_crs: Union[str, Dict[str, Any]],
                    resampling: str = 'nearest') -> None:
    """ラスターデータの座標参照系を変換する
    
    Parameters
    ----------
    src_path : str
        入力ラスターファイルのパス
    dst_path : str
        出力ラスターファイルのパス
    target_crs : Union[str, Dict[str, Any]]
        変換先の座標参照系
    resampling : str, optional
        リサンプリング方法
    """
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    
    # リサンプリング方法の設定
    resampling_methods = {
        'nearest': rasterio.warp.Resampling.nearest,
        'bilinear': rasterio.warp.Resampling.bilinear,
        'cubic': rasterio.warp.Resampling.cubic,
        'cubic_spline': rasterio.warp.Resampling.cubic_spline,
        'lanczos': rasterio.warp.Resampling.lanczos,
        'average': rasterio.warp.Resampling.average,
        'mode': rasterio.warp.Resampling.mode,
        'max': rasterio.warp.Resampling.max,
        'min': rasterio.warp.Resampling.min,
        'med': rasterio.warp.Resampling.med,
        'q1': rasterio.warp.Resampling.q1,
        'q3': rasterio.warp.Resampling.q3
    }
    
    resampling_method = resampling_methods.get(resampling, rasterio.warp.Resampling.nearest)
    
    # 座標参照系の変換
    with rasterio.open(src_path) as src:
        # 変換先の座標参照系
        dst_crs = target_crs
        
        # 変換先の変換行列とサイズを計算
        dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        
        # 変換先のメタデータ
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({
            'crs': dst_crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height
        })
        
        # 変換先のラスターファイルを作成
        with rasterio.open(dst_path, 'w', **dst_kwargs) as dst:
            for i in range(1, src.count + 1):
                # バンドごとに変換
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling_method)
    
    print(f"Raster reprojected to {dst_path}")


def buffer_vector(gdf: gpd.GeoDataFrame, 
                 distance: float,
                 dissolve: bool = False) -> gpd.GeoDataFrame:
    """ベクターデータにバッファを適用する
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        ベクターデータ
    distance : float
        バッファ距離
    dissolve : bool, optional
        バッファを結合するかどうか
        
    Returns
    -------
    gpd.GeoDataFrame
        バッファを適用したベクターデータ
    """
    # バッファの適用
    buffered = gdf.copy()
    buffered['geometry'] = buffered.geometry.buffer(distance)
    
    # バッファの結合
    if dissolve:
        buffered = buffered.dissolve()
    
    return buffered


def rasterize_vector(gdf: gpd.GeoDataFrame,
                    out_shape: Tuple[int, int],
                    transform: Optional[Any] = None,
                    fill: float = 0,
                    value_column: Optional[str] = None,
                    all_touched: bool = False) -> np.ndarray:
    """ベクターデータをラスタライズする
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        ベクターデータ
    out_shape : Tuple[int, int]
        出力ラスターの形状 (height, width)
    transform : Optional[Any], optional
        変換行列
    fill : float, optional
        塗りつぶし値
    value_column : Optional[str], optional
        値のカラム
    all_touched : bool, optional
        ピクセルに触れるすべてのジオメトリを含めるかどうか
        
    Returns
    -------
    np.ndarray
        ラスタライズされたデータ
    """
    # 変換行列の設定
    if transform is None:
        # デフォルトの変換行列
        bounds = gdf.total_bounds
        xmin, ymin, xmax, ymax = bounds
        width, height = out_shape[1], out_shape[0]
        
        # ピクセルサイズの計算
        pixel_width = (xmax - xmin) / width
        pixel_height = (ymax - ymin) / height
        
        # 変換行列の作成
        transform = from_origin(xmin, ymax, pixel_width, pixel_height)
    
    # 値の設定
    if value_column is None:
        # デフォルト値（1）を使用
        shapes = [(geom, 1) for geom in gdf.geometry]
    else:
        # 指定されたカラムの値を使用
        shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[value_column])]
    
    # ラスタライズ
    rasterized = features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=fill,
        all_touched=all_touched,
        dtype=np.float32
    )
    
    return rasterized


def vectorize_raster(raster_data: np.ndarray,
                    transform: Any,
                    crs: Any,
                    mask: Optional[np.ndarray] = None,
                    connectivity: int = 4) -> gpd.GeoDataFrame:
    """ラスターデータをベクトライズする
    
    Parameters
    ----------
    raster_data : np.ndarray
        ラスターデータ
    transform : Any
        変換行列
    crs : Any
        座標参照系
    mask : Optional[np.ndarray], optional
        マスク
    connectivity : int, optional
        接続性
        
    Returns
    -------
    gpd.GeoDataFrame
        ベクトライズされたデータ
    """
    # マスクの設定
    if mask is None:
        # デフォルトのマスク（0以外の値）
        mask = raster_data > 0
    
    # ベクトライズ
    shapes = features.shapes(
        raster_data,
        mask=mask,
        transform=transform,
        connectivity=connectivity
    )
    
    # ジオメトリとプロパティの抽出
    geometries = []
    properties = []
    
    for geom, value in shapes:
        geometries.append(Polygon(geom['coordinates'][0]))
        properties.append({'value': value})
    
    # GeoDataFrameの作成
    gdf = gpd.GeoDataFrame(
        properties,
        geometry=geometries,
        crs=crs
    )
    
    return gdf


def clip_raster_by_vector(raster_data: np.ndarray,
                         transform: Any,
                         vector_data: gpd.GeoDataFrame,
                         all_touched: bool = False) -> np.ndarray:
    """ベクターデータでラスターデータをクリップする
    
    Parameters
    ----------
    raster_data : np.ndarray
        ラスターデータ
    transform : Any
        変換行列
    vector_data : gpd.GeoDataFrame
        ベクターデータ
    all_touched : bool, optional
        ピクセルに触れるすべてのジオメトリを含めるかどうか
        
    Returns
    -------
    np.ndarray
        クリップされたラスターデータ
    """
    # ジオメトリの結合
    if len(vector_data) > 1:
        geometry = unary_union(vector_data.geometry)
    else:
        geometry = vector_data.geometry.iloc[0]
    
    # マスクの作成
    mask = rasterio.features.geometry_mask(
        [geometry],
        out_shape=raster_data.shape,
        transform=transform,
        invert=True,
        all_touched=all_touched
    )
    
    # クリップ
    clipped_data = raster_data.copy()
    clipped_data[~mask] = 0
    
    return clipped_data


def clip_vector_by_vector(target_gdf: gpd.GeoDataFrame,
                         clip_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """ベクターデータでベクターデータをクリップする
    
    Parameters
    ----------
    target_gdf : gpd.GeoDataFrame
        クリップ対象のベクターデータ
    clip_gdf : gpd.GeoDataFrame
        クリップに使用するベクターデータ
        
    Returns
    -------
    gpd.GeoDataFrame
        クリップされたベクターデータ
    """
    # 座標参照系の一致を確認
    if target_gdf.crs != clip_gdf.crs:
        clip_gdf = clip_gdf.to_crs(target_gdf.crs)
    
    # クリップ
    clipped = gpd.clip(target_gdf, clip_gdf)
    
    return clipped


def create_grid(bounds: Tuple[float, float, float, float],
               resolution: Union[float, Tuple[float, float]],
               crs: Any) -> gpd.GeoDataFrame:
    """グリッドを作成する
    
    Parameters
    ----------
    bounds : Tuple[float, float, float, float]
        範囲 (xmin, ymin, xmax, ymax)
    resolution : Union[float, Tuple[float, float]]
        解像度 (x_res, y_res) または単一の値
    crs : Any
        座標参照系
        
    Returns
    -------
    gpd.GeoDataFrame
        グリッド
    """
    # 解像度の設定
    if isinstance(resolution, (int, float)):
        x_res = y_res = resolution
    else:
        x_res, y_res = resolution
    
    # 範囲の設定
    xmin, ymin, xmax, ymax = bounds
    
    # グリッドの作成
    x_coords = np.arange(xmin, xmax, x_res)
    y_coords = np.arange(ymin, ymax, y_res)
    
    # グリッドセルの作成
    cells = []
    for x in x_coords:
        for y in y_coords:
            cells.append(box(x, y, x + x_res, y + y_res))
    
    # GeoDataFrameの作成
    grid = gpd.GeoDataFrame(
        {'id': range(len(cells))},
        geometry=cells,
        crs=crs
    )
    
    return grid


def calculate_zonal_statistics(raster_data: np.ndarray,
                              transform: Any,
                              vector_data: gpd.GeoDataFrame,
                              stats: List[str] = ['mean', 'std', 'min', 'max', 'sum', 'count']) -> gpd.GeoDataFrame:
    """ゾーン統計を計算する
    
    Parameters
    ----------
    raster_data : np.ndarray
        ラスターデータ
    transform : Any
        変換行列
    vector_data : gpd.GeoDataFrame
        ベクターデータ
    stats : List[str], optional
        計算する統計量
        
    Returns
    -------
    gpd.GeoDataFrame
        ゾーン統計を含むベクターデータ
    """
    # 結果の初期化
    result = vector_data.copy()
    
    # 各ジオメトリに対してゾーン統計を計算
    for i, geom in enumerate(vector_data.geometry):
        # ジオメトリのマスクを作成
        mask = rasterio.features.geometry_mask(
            [geom],
            out_shape=raster_data.shape,
            transform=transform,
            invert=True
        )
        
        # マスク内のピクセル値を取得
        values = raster_data[mask]
        
        # 統計量の計算
        for stat in stats:
            if stat == 'mean':
                result.loc[i, 'mean'] = np.mean(values) if len(values) > 0 else np.nan
            elif stat == 'std':
                result.loc[i, 'std'] = np.std(values) if len(values) > 0 else np.nan
            elif stat == 'min':
                result.loc[i, 'min'] = np.min(values) if len(values) > 0 else np.nan
            elif stat == 'max':
                result.loc[i, 'max'] = np.max(values) if len(values) > 0 else np.nan
            elif stat == 'sum':
                result.loc[i, 'sum'] = np.sum(values) if len(values) > 0 else np.nan
            elif stat == 'count':
                result.loc[i, 'count'] = len(values)
    
    return result


def calculate_distance_raster(source_gdf: gpd.GeoDataFrame,
                             shape: Tuple[int, int],
                             transform: Any,
                             max_distance: Optional[float] = None) -> np.ndarray:
    """距離ラスターを計算する
    
    Parameters
    ----------
    source_gdf : gpd.GeoDataFrame
        距離の基準となるベクターデータ
    shape : Tuple[int, int]
        出力ラスターの形状 (height, width)
    transform : Any
        変換行列
    max_distance : Optional[float], optional
        最大距離
        
    Returns
    -------
    np.ndarray
        距離ラスター
    """
    # ソースのラスタライズ
    source_raster = rasterize_vector(
        source_gdf,
        shape,
        transform,
        fill=0,
        value_column=None,
        all_touched=True
    )
    
    # 距離変換
    from scipy.ndimage import distance_transform_edt
    
    # ピクセルサイズの計算
    pixel_width = transform[0]
    pixel_height = -transform[4]
    
    # ピクセル単位の距離を計算
    distance_pixels = distance_transform_edt(source_raster == 0)
    
    # 実際の距離に変換
    distance = distance_pixels * np.sqrt((pixel_width**2 + pixel_height**2) / 2)
    
    # 最大距離の設定
    if max_distance is not None:
        distance[distance > max_distance] = max_distance
    
    return distance


def calculate_slope_aspect(dem: np.ndarray,
                          resolution: Union[float, Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """傾斜と方位を計算する
    
    Parameters
    ----------
    dem : np.ndarray
        数値標高モデル
    resolution : Union[float, Tuple[float, float]]
        解像度 (x_res, y_res) または単一の値
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        傾斜と方位
    """
    # 解像度の設定
    if isinstance(resolution, (int, float)):
        x_res = y_res = resolution
    else:
        x_res, y_res = resolution
    
    # 勾配の計算
    from scipy.ndimage import sobel
    
    # x方向の勾配
    dx = sobel(dem, axis=1) / (8 * x_res)
    # y方向の勾配
    dy = sobel(dem, axis=0) / (8 * y_res)
    
    # 傾斜の計算（ラジアン）
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    # 傾斜の計算（度）
    slope_deg = np.degrees(slope_rad)
    
    # 方位の計算（ラジアン）
    aspect_rad = np.arctan2(dy, dx)
    # 方位の計算（度）
    aspect_deg = np.degrees(aspect_rad)
    # 方位の調整（0-360度）
    aspect_deg = (aspect_deg + 90) % 360
    
    return slope_deg, aspect_deg


def calculate_hillshade(dem: np.ndarray,
                       resolution: Union[float, Tuple[float, float]],
                       azimuth: float = 315,
                       altitude: float = 45) -> np.ndarray:
    """陰影起伏図を計算する
    
    Parameters
    ----------
    dem : np.ndarray
        数値標高モデル
    resolution : Union[float, Tuple[float, float]]
        解像度 (x_res, y_res) または単一の値
    azimuth : float, optional
        方位角（度）
    altitude : float, optional
        高度角（度）
        
    Returns
    -------
    np.ndarray
        陰影起伏図
    """
    # 傾斜と方位の計算
    slope, aspect = calculate_slope_aspect(dem, resolution)
    
    # ラジアンに変換
    slope_rad = np.radians(slope)
    aspect_rad = np.radians(aspect)
    azimuth_rad = np.radians(azimuth)
    altitude_rad = np.radians(altitude)
    
    # 陰影起伏図の計算
    hillshade = np.sin(altitude_rad) * np.sin(slope_rad) + \
                np.cos(altitude_rad) * np.cos(slope_rad) * np.cos(azimuth_rad - aspect_rad)
    
    # 0-255の範囲に正規化
    hillshade = 255 * (hillshade + 1) / 2
    
    return hillshade
