# DEMO FILE: 入出力ユーティリティ

"""
io: 入出力ユーティリティモジュール

このモジュールは、ファイルの読み書きや、データの入出力に関するユーティリティ関数を提供します。
"""

import os
import json
import yaml
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import xarray as xr
from typing import Dict, List, Tuple, Union, Optional, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """設定ファイルを読み込む
    
    Parameters
    ----------
    config_path : str
        設定ファイルのパス
        
    Returns
    -------
    Dict[str, Any]
        設定情報
    """
    # ファイル拡張子の取得
    _, ext = os.path.splitext(config_path)
    
    # ファイル形式に応じた読み込み
    if ext.lower() in ['.yml', '.yaml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif ext.lower() == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {ext}")
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """設定ファイルを保存する
    
    Parameters
    ----------
    config : Dict[str, Any]
        設定情報
    output_path : str
        出力ファイルパス
    """
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # ファイル拡張子の取得
    _, ext = os.path.splitext(output_path)
    
    # ファイル形式に応じた保存
    if ext.lower() in ['.yml', '.yaml']:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=False)
    elif ext.lower() == '.json':
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config file format: {ext}")
    
    print(f"Config saved to {output_path}")


def load_raster(raster_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """ラスターデータを読み込む
    
    Parameters
    ----------
    raster_path : str
        ラスターファイルのパス
        
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        ラスターデータと属性情報
    """
    with rasterio.open(raster_path) as src:
        # ラスターデータの読み込み
        data = src.read(1)  # 最初のバンドを読み込む
        
        # 属性情報の取得
        meta = src.meta.copy()
        
        # 追加情報
        info = {
            'bounds': src.bounds,
            'crs': src.crs.to_string(),
            'transform': src.transform,
            'nodata': src.nodata
        }
    
    return data, {**meta, **info}


def save_raster(data: np.ndarray, 
               meta: Dict[str, Any],
               output_path: str) -> None:
    """ラスターデータを保存する
    
    Parameters
    ----------
    data : np.ndarray
        ラスターデータ
    meta : Dict[str, Any]
        属性情報
    output_path : str
        出力ファイルパス
    """
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 必要な属性情報の抽出
    required_meta = {
        'driver': meta.get('driver', 'GTiff'),
        'height': data.shape[0],
        'width': data.shape[1],
        'count': 1,
        'dtype': data.dtype,
        'crs': meta.get('crs', None),
        'transform': meta.get('transform', None),
        'nodata': meta.get('nodata', None)
    }
    
    # ラスターデータの保存
    with rasterio.open(output_path, 'w', **required_meta) as dst:
        dst.write(data, 1)
    
    print(f"Raster saved to {output_path}")


def load_vector(vector_path: str) -> gpd.GeoDataFrame:
    """ベクターデータを読み込む
    
    Parameters
    ----------
    vector_path : str
        ベクターファイルのパス
        
    Returns
    -------
    gpd.GeoDataFrame
        ベクターデータ
    """
    # ファイル拡張子の取得
    _, ext = os.path.splitext(vector_path)
    
    # ファイル形式に応じた読み込み
    if ext.lower() in ['.shp', '.geojson', '.gpkg']:
        gdf = gpd.read_file(vector_path)
    elif ext.lower() == '.csv':
        # CSVファイルの場合、緯度経度カラムを指定
        df = pd.read_csv(vector_path)
        
        # 緯度経度カラムの推定
        lon_cols = ['lon', 'longitude', 'long', 'x']
        lat_cols = ['lat', 'latitude', 'y']
        
        lon_col = next((col for col in lon_cols if col in df.columns), None)
        lat_col = next((col for col in lat_cols if col in df.columns), None)
        
        if lon_col and lat_col:
            # 緯度経度からジオメトリを作成
            from shapely.geometry import Point
            geometry = [Point(x, y) for x, y in zip(df[lon_col], df[lat_col])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        else:
            raise ValueError(f"Could not find latitude and longitude columns in {vector_path}")
    else:
        raise ValueError(f"Unsupported vector file format: {ext}")
    
    return gdf


def save_vector(gdf: gpd.GeoDataFrame, 
               output_path: str,
               driver: Optional[str] = None) -> None:
    """ベクターデータを保存する
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        ベクターデータ
    output_path : str
        出力ファイルパス
    driver : Optional[str], optional
        ドライバー
    """
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # ファイル拡張子の取得
    _, ext = os.path.splitext(output_path)
    
    # ドライバーの設定
    if driver is None:
        if ext.lower() == '.shp':
            driver = 'ESRI Shapefile'
        elif ext.lower() == '.geojson':
            driver = 'GeoJSON'
        elif ext.lower() == '.gpkg':
            driver = 'GPKG'
        else:
            driver = 'GeoJSON'
    
    # ベクターデータの保存
    gdf.to_file(output_path, driver=driver)
    
    print(f"Vector data saved to {output_path}")


def load_netcdf(netcdf_path: str) -> xr.Dataset:
    """NetCDFデータを読み込む
    
    Parameters
    ----------
    netcdf_path : str
        NetCDFファイルのパス
        
    Returns
    -------
    xr.Dataset
        NetCDFデータ
    """
    # NetCDFデータの読み込み
    ds = xr.open_dataset(netcdf_path)
    
    return ds


def save_netcdf(ds: xr.Dataset, 
               output_path: str,
               encoding: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    """NetCDFデータを保存する
    
    Parameters
    ----------
    ds : xr.Dataset
        NetCDFデータ
    output_path : str
        出力ファイルパス
    encoding : Optional[Dict[str, Dict[str, Any]]], optional
        エンコーディング
    """
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # NetCDFデータの保存
    ds.to_netcdf(output_path, encoding=encoding)
    
    print(f"NetCDF data saved to {output_path}")


def load_pickle(pickle_path: str) -> Any:
    """Pickleファイルを読み込む
    
    Parameters
    ----------
    pickle_path : str
        Pickleファイルのパス
        
    Returns
    -------
    Any
        Pickleデータ
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def save_pickle(data: Any, 
               output_path: str) -> None:
    """Pickleファイルを保存する
    
    Parameters
    ----------
    data : Any
        保存するデータ
    output_path : str
        出力ファイルパス
    """
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Pickleファイルの保存
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Pickle data saved to {output_path}")
