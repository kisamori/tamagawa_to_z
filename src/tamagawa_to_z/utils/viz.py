# DEMO FILE: 可視化ユーティリティ

"""
viz: 可視化ユーティリティモジュール

このモジュールは、データの可視化に関するユーティリティ関数を提供します。
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any, Callable


def setup_figure(figsize: Tuple[float, float] = (10, 8),
                style: str = 'default',
                dpi: int = 100) -> Tuple[Figure, Axes]:
    """図の設定を行う
    
    Parameters
    ----------
    figsize : Tuple[float, float], optional
        図のサイズ
    style : str, optional
        スタイル
    dpi : int, optional
        解像度
        
    Returns
    -------
    Tuple[Figure, Axes]
        図とAxes
    """
    # スタイルの設定
    if style == 'default':
        plt.style.use('default')
    elif style == 'dark':
        plt.style.use('dark_background')
    elif style == 'seaborn':
        plt.style.use('seaborn-v0_8-whitegrid')
    
    # 図の作成
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    return fig, ax


def plot_raster(data: np.ndarray,
               ax: Optional[Axes] = None,
               cmap: str = 'viridis',
               vmin: Optional[float] = None,
               vmax: Optional[float] = None,
               title: Optional[str] = None,
               colorbar: bool = True,
               colorbar_label: Optional[str] = None,
               alpha: float = 1.0) -> Tuple[Figure, Axes]:
    """ラスターデータをプロットする
    
    Parameters
    ----------
    data : np.ndarray
        ラスターデータ
    ax : Optional[Axes], optional
        Axes
    cmap : str, optional
        カラーマップ
    vmin : Optional[float], optional
        最小値
    vmax : Optional[float], optional
        最大値
    title : Optional[str], optional
        タイトル
    colorbar : bool, optional
        カラーバーを表示するかどうか
    colorbar_label : Optional[str], optional
        カラーバーのラベル
    alpha : float, optional
        透明度
        
    Returns
    -------
    Tuple[Figure, Axes]
        図とAxes
    """
    # Axesの設定
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    # データの範囲
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    
    # ラスターデータのプロット
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
    
    # タイトルの設定
    if title:
        ax.set_title(title)
    
    # カラーバーの設定
    if colorbar:
        cbar = fig.colorbar(im, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    
    return fig, ax


def plot_vector(gdf: gpd.GeoDataFrame,
               ax: Optional[Axes] = None,
               column: Optional[str] = None,
               cmap: str = 'viridis',
               vmin: Optional[float] = None,
               vmax: Optional[float] = None,
               markersize: float = 50,
               linewidth: float = 1.0,
               alpha: float = 0.8,
               legend: bool = True,
               legend_title: Optional[str] = None,
               title: Optional[str] = None) -> Tuple[Figure, Axes]:
    """ベクターデータをプロットする
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        ベクターデータ
    ax : Optional[Axes], optional
        Axes
    column : Optional[str], optional
        色分けするカラム
    cmap : str, optional
        カラーマップ
    vmin : Optional[float], optional
        最小値
    vmax : Optional[float], optional
        最大値
    markersize : float, optional
        マーカーサイズ
    linewidth : float, optional
        線の太さ
    alpha : float, optional
        透明度
    legend : bool, optional
        凡例を表示するかどうか
    legend_title : Optional[str], optional
        凡例のタイトル
    title : Optional[str], optional
        タイトル
        
    Returns
    -------
    Tuple[Figure, Axes]
        図とAxes
    """
    # Axesの設定
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    # ベクターデータのプロット
    gdf.plot(column=column, cmap=cmap, vmin=vmin, vmax=vmax,
             markersize=markersize, linewidth=linewidth, alpha=alpha,
             legend=legend, legend_kwds={'title': legend_title} if legend_title else None,
             ax=ax)
    
    # タイトルの設定
    if title:
        ax.set_title(title)
    
    # 軸の設定
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    return fig, ax


def plot_raster_with_vector(raster_data: np.ndarray,
                           vector_data: gpd.GeoDataFrame,
                           raster_transform: Optional[Any] = None,
                           raster_crs: Optional[Any] = None,
                           ax: Optional[Axes] = None,
                           raster_cmap: str = 'viridis',
                           vector_column: Optional[str] = None,
                           vector_cmap: str = 'plasma',
                           raster_alpha: float = 1.0,
                           vector_alpha: float = 0.8,
                           title: Optional[str] = None) -> Tuple[Figure, Axes]:
    """ラスターデータとベクターデータを重ねてプロットする
    
    Parameters
    ----------
    raster_data : np.ndarray
        ラスターデータ
    vector_data : gpd.GeoDataFrame
        ベクターデータ
    raster_transform : Optional[Any], optional
        ラスターデータの変換行列
    raster_crs : Optional[Any], optional
        ラスターデータの座標参照系
    ax : Optional[Axes], optional
        Axes
    raster_cmap : str, optional
        ラスターデータのカラーマップ
    vector_column : Optional[str], optional
        ベクターデータの色分けするカラム
    vector_cmap : str, optional
        ベクターデータのカラーマップ
    raster_alpha : float, optional
        ラスターデータの透明度
    vector_alpha : float, optional
        ベクターデータの透明度
    title : Optional[str], optional
        タイトル
        
    Returns
    -------
    Tuple[Figure, Axes]
        図とAxes
    """
    # Axesの設定
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.figure
    
    # ラスターデータのプロット
    if raster_transform is not None and raster_crs is not None:
        # rasterioを使用してプロット
        import rasterio
        from rasterio.plot import show
        
        # ダミーのラスターデータセットを作成
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=raster_data.shape[0],
                width=raster_data.shape[1],
                count=1,
                dtype=raster_data.dtype,
                crs=raster_crs,
                transform=raster_transform
            ) as dataset:
                dataset.write(raster_data, 1)
            
            # ラスターデータのプロット
            show(dataset, ax=ax, cmap=raster_cmap, alpha=raster_alpha)
    else:
        # matplotlibを使用してプロット
        im = ax.imshow(raster_data, cmap=raster_cmap, alpha=raster_alpha)
        fig.colorbar(im, ax=ax)
    
    # ベクターデータのプロット
    vector_data.plot(column=vector_column, cmap=vector_cmap, alpha=vector_alpha, ax=ax)
    
    # タイトルの設定
    if title:
        ax.set_title(title)
    
    return fig, ax


def plot_time_series(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                    x: Optional[Union[str, np.ndarray]] = None,
                    y: Optional[Union[str, List[str]]] = None,
                    ax: Optional[Axes] = None,
                    title: Optional[str] = None,
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    legend: bool = True,
                    grid: bool = True,
                    style: Optional[str] = None) -> Tuple[Figure, Axes]:
    """時系列データをプロットする
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, np.ndarray]
        時系列データ
    x : Optional[Union[str, np.ndarray]], optional
        x軸のデータ
    y : Optional[Union[str, List[str]]], optional
        y軸のデータ
    ax : Optional[Axes], optional
        Axes
    title : Optional[str], optional
        タイトル
    xlabel : Optional[str], optional
        x軸のラベル
    ylabel : Optional[str], optional
        y軸のラベル
    legend : bool, optional
        凡例を表示するかどうか
    grid : bool, optional
        グリッドを表示するかどうか
    style : Optional[str], optional
        スタイル
        
    Returns
    -------
    Tuple[Figure, Axes]
        図とAxes
    """
    # Axesの設定
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure
    
    # データの種類に応じたプロット
    if isinstance(data, pd.DataFrame):
        if y is None:
            # すべての列をプロット
            data.plot(x=x, ax=ax, style=style)
        else:
            # 指定された列をプロット
            if isinstance(y, list):
                data[y].plot(x=x, ax=ax, style=style)
            else:
                data[y].plot(x=x, ax=ax, style=style)
    
    elif isinstance(data, pd.Series):
        data.plot(ax=ax, style=style)
    
    elif isinstance(data, np.ndarray):
        if x is None:
            # インデックスをx軸として使用
            x = np.arange(len(data))
        
        ax.plot(x, data, style)
    
    # タイトルと軸ラベルの設定
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    # 凡例とグリッドの設定
    if legend:
        ax.legend()
    if grid:
        ax.grid(True)
    
    return fig, ax


def plot_histogram(data: Union[np.ndarray, pd.Series, List[float]],
                  ax: Optional[Axes] = None,
                  bins: Union[int, str, List[float]] = 'auto',
                  density: bool = False,
                  title: Optional[str] = None,
                  xlabel: Optional[str] = None,
                  ylabel: Optional[str] = None,
                  color: Optional[str] = None,
                  alpha: float = 0.7,
                  grid: bool = True,
                  kde: bool = False) -> Tuple[Figure, Axes]:
    """ヒストグラムをプロットする
    
    Parameters
    ----------
    data : Union[np.ndarray, pd.Series, List[float]]
        データ
    ax : Optional[Axes], optional
        Axes
    bins : Union[int, str, List[float]], optional
        ビンの数または範囲
    density : bool, optional
        密度を表示するかどうか
    title : Optional[str], optional
        タイトル
    xlabel : Optional[str], optional
        x軸のラベル
    ylabel : Optional[str], optional
        y軸のラベル
    color : Optional[str], optional
        色
    alpha : float, optional
        透明度
    grid : bool, optional
        グリッドを表示するかどうか
    kde : bool, optional
        KDEを表示するかどうか
        
    Returns
    -------
    Tuple[Figure, Axes]
        図とAxes
    """
    # Axesの設定
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # データの種類に応じたプロット
    if kde:
        # seabornを使用してKDEを表示
        sns.histplot(data, bins=bins, kde=True, ax=ax, color=color, alpha=alpha, stat='density' if density else 'count')
    else:
        # matplotlibを使用してヒストグラムを表示
        ax.hist(data, bins=bins, density=density, color=color, alpha=alpha)
    
    # タイトルと軸ラベルの設定
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    # グリッドの設定
    if grid:
        ax.grid(True)
    
    return fig, ax


def plot_scatter(x: Union[np.ndarray, pd.Series, List[float]],
                y: Union[np.ndarray, pd.Series, List[float]],
                c: Optional[Union[np.ndarray, pd.Series, List[float]]] = None,
                s: Optional[Union[float, np.ndarray, pd.Series, List[float]]] = None,
                ax: Optional[Axes] = None,
                cmap: str = 'viridis',
                title: Optional[str] = None,
                xlabel: Optional[str] = None,
                ylabel: Optional[str] = None,
                colorbar: bool = True,
                colorbar_label: Optional[str] = None,
                alpha: float = 0.7,
                grid: bool = True) -> Tuple[Figure, Axes]:
    """散布図をプロットする
    
    Parameters
    ----------
    x : Union[np.ndarray, pd.Series, List[float]]
        x軸のデータ
    y : Union[np.ndarray, pd.Series, List[float]]
        y軸のデータ
    c : Optional[Union[np.ndarray, pd.Series, List[float]]], optional
        色のデータ
    s : Optional[Union[float, np.ndarray, pd.Series, List[float]]], optional
        サイズのデータ
    ax : Optional[Axes], optional
        Axes
    cmap : str, optional
        カラーマップ
    title : Optional[str], optional
        タイトル
    xlabel : Optional[str], optional
        x軸のラベル
    ylabel : Optional[str], optional
        y軸のラベル
    colorbar : bool, optional
        カラーバーを表示するかどうか
    colorbar_label : Optional[str], optional
        カラーバーのラベル
    alpha : float, optional
        透明度
    grid : bool, optional
        グリッドを表示するかどうか
        
    Returns
    -------
    Tuple[Figure, Axes]
        図とAxes
    """
    # Axesの設定
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    # 散布図のプロット
    sc = ax.scatter(x, y, c=c, s=s, cmap=cmap, alpha=alpha)
    
    # タイトルと軸ラベルの設定
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    # カラーバーの設定
    if colorbar and c is not None:
        cbar = fig.colorbar(sc, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    
    # グリッドの設定
    if grid:
        ax.grid(True)
    
    return fig, ax


def save_figure(fig: Figure,
               output_path: str,
               dpi: int = 300,
               bbox_inches: str = 'tight',
               pad_inches: float = 0.1,
               transparent: bool = False) -> None:
    """図を保存する
    
    Parameters
    ----------
    fig : Figure
        図
    output_path : str
        出力ファイルパス
    dpi : int, optional
        解像度
    bbox_inches : str, optional
        境界ボックス
    pad_inches : float, optional
        パディング
    transparent : bool, optional
        透明にするかどうか
    """
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 図の保存
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, transparent=transparent)
    
    print(f"Figure saved to {output_path}")
