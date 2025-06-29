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
import os
import time

# GDALのメモリ設定を最適化
os.environ['GDAL_CACHEMAX'] = '256'  # さらに少なく（256MB）
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['VSI_CACHE'] = 'FALSE'
os.environ['GDAL_MAX_DATASET_POOL_SIZE'] = '50'


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
    
    # 空のGeoDataFrameの場合は処理をスキップ
    if result_gdf.empty:
        print("警告: 水域頻度計算の入力GeoDataFrameが空です。処理をスキップします。")
        # 空のGeoDataFrameに'occ_pct'カラムを追加して返す
        result_gdf["occ_pct"] = []
        return result_gdf
    
    try:
        # メモリ効率化のため、バッチ処理を実装
        batch_size = 5  # 一度に処理するジオメトリの数を制限（さらに小さく）
        occ_values = []
        
        print(f"水域頻度計算: {len(result_gdf)}件のポイントを{batch_size}件ずつ処理中...")
        
        for i in range(0, len(result_gdf), batch_size):
            batch_end = min(i + batch_size, len(result_gdf))
            batch_gdf = result_gdf.iloc[i:batch_end]
            
            print(f"  バッチ {i//batch_size + 1}/{(len(result_gdf)-1)//batch_size + 1}: {i+1}-{batch_end}件目を処理中...")
            
            try:
                # バッチごとにゾーン統計の計算
                # 最も安全なアプローチ：1つずつ処理 + タイムアウト
                batch_stats = []
                for idx, geom in batch_gdf["geometry"].items():
                    try:
                        start_time = time.time()
                        single_stat = zonal_stats(
                            vectors=[geom],
                            raster=gsw_tif,
                            stats=["mean"],
                            nodata=255,
                            all_touched=True
                        )
                        end_time = time.time()
                        
                        # 処理時間が30秒を超えたら警告
                        if end_time - start_time > 30:
                            print(f"    ジオメトリ {idx} の処理時間: {end_time - start_time:.2f}秒")
                        
                        batch_stats.extend(single_stat)
                        
                    except Exception as single_e:
                        print(f"    ジオメトリ {idx} でエラー: {single_e}")
                        batch_stats.append({"mean": 0})
                    
                    # 少し待機してメモリを解放
                    time.sleep(0.1)
                
                # バッチ結果を追加
                batch_values = [s["mean"] or 0 for s in batch_stats]
                occ_values.extend(batch_values)
                
            except Exception as batch_e:
                print(f"  バッチ {i//batch_size + 1} でエラー発生: {batch_e}")
                # エラーの場合は0で埋める
                batch_values = [0] * len(batch_gdf)
                occ_values.extend(batch_values)
        
        # 結果を設定
        result_gdf["occ_pct"] = occ_values
        print(f"水域頻度計算完了: {len(occ_values)}件処理")
        
    except Exception as e:
        print(f"水域頻度計算中にエラーが発生しました: {e}")
        # エラーが発生した場合でも処理を続行できるように'occ_pct'カラムを追加
        result_gdf["occ_pct"] = [0] * len(result_gdf)
    
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
    
    # 空のGeoDataFrameの場合は処理をスキップ
    if result_gdf.empty:
        print("警告: バッファ水域頻度計算の入力GeoDataFrameが空です。処理をスキップします。")
        # 空のGeoDataFrameに必要なカラムを追加して返す
        result_gdf["buffer_occ_mean"] = []
        result_gdf["buffer_occ_max"] = []
        result_gdf["buffer_occ_min"] = []
        result_gdf["buffer_occ_count"] = []
        return result_gdf
    
    try:
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
    except Exception as e:
        print(f"バッファ水域頻度計算中にエラーが発生しました: {e}")
        # エラーが発生した場合でも処理を続行できるように必要なカラムを追加
        result_gdf["buffer_occ_mean"] = [0] * len(result_gdf)
        result_gdf["buffer_occ_max"] = [0] * len(result_gdf)
        result_gdf["buffer_occ_min"] = [0] * len(result_gdf)
        result_gdf["buffer_occ_count"] = [0] * len(result_gdf)
    
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
    # 空のGeoDataFrameの場合は処理をスキップ
    if gdf.empty:
        print("警告: 候補地点抽出の入力GeoDataFrameが空です。空の結果を返します。")
        # 空のGeoDataFrameをそのまま返す
        return gdf.copy()
    
    try:
        # 必要なカラムの存在確認
        required_cols = ["dist_km", "occ_pct"]
        for col in required_cols:
            if col not in gdf.columns:
                print(f"警告: 必要なカラム '{col}' がGeoDataFrameに存在しません。空の結果を返します。")
                return gpd.GeoDataFrame([], columns=gdf.columns, crs=gdf.crs)
        
        # 条件に基づくフィルタリング
        # 1. 現河道から離れている（dist_km >= dist_threshold）
        # 2. 水域頻度が低い（occ_pct < occ_threshold）
        candidates = gdf[(gdf["dist_km"] >= dist_threshold) & 
                         (gdf["occ_pct"] < occ_threshold)].copy()
        
        # 候補フラグの追加
        candidates["is_candidate"] = True
    except Exception as e:
        print(f"候補地点抽出中にエラーが発生しました: {e}")
        # エラーが発生した場合は空のGeoDataFrameを返す
        return gpd.GeoDataFrame([], columns=gdf.columns, crs=gdf.crs)
    
    return candidates
