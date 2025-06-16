"""
candidate: 候補地点評価モジュール

このモジュールは、閾値ベースで候補地点を評価するための機能を提供します。
S-5後半: "川が無いのに川名が残る"ポイント抽出（閾値ベース判定）
"""

import pandas as pd
import geopandas as gpd
from typing import Union


def filter_candidates(df: Union[pd.DataFrame, gpd.GeoDataFrame],
                      dist_threshold: float = 3.0,
                      occ_threshold: float = 5.0) -> gpd.GeoDataFrame:
    """閾値ベースで候補地点をフィルタリングする
    
    Parameters
    ----------
    df : Union[pd.DataFrame, gpd.GeoDataFrame]
        候補地点データ
    dist_threshold : float, optional
        距離の閾値（キロメートル）
    occ_threshold : float, optional
        水域頻度の閾値（パーセント）
        
    Returns
    -------
    gpd.GeoDataFrame
        フィルタリングされた候補地点データ
    """
    # 必要なカラムの存在確認
    required_cols = ["dist_km", "occ_pct"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must have a '{col}' column")
    
    # 条件に基づくフィルタリング
    # 1. 現河道から離れている（dist_km >= dist_threshold）
    # 2. 水域頻度が低い（occ_pct < occ_threshold）
    candidates = df[(df["dist_km"] >= dist_threshold) & 
                    (df["occ_pct"] < occ_threshold)].copy()
    
    # 候補フラグの追加
    candidates["is_candidate"] = True
    
    return candidates


def score_candidates(df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """候補地点にスコアを付ける
    
    Parameters
    ----------
    df : Union[pd.DataFrame, gpd.GeoDataFrame]
        候補地点データ
        
    Returns
    -------
    gpd.GeoDataFrame
        スコア付きの候補地点データ
    """
    # データのコピーを作成
    result_df = df.copy()
    
    # 必要なカラムの存在確認
    required_cols = ["dist_km", "occ_pct"]
    for col in required_cols:
        if col not in result_df.columns:
            raise ValueError(f"DataFrame must have a '{col}' column")
    
    # スコアの計算
    # 1. 距離スコア: 距離が遠いほど高スコア（最大1.0）
    result_df["dist_score"] = result_df["dist_km"].clip(upper=10) / 10
    
    # 2. 水域頻度スコア: 頻度が低いほど高スコア（最大1.0）
    result_df["occ_score"] = 1 - (result_df["occ_pct"].clip(upper=100) / 100)
    
    # 3. 総合スコア: 距離スコアと水域頻度スコアの加重平均
    result_df["total_score"] = 0.6 * result_df["dist_score"] + 0.4 * result_df["occ_score"]
    
    # スコアでソート
    result_df = result_df.sort_values("total_score", ascending=False)
    
    return result_df
