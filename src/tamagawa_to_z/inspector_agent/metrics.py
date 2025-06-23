"""Inspector-Validator Agent メトリクス計算モジュール

このモジュールは、多言語トポニム解析の結果を評価するための
メトリクス計算機能を提供します。
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import Dict, List, Tuple, Optional


def _to_geodf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """CSV DataFrameをGeoDataFrameに変換する
    
    Parameters
    ----------
    df : pd.DataFrame
        geometry列(WKT形式)を含むDataFrame
        
    Returns
    -------
    gpd.GeoDataFrame
        変換されたGeoDataFrame
    """
    if "geometry" not in df.columns:
        raise ValueError("DataFrame must have 'geometry' column with WKT strings")
    
    return gpd.GeoDataFrame(
        df, 
        geometry=gpd.GeoSeries.from_wkt(df["geometry"]), 
        crs="EPSG:4326"
    )


def _reproject_for_distance(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """距離計算用に適切な投影座標系に変換する
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        地理座標系のGeoDataFrame
        
    Returns
    -------
    gpd.GeoDataFrame
        投影座標系に変換されたGeoDataFrame
    """
    if gdf.crs and gdf.crs.is_geographic:
        # アマゾン地域用のUTM座標系を使用 (Zone 21S for most of Amazon basin)
        return gdf.to_crs("EPSG:32721")  # UTM Zone 21S
    return gdf


def recall_at_k(candidates, known: gpd.GeoDataFrame, k: int = 100) -> float:
    """Recall@K指標を計算する
    
    上位K件の候補の中に既知遺跡がどの程度含まれているかを計算します。
    
    Parameters
    ----------
    candidates : pd.DataFrame or gpd.GeoDataFrame
        候補データ（total_score列でソート済み想定）
    known : gpd.GeoDataFrame
        既知遺跡のGeoDataFrame
    k : int, optional
        上位K件（デフォルト: 100）
        
    Returns
    -------
    float
        Recall@K値（0-1の範囲）
    """
    if len(candidates) == 0 or len(known) == 0:
        return 0.0
    
    # 候補をGeoDataFrameに変換
    if isinstance(candidates, gpd.GeoDataFrame):
        cand_gdf = candidates
    else:
        cand_gdf = _to_geodf(candidates)
    
    # 上位K件を取得
    top_k = cand_gdf.nlargest(min(k, len(cand_gdf)), "total_score")
    
    # 距離計算用に投影座標系に変換
    known_proj = _reproject_for_distance(known)
    top_k_proj = _reproject_for_distance(top_k)
    
    # 既知遺跡との最近傍結合（500m以内）
    matched = gpd.sjoin_nearest(
        known_proj, 
        top_k_proj, 
        how="inner", 
        max_distance=500  # 500m以内
    )
    
    # Recall@K = マッチした既知遺跡数 / 全既知遺跡数
    return len(matched) / len(known)


def map_score(candidates, known: gpd.GeoDataFrame) -> float:
    """Mean Average Precision (mAP) を計算する
    
    Parameters
    ----------
    candidates : pd.DataFrame or gpd.GeoDataFrame
        候補データ（total_score列でソート済み想定）
    known : gpd.GeoDataFrame
        既知遺跡のGeoDataFrame
        
    Returns
    -------
    float
        mAP値（0-1の範囲）
    """
    if len(candidates) == 0 or len(known) == 0:
        return 0.0
    
    # 候補をGeoDataFrameに変換
    if isinstance(candidates, gpd.GeoDataFrame):
        cand_gdf = candidates
    else:
        cand_gdf = _to_geodf(candidates)
    
    # total_scoreで降順ソート
    cand_sorted = cand_gdf.sort_values("total_score", ascending=False).reset_index(drop=True)
    
    # 距離計算用に投影座標系に変換
    known_proj = _reproject_for_distance(known)
    cand_sorted_proj = _reproject_for_distance(cand_sorted)
    
    # 各候補について既知遺跡との距離を計算
    precisions = []
    tp_count = 0
    
    for i, candidate in cand_sorted_proj.iterrows():
        # 候補点から500m以内に既知遺跡があるかチェック
        candidate_point = candidate.geometry
        is_hit = any(
            candidate_point.distance(known_site.geometry) <= 500  # 500m（メートル単位）
            for _, known_site in known_proj.iterrows()
        )
        
        if is_hit:
            tp_count += 1
        
        # Precision@(i+1)を計算
        precision_at_i = tp_count / (i + 1)
        precisions.append(precision_at_i)
    
    # mAPを計算（簡易版：各位置でのprecisionの平均）
    return np.mean(precisions) if precisions else 0.0


def workload(candidates: pd.DataFrame) -> int:
    """候補総数（作業負荷指標）を計算する
    
    Parameters
    ----------
    candidates : pd.DataFrame
        候補データ
        
    Returns
    -------
    int
        候補総数
    """
    return len(candidates)


def root_diversity(candidates: pd.DataFrame, root_column: str = "root") -> float:
    """語根の多様性指標（シャノン多様度）を計算する
    
    Parameters
    ----------
    candidates : pd.DataFrame
        候補データ
    root_column : str, optional
        語根を示すカラム名（デフォルト: "root"）
        
    Returns
    -------
    float
        シャノン多様度
    """
    if root_column not in candidates.columns or len(candidates) == 0:
        return 0.0
    
    # 語根の頻度を計算
    root_counts = candidates[root_column].value_counts()
    total = len(candidates)
    
    # シャノン多様度を計算
    shannon_entropy = 0.0
    for count in root_counts:
        if count > 0:
            p = count / total
            shannon_entropy -= p * np.log(p)
    
    return shannon_entropy


def calculate_all_metrics(
    candidates: pd.DataFrame, 
    known: gpd.GeoDataFrame,
    k_values: List[int] = [5, 10, 20, 50, 100]
) -> Dict[str, float]:
    """全ての評価指標を一括計算する
    
    Parameters
    ----------
    candidates : pd.DataFrame
        候補データ
    known : gpd.GeoDataFrame
        既知遺跡データ
    k_values : List[int], optional
        Recall@Kで使用するK値のリスト
        
    Returns
    -------
    Dict[str, float]
        各種評価指標の辞書
    """
    metrics = {}
    
    # Recall@K for multiple K values
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(candidates, known, k)
    
    # mAP
    metrics["map"] = map_score(candidates, known)
    
    # Workload
    metrics["workload"] = workload(candidates)
    
    # Root diversity
    metrics["root_diversity"] = root_diversity(candidates)
    
    return metrics


def analyze_spatial_distribution(candidates: pd.DataFrame) -> Dict[str, float]:
    """候補の空間分布を分析する
    
    Parameters
    ----------
    candidates : pd.DataFrame
        候補データ
        
    Returns
    -------
    Dict[str, float]
        空間分布の統計情報
    """
    if len(candidates) == 0:
        return {}
    
    cand_gdf = _to_geodf(candidates)
    
    # 境界ボックス
    bounds = cand_gdf.total_bounds
    
    # 座標の分散
    coords = np.array([[point.x, point.y] for point in cand_gdf.geometry])
    coord_std = np.std(coords, axis=0)
    
    return {
        "bbox_width": bounds[2] - bounds[0],  # xmax - xmin
        "bbox_height": bounds[3] - bounds[1],  # ymax - ymin
        "coord_std_x": coord_std[0],
        "coord_std_y": coord_std[1],
        "centroid_x": np.mean(coords[:, 0]),
        "centroid_y": np.mean(coords[:, 1])
    }