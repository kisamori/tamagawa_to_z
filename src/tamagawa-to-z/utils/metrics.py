# DEMO FILE: 評価指標ユーティリティ

"""
metrics: 評価指標ユーティリティモジュール

このモジュールは、モデルの評価や指標の計算に関するユーティリティ関数を提供します。
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple, Union, Optional, Any


def calculate_binary_metrics(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            y_score: Optional[np.ndarray] = None,
                            threshold: float = 0.5) -> Dict[str, float]:
    """二値分類の評価指標を計算する
    
    Parameters
    ----------
    y_true : np.ndarray
        真のラベル
    y_pred : np.ndarray
        予測ラベル
    y_score : Optional[np.ndarray], optional
        予測スコア
    threshold : float, optional
        閾値
        
    Returns
    -------
    Dict[str, float]
        評価指標
    """
    # 予測ラベルの作成
    if y_score is not None:
        y_pred = (y_score >= threshold).astype(int)
    
    # 混同行列の計算
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 評価指標の計算
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # ROC AUCの計算
    auc = roc_auc_score(y_true, y_score) if y_score is not None else None
    
    # 結果の辞書
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }
    
    if auc is not None:
        metrics['auc'] = auc
    
    return metrics


def calculate_regression_metrics(y_true: np.ndarray,
                               y_pred: np.ndarray) -> Dict[str, float]:
    """回帰の評価指標を計算する
    
    Parameters
    ----------
    y_true : np.ndarray
        真の値
    y_pred : np.ndarray
        予測値
        
    Returns
    -------
    Dict[str, float]
        評価指標
    """
    # 平均二乗誤差
    mse = np.mean((y_true - y_pred) ** 2)
    # 平均絶対誤差
    mae = np.mean(np.abs(y_true - y_pred))
    # 平均二乗誤差の平方根
    rmse = np.sqrt(mse)
    # 決定係数
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    # 結果の辞書
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    return metrics


def calculate_iou(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 threshold: float = 0.5) -> float:
    """IoU（Intersection over Union）を計算する
    
    Parameters
    ----------
    y_true : np.ndarray
        真のマスク
    y_pred : np.ndarray
        予測マスク
    threshold : float, optional
        閾値
        
    Returns
    -------
    float
        IoU
    """
    # 二値化
    y_true_binary = (y_true > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # 積集合と和集合の計算
    intersection = np.logical_and(y_true_binary, y_pred_binary).sum()
    union = np.logical_or(y_true_binary, y_pred_binary).sum()
    
    # IoUの計算
    iou = intersection / union if union > 0 else 0
    
    return iou


def calculate_dice_coefficient(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              threshold: float = 0.5) -> float:
    """Diceの類似係数を計算する
    
    Parameters
    ----------
    y_true : np.ndarray
        真のマスク
    y_pred : np.ndarray
        予測マスク
    threshold : float, optional
        閾値
        
    Returns
    -------
    float
        Diceの類似係数
    """
    # 二値化
    y_true_binary = (y_true > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # 積集合の計算
    intersection = np.logical_and(y_true_binary, y_pred_binary).sum()
    
    # Diceの類似係数の計算
    dice = 2 * intersection / (y_true_binary.sum() + y_pred_binary.sum()) if (y_true_binary.sum() + y_pred_binary.sum()) > 0 else 0
    
    return dice


def calculate_ensemble_metrics(ensemble_predictions: np.ndarray,
                              y_true: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """アンサンブル予測の評価指標を計算する
    
    Parameters
    ----------
    ensemble_predictions : np.ndarray
        アンサンブル予測（shape: (n_ensemble, ...)）
    y_true : Optional[np.ndarray], optional
        真の値
        
    Returns
    -------
    Dict[str, Any]
        評価指標
    """
    # アンサンブル平均
    ensemble_mean = np.mean(ensemble_predictions, axis=0)
    # アンサンブル標準偏差
    ensemble_std = np.std(ensemble_predictions, axis=0)
    # アンサンブル変動係数
    ensemble_cv = ensemble_std / ensemble_mean if np.any(ensemble_mean != 0) else np.zeros_like(ensemble_std)
    
    # 結果の辞書
    metrics = {
        'ensemble_mean': ensemble_mean,
        'ensemble_std': ensemble_std,
        'ensemble_cv': ensemble_cv
    }
    
    # 真の値が与えられた場合は追加の指標を計算
    if y_true is not None:
        # 平均二乗誤差
        mse = np.mean((y_true - ensemble_mean) ** 2)
        # 平均絶対誤差
        mae = np.mean(np.abs(y_true - ensemble_mean))
        # 平均二乗誤差の平方根
        rmse = np.sqrt(mse)
        
        # 結果の辞書に追加
        metrics.update({
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        })
    
    return metrics


def calculate_spatial_metrics(gdf_true: gpd.GeoDataFrame,
                             gdf_pred: gpd.GeoDataFrame,
                             buffer_distance: float = 0.001) -> Dict[str, float]:
    """空間的な評価指標を計算する
    
    Parameters
    ----------
    gdf_true : gpd.GeoDataFrame
        真のベクターデータ
    gdf_pred : gpd.GeoDataFrame
        予測ベクターデータ
    buffer_distance : float, optional
        バッファ距離
        
    Returns
    -------
    Dict[str, float]
        評価指標
    """
    # 座標参照系の一致を確認
    if gdf_true.crs != gdf_pred.crs:
        gdf_pred = gdf_pred.to_crs(gdf_true.crs)
    
    # バッファの作成
    gdf_true_buffer = gdf_true.copy()
    gdf_true_buffer['geometry'] = gdf_true_buffer.geometry.buffer(buffer_distance)
    
    gdf_pred_buffer = gdf_pred.copy()
    gdf_pred_buffer['geometry'] = gdf_pred_buffer.geometry.buffer(buffer_distance)
    
    # 空間的な一致の計算
    # 真陽性：予測ジオメトリが真のジオメトリと交差する
    tp = sum(1 for pred_geom in gdf_pred.geometry for true_geom in gdf_true_buffer.geometry if pred_geom.intersects(true_geom))
    
    # 偽陽性：予測ジオメトリが真のジオメトリと交差しない
    fp = len(gdf_pred) - tp
    
    # 偽陰性：真のジオメトリが予測ジオメトリと交差しない
    fn = sum(1 for true_geom in gdf_true.geometry for pred_geom in gdf_pred_buffer.geometry if not true_geom.intersects(pred_geom))
    
    # 評価指標の計算
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 結果の辞書
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }
    
    return metrics


def calculate_hit_rate(sites_gdf: gpd.GeoDataFrame,
                      prediction_raster: np.ndarray,
                      transform: Any,
                      threshold: float = 0.5) -> Dict[str, float]:
    """遺跡ヒットレートを計算する
    
    Parameters
    ----------
    sites_gdf : gpd.GeoDataFrame
        遺跡のベクターデータ
    prediction_raster : np.ndarray
        予測ラスター
    transform : Any
        変換行列
    threshold : float, optional
        閾値
        
    Returns
    -------
    Dict[str, float]
        評価指標
    """
    # 予測ラスターの二値化
    prediction_binary = prediction_raster >= threshold
    
    # 各遺跡に対してヒットを確認
    hits = 0
    total_sites = len(sites_gdf)
    
    for idx, site in sites_gdf.iterrows():
        # 遺跡の座標を取得
        x, y = site.geometry.x, site.geometry.y
        
        # 座標をピクセル座標に変換
        col, row = ~transform * (x, y)
        col, row = int(col), int(row)
        
        # ピクセル座標が有効範囲内かつ予測が陽性の場合はヒット
        if (0 <= row < prediction_binary.shape[0] and 
            0 <= col < prediction_binary.shape[1] and 
            prediction_binary[row, col]):
            hits += 1
    
    # ヒットレートの計算
    hit_rate = hits / total_sites if total_sites > 0 else 0
    
    # 結果の辞書
    metrics = {
        'hit_rate': hit_rate,
        'hits': hits,
        'total_sites': total_sites
    }
    
    return metrics


def calculate_priority_grid_metrics(priority_grid: gpd.GeoDataFrame,
                                   sites_gdf: gpd.GeoDataFrame,
                                   priority_column: str = 'priority') -> Dict[str, Dict[str, float]]:
    """優先度グリッドの評価指標を計算する
    
    Parameters
    ----------
    priority_grid : gpd.GeoDataFrame
        優先度グリッド
    sites_gdf : gpd.GeoDataFrame
        遺跡のベクターデータ
    priority_column : str, optional
        優先度のカラム
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        評価指標
    """
    # 座標参照系の一致を確認
    if priority_grid.crs != sites_gdf.crs:
        sites_gdf = sites_gdf.to_crs(priority_grid.crs)
    
    # 優先度の一覧
    priorities = sorted(priority_grid[priority_column].unique())
    
    # 各優先度に対して評価指標を計算
    metrics = {}
    
    for priority in priorities:
        # 優先度のグリッドを抽出
        grid_priority = priority_grid[priority_grid[priority_column] == priority]
        
        # グリッド内の遺跡数を計算
        sites_in_grid = gpd.sjoin(sites_gdf, grid_priority, how='inner', predicate='within')
        sites_in_grid_count = len(sites_in_grid)
        
        # グリッドの面積を計算
        grid_area = grid_priority.geometry.area.sum()
        
        # グリッドの割合を計算
        grid_ratio = len(grid_priority) / len(priority_grid)
        
        # 遺跡密度を計算
        site_density = sites_in_grid_count / grid_area if grid_area > 0 else 0
        
        # ヒットレートを計算
        hit_rate = sites_in_grid_count / len(sites_gdf) if len(sites_gdf) > 0 else 0
        
        # 効率を計算
        efficiency = hit_rate / grid_ratio if grid_ratio > 0 else 0
        
        # 結果の辞書
        metrics[f'priority_{priority}'] = {
            'hit_rate': hit_rate,
            'sites_count': sites_in_grid_count,
            'grid_count': len(grid_priority),
            'grid_ratio': grid_ratio,
            'site_density': site_density,
            'efficiency': efficiency
        }
    
    return metrics


def calculate_convergence_metrics(iterations: List[int],
                                 errors: List[float]) -> Dict[str, float]:
    """収束の評価指標を計算する
    
    Parameters
    ----------
    iterations : List[int]
        反復回数
    errors : List[float]
        誤差
        
    Returns
    -------
    Dict[str, float]
        評価指標
    """
    # 収束までの反復回数
    convergence_iterations = len(iterations)
    
    # 最終誤差
    final_error = errors[-1] if errors else None
    
    # 収束率の計算
    if len(errors) >= 2:
        error_ratio = errors[-1] / errors[0] if errors[0] != 0 else 0
        convergence_rate = 1 - error_ratio
    else:
        convergence_rate = 0
    
    # 結果の辞書
    metrics = {
        'convergence_iterations': convergence_iterations,
        'final_error': final_error,
        'convergence_rate': convergence_rate
    }
    
    return metrics


def calculate_cost_function(dem_error: float,
                           watermask_iou: float,
                           toponym_correlation: float,
                           dem_weight: float = 0.4,
                           watermask_weight: float = 0.3,
                           toponym_weight: float = 0.3) -> float:
    """コスト関数を計算する
    
    Parameters
    ----------
    dem_error : float
        DEMの誤差
    watermask_iou : float
        水域マスクのIoU
    toponym_correlation : float
        地名の相関
    dem_weight : float, optional
        DEMの重み
    watermask_weight : float, optional
        水域マスクの重み
    toponym_weight : float, optional
        地名の重み
        
    Returns
    -------
    float
        コスト
    """
    # 各項の計算
    dem_cost = dem_error
    watermask_cost = 1 - watermask_iou
    toponym_cost = 1 - toponym_correlation
    
    # 重み付き合計
    cost = (dem_weight * dem_cost + 
            watermask_weight * watermask_cost + 
            toponym_weight * toponym_cost)
    
    return cost
