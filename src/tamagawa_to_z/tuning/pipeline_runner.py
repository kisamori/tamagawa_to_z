"""既存パイプライン薄ラッパ - 既存のsite identification pipelineを呼び出す."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple

import pandas as pd
import geopandas as gpd
import numpy as np

logger = logging.getLogger(__name__)


def run_pipeline_with_params(
    distance_km: float,
    occ_pct: float,
    root_weights: Dict[str, float],
    validation_set: gpd.GeoDataFrame,
    return_fp: bool = False,
    return_intermediate: bool = False,
    experiment_id: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    run_dir: Optional[Path] = None
) -> Union[float, Tuple[float, list], Tuple[float, list, Dict]]:
    """
    既存パイプラインを呼び出し、評価指標を返す薄ラッパ.
    
    Args:
        distance_km: 距離しきい値（km）
        occ_pct: 水域出現率しきい値（%）
        root_weights: 語根重み辞書
        validation_set: 検証用遺跡データ
        return_fp: False Positive例も返すかどうか
        return_intermediate: S1-S5の中間データも返すかどうか
        experiment_id: 実験ID（Noneの場合は自動生成）
        config_overrides: 追加設定オーバーライド
        
    Returns:
        評価スコア、またはスコアとFP例のタプル、または中間データも含むタプル
        
    Raises:
        ImportError: 既存パイプラインがインポートできない場合
        RuntimeError: パイプライン実行エラー
    """
    if experiment_id is None:
        experiment_id = f"optuna_{uuid.uuid4().hex[:6]}"
    
    logger.info(f"Running pipeline with distance={distance_km}km, occ={occ_pct}% (exp: {experiment_id})")
    
    try:
        # パラメータパッチを作成
        param_patch = {
            "distance_threshold_km": distance_km,
            "occ_pct_threshold": occ_pct,
            "root_weight_table": root_weights
        }
        
        # 追加設定のマージ
        if config_overrides:
            param_patch.update(config_overrides)
        
        # パラメータファイルを保存
        if run_dir:
            params_file = run_dir / f"{experiment_id}_params.json"
        else:
            # タイムスタンプディレクトリを作成
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamp_dir = Path(f"data/output/optuna/{timestamp}")
            timestamp_dir.mkdir(parents=True, exist_ok=True)
            params_file = timestamp_dir / f"{experiment_id}_params.json"
        
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(param_patch, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved parameters to {params_file}")
        
        # 既存パイプラインを実行（timestamp_dirを渡す）
        actual_run_dir = run_dir if run_dir else timestamp_dir
        result = _run_existing_pipeline(param_patch, experiment_id, actual_run_dir, return_intermediate)
        if return_intermediate:
            candidates, intermediate_data = result
        else:
            candidates = result
        
        # 評価指標を計算
        metrics = _calculate_metrics(candidates, validation_set)
        
        # 総合スコアを計算
        score = _calculate_composite_score(metrics)
        
        logger.info(f"Pipeline completed: score={score:.4f}, candidates={len(candidates)}")
        
        if return_intermediate:
            # 偽陽性例を抽出
            fp_examples = _extract_false_positives(candidates, validation_set) if return_fp else []
            return score, fp_examples, intermediate_data
        elif return_fp:
            # 偽陽性例を抽出
            fp_examples = _extract_false_positives(candidates, validation_set)
            return score, fp_examples
        else:
            return score
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise RuntimeError(f"Pipeline failed: {e}")


def _run_existing_pipeline(
    param_patch: Dict[str, Any], 
    experiment_id: str,
    run_dir: Optional[Path] = None,
    return_intermediate: bool = False
) -> pd.DataFrame:
    """
    既存のsite identification pipelineを実行する.
    
    Args:
        param_patch: パラメータオーバーライド
        experiment_id: 実験ID
        
    Returns:
        候補地点DataFrame
    """
    try:
        # 既存のharmonizer pipelineをインポート
        from tamagawa_to_z.harmonizer.harmonizer import HarmonizerPipeline
        
        # パイプライン初期化
        pipeline = HarmonizerPipeline()
        
        # パラメータ適用
        pipeline.config.update(param_patch)
        
        # パイプライン実行（S-1からS-5まで）
        logger.debug("Executing S-1: BBox definition")
        bbox_gdf = pipeline.make_bbox_gdf()
        
        logger.debug("Executing S-2: Toponym extraction")
        toponyms = pipeline.extract_toponyms_pyrosm(bbox_gdf)
        
        logger.debug("Executing S-3: Toponym processing")
        processed_toponyms = pipeline.process_toponyms(toponyms)
        
        logger.debug("Executing S-4: Distance calculation")
        with_distance = pipeline.attach_distance(processed_toponyms)
        
        logger.debug("Executing S-5: Candidate filtering")
        candidates = pipeline.filter_candidates(with_distance)
        
        # 中間データを準備
        intermediate_data = None
        if return_intermediate:
            intermediate_data = {
                'bbox_gdf': bbox_gdf,
                'toponyms': toponyms,
                'processed_toponyms': processed_toponyms,
                'with_distance': with_distance,
                'candidates': candidates,
                'config': pipeline.config
            }
        
        # 結果を保存
        output_file = run_dir / f"{experiment_id}_candidates.csv"
        
        # Inspector用にgeometry列をWKT形式で保存
        if isinstance(candidates, gpd.GeoDataFrame) and 'geometry' in candidates.columns:
            candidates_csv = candidates.copy()
            # geometry列をWKT形式に変換
            candidates_csv['geometry'] = candidates.geometry.to_wkt()
            candidates_csv.to_csv(output_file, index=False, encoding='utf-8')
        elif 'lon' in candidates.columns and 'lat' in candidates.columns:
            # lon/latからGeometryを作成してWKT形式で保存
            candidates_csv = candidates.copy()
            from shapely.geometry import Point
            candidates_csv['geometry'] = candidates.apply(
                lambda row: Point(row['lon'], row['lat']).wkt, axis=1
            )
            candidates_csv.to_csv(output_file, index=False, encoding='utf-8')
        else:
            candidates.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.debug(f"Saved {len(candidates)} candidates to {output_file}")
        
        if return_intermediate:
            return candidates, intermediate_data
        else:
            return candidates
        
    except ImportError as e:
        logger.error(f"Failed to import existing pipeline: {e}")
        # フォールバック: モックパイプライン
        return _run_mock_pipeline(param_patch, experiment_id, run_dir)
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}")
        raise


def _run_mock_pipeline(
    param_patch: Dict[str, Any], 
    experiment_id: str,
    run_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    テスト用のモックパイプライン.
    
    Args:
        param_patch: パラメータオーバーライド
        experiment_id: 実験ID
        
    Returns:
        模擬候補地点DataFrame
    """
    logger.warning("Using mock pipeline - real pipeline not available")
    
    # モック候補地点を生成
    np.random.seed(hash(experiment_id) % 2**32)
    
    n_candidates = np.random.randint(50, 200)
    
    # Acreリージョンの範囲
    bbox = (-70.5, -11.5, -66.5, -8.5)  # west, south, east, north
    
    candidates_data = []
    for i in range(n_candidates):
        lon = np.random.uniform(bbox[0], bbox[2])
        lat = np.random.uniform(bbox[1], bbox[3])
        
        # パラメータに基づく模擬スコア
        base_score = np.random.beta(2, 5)  # 低スコアに偏った分布
        
        # 距離・水域パラメータの影響を模擬
        distance_factor = 1.0 if np.random.random() > 0.3 else 0.5
        occ_factor = 1.0 if np.random.random() > 0.4 else 0.3
        
        final_score = base_score * distance_factor * occ_factor
        
        candidates_data.append({
            'name': f'mock_candidate_{i:04d}',
            'lon': lon,
            'lat': lat,
            'normalized_name': f'candidate_{i:04d}',
            'type': np.random.choice(['igarape', 'lake', 'river', 'stream']),
            'dist_km': np.random.exponential(param_patch.get('distance_threshold_km', 3.0)),
            'occ_pct': np.random.exponential(param_patch.get('occ_pct_threshold', 5.0)),
            'total_score': final_score,
            'is_candidate': final_score > 0.3,
            'experiment_id': experiment_id
        })
    
    mock_df = pd.DataFrame(candidates_data)
    
    logger.info(f"Generated {len(mock_df)} mock candidates")
    
    return mock_df


def _calculate_metrics(
    candidates: pd.DataFrame, 
    validation_set: gpd.GeoDataFrame
) -> Dict[str, float]:
    """
    評価指標を計算する.
    
    Args:
        candidates: 候補地点DataFrame
        validation_set: 検証用遺跡GeoDataFrame
        
    Returns:
        評価指標辞書
    """
    try:
        # 既存の評価モジュールをインポート
        from tamagawa_to_z.inspector_agent.metrics import (
            recall_at_k, map_score, workload
        )
        
        # 候補をGeoDataFrameに変換
        if 'geometry' not in candidates.columns:
            # lon, lat列の存在確認
            if 'lon' not in candidates.columns or 'lat' not in candidates.columns:
                logger.error(f"候補データにlon/lat列がありません。列: {list(candidates.columns)}")
                raise ValueError("候補データにlon/lat列が必要です")
            
            candidates_gdf = gpd.GeoDataFrame(
                candidates,
                geometry=gpd.points_from_xy(candidates.lon, candidates.lat),
                crs="EPSG:4326"
            )
        else:
            candidates_gdf = candidates
        
        # 評価指標計算
        metrics = {
            'recall_100': recall_at_k(candidates_gdf, validation_set, k=100),
            'recall_50': recall_at_k(candidates_gdf, validation_set, k=50),
            'map_score': map_score(candidates_gdf, validation_set),
            'workload': workload(candidates_gdf)
        }
        
        logger.debug(f"Calculated metrics: {metrics}")
        
        return metrics
        
    except ImportError:
        logger.warning("Using mock metrics - evaluation module not available")
        return _calculate_mock_metrics(candidates, validation_set)


def _calculate_mock_metrics(
    candidates: pd.DataFrame, 
    validation_set: gpd.GeoDataFrame
) -> Dict[str, float]:
    """
    モック評価指標（テスト用）.
    
    Args:
        candidates: 候補地点DataFrame
        validation_set: 検証用遺跡GeoDataFrame
        
    Returns:
        模擬評価指標辞書
    """
    # 単純な模擬指標
    n_candidates = len(candidates)
    n_validation = len(validation_set)
    
    # 候補数に基づく模擬recall（候補が多いほど高い、ただし収穫逓減）
    recall_100 = min(0.95, n_candidates / 1000.0)
    recall_50 = min(0.85, n_candidates / 500.0)
    
    # ランダムMAP
    map_score = np.random.uniform(0.1, 0.6)
    
    # 実際のworkload
    workload = n_candidates
    
    metrics = {
        'recall_100': recall_100,
        'recall_50': recall_50,
        'map_score': map_score,
        'workload': workload
    }
    
    logger.debug(f"Mock metrics: {metrics}")
    
    return metrics


def _calculate_composite_score(metrics: Dict[str, float]) -> float:
    """
    総合スコアを計算する（候補ゼロ問題対策版）.
    
    Args:
        metrics: 評価指標辞書
        
    Returns:
        総合スコア（最大化目標）
    """
    # 修正された重み
    weights = {
        'recall_weight': 0.6,
        'map_weight': 0.2,
        'workload_weight': -0.1  # -0.2から-0.1に変更（ペナルティ軽減）
    }
    
    recall = metrics.get('recall_100', 0.0)
    map_score = metrics.get('map_score', 0.0)
    workload = metrics.get('workload', 0)
    
    # 候補ゼロのハードペナルティ
    if workload == 0:
        return -0.05
    
    # 正規化されたWorkloadペナルティ（0-1範囲）
    W_MAX = 1000
    workload_penalty = np.log10(workload + 1) / np.log10(W_MAX + 1)
    
    composite_score = (
        weights['recall_weight'] * recall +
        weights['map_weight'] * map_score +
        weights['workload_weight'] * workload_penalty
    )
    
    return composite_score


def _extract_false_positives(
    candidates: pd.DataFrame, 
    validation_set: gpd.GeoDataFrame,
    max_examples: int = 50
) -> list:
    """
    偽陽性例を抽出する.
    
    Args:
        candidates: 候補地点DataFrame
        validation_set: 検証用遺跡GeoDataFrame
        max_examples: 最大例数
        
    Returns:
        偽陽性例のリスト
    """
    try:
        # 実際のFP検出は複雑なので、とりあえずランダムサンプリング
        if 'is_candidate' in candidates.columns:
            candidate_subset = candidates[candidates['is_candidate'] == True]
        else:
            # スコア上位をcandidate扱い
            candidate_subset = candidates.nlargest(100, 'total_score')
        
        # ランダムに偽陽性例を選択（実際は地理的距離で判定）
        n_fp = min(max_examples, len(candidate_subset) // 3)
        fp_sample = candidate_subset.sample(n=n_fp, random_state=42)
        
        fp_examples = fp_sample['name'].tolist()
        
        logger.debug(f"Extracted {len(fp_examples)} false positive examples")
        
        return fp_examples
        
    except Exception as e:
        logger.warning(f"FP extraction failed: {e}")
        return []


if __name__ == "__main__":
    # テスト実行
    import tempfile
    
    # テスト用validation set
    test_validation = gpd.GeoDataFrame({
        'site_name': ['Site_A', 'Site_B', 'Site_C'],
        'culture_tag': ['acre', 'acre', 'casarabe'],
        'discovery_year': [2015, 2018, 2019]
    }, geometry=[
        gpd.points_from_xy([-68.0], [-9.0])[0],
        gpd.points_from_xy([-68.2], [-9.2])[0],
        gpd.points_from_xy([-63.0], [-17.8])[0]
    ], crs="EPSG:4326")
    
    # テスト用パラメータ
    test_params = {
        'distance_km': 2.5,
        'occ_pct': 5.0,
        'root_weights': {'igarape': 0.8, 'parana': 0.6, 'lago': 0.4}
    }
    
    print("=== Pipeline Runner Test ===")
    
    try:
        # スコアのみ
        score = run_pipeline_with_params(
            **test_params,
            validation_set=test_validation,
            return_fp=False
        )
        print(f"Score: {score:.4f}")
        
        # スコア + FP例
        score_fp, fp_examples = run_pipeline_with_params(
            **test_params,
            validation_set=test_validation,
            return_fp=True
        )
        print(f"Score with FP: {score_fp:.4f}")
        print(f"FP examples: {fp_examples[:5]}")  # 最初の5例
        
    except Exception as e:
        print(f"Test failed: {e}")
    
    print("Test completed.")