#!/usr/bin/env python3
"""
Real Optuna Hybrid: 実パイプラインを使ったハイブリッド最適化（最適化版）

このモジュールは、重いS1-S5計算を事前実行し、目的関数では
threshold適用と評価のみを行う高速化版RealHybridBOです。
"""

import logging
import json
import time
import optuna
import pandas as pd
import geopandas as gpd
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

from .real_pipeline_runner import RealPipelineRunner, PipelineRunnerConfig
from .llm_root import get_root_weights  # 既存のLLM統合をそのまま使用

logger = logging.getLogger(__name__)


class RealHybridBO:
    """
    実パイプラインを使ったハイブリッドベイズ最適化（最適化版）
    
    重いS1-S5計算を初期化時に一度だけ実行し、目的関数では
    threshold適用と評価のみを行うことで大幅に高速化します。
    """
    
    def __init__(self,
                 script_path: str,
                 validation_sites: gpd.GeoDataFrame,
                 config_path: Path,
                 n_trials: int = 50,
                 base_config: Optional[Dict[str, Any]] = None,
                 resume: bool = False,
                 timeout_per_trial: int = 1800):
        """
        Parameters
        ----------
        script_path : str
            run_site_identification.py のパス
        validation_sites : gpd.GeoDataFrame
            既知の考古学サイト（検証用）
        config_path : Path
            Optuna設定ファイルパス
        n_trials : int
            最適化試行回数
        base_config : Dict[str, Any], optional
            ベース設定（データパス等）
        resume : bool
            既存スタディを再開するか
        timeout_per_trial : int
            1試行あたりのタイムアウト時間（秒）
        """
        self.script_path = script_path
        self.validation_sites = validation_sites
        self.n_trials = n_trials
        self.timeout_per_trial = timeout_per_trial
        self.resume = resume
        
        # 設定読み込み
        self.config = self._load_config(config_path)
        
        # ベース設定（データパス等）
        if base_config is None:
            # デフォルト設定を生成
            project_root = Path(script_path).parents[1]
            base_config = PipelineRunnerConfig.create_amazon_config(project_root)
        self.base_config = base_config
        
        # パイプラインランナー初期化
        self.pipeline_runner = RealPipelineRunner(
            script_path=script_path,
            base_config=base_config,
            timeout_seconds=timeout_per_trial
        )
        
        # LLM統合（既存のllm_rootモジュールを使用）
        self.has_llm = True
        self.toponym_stats = {}  # 地名統計（事前計算後に設定）
        self.fp_log = []  # 偽陽性ログ
        
        # 実行ディレクトリ設定
        self.run_dir = Path(f"data/output/optuna/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 結果記録
        self.trial_history = []
        self.best_params = None
        self.best_score = float('-inf')
        
        # 事前計算結果を保存
        self.precomputed_candidates = None
        self.precomputation_done = False
        
        logger.info(f"RealHybridBO initialized: {n_trials} trials, timeout={timeout_per_trial}s")
        logger.info(f"Output directory: {self.run_dir}")
    
    def run(self) -> Dict[str, Any]:
        """最適化を実行"""
        logger.info(f"Starting optimized pipeline optimization with {self.n_trials} trials")
        
        # 事前計算実行（S1-S5の重い処理を一度だけ実行）
        logger.info("🔄 Pre-computing heavy S1-S5 calculations...")
        self._precompute_heavy_calculations()
        logger.info("✅ Pre-computation completed. Starting optimization...")
        
        # Optuna Study作成
        study_name = f"real_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        storage = f"sqlite:///{self.run_dir}/optuna.db"
        
        if self.resume:
            study = optuna.load_study(study_name=study_name, storage=storage)
            logger.info(f"Resumed study: {len(study.trials)} existing trials")
        else:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(multivariate=True)
            )
        
        # 最適化実行
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # 結果保存
        result = self._save_results(study)
        
        logger.info(f"Optimization completed: best score = {result['score']:.4f}")
        return result
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna目的関数（最適化版：事前計算済みデータを使用）"""
        trial_start_time = time.time()
        
        try:
            # パラメータサンプリング
            search_space = self.config.get("search_space", self.config)
            distance_km = trial.suggest_float("distance_km", 
                                            search_space["distance_km"]["low"],
                                            search_space["distance_km"]["high"])
            occ_pct = trial.suggest_float("occ_pct",
                                        search_space["occ_pct"]["low"], 
                                        search_space["occ_pct"]["high"])
            
            # LLM による語根ウェイト生成
            root_weights = self._get_root_weights(trial, distance_km, occ_pct)
            
            logger.info(f"🔬 Trial {trial.number} STARTING (fast mode):")
            logger.info(f"   📏 Distance threshold: {distance_km:.2f} km")
            logger.info(f"   💧 Water occurrence: {occ_pct:.2f} %")
            logger.info(f"   🔤 Root weights: {len(root_weights)} weights")
            
            # 高速評価：事前計算済みデータにthresholdを適用
            evaluation_start = time.time()
            score, n_candidates, metrics = self._evaluate_fast(
                distance_km=distance_km,
                occ_pct=occ_pct,
                root_weights=root_weights
            )
            evaluation_time = time.time() - evaluation_start
            
            logger.info(f"🏁 Trial {trial.number} COMPLETED (fast):")
            logger.info(f"   ⏱️  Evaluation time: {evaluation_time:.2f}s")
            logger.info(f"   🎯 Score: {score:.4f}")
            logger.info(f"   📊 Candidates: {n_candidates}")
            logger.info(f"   {'🏆 NEW BEST!' if score > self.best_score else '   Previous best'}")
            
            # 実行時間記録
            execution_time = time.time() - trial_start_time
            
            # 試行結果記録
            trial_result = {
                "trial_number": trial.number,
                "distance_km": distance_km,
                "occ_pct": occ_pct,
                "root_weights": root_weights,
                "score": score,
                "n_candidates": n_candidates,
                "execution_time": execution_time,
                "metrics": metrics
            }
            self.trial_history.append(trial_result)
            
            # ベストスコア更新
            if score > self.best_score:
                self.best_score = score
                self.best_params = trial_result
                logger.info(f"New best score: {score:.4f} (trial {trial.number})")
            
            # 中間結果保存
            self._save_trial_result(trial_result)
            
            return score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # 失敗した試行も記録
            failed_result = {
                "trial_number": trial.number,
                "error": str(e),
                "execution_time": time.time() - trial_start_time
            }
            self.trial_history.append(failed_result)
            
            # 大きなペナルティスコア
            return -1.0
    
    def _get_root_weights(self, trial: optuna.Trial, distance_km: float, occ_pct: float) -> Dict[str, float]:
        """語根ウェイトを取得（LLMまたはデフォルト）"""
        
        if self.has_llm:
            try:
                # LLMコンテキスト作成（完全版）
                context = {
                    "trial_number": trial.number,
                    "distance_km": distance_km,
                    "occ_pct": occ_pct,
                    "validation_sites_count": len(self.validation_sites),
                    "previous_trials": self.trial_history[-5:] if self.trial_history else [],
                    # LLMが期待する必須キーを追加
                    "toponym_stats": self.toponym_stats,
                    "false_pos": self.fp_log[-10:] if self.fp_log else [],
                    "active_categories": list(self.toponym_stats.keys()) if self.toponym_stats else []
                }
                
                # LLM問い合わせ（既存の関数を使用）
                root_weights = get_root_weights(context)
                logger.debug(f"LLM generated {len(root_weights)} root weights")
                
                return root_weights
                
            except Exception as e:
                logger.warning(f"LLM root weight generation failed: {e}")
                logger.debug(f"LLM error details: {type(e).__name__}: {e}")
        
        # フォールバック：デフォルトウェイト
        default_weights = {
            "rio": 0.8,
            "igarape": 0.9,
            "lago": 0.7,
            "parana": 0.8,
            "cachoeira": 0.6,
            "lagoa": 0.7,
            "canal": 0.5,
            "furo": 0.6,
            "travessa": 0.3,
            "ramal": 0.3
        }
        
        return default_weights
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Optuna設定を読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix == '.json':
                    config = json.load(f)
                else:
                    import yaml
                    config = yaml.safe_load(f)
            
            logger.info(f"Loaded config from {config_path}")
            return config
            
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return {
                "distance_km": {"low": 0.5, "high": 15.0},
                "occ_pct": {"low": 0.5, "high": 20.0}
            }
    
    def _save_trial_result(self, trial_result: Dict[str, Any]):
        """個別試行結果を保存"""
        trial_file = self.run_dir / f"trial_{trial_result['trial_number']:04d}.json"
        
        try:
            with open(trial_file, 'w', encoding='utf-8') as f:
                json.dump(trial_result, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save trial result: {e}")
    
    def _save_results(self, study: optuna.Study) -> Dict[str, Any]:
        """最終結果を保存"""
        
        # ベストパラメータ
        best_trial = study.best_trial
        best_params = best_trial.params.copy()
        
        # 語根ウェイトを追加（記録されている場合）
        if self.best_params and "root_weights" in self.best_params:
            best_params["root_weights"] = self.best_params["root_weights"]
        
        # ベストパラメータで候補地を再計算
        best_candidates, best_metrics = self._get_best_candidates_details(
            best_params.get("distance_km", 0.0),
            best_params.get("occ_pct", 0.0),
            best_params.get("root_weights", {})
        )
        
        # 結果サマリー
        result = {
            "score": study.best_value if study.best_value is not None else -1.0,
            "distance_km": best_params.get("distance_km", 0.0),
            "occ_pct": best_params.get("occ_pct", 0.0),
            "root_weights": best_params.get("root_weights", {}),
            "trial_number": best_trial.number if best_trial else 0,
            "optimization_history": [
                {"trial_number": t.get("trial_number"), "score": t.get("score")}
                for t in self.trial_history if t.get("score") is not None
            ],
            "best_candidates": best_candidates,
            "objective_components": best_metrics
        }
        
        # ファイル保存
        result_file = self.run_dir / "best_params.json"
        history_file = self.run_dir / "optimization_history.json"
        
        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.trial_history, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Results saved to {self.run_dir}")
            
            # 詳細な結果出力
            logger.info("=== 最適化結果 ===")
            logger.info(f"Best score: {result['score']:.4f}")
            logger.info(f"Best parameters:")
            logger.info(f"  Distance threshold: {result['distance_km']:.2f} km")
            logger.info(f"  Water occurrence: {result['occ_pct']:.2f} %")
            logger.info(f"  Root weights: {len(result['root_weights'])} weights")
            
            # 候補地詳細
            if best_candidates:
                logger.info(f"Best parameter candidates: {len(best_candidates)} sites")
                for i, candidate in enumerate(best_candidates[:5]):  # 上位5件のみ表示
                    logger.info(f"  {i+1}. {candidate.get('name', 'Unknown')} "
                              f"({candidate.get('lat', 0):.4f}, {candidate.get('lon', 0):.4f})")
                if len(best_candidates) > 5:
                    logger.info(f"  ... and {len(best_candidates) - 5} more candidates")
            
            # 目的関数構成要素
            if best_metrics:
                recall_k = self.config.get('objective', {}).get('recall_k', 100)
                logger.info("Objective function components:")
                logger.info(f"  Recall@{recall_k}: {best_metrics.get(f'recall_{recall_k}', 0):.4f}")
                logger.info(f"  MAP score: {best_metrics.get('map_score', 0):.4f}")
                logger.info(f"  Workload: {best_metrics.get('workload', 0)}")
                logger.info(f"  Composite score: {best_metrics.get('composite_score', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
        
        return result
    
    def _get_best_candidates_details(self, distance_km: float, occ_pct: float, 
                                   root_weights: Dict[str, float]) -> Tuple[List[Dict], Dict]:
        """ベストパラメータで候補地詳細を取得"""
        try:
            # 高速評価で候補地を取得
            _, n_candidates, metrics = self._evaluate_fast(distance_km, occ_pct, root_weights)
            
            # フィルタリング済み候補データを取得
            candidates_df = self.precomputed_candidates.copy()
            
            # 同じフィルタリングを適用
            distance_col = 'dist_km' if 'dist_km' in candidates_df.columns else 'distance_to_river_km'
            if distance_col in candidates_df.columns:
                candidates_df = candidates_df[candidates_df[distance_col] >= distance_km]
            
            # 水域出現率のフィルタリング（列名を確認）
            occ_col = 'occ_pct' if 'occ_pct' in candidates_df.columns else 'water_occurrence_pct'
            if occ_col in candidates_df.columns:
                candidates_df = candidates_df[candidates_df[occ_col] <= occ_pct]
            
            # 語根ウェイトに基づくスコア計算でソート
            root_col = 'type' if 'type' in candidates_df.columns else 'root'
            if root_col in candidates_df.columns and root_weights:
                candidates_df['root_score'] = candidates_df[root_col].map(
                    lambda x: root_weights.get(x, 0.1) if pd.notna(x) else 0.1
                )
                
                import numpy as np
                distance_vals = candidates_df[distance_col] if distance_col in candidates_df.columns else 5.0
                water_vals = candidates_df[occ_col] if occ_col in candidates_df.columns else 5.0
                
                candidates_df['total_score'] = (
                    candidates_df['root_score'] * 0.6 +
                    np.clip(20.0 - distance_vals, 0, 20) / 20.0 * 0.3 +
                    np.clip(20.0 - water_vals, 0, 20) / 20.0 * 0.1
                )
                
                candidates_df = candidates_df.sort_values('total_score', ascending=False)
            
            # 候補地詳細を抽出
            candidate_details = []
            for _, row in candidates_df.head(20).iterrows():  # 上位20件
                candidate_info = {
                    'name': row.get('name', row.get('normalized_name', 'Unknown')),
                    'lat': self._extract_coordinate(row, 'lat'),
                    'lon': self._extract_coordinate(row, 'lon'),
                    'type': row.get('type', row.get('root', 'unknown')),
                    'total_score': row.get('total_score', 0.0),
                    'distance_km': row.get(distance_col, 0.0),
                    'water_occurrence_pct': row.get(occ_col, 0.0)
                }
                candidate_details.append(candidate_info)
            
            return candidate_details, metrics
            
        except Exception as e:
            logger.error(f"Failed to get best candidates details: {e}")
            return [], {}
    
    def _extract_coordinate(self, row: pd.Series, coord_type: str) -> float:
        """座標値を抽出"""
        try:
            # 直接的な座標列
            if coord_type in row and pd.notna(row[coord_type]):
                return float(row[coord_type])
            
            # geometry列からの抽出
            if 'geometry' in row and pd.notna(row['geometry']):
                geometry_str = str(row['geometry']).strip()
                if 'POINT' in geometry_str:
                    # "POINT (-68.123 -9.456)" から座標を抽出（スペースがあることに注意）
                    import re
                    # POINT (lon lat) または POINT(lon lat) 形式に対応
                    match = re.search(r'POINT\s*\(\s*([^)]+)\)', geometry_str)
                    if match:
                        coords_str = match.group(1).strip()
                        coords = coords_str.split()
                        if len(coords) >= 2:
                            try:
                                lon = float(coords[0])
                                lat = float(coords[1])
                                if coord_type == 'lon':
                                    return lon
                                elif coord_type == 'lat':
                                    return lat
                            except ValueError:
                                logger.debug(f"Invalid coordinate values: {coords}")
            
            return 0.0
            
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to extract {coord_type} coordinate from row: {e}")
            return 0.0
    
    def _precompute_heavy_calculations(self):
        """S1-S5の重い計算を事前実行"""
        if self.precomputation_done:
            logger.info("Pre-computation already completed")
            return
        
        # 設定ハッシュをチェックして、既存の事前計算結果が使用可能か確認
        config_hash = self._get_config_hash()
        existing_result = self._check_existing_precomputation(config_hash)
        
        if existing_result is not None:
            logger.info(f"✅ Using existing pre-computation: {len(existing_result)} candidates")
            self.precomputed_candidates = existing_result
            # 地名統計を抽出（LLM用）
            self._extract_toponym_stats()
            self.precomputation_done = True
            logger.info(f"   📊 Toponym stats: {len(self.toponym_stats)} categories")
            return
        
        logger.info("Executing heavy S1-S5 calculations once...")
        
        # 事前計算用の出力パス
        precomp_dir = self.run_dir / "precomputation"
        precomp_dir.mkdir(exist_ok=True)
        
        candidates_csv = precomp_dir / "all_candidates_precomputed.csv"
        metrics_json = precomp_dir / "precomputed_metrics.json"
        
        # S1-S5までの全計算を実行（threshold適用前）
        cmd = [
            "python", self.script_path,
            "--precompute-only",  # 事前計算モード
            "--output-path", str(candidates_csv),
            "--output-metrics-json", str(metrics_json),
            "--quiet",
            "--no-visualize"
        ]
        
        # ベース設定追加
        for key, value in self.base_config.items():
            if key in ['precompute-only', 'quiet', 'no-visualize']:
                # 事前計算専用オプションは既に追加済みなのでスキップ
                continue
            elif isinstance(value, list):
                # BBOXのような複数値オプション
                cmd.extend([f"--{key}"] + [str(v) for v in value])
            elif isinstance(value, bool):
                # フラグオプション
                if value:
                    cmd.append(f"--{key}")
            elif value is not None:
                # 通常のキー=値オプション
                cmd.extend([f"--{key}", str(value)])
        
        try:
            logger.info(f"Executing pre-computation command: {' '.join(cmd[:10])}...")
            logger.debug(f"Full command: {' '.join(cmd)}")
            logger.info(f"Working directory: {Path(self.script_path).parent.parent}")
            logger.info(f"Expected output file: {candidates_csv}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_per_trial * 2,  # 事前計算は長めのタイムアウト
                cwd=Path(self.script_path).parent.parent
            )
            
            logger.info(f"Pre-computation return code: {result.returncode}")
            if result.stdout:
                logger.debug(f"Pre-computation stdout: {result.stdout[-500:]}")  # 最後の500文字のみ
            if result.stderr:
                logger.warning(f"Pre-computation stderr: {result.stderr[-500:]}")  # 最後の500文字のみ
            
            if result.returncode != 0:
                raise RuntimeError(f"Pre-computation failed: {result.stderr}")
            
            # 事前計算結果を読み込み
            if not candidates_csv.exists():
                # フォールバック: デフォルトの出力パスもチェック
                fallback_path = Path(self.base_config.get('output-path', 'data/output/candidates/paleochannel_candidates.csv'))
                if fallback_path.exists():
                    logger.warning(f"Pre-computation output found at fallback path: {fallback_path}")
                    self.precomputed_candidates = pd.read_csv(fallback_path)
                else:
                    raise RuntimeError(f"Pre-computation output file not created: {candidates_csv}")
            else:
                self.precomputed_candidates = pd.read_csv(candidates_csv)
            
            # データの基本検証
            if len(self.precomputed_candidates) == 0:
                raise RuntimeError("Pre-computation produced empty results")
                
            logger.info(f"Pre-computed data columns: {list(self.precomputed_candidates.columns)}")
            
            # 必要な列の確認
            required_cols = ['name', 'type']
            missing_cols = [col for col in required_cols if col not in self.precomputed_candidates.columns]
            if missing_cols:
                logger.warning(f"Missing expected columns: {missing_cols}")
            
            # 距離・水域頻度列の確認
            has_distance = ('dist_km' in self.precomputed_candidates.columns or 
                          'distance_to_river_km' in self.precomputed_candidates.columns)
            has_occurrence = ('occ_pct' in self.precomputed_candidates.columns or 
                            'water_occurrence_pct' in self.precomputed_candidates.columns)
            
            if not has_distance:
                logger.warning("No distance column found in pre-computed data")
            if not has_occurrence:
                logger.warning("No water occurrence column found in pre-computed data")
            
            # メトリクス読み込み（必要に応じて）
            if metrics_json.exists():
                with open(metrics_json, 'r', encoding='utf-8') as f:
                    _ = json.load(f)  # 現在は使わないが将来用
            
            self.precomputation_done = True
            
            # 地名統計を抽出（LLM用）
            self._extract_toponym_stats()
            
            # キャッシュに保存
            self._save_precomputation_cache(config_hash)
            
            logger.info(f"✅ Pre-computation completed: {len(self.precomputed_candidates)} candidates")
            logger.info(f"   Data includes all toponyms with distance/occurrence values")
            logger.info(f"   📊 Toponym stats: {len(self.toponym_stats)} categories")
            
        except Exception as e:
            logger.error(f"Pre-computation failed: {e}")
            raise RuntimeError(f"Failed to pre-compute heavy calculations: {e}")
    
    def _evaluate_fast(self, distance_km: float, occ_pct: float, 
                      root_weights: Dict[str, float]) -> Tuple[float, int, Dict]:
        """事前計算済みデータを使って高速評価（元のRecall@100ベース目的関数）"""
        if not self.precomputation_done or self.precomputed_candidates is None:
            raise RuntimeError("Pre-computation not completed")
        
        # 事前計算済みデータにthresholdを適用
        candidates_df = self.precomputed_candidates.copy()
        
        # 距離フィルタ（川から遠い = 古河道候補）
        distance_col = 'dist_km' if 'dist_km' in candidates_df.columns else 'distance_to_river_km'
        if distance_col in candidates_df.columns:
            candidates_df = candidates_df[candidates_df[distance_col] >= distance_km]
        
        # 水域出現率フィルタ（水域頻度が低い = 古河道候補）
        occ_col = 'occ_pct' if 'occ_pct' in candidates_df.columns else 'water_occurrence_pct'
        if occ_col in candidates_df.columns:
            candidates_df = candidates_df[candidates_df[occ_col] <= occ_pct]
        
        # 語根ウェイトに基づくスコア計算でソート
        root_col = 'type' if 'type' in candidates_df.columns else 'root'
        if root_col in candidates_df.columns and root_weights:
            candidates_df['root_score'] = candidates_df[root_col].map(
                lambda x: root_weights.get(x, 0.1) if pd.notna(x) else 0.1
            )
            
            # 総合スコア計算（語根重み中心）
            import numpy as np
            distance_vals = candidates_df[distance_col] if distance_col in candidates_df.columns else 5.0
            water_vals = candidates_df[occ_col] if occ_col in candidates_df.columns else 5.0
            
            candidates_df['total_score'] = (
                candidates_df['root_score'] * 0.6 +
                np.clip(20.0 - distance_vals, 0, 20) / 20.0 * 0.3 +
                np.clip(20.0 - water_vals, 0, 20) / 20.0 * 0.1
            )
            
            # スコア順でソート
            candidates_df = candidates_df.sort_values('total_score', ascending=False)
        
        n_candidates = len(candidates_df)
        
        # 元の目的関数に基づく評価スコア計算
        metrics = self._calculate_evaluation_metrics(candidates_df)
        composite_score = self._calculate_composite_score(metrics)
        
        # デバッグ用詳細メトリクス
        recall_k = self.config.get('objective', {}).get('recall_k', 100)
        detailed_metrics = {
            'n_candidates_after_filter': n_candidates,
            'distance_threshold_km': distance_km,
            'occurrence_threshold_pct': occ_pct,
            'n_root_weights': len(root_weights),
            'evaluation_mode': 'fast_precomputed',
            f'recall_{recall_k}': metrics.get(f'recall_{recall_k}', 0.0),
            'map_score': metrics.get('map_score', 0.0),
            'workload': metrics.get('workload', 0),
            'composite_score': composite_score
        }
        
        return composite_score, n_candidates, detailed_metrics
    
    def _calculate_score_fast(self, candidates_df: pd.DataFrame, n_candidates: int) -> float:
        """高速スコア計算"""
        # 1. 候補数スコア（最適範囲: 50-500件）
        if n_candidates == 0:
            candidate_score = 0.0
        elif n_candidates < 50:
            candidate_score = n_candidates / 50.0
        elif n_candidates <= 500:
            candidate_score = 1.0
        else:
            candidate_score = max(0.1, 1.0 - (n_candidates - 500) / 1000.0)
        
        # 2. 地理的多様性スコア
        diversity_score = 0.5  # デフォルト値
        if len(candidates_df) >= 2 and 'lat' in candidates_df.columns and 'lon' in candidates_df.columns:
            try:
                lat_std = candidates_df['lat'].std()
                lon_std = candidates_df['lon'].std()
                diversity_score = min(1.0, (lat_std + lon_std) / 2.0)
            except:
                pass
        
        # 3. 検証スコア（既知サイトとの一致）
        validation_score = 0.5  # デフォルト値
        if (self.validation_sites is not None and len(self.validation_sites) > 0 and 
            len(candidates_df) > 0 and 'lat' in candidates_df.columns and 'lon' in candidates_df.columns):
            try:
                cand_lat_range = (candidates_df['lat'].min(), candidates_df['lat'].max())
                cand_lon_range = (candidates_df['lon'].min(), candidates_df['lon'].max())
                
                in_range = self.validation_sites[
                    (self.validation_sites['lat'] >= cand_lat_range[0]) &
                    (self.validation_sites['lat'] <= cand_lat_range[1]) &
                    (self.validation_sites['lon'] >= cand_lon_range[0]) &
                    (self.validation_sites['lon'] <= cand_lon_range[1])
                ]
                
                if len(self.validation_sites) > 0:
                    overlap_ratio = len(in_range) / len(self.validation_sites)
                    validation_score = min(1.0, overlap_ratio * 2.0)
            except:
                pass
        
        # 総合スコア（重み付き平均）
        total_score = (
            0.4 * candidate_score +
            0.3 * diversity_score +
            0.3 * validation_score
        )
        
        return total_score
    
    def _calculate_evaluation_metrics(self, candidates_df: pd.DataFrame) -> Dict[str, float]:
        """元のパイプラインと同じ評価メトリクスを計算"""
        try:
            # 候補をGeoDataFrameに変換（座標抽出）
            candidates_gdf = self._convert_to_geodataframe(candidates_df)
            
            # 元のメトリクス計算を呼び出し
            from tamagawa_to_z.inspector_agent.metrics import (
                recall_at_k, map_score, workload
            )
            
            # YAMLファイルの設定からrecall_kを取得
            recall_k = self.config.get('objective', {}).get('recall_k', 100)
            
            metrics = {
                f'recall_{recall_k}': recall_at_k(candidates_gdf, self.validation_sites, k=recall_k),
                'recall_50': recall_at_k(candidates_gdf, self.validation_sites, k=50),
                'map_score': map_score(candidates_gdf, self.validation_sites),
                'workload': workload(candidates_gdf)
            }
            
            return metrics
            
        except ImportError:
            logger.warning("Evaluation module not available, using mock metrics")
            return self._calculate_mock_metrics(candidates_df)
        except Exception as e:
            logger.warning(f"Metrics calculation failed: {e}, using mock metrics")
            return self._calculate_mock_metrics(candidates_df)
    
    def _convert_to_geodataframe(self, candidates_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """候補データをGeoDataFrameに変換"""
        if 'geometry' in candidates_df.columns:
            # 既にgeometry列がある場合はそのまま使用
            if isinstance(candidates_df, gpd.GeoDataFrame):
                return candidates_df
            else:
                return gpd.GeoDataFrame(candidates_df, crs="EPSG:4326")
        else:
            # lat/lon列がある場合はgeometryを作成
            if 'lat' in candidates_df.columns and 'lon' in candidates_df.columns:
                return gpd.GeoDataFrame(
                    candidates_df,
                    geometry=gpd.points_from_xy(candidates_df.lon, candidates_df.lat),
                    crs="EPSG:4326"
                )
            else:
                raise ValueError("No geometry or lat/lon columns found in candidates data")
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """元のパイプラインと同じ複合スコア計算"""
        # YAMLファイルからの重み設定
        objective_config = self.config.get('objective', {})
        weights = {
            'recall_weight': objective_config.get('recall_weight', 0.6),
            'map_weight': objective_config.get('map_weight', 0.2),
            'workload_weight': objective_config.get('workload_weight', -0.1)
        }
        
        # 動的にrecall_kを取得
        recall_k = objective_config.get('recall_k', 100)
        recall = metrics.get(f'recall_{recall_k}', 0.0)
        map_score_val = metrics.get('map_score', 0.0)
        workload = metrics.get('workload', 0)
        
        # 候補ゼロのハードペナルティ
        if workload == 0:
            return -0.05
        
        # 正規化されたWorkloadペナルティ（0-1範囲）
        W_MAX = 1000
        import numpy as np
        workload_penalty = np.log10(workload + 1) / np.log10(W_MAX + 1)
        
        composite_score = (
            weights['recall_weight'] * recall +
            weights['map_weight'] * map_score_val +
            weights['workload_weight'] * workload_penalty
        )
        
        return composite_score
    
    def _calculate_mock_metrics(self, candidates_df: pd.DataFrame) -> Dict[str, float]:
        """モック評価メトリクス（評価モジュールが利用できない場合）"""
        n_candidates = len(candidates_df)
        recall_k = self.config.get('objective', {}).get('recall_k', 100)
        
        # 簡単なモックメトリクス
        mock_recall = min(1.0, n_candidates / max(recall_k, 1)) if n_candidates > 0 else 0.0
        mock_map = 0.5 if n_candidates > 0 else 0.0
        
        return {
            f'recall_{recall_k}': mock_recall,
            'recall_50': mock_recall * 0.8,
            'map_score': mock_map,
            'workload': n_candidates
        }
    
    def _extract_toponym_stats(self):
        """事前計算済みデータから地名統計を抽出（LLM用）"""
        if self.precomputed_candidates is None:
            return
        
        # 語根タイプ別の統計を計算
        root_col = 'type' if 'type' in self.precomputed_candidates.columns else 'root'
        if root_col in self.precomputed_candidates.columns:
            type_counts = self.precomputed_candidates[root_col].value_counts()
            self.toponym_stats = type_counts.to_dict()
        else:
            self.toponym_stats = {}
        
        logger.debug(f"Extracted toponym stats: {self.toponym_stats}")
    
    def _get_config_hash(self) -> str:
        """設定のハッシュ値を生成（事前計算の再利用判定用）"""
        import hashlib
        
        # 事前計算に影響する設定のみを対象
        relevant_config = {
            'bbox': self.base_config.get('bbox'),
            'pbf-path': self.base_config.get('pbf-path'),
            'rivers-path': self.base_config.get('rivers-path'),
            'gsw-path': self.base_config.get('gsw-path'),
            'skip-water-freq': self.base_config.get('skip-water-freq', False)
        }
        
        config_str = json.dumps(relevant_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def _check_existing_precomputation(self, config_hash: str) -> Optional[pd.DataFrame]:
        """既存の事前計算結果をチェック"""
        # グローバル事前計算キャッシュディレクトリ
        cache_dir = Path("data/cache/precomputation")
        cache_file = cache_dir / f"precomputed_{config_hash}.csv"
        
        if cache_file.exists():
            try:
                logger.info(f"Found existing pre-computation cache: {cache_file}")
                return pd.read_csv(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def _save_precomputation_cache(self, config_hash: str):
        """事前計算結果をキャッシュに保存"""
        if self.precomputed_candidates is None:
            return
        
        cache_dir = Path("data/cache/precomputation")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"precomputed_{config_hash}.csv"
        
        try:
            self.precomputed_candidates.to_csv(cache_file, index=False)
            logger.info(f"Saved pre-computation cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get_study_info(self) -> Dict[str, Any]:
        """Study情報を取得"""
        return {
            "n_trials": len(self.trial_history),
            "best_value": self.best_score,
            "run_dir": str(self.run_dir)
        }


# ===== 使用例とテスト =====

def create_test_validation_sites() -> gpd.GeoDataFrame:
    """テスト用の検証サイトを作成"""
    import geopandas as gpd
    from shapely.geometry import Point
    
    # アクレ州内のダミーサイト
    sites_data = [
        {"name": "Site A", "lat": -9.5, "lon": -68.0},
        {"name": "Site B", "lat": -10.0, "lon": -67.5},
        {"name": "Site C", "lat": -9.8, "lon": -68.3}
    ]
    
    geometry = [Point(s["lon"], s["lat"]) for s in sites_data]
    gdf = gpd.GeoDataFrame(sites_data, geometry=geometry, crs="EPSG:4326")
    
    return gdf


if __name__ == "__main__":
    # テスト実行例
    import sys
    from pathlib import Path
    
    # パス設定
    project_root = Path(__file__).parents[3]
    script_path = str(project_root / "scripts/run_site_identification.py")
    config_path = project_root / "configs/optuna_space.yaml"
    
    # 検証サイト
    validation_sites = create_test_validation_sites()
    
    # 高速テスト設定
    test_config = PipelineRunnerConfig.create_test_config()
    
    try:
        # 最適化実行
        optimizer = RealHybridBO(
            script_path=script_path,
            validation_sites=validation_sites,
            config_path=config_path,
            n_trials=3,  # テスト用に少数
            base_config=test_config,
            timeout_per_trial=300  # 5分
        )
        
        result = optimizer.run()
        
        print("Optimization completed!")
        print(f"Best score: {result['score']:.4f}")
        print(f"Best distance: {result['distance_km']:.2f} km")
        print(f"Best occ_pct: {result['occ_pct']:.2f} %")
        print(f"Output directory: {optimizer.run_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)