#!/usr/bin/env python3
"""
Real Pipeline Runner: run_site_identification.py をサブプロセスとして実行

このモジュールは、実際の地理空間解析パイプラインをサブプロセスとして実行し、
Optunaベースのパラメータ最適化と統合するためのインターフェースを提供します。
"""

import subprocess
import json
import tempfile
import logging
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)


class RealPipelineRunner:
    """run_site_identification.py をサブプロセスとして実行するラッパー"""
    
    def __init__(self, 
                 script_path: str,
                 base_config: Optional[Dict[str, Any]] = None,
                 timeout_seconds: int = 3600):
        """
        Parameters
        ----------
        script_path : str
            run_site_identification.py のパス
        base_config : Dict[str, Any], optional
            ベース設定（bbox, data paths等）
        timeout_seconds : int
            サブプロセスのタイムアウト時間（秒）
        """
        self.script_path = Path(script_path)
        self.base_config = base_config or {}
        self.timeout_seconds = timeout_seconds
        
        if not self.script_path.exists():
            raise FileNotFoundError(f"Script not found: {self.script_path}")
    
    def run_with_params(self, 
                       distance_km: float,
                       occ_pct: float, 
                       root_weights: Dict[str, float],
                       experiment_id: str,
                       validation_sites: Optional[pd.DataFrame] = None) -> Tuple[float, int, Dict]:
        """
        指定されたパラメータでパイプラインを実行
        
        Parameters
        ----------
        distance_km : float
            距離閾値（km）
        occ_pct : float
            水域出現率閾値（%）
        root_weights : Dict[str, float]
            語根ウェイト辞書
        experiment_id : str
            実験ID
        validation_sites : pd.DataFrame, optional
            検証用の既知サイト
            
        Returns
        -------
        Tuple[float, int, Dict]
            (評価スコア, 候補数, メトリクス詳細)
        """
        start_time = time.time()
        
        # 一時ディレクトリ作成
        with tempfile.TemporaryDirectory(prefix=f"optuna_{experiment_id}_") as temp_dir:
            temp_path = Path(temp_dir)
            
            # 出力パス設定
            output_csv = temp_path / f"{experiment_id}_candidates.csv"
            metrics_json = temp_path / f"{experiment_id}_metrics.json"
            
            try:
                # コマンド構築
                cmd = self._build_command(
                    distance_km=distance_km,
                    occ_pct=occ_pct,
                    root_weights=root_weights,
                    output_csv=output_csv,
                    metrics_json=metrics_json
                )
                
                logger.debug(f"Executing command: {' '.join(cmd)}")
                
                # サブプロセス実行（リアルタイムログ出力）
                logger.info(f"Executing pipeline command: {' '.join(cmd[:5])}...")
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # stderrもstdoutに統合
                    text=True,
                    cwd=self.script_path.parent.parent,  # プロジェクトルート
                    bufsize=1,  # 行バッファリング
                    universal_newlines=True
                )
                
                # リアルタイムでログ出力
                output_lines = []
                while True:
                    line = process.stdout.readline()
                    if line:
                        # パイプラインのログを転送（プレフィックス付き）
                        logger.info(f"[Pipeline] {line.rstrip()}")
                        output_lines.append(line)
                    elif process.poll() is not None:
                        break
                
                # プロセス完了待ち
                try:
                    return_code = process.wait(timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                    raise subprocess.TimeoutExpired(cmd, self.timeout_seconds)
                
                # 結果オブジェクト作成（subprocess.run互換）
                class MockResult:
                    def __init__(self, returncode, stdout):
                        self.returncode = returncode
                        self.stdout = stdout
                        self.stderr = ""
                
                result = MockResult(return_code, ''.join(output_lines))
                
                # エラーハンドリング
                if result.returncode != 0:
                    error_msg = f"Pipeline failed (code {result.returncode}): {result.stderr}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # 結果読み込み
                candidates_df, metrics = self._load_results(output_csv, metrics_json)
                
                # 評価スコア計算
                score = self._calculate_score(candidates_df, metrics, validation_sites)
                n_candidates = len(candidates_df)
                
                # 実行時間記録
                execution_time = time.time() - start_time
                metrics['execution_time_seconds'] = execution_time
                metrics['experiment_id'] = experiment_id
                
                logger.info(f"Pipeline completed: {n_candidates} candidates, "
                           f"score={score:.4f}, time={execution_time:.1f}s")
                
                return score, n_candidates, metrics
                
            except subprocess.TimeoutExpired:
                error_msg = f"Pipeline timeout after {self.timeout_seconds} seconds"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            except Exception as e:
                logger.error(f"Pipeline execution failed: {e}")
                raise RuntimeError(f"Pipeline execution failed: {e}")
    
    def _build_command(self,
                      distance_km: float,
                      occ_pct: float,
                      root_weights: Dict[str, float],
                      output_csv: Path,
                      metrics_json: Path) -> List[str]:
        """コマンドライン引数を構築"""
        
        cmd = [
            "python", str(self.script_path),
            "--dist-threshold", str(distance_km),
            "--occ-threshold", str(occ_pct),
            "--root-weights-json", json.dumps(root_weights),
            "--output-path", str(output_csv),
            "--output-metrics-json", str(metrics_json),
            "--quiet",  # ログ抑制
            "--no-visualize"  # 可視化無効
        ]
        
        # ベース設定追加
        for key, value in self.base_config.items():
            if isinstance(value, list):
                # bbox などのリスト値の場合
                cmd.extend([f"--{key}"] + [str(v) for v in value])
            elif isinstance(value, bool):
                # boolean値の場合はTrueの時だけフラグを追加
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
        
        return cmd
    
    def _load_results(self, output_csv: Path, metrics_json: Path) -> Tuple[pd.DataFrame, Dict]:
        """結果ファイルを読み込み"""
        
        # 候補CSV読み込み
        if not output_csv.exists():
            raise RuntimeError(f"Output CSV not found: {output_csv}")
        
        candidates_df = pd.read_csv(output_csv)
        
        # メトリクス読み込み
        metrics = {}
        if metrics_json.exists():
            try:
                with open(metrics_json, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metrics JSON: {e}")
        
        return candidates_df, metrics
    
    def _calculate_score(self, 
                        candidates_df: pd.DataFrame, 
                        metrics: Dict,
                        validation_sites: Optional[pd.DataFrame] = None) -> float:
        """
        評価スコアを計算
        
        複数の指標を組み合わせた総合スコア:
        1. 候補数（多すぎず少なすぎず）
        2. 地理的多様性
        3. 既知サイトとの一致度（validation_sites提供時）
        """
        
        n_candidates = len(candidates_df)
        
        # 1. 候補数スコア（最適範囲: 50-500件）
        if n_candidates == 0:
            candidate_score = 0.0
        elif n_candidates < 50:
            candidate_score = n_candidates / 50.0  # 0-1の範囲
        elif n_candidates <= 500:
            candidate_score = 1.0  # 最適範囲
        else:
            candidate_score = max(0.1, 1.0 - (n_candidates - 500) / 1000.0)  # 減衰
        
        # 2. 地理的多様性スコア
        diversity_score = self._calculate_diversity_score(candidates_df)
        
        # 3. 検証スコア（既知サイトとの一致）
        validation_score = 0.5  # デフォルト
        if validation_sites is not None and len(validation_sites) > 0:
            validation_score = self._calculate_validation_score(candidates_df, validation_sites)
        
        # 総合スコア（重み付き平均）
        total_score = (
            0.4 * candidate_score +      # 候補数の適切性
            0.3 * diversity_score +      # 地理的多様性
            0.3 * validation_score       # 既知サイトとの一致
        )
        
        return total_score
    
    def _calculate_diversity_score(self, candidates_df: pd.DataFrame) -> float:
        """地理的多様性スコアを計算"""
        if len(candidates_df) < 2:
            return 0.0
        
        try:
            # 緯度経度の標準偏差をもとに多様性を評価
            if 'lat' in candidates_df.columns and 'lon' in candidates_df.columns:
                lat_std = candidates_df['lat'].std()
                lon_std = candidates_df['lon'].std()
                
                # 正規化（アマゾン地域のスケールに基づく）
                normalized_diversity = min(1.0, (lat_std + lon_std) / 2.0)
                return normalized_diversity
            else:
                return 0.5  # デフォルト値
                
        except Exception as e:
            logger.warning(f"Failed to calculate diversity score: {e}")
            return 0.5
    
    def _calculate_validation_score(self, 
                                   candidates_df: pd.DataFrame, 
                                   validation_sites: pd.DataFrame) -> float:
        """既知サイトとの一致度スコアを計算"""
        if len(candidates_df) == 0 or len(validation_sites) == 0:
            return 0.0
        
        try:
            # 簡単な距離ベース一致評価
            # より精密な評価は地理空間ライブラリが必要
            
            # 候補地の範囲内にある既知サイト数
            if all(col in candidates_df.columns for col in ['lat', 'lon']) and \
               all(col in validation_sites.columns for col in ['lat', 'lon']):
                
                cand_lat_range = (candidates_df['lat'].min(), candidates_df['lat'].max())
                cand_lon_range = (candidates_df['lon'].min(), candidates_df['lon'].max())
                
                # 既知サイトのうち候補地域内にあるものの割合
                in_range = validation_sites[
                    (validation_sites['lat'] >= cand_lat_range[0]) &
                    (validation_sites['lat'] <= cand_lat_range[1]) &
                    (validation_sites['lon'] >= cand_lon_range[0]) &
                    (validation_sites['lon'] <= cand_lon_range[1])
                ]
                
                if len(validation_sites) > 0:
                    overlap_ratio = len(in_range) / len(validation_sites)
                    return min(1.0, overlap_ratio * 2.0)  # 0.5で最大スコア
                
            return 0.5  # デフォルト値
            
        except Exception as e:
            logger.warning(f"Failed to calculate validation score: {e}")
            return 0.5


class PipelineRunnerConfig:
    """パイプライン実行設定のヘルパークラス"""
    
    @staticmethod
    def create_amazon_config(project_root: Path) -> Dict[str, Any]:
        """アマゾン地域用のデフォルト設定"""
        return {
            "bbox": [-70.5, -11.5, -66.5, -8.5],  # アクレ州周辺
            "pbf-path": str(project_root / "data/raw/osm/norte-latest.osm.pbf"),
            "rivers-path": str(project_root / "data/raw/hydrorivers_sahydrorivers_sa/HydroRIVERS_v10_sa.shp"),
            "gsw-path": str(project_root / "data/raw/GSW_occurrence/occurrence_70W_10Sv1_4_2021.tif")
        }
    
    @staticmethod
    def create_test_config() -> Dict[str, Any]:
        """テスト用の最小設定"""
        return {
            "bbox": [-68.0, -10.0, -67.0, -9.0],  # 小範囲
            "skip-water-freq": True  # 高速化のため水域頻度計算スキップ
        }


# 使用例とテスト用コード
if __name__ == "__main__":
    # テスト実行例
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parents[3]
    script_path = project_root / "scripts/run_site_identification.py"
    
    # 設定
    config = PipelineRunnerConfig.create_test_config()
    
    # ランナー初期化
    runner = RealPipelineRunner(
        script_path=str(script_path),
        base_config=config,
        timeout_seconds=600  # 10分
    )
    
    # テスト実行
    try:
        score, n_candidates, metrics = runner.run_with_params(
            distance_km=3.5,
            occ_pct=7.2,
            root_weights={"rio": 0.8, "igarape": 0.9, "lago": 0.7},
            experiment_id="test_001"
        )
        
        print(f"Score: {score:.4f}")
        print(f"Candidates: {n_candidates}")
        print(f"Metrics: {json.dumps(metrics, indent=2)}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)