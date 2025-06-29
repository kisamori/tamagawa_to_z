"""Hybrid-BO実装 - OptunaとLLMを組み合わせたハイブリッド最適化."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter

import optuna
import yaml
import numpy as np
import geopandas as gpd

from .llm_root import get_root_weights
from .pipeline_runner import run_pipeline_with_params
from .history_summarizer import HistorySummarizer

logger = logging.getLogger(__name__)


class HybridBO:
    """
    Hybrid Bayesian Optimization クラス.
    
    距離・湿地はOptuna、語根ウェイトはLLMで最適化する。
    """
    
    def __init__(
        self,
        data_splitter,
        toponym_stats: Dict[str, int],
        config_path: Optional[Path] = None,
        n_trials: int = 50,
        resume: bool = False
    ):
        """
        初期化.
        
        Args:
            data_splitter: DataSplitterインスタンス
            toponym_stats: 語根の出現統計
            config_path: 設定ファイルパス（optuna_space.yaml）
            n_trials: 試行回数
            resume: 既存studyを再開するかどうか
        """
        self.data_splitter = data_splitter
        self.toponym_stats = toponym_stats
        self.n_trials = n_trials
        self.resume = resume
        
        # 設定読み込み
        if config_path is None:
            config_path = Path("configs/optuna_space.yaml")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # データ分割
        self.splits = data_splitter.split()
        self.val_gdf = self.splits["val"]
        
        # False Positive ログ
        self.fp_log: List[str] = []
        
        # History Summarizer
        self.summarizer = HistorySummarizer()
        
        # タイムスタンプディレクトリを作成
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(self.config["experiment"]["results_dir"]) / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Optuna study作成
        self._setup_optuna_study()
        
        logger.info(f"Initialized HybridBO with {len(self.val_gdf)} validation sites")
        
    def run(self) -> Dict[str, Any]:
        """
        ハイブリッド最適化を実行する.
        
        Returns:
            最適化結果辞書
        """
        logger.info(f"Starting Hybrid-BO with {self.n_trials} trials")
        
        try:
            # 最適化実行
            self.study.optimize(self._objective, n_trials=self.n_trials)
            
            # 結果の整理
            best_result = self._process_results()
            
            logger.info(f"Optimization completed. Best score: {best_result['score']:.4f}")
            
            return best_result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    def get_study_info(self) -> Dict[str, Any]:
        """
        Optuna studyの情報を取得する.
        
        Returns:
            Study情報辞書
        """
        try:
            best_value = self.study.best_value if len(self.study.trials) > 0 else None
            best_params = self.study.best_params if len(self.study.trials) > 0 else None
        except ValueError:
            # 成功したtrialがない場合
            best_value = None
            best_params = None
            
        return {
            "study_name": self.study.study_name,
            "direction": self.study.direction.name,
            "n_trials": len(self.study.trials),
            "best_value": best_value,
            "best_params": best_params
        }
    
    def _setup_optuna_study(self):
        """Optuna studyをセットアップする."""
        storage_url = self.config["storage"]["url"]
        base_study_name = self.config["storage"]["study_name"]
        
        # resumeしない場合はタイムスタンプ付きの新しいstudy名を生成
        if self.resume:
            study_name = base_study_name
        else:
            study_name = f"{base_study_name}_{self.timestamp}"
        
        # データベースディレクトリ作成
        if storage_url.startswith("sqlite:///"):
            db_path = Path(storage_url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sampler設定
        sampler_config = self.config["optimization"]["sampler"]
        if sampler_config["type"] == "TPESampler":
            sampler = optuna.samplers.TPESampler(
                multivariate=sampler_config.get("multivariate", True),
                n_ei_candidates=sampler_config.get("n_ei_candidates", 24)
            )
        else:
            sampler = optuna.samplers.RandomSampler()
        
        # Pruner設定
        pruning_config = self.config["optimization"]["pruning"]
        if pruning_config.get("enabled", True):
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=pruning_config.get("n_startup_trials", 10),
                n_warmup_steps=pruning_config.get("n_warmup_steps", 5)
            )
        else:
            pruner = optuna.pruners.NopPruner()
        
        # Study作成
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction=self.config["optimization"]["direction"],
            sampler=sampler,
            pruner=pruner,
            load_if_exists=self.resume
        )
        
        logger.info(f"Created Optuna study: {study_name} (resume={self.resume})")
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        最適化の目的関数.
        
        Args:
            trial: Optuna trial
            
        Returns:
            目的関数値（最大化）
        """
        # 距離・湿地パラメータをOptunaでサンプリング
        search_space = self.config["search_space"]
        
        distance_km = trial.suggest_float(
            "distance_km",
            search_space["distance_km"]["low"],
            search_space["distance_km"]["high"],
            log=search_space["distance_km"].get("log", False)
        )
        
        occ_pct = trial.suggest_float(
            "occ_pct", 
            search_space["occ_pct"]["low"],
            search_space["occ_pct"]["high"],
            log=search_space["occ_pct"].get("log", False)
        )
        
        # 履歴要約を取得
        context = self.summarizer.to_context()
        
        # LLMで語根ウェイトを推論
        llm_context = {
            "distance_km": distance_km,
            "occ_pct": occ_pct,
            "toponym_stats": self.toponym_stats,  # 地名統計を明示的に追加
            "false_pos": self.fp_log[-10:] if self.fp_log else [],  # 最新10件（テンプレートと一致）
            **context,  # 履歴要約を追加
        }
        
        try:
            # LLM設定
            llm_config = self.config["llm"]
            root_weights = get_root_weights(
                context=llm_context,
                model=llm_config["model"],
                temperature=llm_config["temperature"],
                max_tokens=llm_config.get("max_tokens", 1000),
                cache_dir=self.config["experiment"]["cache_dir"],
                system_prompt=llm_config.get("system_prompt"),
                user_prompt_template=llm_config.get("user_prompt_template")
            )
            
        except Exception as e:
            logger.warning(f"LLM query failed, using default weights: {e}")
            root_weights = self._get_default_weights()
        
        # パイプライン実行
        try:
            score, fp_examples = run_pipeline_with_params(
                distance_km=distance_km,
                occ_pct=occ_pct,
                root_weights=root_weights,
                validation_set=self.val_gdf,
                return_fp=True,
                experiment_id=f"trial_{trial.number:04d}",
                run_dir=self.run_dir
            )
            
            # FPログを更新
            self.fp_log.extend(fp_examples)
            
            # 履歴へ追記
            self.summarizer.update(
                distance_km=distance_km,
                occ_pct=occ_pct,
                score=score,
                weights=root_weights,
                fp_roots=Counter(fp_examples)
            )
            
            # 中間値を記録（pruning用）
            trial.report(score, step=0)
            
            # Pruning判定
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            logger.debug(f"Trial {trial.number}: score={score:.4f}, "
                        f"distance={distance_km:.2f}, occ={occ_pct:.2f}")
            
            return score
            
        except Exception as e:
            logger.error(f"Pipeline execution failed in trial {trial.number}: {e}")
            # 失敗した場合は低いスコアを返す
            return -1.0
    
    def _get_default_weights(self) -> Dict[str, float]:
        """デフォルトの語根ウェイトを取得する."""
        # 統計に基づく単純なヒューリスティック
        default_weights = {}
        
        for root, count in self.toponym_stats.items():
            # 出現頻度に基づく重み（ログスケール）
            if count > 0:
                weight = min(1.0, max(0.1, np.log10(count + 1) / 3.0))
            else:
                weight = 0.1
            default_weights[root] = round(weight, 2)
        
        logger.debug(f"Generated default weights for {len(default_weights)} roots")
        return default_weights
    
    def _process_results(self) -> Dict[str, Any]:
        """最適化結果を処理する."""
        if not self.study.best_trial:
            raise RuntimeError("No successful trials found")
        
        best_trial = self.study.best_trial
        
        # 最良パラメータでLLM語根ウェイトを再取得
        history_context = self.summarizer.to_context()
        best_context = {
            "distance_km": best_trial.params["distance_km"],
            "occ_pct": best_trial.params["occ_pct"],
            "toponym_stats": self.toponym_stats,  # 地名統計を明示的に追加
            "false_pos": self.fp_log[-10:] if self.fp_log else [],  # 最新10件（テンプレートと一致）
            **history_context,  # 履歴要約を追加
        }
        
        try:
            llm_config = self.config["llm"]
            best_root_weights = get_root_weights(
                context=best_context,
                model=llm_config["model"],
                temperature=llm_config["temperature"],
                cache_dir=self.config["experiment"]["cache_dir"]
            )
        except Exception as e:
            logger.warning(f"Failed to get best root weights: {e}")
            best_root_weights = self._get_default_weights()
        
        # 結果をまとめる
        result = {
            "score": best_trial.value,
            "distance_km": best_trial.params["distance_km"],
            "occ_pct": best_trial.params["occ_pct"],
            "root_weights": best_root_weights,
            "trial_number": best_trial.number,
            "optimization_history": self._get_optimization_history()
        }
        
        # 結果を保存
        self._save_results(result)
        
        return result
    
    def _get_optimization_history(self) -> List[Dict[str, Any]]:
        """最適化履歴を取得する."""
        history = []
        
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    "trial_number": trial.number,
                    "score": trial.value,
                    "params": trial.params,
                    "datetime": trial.datetime_start.isoformat() if trial.datetime_start else None
                })
        
        # スコア順でソート
        history.sort(key=lambda x: x["score"], reverse=True)
        
        return history
    
    def _save_results(self, result: Dict[str, Any]):
        """結果をファイルに保存する."""
        # タイムスタンプディレクトリを使用
        output_dir = self.run_dir
        
        # 最良パラメータ保存
        best_params_file = output_dir / "best_params.json"
        with open(best_params_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved best parameters to {best_params_file}")
        
        # 最適化履歴保存
        history_file = output_dir / "optimization_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(result["optimization_history"], f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved optimization history to {history_file}")


def create_toponym_stats_from_all_roots() -> Dict[str, int]:
    """
    all_roots.csvから語根統計を作成する（実データベース）.
    
    Returns:
        語根統計辞書 {root: estimated_count}
    """
    try:
        # all_roots.csvから語根データを読み込み
        from pathlib import Path
        import pandas as pd
        
        project_root = Path(__file__).resolve().parents[3]
        all_roots_path = project_root / "data" / "dict" / "all_roots.csv"
        
        if not all_roots_path.exists():
            logger.warning(f"all_roots.csv not found at {all_roots_path}, using fallback sample data")
            return _create_fallback_sample_stats()
        
        df = pd.read_csv(all_roots_path)
        
        if df.empty:
            logger.warning("all_roots.csv is empty, using fallback sample data")
            return _create_fallback_sample_stats()
        
        logger.info(f"Loading toponym stats from all_roots.csv: {len(df)} roots")
        
        # カテゴリ別の推定出現頻度を設定
        category_weights = {
            'water': 15,      # 水系語根は比較的高頻度
            'terrain': 8,     # 地形語根は中頻度
            'flora': 6,       # 植生語根は中頻度
            'culture': 4,     # 文化語根は低頻度
            'resource': 3,    # 資源語根は低頻度
            'temporal': 5     # 時間語根は中頻度
        }
        
        stats = {}
        for _, row in df.iterrows():
            root = row.get('root', '')
            category = row.get('category', 'water')
            weight = float(row.get('weight', 0.5))
            
            if root:
                # カテゴリ別基準頻度 × 語根重み × ランダム要素（正規化）
                base_freq = category_weights.get(category, 5)
                # ハッシュ値を0-1の範囲に正規化
                random_factor = 0.8 + 0.4 * abs(hash(root) % 1000) / 1000.0
                estimated_count = max(1, int(base_freq * weight * random_factor))
                stats[root] = estimated_count
        
        # カテゴリ別統計をログ出力
        if 'category' in df.columns:
            category_stats = df['category'].value_counts().to_dict()
            logger.info(f"Category distribution: {category_stats}")
        
        total_count = sum(stats.values())
        logger.info(f"Generated toponym stats: {len(stats)} roots, total estimated count: {total_count}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to load from all_roots.csv: {e}")
        logger.warning("Falling back to sample data")
        return _create_fallback_sample_stats()


def _create_fallback_sample_stats() -> Dict[str, int]:
    """
    フォールバック用のサンプル統計（all_roots.csv読み込み失敗時のみ使用）.
    
    Returns:
        サンプル地名統計辞書
    """
    logger.warning("Using fallback sample toponym statistics")
    sample_stats = {
        "igarape": 25,
        "parana": 18,
        "lago": 12,
        "rio": 45,
        "acu": 8,
        "mirim": 15,
        "guacu": 6,
        "tinga": 4,
        "preta": 3,
        "cocha": 7,
        "quebrada": 9,
        "cano": 11,
        "pozo": 5,
        "agua": 13
    }
    
    return sample_stats


if __name__ == "__main__":
    # テスト実行
    from ..dataset.splitter import DataSplitter, create_sample_master_csv
    import tempfile
    
    # テスト用データ作成
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_csv_path = Path(f.name)
    
    create_sample_master_csv(sample_csv_path)
    
    # 設定ファイルパス
    config_path = Path("configs/dataset_split.yaml")
    optuna_config_path = Path("configs/optuna_space.yaml")
    
    try:
        print("=== Hybrid-BO Test ===")
        
        # データ分割器作成
        splitter = DataSplitter(config_path, sample_csv_path)
        
        # 地名統計作成
        toponym_stats = create_sample_toponym_stats()
        
        # Hybrid-BO実行
        hybrid_bo = HybridBO(
            data_splitter=splitter,
            toponym_stats=toponym_stats,
            config_path=optuna_config_path,
            n_trials=5  # テスト用に短縮
        )
        
        # 最適化実行
        result = hybrid_bo.run()
        
        print(f"Best score: {result['score']:.4f}")
        print(f"Best params: distance={result['distance_km']:.2f}km, occ={result['occ_pct']:.2f}%")
        print(f"Root weights: {list(result['root_weights'].keys())[:5]}...")  # 最初の5個
        
        # Study情報
        study_info = hybrid_bo.get_study_info()
        print(f"Study info: {study_info}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # クリーンアップ
        sample_csv_path.unlink()
    
    print("Test completed.")