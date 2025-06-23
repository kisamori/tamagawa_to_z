"""History Summarizer - 全 Optuna trial の履歴を JSONL で保持し、LLM 用に要約する."""

from __future__ import annotations

import json
import hashlib
import statistics
import collections
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class HistorySummarizer:
    """全 Optuna trial の履歴を JSONL で保持し、LLM 用に要約する."""

    def __init__(self, log_path: Path = Path("data/output/history.jsonl")):
        """
        初期化.
        
        Args:
            log_path: 履歴ログファイルのパス
        """
        self.log_path = log_path
        self.trials: List[Dict[str, Any]] = []
        
        # 既存ログがあれば読み込み
        if log_path.exists():
            try:
                with log_path.open('r', encoding='utf-8') as f:
                    self.trials = [json.loads(line.strip()) for line in f if line.strip()]
                logger.info(f"Loaded {len(self.trials)} trials from {log_path}")
            except Exception as e:
                logger.warning(f"Failed to load history from {log_path}: {e}")
                self.trials = []
        else:
            logger.info(f"No existing history found at {log_path}")
            # ディレクトリが存在しない場合は作成
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def update(self,
               distance_km: float,
               occ_pct: float,
               score: float,
               weights: Dict[str, float],
               fp_roots: collections.Counter) -> None:
        """
        新しいトライアル結果を履歴に追加.
        
        Args:
            distance_km: 距離閾値
            occ_pct: 水域出現率閾値
            score: スコア
            weights: 語根ウェイト辞書
            fp_roots: False Positive 語根のカウンター
        """
        trial = {
            "id": len(self.trials) + 1,
            "D": distance_km,
            "O": occ_pct,
            "score": score,
            "weights": weights,
            "fp_roots": dict(fp_roots)  # Counterを辞書に変換
        }
        
        self.trials.append(trial)
        
        # ファイルに追記
        try:
            with self.log_path.open("a", encoding='utf-8') as f:
                f.write(json.dumps(trial, ensure_ascii=False) + "\n")
            logger.debug(f"Appended trial {trial['id']} to history")
        except Exception as e:
            logger.error(f"Failed to write trial to history: {e}")

    def to_context(self, top_m: int = 5) -> Dict[str, Any]:
        """
        LLM に渡す compact JSON (≤2 KB).
        
        Args:
            top_m: 上位何件のトライアルを含めるか
            
        Returns:
            LLM用のコンテキスト辞書
        """
        if not self.trials:
            return {}

        # 1. 上位 M trial (スコア順)
        best = sorted(self.trials, key=lambda t: -t["score"])[:top_m]

        # 2. Root ごとの平均 gain/loss
        delta = collections.defaultdict(list)
        
        if len(self.trials) > 1:
            # 基準: 最新 trial の weights
            latest_w = self.trials[-1]["weights"]
            
            # 直近 20 trial だけで勾配近似
            recent_trials = self.trials[-20:] if len(self.trials) > 20 else self.trials
            
            for t in recent_trials:
                for root, w in t["weights"].items():
                    baseline_w = latest_w.get(root, 0)
                    delta[root].append(w - baseline_w)

        avg_gain = {}
        for root, values in delta.items():
            if values:
                avg_gain[root] = round(statistics.mean(values), 3)

        # 3. 累積 FP 頻度（語根ランキング）
        fp_counter = collections.Counter()
        for t in self.trials:
            fp_roots_dict = t.get("fp_roots", {})
            # 辞書形式のfp_rootsをCounterに追加
            for root, count in fp_roots_dict.items():
                fp_counter[root] += count
                
        fp_rank = [root for root, _ in fp_counter.most_common(10)]

        # 4. ハッシュ (キャッシュキー用)
        history_hash = self._generate_history_hash()

        return {
            "best_trials": best,
            "avg_gain_per_root": avg_gain,
            "cumulative_fp_roots": fp_rank,
            "history_hash": history_hash
        }

    def _generate_history_hash(self) -> str:
        """履歴の状態を表すハッシュを生成."""
        if not self.log_path.exists():
            return "empty"
        
        try:
            # ファイルサイズとトライアル数から簡易ハッシュ
            file_size = self.log_path.stat().st_size
            n_trials = len(self.trials)
            hash_input = f"{file_size}_{n_trials}"
            return hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        except Exception as e:
            logger.warning(f"Failed to generate history hash: {e}")
            return "error"

    def get_stats(self) -> Dict[str, Any]:
        """履歴の統計情報を取得."""
        if not self.trials:
            return {"n_trials": 0}
        
        scores = [t["score"] for t in self.trials]
        distances = [t["D"] for t in self.trials]
        occ_pcts = [t["O"] for t in self.trials]
        
        return {
            "n_trials": len(self.trials),
            "best_score": max(scores),
            "avg_score": statistics.mean(scores),
            "score_std": statistics.stdev(scores) if len(scores) > 1 else 0,
            "distance_range": [min(distances), max(distances)],
            "occ_range": [min(occ_pcts), max(occ_pcts)],
            "unique_roots": len(set().union(*(t["weights"].keys() for t in self.trials)))
        }

    def clear(self) -> None:
        """履歴をクリア."""
        self.trials = []
        if self.log_path.exists():
            self.log_path.unlink()
        logger.info("Cleared history")


def create_sample_history() -> HistorySummarizer:
    """テスト用のサンプル履歴を作成."""
    import tempfile
    import random
    
    # 一時ファイルで履歴作成
    temp_dir = Path(tempfile.mkdtemp())
    summarizer = HistorySummarizer(temp_dir / "test_history.jsonl")
    
    # サンプルデータ追加
    roots = ["rio", "lago", "igarape", "parana", "acu", "mirim"]
    
    for i in range(10):
        weights = {root: round(random.uniform(0.1, 1.0), 2) for root in roots}
        fp_counter = collections.Counter({
            random.choice(roots): random.randint(1, 5) 
            for _ in range(random.randint(0, 3))
        })
        
        summarizer.update(
            distance_km=random.uniform(1.0, 5.0),
            occ_pct=random.uniform(2.0, 10.0),
            score=random.uniform(0.3, 0.8),
            weights=weights,
            fp_roots=fp_counter
        )
    
    return summarizer


if __name__ == "__main__":
    # テスト実行
    print("=== History Summarizer Test ===")
    
    # サンプル履歴作成
    summarizer = create_sample_history()
    
    # 統計表示
    stats = summarizer.get_stats()
    print(f"Stats: {stats}")
    
    # コンテキスト生成
    context = summarizer.to_context()
    print(f"Context keys: {list(context.keys())}")
    print(f"Context size: {len(json.dumps(context, ensure_ascii=False))} bytes")
    print(f"Best trials: {len(context.get('best_trials', []))}")
    print(f"History hash: {context.get('history_hash')}")
    
    # クリーンアップ
    summarizer.clear()
    
    print("Test completed.")