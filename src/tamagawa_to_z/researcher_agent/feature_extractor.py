"""最適化ログからの実用的特徴量抽出

実装コストと効果のバランスを考慮し、現実的に計算可能で
o3-pro分析に有用な特徴量を厳選して実装します。
"""

from __future__ import annotations

import numpy as np
import sys
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass

from .schemas.optimization_schema import OptimizationSummary, OptimizationTrial


@dataclass
class FeatureSet:
    """抽出された特徴量セット"""
    basic_stats: Dict[str, Any]
    parameter_exploration: Dict[str, Any]
    optimization_efficiency: Dict[str, Any]
    trial_quality: Dict[str, Any]
    domain_specific: Dict[str, Any]
    temporal_patterns: Dict[str, Any]
    meta_features: Dict[str, Any]


class OptimizationFeatureExtractor:
    """実用的な最適化特徴量抽出器"""
    
    def extract_comprehensive_features(
        self, 
        logs: OptimizationSummary,
        domain_context: Optional[Dict[str, Any]] = None
    ) -> FeatureSet:
        """
        包括的特徴量抽出
        
        Parameters
        ----------
        logs : OptimizationSummary
            最適化ログ
        domain_context : Optional[Dict[str, Any]]
            ドメイン固有コンテキスト
            
        Returns
        -------
        FeatureSet
            抽出された特徴量セット
        """
        
        trials = logs.trials
        completed_trials = [t for t in trials if t.status == "completed" and t.score is not None]
        
        if not completed_trials:
            return self._empty_feature_set()
        
        return FeatureSet(
            basic_stats=self._extract_basic_statistics(completed_trials),
            parameter_exploration=self._extract_parameter_exploration(completed_trials, logs.search_space.parameters),
            optimization_efficiency=self._extract_optimization_efficiency(completed_trials),
            trial_quality=self._extract_trial_quality(trials),
            domain_specific=self._extract_domain_specific_features(completed_trials, domain_context),
            temporal_patterns=self._extract_temporal_patterns(completed_trials),
            meta_features={}  # 他の特徴量計算後に算出
        )
    
    def _extract_basic_statistics(self, trials: List[OptimizationTrial]) -> Dict[str, Any]:
        """基本統計特徴量（実装コスト: 低）"""
        
        scores = [t.score for t in trials]
        candidates_counts = [t.candidates_count for t in trials]
        
        # スコア統計
        score_stats = {
            "count": len(scores),
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "median": np.median(scores),
            "q25": np.percentile(scores, 25),
            "q75": np.percentile(scores, 75),
            "iqr": np.percentile(scores, 75) - np.percentile(scores, 25),
            "skewness": self._calculate_skewness(scores),
            "kurtosis": self._calculate_kurtosis(scores)
        }
        
        # 候補数統計
        candidates_stats = {
            "mean": np.mean(candidates_counts),
            "std": np.std(candidates_counts),
            "min": np.min(candidates_counts),
            "max": np.max(candidates_counts),
            "median": np.median(candidates_counts),
            "zero_count": sum(1 for c in candidates_counts if c == 0),
            "zero_rate": sum(1 for c in candidates_counts if c == 0) / len(candidates_counts),
            "non_zero_mean": np.mean([c for c in candidates_counts if c > 0]) if any(c > 0 for c in candidates_counts) else 0
        }
        
        # スコア分布
        positive_scores = [s for s in scores if s > 0]
        negative_scores = [s for s in scores if s < 0]
        zero_scores = [s for s in scores if s == 0]
        
        score_distribution = {
            "positive_count": len(positive_scores),
            "negative_count": len(negative_scores),
            "zero_count": len(zero_scores),
            "positive_rate": len(positive_scores) / len(scores),
            "negative_rate": len(negative_scores) / len(scores),
            "zero_rate": len(zero_scores) / len(scores)
        }
        
        # 相関分析
        correlations = {}
        if len(set(candidates_counts)) > 1 and len(set(scores)) > 1:
            correlations["score_candidates_correlation"] = np.corrcoef(scores, candidates_counts)[0, 1]
        
        return {
            "score_statistics": score_stats,
            "candidates_statistics": candidates_stats,
            "score_distribution": score_distribution,
            "correlations": correlations
        }
    
    def _extract_parameter_exploration(
        self, 
        trials: List[OptimizationTrial],
        search_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """パラメータ探索特徴量（実装コスト: 低〜中）"""
        
        if not trials:
            return {}
        
        # パラメータ値の抽出
        param_values = defaultdict(list)
        for trial in trials:
            for param_name, value in trial.params.items():
                if value is not None:
                    param_values[param_name].append(value)
        
        # パラメータ統計
        param_stats = {}
        param_utilization = {}
        
        for param_name, values in param_values.items():
            if not values:
                continue
                
            param_config = search_space.get(param_name, {})
            
            # 基本統計
            param_stats[param_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "unique_count": len(set(values)),
                "unique_rate": len(set(values)) / len(values)
            }
            
            # 範囲利用率
            if param_config.get("type") in ["float", "int"]:
                defined_min = param_config.get("low", np.min(values))
                defined_max = param_config.get("high", np.max(values))
                
                if defined_max > defined_min:
                    actual_range = np.max(values) - np.min(values)
                    defined_range = defined_max - defined_min
                    utilization_rate = actual_range / defined_range
                    
                    param_utilization[param_name] = {
                        "defined_range": [defined_min, defined_max],
                        "actual_range": [np.min(values), np.max(values)],
                        "utilization_rate": utilization_rate,
                        "boundary_exploration": {
                            "near_min": sum(1 for v in values if abs(v - defined_min) / defined_range < 0.1) / len(values),
                            "near_max": sum(1 for v in values if abs(v - defined_max) / defined_range < 0.1) / len(values)
                        }
                    }
        
        # パラメータ間相関
        param_correlations = {}
        param_names = list(param_values.keys())
        
        for i, param1 in enumerate(param_names):
            for j, param2 in enumerate(param_names[i+1:], i+1):
                values1 = param_values[param1]
                values2 = param_values[param2]
                
                if len(values1) == len(values2) and len(set(values1)) > 1 and len(set(values2)) > 1:
                    corr = np.corrcoef(values1, values2)[0, 1]
                    param_correlations[f"{param1}_{param2}"] = corr
        
        # 探索多様性
        unique_combinations = len(set(tuple(sorted(t.params.items())) for t in trials))
        diversity_metrics = {
            "unique_combinations": unique_combinations,
            "diversity_ratio": unique_combinations / len(trials),
            "repetition_rate": (len(trials) - unique_combinations) / len(trials)
        }
        
        return {
            "parameter_statistics": param_stats,
            "parameter_utilization": param_utilization,
            "parameter_correlations": param_correlations,
            "diversity_metrics": diversity_metrics
        }
    
    def _extract_optimization_efficiency(self, trials: List[OptimizationTrial]) -> Dict[str, Any]:
        """最適化効率特徴量（実装コスト: 中）"""
        
        scores = [t.score for t in trials]
        
        # 改善効率
        best_scores = []
        current_best = float('-inf')
        
        for score in scores:
            current_best = max(current_best, score)
            best_scores.append(current_best)
        
        total_improvement = best_scores[-1] - best_scores[0] if len(best_scores) > 1 else 0
        
        # 早期改善vs後期改善
        quarter_point = len(scores) // 4
        if quarter_point > 0:
            early_improvement = best_scores[quarter_point-1] - best_scores[0]
            late_improvement = best_scores[-1] - best_scores[-quarter_point]
        else:
            early_improvement = late_improvement = 0
        
        # 改善頻度
        improvements = []
        for i in range(1, len(scores)):
            if scores[i] > scores[i-1]:
                improvements.append(i)
        
        improvement_frequency = len(improvements) / (len(scores) - 1) if len(scores) > 1 else 0
        
        # 停滞期間
        stagnation_periods = self._detect_stagnation_periods(best_scores)
        
        return {
            "improvement_efficiency": {
                "total_improvement": total_improvement,
                "improvement_per_trial": total_improvement / len(trials),
                "early_improvement": early_improvement,
                "late_improvement": late_improvement,
                "improvement_frequency": improvement_frequency
            },
            "convergence_characteristics": {
                "stagnation_periods": stagnation_periods,
                "longest_stagnation": max(stagnation_periods) if stagnation_periods else 0,
                "convergence_speed": self._estimate_convergence_speed(best_scores)
            }
        }
    
    def _extract_trial_quality(self, trials: List[OptimizationTrial]) -> Dict[str, Any]:
        """試行品質特徴量（実装コスト: 低）"""
        
        total_trials = len(trials)
        completed_trials = sum(1 for t in trials if t.status == "completed")
        failed_trials = sum(1 for t in trials if t.status == "failed")
        
        # 実行時間分析（利用可能な場合）
        durations = [t.duration_seconds for t in trials if t.duration_seconds is not None]
        duration_stats = {}
        
        if durations:
            duration_stats = {
                "mean_duration": np.mean(durations),
                "std_duration": np.std(durations),
                "min_duration": np.min(durations),
                "max_duration": np.max(durations)
            }
        
        # エラーパターン分析
        error_messages = [t.error_message for t in trials if t.error_message]
        error_patterns = Counter(error_messages) if error_messages else {}
        
        return {
            "success_metrics": {
                "completion_rate": completed_trials / total_trials,
                "failure_rate": failed_trials / total_trials,
                "total_trials": total_trials
            },
            "execution_quality": duration_stats,
            "error_analysis": {
                "error_count": len(error_messages),
                "unique_errors": len(set(error_messages)),
                "error_patterns": dict(error_patterns.most_common(5))
            }
        }
    
    def _extract_domain_specific_features(
        self, 
        trials: List[OptimizationTrial],
        domain_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """ドメイン固有特徴量（実装コスト: 低〜中）"""
        
        candidates_counts = [t.candidates_count for t in trials]
        
        # 候補数分布の詳細
        candidates_distribution = {
            "zero_candidates": sum(1 for c in candidates_counts if c == 0),
            "tiny_candidates": sum(1 for c in candidates_counts if 1 <= c <= 5),
            "small_candidates": sum(1 for c in candidates_counts if 6 <= c <= 20),
            "medium_candidates": sum(1 for c in candidates_counts if 21 <= c <= 100),
            "large_candidates": sum(1 for c in candidates_counts if c > 100)
        }
        
        # 候補数とスコアの関係
        score_by_candidates = defaultdict(list)
        for trial in trials:
            bucket = self._categorize_candidates_count(trial.candidates_count)
            score_by_candidates[bucket].append(trial.score)
        
        candidates_score_relationship = {}
        for bucket, scores in score_by_candidates.items():
            if scores:
                candidates_score_relationship[bucket] = {
                    "mean_score": np.mean(scores),
                    "count": len(scores)
                }
        
        return {
            "candidates_distribution": candidates_distribution,
            "candidates_score_relationship": candidates_score_relationship,
            "domain_context": domain_context or {}
        }
    
    def _extract_temporal_patterns(self, trials: List[OptimizationTrial]) -> Dict[str, Any]:
        """時系列パターン特徴量（実装コスト: 中）"""
        
        scores = [t.score for t in trials]
        
        if len(scores) < 3:
            return {"insufficient_data": True}
        
        # 傾向分析
        recent_third = len(scores) * 2 // 3
        early_scores = scores[:len(scores)//3]
        recent_scores = scores[recent_third:]
        
        trend_analysis = {}
        if early_scores and recent_scores:
            trend_analysis = {
                "early_mean": np.mean(early_scores),
                "recent_mean": np.mean(recent_scores),
                "overall_trend": "improving" if np.mean(recent_scores) > np.mean(early_scores) else "declining",
                "trend_strength": abs(np.mean(recent_scores) - np.mean(early_scores)) / (np.std(scores) + 1e-8)
            }
        
        # 変動性分析
        score_changes = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        volatility_metrics = {
            "score_volatility": np.std(score_changes) if score_changes else 0,
            "positive_changes": sum(1 for c in score_changes if c > 0),
            "negative_changes": sum(1 for c in score_changes if c < 0),
            "zero_changes": sum(1 for c in score_changes if abs(c) < 1e-8)
        }
        
        return {
            "trend_analysis": trend_analysis,
            "volatility_metrics": volatility_metrics,
            "temporal_statistics": {
                "total_periods": len(scores),
                "analysis_window": len(scores) // 3
            }
        }
    
    # ヘルパーメソッド
    def _calculate_skewness(self, values: List[float]) -> float:
        """歪度計算"""
        if len(values) < 3:
            return 0.0
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val == 0:
            return 0.0
        return np.mean([((x - mean_val) / std_val) ** 3 for x in values])
    
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """尖度計算"""
        if len(values) < 4:
            return 0.0
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val == 0:
            return 0.0
        return np.mean([((x - mean_val) / std_val) ** 4 for x in values]) - 3
    
    def _detect_stagnation_periods(self, best_scores: List[float], tolerance: float = 1e-6) -> List[int]:
        """停滞期間の検出"""
        stagnation_periods = []
        current_period = 0
        
        for i in range(1, len(best_scores)):
            if abs(best_scores[i] - best_scores[i-1]) < tolerance:
                current_period += 1
            else:
                if current_period > 0:
                    stagnation_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            stagnation_periods.append(current_period)
        
        return stagnation_periods
    
    def _estimate_convergence_speed(self, best_scores: List[float]) -> float:
        """収束速度の推定"""
        if len(best_scores) < 2:
            return 0.0
        
        total_improvement = best_scores[-1] - best_scores[0]
        if total_improvement <= 0:
            return 0.0
        
        # 50%改善到達時点
        target_score = best_scores[0] + total_improvement * 0.5
        
        for i, score in enumerate(best_scores):
            if score >= target_score:
                return i / len(best_scores)  # 正規化された収束速度
        
        return 1.0  # 50%改善に到達しなかった
    
    def _categorize_candidates_count(self, count: int) -> str:
        """候補数のカテゴリ化"""
        if count == 0:
            return "zero"
        elif count <= 5:
            return "tiny"
        elif count <= 20:
            return "small"
        elif count <= 100:
            return "medium"
        else:
            return "large"
    
    def _empty_feature_set(self) -> FeatureSet:
        """空の特徴量セット"""
        return FeatureSet(
            basic_stats={},
            parameter_exploration={},
            optimization_efficiency={},
            trial_quality={},
            domain_specific={},
            temporal_patterns={},
            meta_features={}
        )