"""最適化ログの標準化スキーマ定義

このモジュールは、Optuna、Grid Search、Random Search、Manual Tuning等の
異なる最適化手法のログを統一的に扱うための標準化フォーマットを定義します。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import yaml
import json
from pathlib import Path


@dataclass
class OptimizationTrial:
    """単一の最適化試行を表すデータクラス"""
    trial_id: int
    score: Optional[float]
    candidates_count: int
    params: Dict[str, Any]
    timestamp: Optional[str] = None
    status: str = "completed"  # completed, failed, pruned
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class OptimizationObjective:
    """目的関数の定義"""
    direction: str  # maximize, minimize
    function_design: str  # weighted_composite, single_metric, custom
    weights: Dict[str, float]
    thresholds: Optional[Dict[str, float]] = None


@dataclass
class OptimizationSearchSpace:
    """探索空間の定義"""
    parameters: Dict[str, Dict[str, Any]]
    
    def __post_init__(self):
        """パラメータ定義の正規化"""
        for param_name, param_def in self.parameters.items():
            if isinstance(param_def, dict):
                # 必要なフィールドのデフォルト値を設定
                param_def.setdefault("type", "float")
                param_def.setdefault("log", False)


@dataclass
class OptimizationPatterns:
    """自動検出される問題パターン"""
    candidates_zero_rate: float
    negative_score_rate: float
    convergence_issues: bool
    best_score: Optional[float] = None
    score_distribution: Optional[Dict[str, float]] = None
    parameter_correlation: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """パターン分析の追加計算"""
        if self.score_distribution is None:
            self.score_distribution = {}


@dataclass
class OptimizationSummary:
    """最適化プロセス全体のサマリー"""
    method: str  # optuna_tpe, grid_search, random_search, manual
    status: str  # running, completed, failed
    total_trials: int
    successful_trials: int
    objective: OptimizationObjective
    search_space: OptimizationSearchSpace
    trials: List[OptimizationTrial]
    patterns: OptimizationPatterns
    
    # メタデータ
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    total_duration_seconds: Optional[float] = None
    study_name: Optional[str] = None
    
    def to_yaml(self, output_path: Path) -> None:
        """YAML形式でファイルに保存"""
        data = {
            "optimization": {
                "method": self.method,
                "status": self.status,
                "total_trials": self.total_trials,
                "successful_trials": self.successful_trials,
                "study_name": self.study_name,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_duration_seconds": self.total_duration_seconds
            },
            "objective": {
                "direction": self.objective.direction,
                "function_design": self.objective.function_design,
                "weights": self.objective.weights,
                "thresholds": self.objective.thresholds
            },
            "search_space": {
                param_name: param_def 
                for param_name, param_def in self.search_space.parameters.items()
            },
            "trials": [
                {
                    "trial_id": trial.trial_id,
                    "score": trial.score,
                    "candidates_count": trial.candidates_count,
                    "params": trial.params,
                    "timestamp": trial.timestamp,
                    "status": trial.status,
                    "duration_seconds": trial.duration_seconds,
                    "error_message": trial.error_message
                }
                for trial in self.trials
            ],
            "patterns": {
                "candidates_zero_rate": self.patterns.candidates_zero_rate,
                "negative_score_rate": self.patterns.negative_score_rate,
                "convergence_issues": self.patterns.convergence_issues,
                "best_score": self.patterns.best_score,
                "score_distribution": self.patterns.score_distribution,
                "parameter_correlation": self.patterns.parameter_correlation
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'OptimizationSummary':
        """YAML形式から読み込み"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # データ構造の再構築
        opt_data = data["optimization"]
        obj_data = data["objective"] 
        space_data = data["search_space"]
        trials_data = data["trials"]
        patterns_data = data["patterns"]
        
        objective = OptimizationObjective(
            direction=obj_data["direction"],
            function_design=obj_data["function_design"],
            weights=obj_data["weights"],
            thresholds=obj_data.get("thresholds")
        )
        
        search_space = OptimizationSearchSpace(parameters=space_data)
        
        trials = [
            OptimizationTrial(
                trial_id=trial["trial_id"],
                score=trial["score"],
                candidates_count=trial["candidates_count"],
                params=trial["params"],
                timestamp=trial.get("timestamp"),
                status=trial.get("status", "completed"),
                duration_seconds=trial.get("duration_seconds"),
                error_message=trial.get("error_message")
            )
            for trial in trials_data
        ]
        
        patterns = OptimizationPatterns(
            candidates_zero_rate=patterns_data["candidates_zero_rate"],
            negative_score_rate=patterns_data["negative_score_rate"],
            convergence_issues=patterns_data["convergence_issues"],
            best_score=patterns_data.get("best_score"),
            score_distribution=patterns_data.get("score_distribution"),
            parameter_correlation=patterns_data.get("parameter_correlation")
        )
        
        return cls(
            method=opt_data["method"],
            status=opt_data["status"],
            total_trials=opt_data["total_trials"],
            successful_trials=opt_data["successful_trials"],
            objective=objective,
            search_space=search_space,
            trials=trials,
            patterns=patterns,
            start_time=opt_data.get("start_time"),
            end_time=opt_data.get("end_time"),
            total_duration_seconds=opt_data.get("total_duration_seconds"),
            study_name=opt_data.get("study_name")
        )


def calculate_optimization_patterns(trials: List[OptimizationTrial]) -> OptimizationPatterns:
    """試行データから問題パターンを自動計算"""
    if not trials:
        return OptimizationPatterns(
            candidates_zero_rate=0.0,
            negative_score_rate=0.0,
            convergence_issues=True
        )
    
    successful_trials = [t for t in trials if t.status == "completed" and t.score is not None]
    
    if not successful_trials:
        return OptimizationPatterns(
            candidates_zero_rate=1.0,
            negative_score_rate=0.0,
            convergence_issues=True
        )
    
    # 候補数0の率
    zero_candidates = sum(1 for t in successful_trials if t.candidates_count == 0)
    candidates_zero_rate = zero_candidates / len(successful_trials)
    
    # 負スコアの率
    negative_scores = sum(1 for t in successful_trials if t.score < 0)
    negative_score_rate = negative_scores / len(successful_trials)
    
    # 収束問題の検出（最近の試行で改善がない）
    convergence_issues = False
    if len(successful_trials) >= 10:
        recent_scores = [t.score for t in successful_trials[-5:]]
        earlier_scores = [t.score for t in successful_trials[-10:-5]]
        if recent_scores and earlier_scores:
            recent_best = max(recent_scores)
            earlier_best = max(earlier_scores)
            convergence_issues = recent_best <= earlier_best
    
    # スコア分布統計
    scores = [t.score for t in successful_trials]
    score_distribution = {
        "min": min(scores),
        "max": max(scores),
        "mean": sum(scores) / len(scores),
        "std": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5
    }
    
    best_score = max(scores) if scores else None
    
    return OptimizationPatterns(
        candidates_zero_rate=candidates_zero_rate,
        negative_score_rate=negative_score_rate,
        convergence_issues=convergence_issues,
        best_score=best_score,
        score_distribution=score_distribution
    )


def validate_optimization_summary(summary: OptimizationSummary) -> List[str]:
    """最適化サマリーの妥当性検証"""
    issues = []
    
    # 基本的な整合性チェック
    if summary.successful_trials > summary.total_trials:
        issues.append("successful_trials が total_trials を超えています")
    
    if len(summary.trials) != summary.total_trials:
        issues.append(f"trials数({len(summary.trials)})と total_trials({summary.total_trials})が一致しません")
    
    # 目的関数の重み合計チェック
    if summary.objective.function_design == "weighted_composite":
        weight_sum = sum(abs(w) for w in summary.objective.weights.values())
        if weight_sum == 0:
            issues.append("目的関数の重みの合計が0です")
    
    # パターン検証
    patterns = summary.patterns
    if patterns.candidates_zero_rate > 0.8:
        issues.append(f"候補数0の率が異常に高いです: {patterns.candidates_zero_rate:.1%}")
    
    if patterns.best_score is not None and patterns.best_score <= 0 and summary.objective.direction == "maximize":
        issues.append("最大化目標なのに最良スコアが0以下です")
    
    return issues