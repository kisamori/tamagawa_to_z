"""最適化プロセス専門分析モジュール

このモジュールは、OptunやGrid Search等の最適化ログを分析し、
目的関数設計やパラメータ空間設定の問題を検出・修正提案します。

o3-pro Response APIを使用した高度な分析を提供します。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from openai import OpenAI

from .schemas.optimization_schema import OptimizationSummary, OptimizationTrial, OptimizationPatterns

logger = logging.getLogger(__name__)


@dataclass
class OptimizationIssue:
    """検出された最適化問題"""
    issue_type: str
    severity: str  # critical, high, medium, low
    description: str
    root_cause: str
    affected_metrics: List[str]
    evidence: Dict[str, Any]
    confidence: float  # 0.0-1.0


@dataclass  
class OptimizationProposal:
    """最適化問題の修正提案"""
    proposal_id: str
    issue_type: str
    action_type: str  # adjust_objective_weights, expand_search_space, redesign_objective
    params: Dict[str, Any]
    rationale: str
    expected_improvement: str
    implementation_effort: str  # low, medium, high
    risk_level: str  # low, medium, high


class OptimizationAnalyzer:
    """最適化プロセス分析エンジン"""
    
    def __init__(self, openai_client: Optional[OpenAI] = None):
        """
        初期化
        
        Parameters
        ----------
        openai_client : Optional[OpenAI]
            OpenAIクライアント。Noneの場合は環境変数から初期化
        """
        self.client = openai_client or OpenAI(
            api_key=os.getenv("OPENAI_API_KEY_TIRE5")
        )
        
        # 問題検出のしきい値
        self.thresholds = {
            "candidates_zero_rate_critical": 0.8,
            "candidates_zero_rate_high": 0.5,
            "negative_score_rate_critical": 0.6,
            "negative_score_rate_high": 0.3,
            "convergence_stagnation_trials": 10
        }
    
    def analyze_optimization_issues(
        self, 
        optimization_logs: OptimizationSummary,
        optimization_config: Optional[Dict[str, Any]] = None
    ) -> List[OptimizationIssue]:
        """
        最適化ログから問題を検出・分析
        
        Parameters
        ----------
        optimization_logs : OptimizationSummary
            標準化された最適化ログ
        optimization_config : Optional[Dict[str, Any]]
            最適化設定（optuna_space.yaml等）
            
        Returns
        -------
        List[OptimizationIssue]
            検出された問題のリスト
        """
        issues = []
        
        # 1. 基本的なパターン分析
        issues.extend(self._detect_basic_patterns(optimization_logs))
        
        # 2. 目的関数設計問題
        issues.extend(self._detect_objective_function_issues(optimization_logs, optimization_config))
        
        # 3. パラメータ空間問題  
        issues.extend(self._detect_search_space_issues(optimization_logs, optimization_config))
        
        # 4. 収束・探索問題
        issues.extend(self._detect_convergence_issues(optimization_logs))
        
        logger.info(f"Detected {len(issues)} optimization issues")
        
        return issues
    
    def suggest_optimization_fixes(
        self,
        issues: List[OptimizationIssue],
        optimization_logs: OptimizationSummary,
        optimization_config: Optional[Dict[str, Any]] = None
    ) -> List[OptimizationProposal]:
        """
        検出された問題に対する修正提案を生成（o3-pro Response API使用）
        
        Parameters
        ----------
        issues : List[OptimizationIssue]
            検出された問題
        optimization_logs : OptimizationSummary
            最適化ログ
        optimization_config : Optional[Dict[str, Any]]
            最適化設定
            
        Returns
        -------
        List[OptimizationProposal]
            修正提案のリスト
        """
        if not issues:
            return []
        
        try:
            # o3-pro Response API で高度な提案生成
            proposals = self._generate_proposals_with_o3_pro(
                issues, optimization_logs, optimization_config
            )
            
            # フォールバック: ルールベース提案
            if not proposals:
                logger.warning("o3-pro提案生成に失敗。ルールベース提案を使用")
                proposals = self._generate_fallback_proposals(issues)
            
            logger.info(f"Generated {len(proposals)} optimization proposals")
            return proposals
            
        except Exception as e:
            logger.error(f"提案生成に失敗: {e}")
            return self._generate_fallback_proposals(issues)
    
    def _detect_basic_patterns(self, logs: OptimizationSummary) -> List[OptimizationIssue]:
        """基本的な問題パターンを検出"""
        issues = []
        patterns = logs.patterns
        
        # 候補数0が支配的な問題
        if patterns.candidates_zero_rate >= self.thresholds["candidates_zero_rate_critical"]:
            issues.append(OptimizationIssue(
                issue_type="candidates_zero_dominant",
                severity="critical",
                description=f"{patterns.candidates_zero_rate:.1%}の試行で候補数が0",
                root_cause="パラメータ設定が厳しすぎる、または抽出ロジックに問題",
                affected_metrics=["recall", "map", "workload"],
                evidence={
                    "candidates_zero_rate": patterns.candidates_zero_rate,
                    "total_trials": logs.total_trials,
                    "zero_trials": int(patterns.candidates_zero_rate * logs.total_trials)
                },
                confidence=0.95
            ))
        elif patterns.candidates_zero_rate >= self.thresholds["candidates_zero_rate_high"]:
            issues.append(OptimizationIssue(
                issue_type="candidates_zero_frequent",
                severity="high", 
                description=f"{patterns.candidates_zero_rate:.1%}の試行で候補数が0",
                root_cause="パラメータ設定がやや厳しい可能性",
                affected_metrics=["recall", "workload"],
                evidence={"candidates_zero_rate": patterns.candidates_zero_rate},
                confidence=0.85
            ))
        
        # 負スコア問題
        if patterns.negative_score_rate >= self.thresholds["negative_score_rate_critical"]:
            issues.append(OptimizationIssue(
                issue_type="negative_scores_dominant", 
                severity="critical",
                description=f"{patterns.negative_score_rate:.1%}の試行で負スコア",
                root_cause="目的関数の重みバランスが不適切（ペナルティ過大）",
                affected_metrics=["composite_score"],
                evidence={"negative_score_rate": patterns.negative_score_rate},
                confidence=0.90
            ))
        
        return issues
    
    def _detect_objective_function_issues(
        self, 
        logs: OptimizationSummary,
        config: Optional[Dict[str, Any]] = None
    ) -> List[OptimizationIssue]:
        """目的関数設計の問題を検出"""
        issues = []
        
        # 候補数0が最適解になっている問題
        best_score = logs.patterns.best_score
        if (best_score is not None and best_score <= 0 and 
            logs.objective.direction == "maximize" and
            logs.patterns.candidates_zero_rate > 0):
            
            # 候補数0の試行のスコア分布を分析
            zero_candidate_scores = [
                trial.score for trial in logs.trials 
                if trial.candidates_count == 0 and trial.score is not None
            ]
            non_zero_candidate_scores = [
                trial.score for trial in logs.trials
                if trial.candidates_count > 0 and trial.score is not None  
            ]
            
            if zero_candidate_scores and non_zero_candidate_scores:
                avg_zero_score = sum(zero_candidate_scores) / len(zero_candidate_scores)
                avg_non_zero_score = sum(non_zero_candidate_scores) / len(non_zero_candidate_scores)
                
                if avg_zero_score >= avg_non_zero_score:
                    issues.append(OptimizationIssue(
                        issue_type="objective_function_design_flaw",
                        severity="critical",
                        description="候補数0が最適解となる目的関数設計",
                        root_cause="workloadペナルティが過大、またはrecall/mAPの重みが過小",
                        affected_metrics=["composite_score", "optimization_direction"],
                        evidence={
                            "avg_zero_score": avg_zero_score,
                            "avg_non_zero_score": avg_non_zero_score,
                            "best_score": best_score,
                            "objective_weights": logs.objective.weights
                        },
                        confidence=0.95
                    ))
        
        return issues
    
    def _detect_search_space_issues(
        self,
        logs: OptimizationSummary, 
        config: Optional[Dict[str, Any]] = None
    ) -> List[OptimizationIssue]:
        """パラメータ空間の問題を検出"""
        issues = []
        
        # パラメータ範囲の分析
        if logs.trials:
            param_analysis = {}
            for param_name in logs.search_space.parameters.keys():
                param_values = [
                    trial.params.get(param_name) for trial in logs.trials
                    if param_name in trial.params and trial.params[param_name] is not None
                ]
                
                if param_values:
                    param_analysis[param_name] = {
                        "min": min(param_values),
                        "max": max(param_values), 
                        "range_utilization": (max(param_values) - min(param_values)) / 
                                            (logs.search_space.parameters[param_name].get("high", 1) - 
                                             logs.search_space.parameters[param_name].get("low", 0))
                    }
            
            # 範囲利用率が低い場合
            underutilized_params = {
                name: analysis for name, analysis in param_analysis.items()
                if analysis["range_utilization"] < 0.3
            }
            
            if underutilized_params and logs.patterns.candidates_zero_rate > 0.5:
                issues.append(OptimizationIssue(
                    issue_type="search_space_too_narrow",
                    severity="high",
                    description="パラメータ空間が狭すぎて十分な探索ができていない",
                    root_cause="パラメータ範囲設定が保守的すぎる",
                    affected_metrics=["exploration_coverage", "candidates_generation"],
                    evidence={
                        "underutilized_params": underutilized_params,
                        "param_analysis": param_analysis
                    },
                    confidence=0.80
                ))
        
        return issues
    
    def _detect_convergence_issues(self, logs: OptimizationSummary) -> List[OptimizationIssue]:
        """収束・探索問題を検出"""
        issues = []
        
        if logs.patterns.convergence_issues and len(logs.trials) >= self.thresholds["convergence_stagnation_trials"]:
            # 最近の改善状況を詳細分析
            recent_trials = logs.trials[-5:]
            earlier_trials = logs.trials[-10:-5] if len(logs.trials) >= 10 else []
            
            if recent_trials and earlier_trials:
                recent_scores = [t.score for t in recent_trials if t.score is not None]
                earlier_scores = [t.score for t in earlier_trials if t.score is not None]
                
                if recent_scores and earlier_scores:
                    recent_best = max(recent_scores)
                    earlier_best = max(earlier_scores)
                    
                    if recent_best <= earlier_best:
                        issues.append(OptimizationIssue(
                            issue_type="convergence_stagnation",
                            severity="medium",
                            description="最近の試行で改善が見られない",
                            root_cause="局所最適に陥っている、または探索空間が枯渇",
                            affected_metrics=["optimization_progress"],
                            evidence={
                                "recent_best": recent_best,
                                "earlier_best": earlier_best,
                                "stagnation_trials": len(recent_trials)
                            },
                            confidence=0.75
                        ))
        
        return issues
    
    def _generate_proposals_with_o3_pro(
        self,
        issues: List[OptimizationIssue],
        logs: OptimizationSummary,
        config: Optional[Dict[str, Any]] = None
    ) -> List[OptimizationProposal]:
        """o3-pro Response APIを使用した高度な提案生成"""
        
        # 分析コンテキストの準備
        analysis_context = {
            "optimization_summary": {
                "method": logs.method,
                "total_trials": logs.total_trials,
                "successful_trials": logs.successful_trials,
                "patterns": {
                    "candidates_zero_rate": logs.patterns.candidates_zero_rate,
                    "negative_score_rate": logs.patterns.negative_score_rate,
                    "best_score": logs.patterns.best_score,
                    "convergence_issues": logs.patterns.convergence_issues
                }
            },
            "objective_function": {
                "direction": logs.objective.direction,
                "weights": logs.objective.weights,
                "design": logs.objective.function_design
            },
            "search_space": logs.search_space.parameters,
            "detected_issues": [
                {
                    "type": issue.issue_type,
                    "severity": issue.severity,
                    "description": issue.description,
                    "root_cause": issue.root_cause,
                    "evidence": issue.evidence,
                    "confidence": issue.confidence
                }
                for issue in issues
            ]
        }
        
        # o3-pro用のプロンプト
        system_prompt = """You are an expert optimization consultant specializing in Bayesian Optimization and hyperparameter tuning for archaeological site identification systems.

Your task is to analyze optimization problems and provide specific, actionable improvement proposals.

Focus on:
1. Objective function design flaws (especially when "no candidates" becomes optimal)
2. Search space configuration issues
3. Parameter weighting problems
4. Convergence and exploration issues

Provide concrete parameter adjustments with rationale."""

        user_prompt = f"""Analyze this optimization run and provide improvement proposals:

OPTIMIZATION ANALYSIS:
{json.dumps(analysis_context, indent=2, ensure_ascii=False)}

SPECIFIC PROBLEMS DETECTED:
{chr(10).join([f"- {issue.issue_type}: {issue.description} (severity: {issue.severity})" for issue in issues])}

Provide 1-3 specific improvement proposals in JSON format:
{{
  "proposals": [
    {{
      "proposal_id": "fix_001",
      "issue_type": "objective_function_design_flaw",
      "action_type": "adjust_objective_weights",
      "params": {{
        "workload_weight": -0.05,
        "recall_weight": 0.8
      }},
      "rationale": "Reduce workload penalty severity...",
      "expected_improvement": "Should eliminate zero-candidate optimal solutions...",
      "implementation_effort": "low",
      "risk_level": "low"
    }}
  ]
}}

Focus on the most critical issues first. Ensure all proposed parameter values are realistic and well-justified."""

        try:
            # o3-pro Response API呼び出し（絶対にChat Completion APIは使わない）
            response = self.client.responses.create(
                model="o3-pro",
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # レスポンス解析 - Response API format
            response_text = None
            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                response_text = response.message.content
            elif hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'choices') and response.choices:
                response_text = response.choices[0].message.content
            else:
                logger.error("Unexpected o3-pro response format")
                return []
            
            if not response_text:
                logger.error("Empty response from o3-pro API")
                return []
            
            # JSON解析
            try:
                response_data = json.loads(response_text)
                proposals_data = response_data.get("proposals", [])
                
                proposals = []
                for i, prop_data in enumerate(proposals_data):
                    proposals.append(OptimizationProposal(
                        proposal_id=prop_data.get("proposal_id", f"o3_proposal_{i}"),
                        issue_type=prop_data.get("issue_type", "unknown"),
                        action_type=prop_data.get("action_type", "unknown"),
                        params=prop_data.get("params", {}),
                        rationale=prop_data.get("rationale", ""),
                        expected_improvement=prop_data.get("expected_improvement", ""),
                        implementation_effort=prop_data.get("implementation_effort", "medium"),
                        risk_level=prop_data.get("risk_level", "medium")
                    ))
                
                logger.info(f"o3-pro generated {len(proposals)} proposals")
                return proposals
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse o3-pro response as JSON: {e}")
                logger.error(f"Raw response: {response_text[:500]}...")
                return []
        
        except Exception as e:
            logger.error(f"o3-pro API call failed: {e}")
            raise
    
    def _generate_fallback_proposals(self, issues: List[OptimizationIssue]) -> List[OptimizationProposal]:
        """ルールベースのフォールバック提案生成"""
        proposals = []
        
        for issue in issues:
            if issue.issue_type == "objective_function_design_flaw":
                proposals.append(OptimizationProposal(
                    proposal_id="fallback_objective_fix",
                    issue_type=issue.issue_type,
                    action_type="adjust_objective_weights",
                    params={
                        "workload_weight": -0.05,  # -0.2から緩和
                        "recall_weight": 0.8       # 0.6から強化
                    },
                    rationale="Workloadペナルティを緩和してRecallを重視することで、候補数0が最適解になることを防ぐ",
                    expected_improvement="候補があるケースでより高いスコアが得られるようになる",
                    implementation_effort="low",
                    risk_level="low"
                ))
            
            elif issue.issue_type in ["candidates_zero_dominant", "search_space_too_narrow"]:
                proposals.append(OptimizationProposal(
                    proposal_id="fallback_search_space_expansion",
                    issue_type=issue.issue_type,
                    action_type="expand_search_space",
                    params={
                        "distance_km": {"low": 1.0, "high": 15.0},  # 10→15に拡大
                        "occ_pct": {"low": 1.0, "high": 20.0}       # 10→20に拡大  
                    },
                    rationale="パラメータ空間を拡大することで、より多くの候補が生成される可能性を高める",
                    expected_improvement="候補数0の試行が減少し、探索範囲が広がる",
                    implementation_effort="low", 
                    risk_level="medium"
                ))
        
        return proposals