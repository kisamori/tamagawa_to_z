"""汎用的な最適化分析エンジン（o3-pro主導設計）

特定の問題パターンにハードコーディングせず、o3-proの高度な分析能力を活用して
あらゆる最適化問題を総合的に検出・分析します。
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
from .feature_extractor import OptimizationFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class OptimizationAnalysis:
    """o3-proによる総合分析結果"""
    overall_assessment: str
    key_findings: List[str]
    identified_issues: List[Dict[str, Any]]
    optimization_health_score: float  # 0.0-1.0
    confidence: float
    analysis_method: str = "o3_pro_comprehensive"


@dataclass
class OptimizationRecommendation:
    """o3-proによる総合的な改善推奨"""
    recommendation_id: str
    category: str  # objective_function, search_space, algorithm, methodology
    priority: str  # critical, high, medium, low
    description: str
    specific_actions: List[Dict[str, Any]]
    expected_outcomes: List[str]
    implementation_complexity: str
    risk_assessment: str
    alternative_approaches: Optional[List[str]] = None


class OptimizationAnalyzerV2:
    """o3-pro主導の汎用最適化分析エンジン"""
    
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
        self.feature_extractor = OptimizationFeatureExtractor()
    
    def comprehensive_analysis(
        self,
        optimization_logs: OptimizationSummary,
        optimization_config: Optional[Dict[str, Any]] = None,
        domain_context: Optional[Dict[str, Any]] = None
    ) -> OptimizationAnalysis:
        """
        o3-proによる総合的な最適化分析
        
        特定の問題パターンを前提とせず、データから包括的に問題を発見・分析
        
        Parameters
        ----------
        optimization_logs : OptimizationSummary
            最適化ログデータ
        optimization_config : Optional[Dict[str, Any]]
            最適化設定
        domain_context : Optional[Dict[str, Any]]
            ドメイン固有のコンテキスト（考古学サイト識別等）
            
        Returns
        -------
        OptimizationAnalysis
            包括的な分析結果
        """
        
        # 包括的特徴量抽出
        features = self.feature_extractor.extract_comprehensive_features(
            optimization_logs, domain_context
        )
        
        # 分析用データ準備
        analysis_data = self._prepare_analysis_data(
            optimization_logs, optimization_config, features
        )
        
        # o3-proによる包括分析
        try:
            analysis = self._perform_o3_pro_comprehensive_analysis(analysis_data)
            logger.info("o3-pro comprehensive analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"o3-pro comprehensive analysis failed: {e}")
            # 最小限のフォールバック分析
            return self._minimal_fallback_analysis(optimization_logs)
    
    def generate_comprehensive_recommendations(
        self,
        analysis: OptimizationAnalysis,
        optimization_logs: OptimizationSummary,
        optimization_config: Optional[Dict[str, Any]] = None
    ) -> List[OptimizationRecommendation]:
        """
        分析結果に基づく総合的な改善推奨を生成
        
        Parameters
        ----------
        analysis : OptimizationAnalysis
            包括分析結果
        optimization_logs : OptimizationSummary
            最適化ログ
        optimization_config : Optional[Dict[str, Any]]
            最適化設定
            
        Returns
        -------
        List[OptimizationRecommendation]
            総合的な改善推奨リスト
        """
        
        recommendation_context = {
            "analysis_results": {
                "overall_assessment": analysis.overall_assessment,
                "key_findings": analysis.key_findings,
                "identified_issues": analysis.identified_issues,
                "health_score": analysis.optimization_health_score
            },
            "optimization_setup": {
                "method": optimization_logs.method,
                "objective": optimization_logs.objective.__dict__,
                "search_space": optimization_logs.search_space.parameters,
                "total_trials": optimization_logs.total_trials
            },
            "configuration": optimization_config or {}
        }
        
        try:
            recommendations = self._generate_o3_pro_recommendations(recommendation_context)
            logger.info(f"Generated {len(recommendations)} comprehensive recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"o3-pro recommendation generation failed: {e}")
            return []
    
    def _prepare_analysis_data(
        self,
        logs: OptimizationSummary,
        config: Optional[Dict[str, Any]],
        features: 'FeatureSet'
    ) -> Dict[str, Any]:
        """抽出された特徴量を使って分析用データを準備"""
        
        return {
            "optimization_metadata": {
                "method": logs.method,
                "status": logs.status,
                "total_trials": logs.total_trials,
                "successful_trials": logs.successful_trials,
                "study_name": logs.study_name
            },
            "objective_function": {
                "direction": logs.objective.direction,
                "design": logs.objective.function_design,
                "weights": logs.objective.weights,
                "thresholds": logs.objective.thresholds
            },
            "search_space": {
                "parameters": logs.search_space.parameters
            },
            "configuration": config or {},
            
            # 抽出された特徴量を統合
            "comprehensive_features": {
                "basic_statistics": features.basic_stats,
                "parameter_exploration": features.parameter_exploration,
                "optimization_efficiency": features.optimization_efficiency,
                "trial_quality": features.trial_quality,
                "domain_specific_features": features.domain_specific,
                "temporal_patterns": features.temporal_patterns,
                "meta_features": features.meta_features
            },
            
            # 従来の検出パターン（互換性のため）
            "detected_patterns": {
                "candidates_zero_rate": logs.patterns.candidates_zero_rate,
                "negative_score_rate": logs.patterns.negative_score_rate,
                "convergence_issues": logs.patterns.convergence_issues,
                "best_score": logs.patterns.best_score,
                "score_distribution": logs.patterns.score_distribution
            }
        }
    
    
    def _perform_o3_pro_comprehensive_analysis(self, analysis_data: Dict[str, Any]) -> OptimizationAnalysis:
        """o3-proによる包括的分析の実行"""
        
        system_prompt = """You are a world-class optimization expert and data scientist specializing in Bayesian Optimization, hyperparameter tuning, and machine learning system diagnosis.

Your task is to perform a comprehensive analysis of an optimization run WITHOUT making assumptions about specific problem types. Analyze the data holistically and identify any issues, patterns, or opportunities for improvement.

Focus on:
1. Overall optimization health and effectiveness
2. Objective function behavior and appropriateness  
3. Search space utilization and efficiency
4. Convergence patterns and exploration vs exploitation balance
5. Any anomalies or unexpected patterns in the data

Provide insights that go beyond surface-level metrics to understand the fundamental dynamics of the optimization process."""

        user_prompt = f"""Analyze this optimization run comprehensively using the rich feature set provided. Do not assume specific problem types - let the comprehensive data speak for itself.

COMPREHENSIVE OPTIMIZATION ANALYSIS:
{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

The data includes:
- Basic Statistics: Score distributions, skewness, kurtosis, percentiles
- Parameter Exploration: Diversity, range utilization, boundary exploration, correlations  
- Optimization Efficiency: Improvement rates, convergence speed, stagnation periods
- Trial Quality: Success rates, execution times, error patterns
- Domain-Specific Features: Candidate distributions, score relationships
- Temporal Patterns: Trends, volatility, improvement dynamics

Provide a comprehensive analysis in JSON format:
{{
  "overall_assessment": "Detailed assessment of optimization effectiveness based on all features...",
  "key_findings": [
    "Finding 1: Multi-dimensional pattern discovery from feature analysis...",
    "Finding 2: Cross-feature insights and correlations..."
  ],
  "identified_issues": [
    {{
      "issue_type": "descriptive_name_based_on_features",
      "severity": "critical|high|medium|low", 
      "description": "Clear description leveraging multiple feature categories",
      "evidence": "Specific feature values and cross-feature patterns supporting this finding",
      "potential_causes": ["Cause 1 from feature analysis", "Cause 2 from correlations"],
      "impact": "How this affects optimization performance across multiple dimensions"
    }}
  ],
  "optimization_health_score": 0.0-1.0,
  "confidence": 0.0-1.0
}}

Focus on:
1. Cross-feature patterns that reveal deeper optimization issues
2. How different feature categories support or contradict each other
3. Multi-dimensional assessment rather than single-metric evaluation
4. Actionable insights that go beyond surface-level observations"""

        try:
            # o3-proのResponse API呼び出し（正しい形式）
            response = self.client.responses.create(
                model="o3-pro",
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # レスポンス解析 - Response API format
            logger.debug(f"o3-pro response type: {type(response)}")
            logger.debug(f"o3-pro response attributes: {dir(response)}")
            
            response_text = None
            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                response_text = response.message.content
            elif hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'choices') and response.choices:
                response_text = response.choices[0].message.content
            else:
                # フォールバック: responseの文字列化を試す
                response_str = str(response)
                logger.debug(f"Response string: {response_str[:500]}")
                raise ValueError(f"Unexpected o3-pro response format. Available attributes: {dir(response)}")
            
            if not response_text:
                raise ValueError("Empty response from o3-pro API")
            
            logger.debug(f"Response text: {response_text[:500]}...")
            
            # JSON解析
            try:
                response_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {response_text[:1000]}")
                raise ValueError(f"Invalid JSON response from o3-pro: {e}")
            
            return OptimizationAnalysis(
                overall_assessment=response_data.get("overall_assessment", ""),
                key_findings=response_data.get("key_findings", []),
                identified_issues=response_data.get("identified_issues", []),
                optimization_health_score=response_data.get("optimization_health_score", 0.0),
                confidence=response_data.get("confidence", 0.0)
            )
            
        except Exception as e:
            logger.error(f"o3-pro comprehensive analysis failed: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _generate_o3_pro_recommendations(self, context: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """o3-proによる総合的な改善推奨生成"""
        
        system_prompt = """You are an optimization consultant providing strategic recommendations for improving hyperparameter optimization and machine learning system performance.

Your recommendations should be:
1. Specific and actionable
2. Prioritized by impact and feasibility  
3. Comprehensive across different aspects (objective function, search space, algorithm)
4. Risk-aware with alternative approaches when appropriate

Focus on both immediate fixes and longer-term strategic improvements."""

        user_prompt = f"""Based on this optimization analysis, provide comprehensive improvement recommendations:

ANALYSIS CONTEXT:
{json.dumps(context, indent=2, ensure_ascii=False)}

Provide recommendations in JSON format:
{{
  "recommendations": [
    {{
      "recommendation_id": "rec_001",
      "category": "objective_function|search_space|algorithm|methodology",
      "priority": "critical|high|medium|low",
      "description": "Clear description of what to do",
      "specific_actions": [
        {{
          "action": "action_type",
          "parameters": {{"param1": "value1"}},
          "rationale": "Why this action is needed"
        }}
      ],
      "expected_outcomes": ["Outcome 1", "Outcome 2"],
      "implementation_complexity": "low|medium|high",
      "risk_assessment": "Assessment of potential risks",
      "alternative_approaches": ["Alternative 1", "Alternative 2"]
    }}
  ]
}}

Prioritize recommendations by potential impact. Address both immediate issues and strategic improvements."""

        try:
            # o3-proのResponse API呼び出し（正しい形式）
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
                raise ValueError(f"Unexpected o3-pro response format. Available attributes: {dir(response)}")
            
            if not response_text:
                raise ValueError("Empty response from o3-pro API")
            
            # JSON解析
            try:
                response_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {response_text[:1000]}")
                raise ValueError(f"Invalid JSON response from o3-pro: {e}")
            
            recommendations_data = response_data.get("recommendations", [])
            
            recommendations = []
            for rec_data in recommendations_data:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=rec_data.get("recommendation_id", f"rec_{len(recommendations)}"),
                    category=rec_data.get("category", "methodology"),
                    priority=rec_data.get("priority", "medium"),
                    description=rec_data.get("description", ""),
                    specific_actions=rec_data.get("specific_actions", []),
                    expected_outcomes=rec_data.get("expected_outcomes", []),
                    implementation_complexity=rec_data.get("implementation_complexity", "medium"),
                    risk_assessment=rec_data.get("risk_assessment", ""),
                    alternative_approaches=rec_data.get("alternative_approaches")
                ))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"o3-pro recommendation generation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _minimal_fallback_analysis(self, logs: OptimizationSummary) -> OptimizationAnalysis:
        """最小限のフォールバック分析"""
        
        return OptimizationAnalysis(
            overall_assessment="最適化分析でエラーが発生したため、基本情報のみ提供",
            key_findings=[
                f"総試行数: {logs.total_trials}",
                f"成功試行数: {logs.successful_trials}",
                f"最良スコア: {logs.patterns.best_score}"
            ],
            identified_issues=[],
            optimization_health_score=0.5,
            confidence=0.3,
            analysis_method="fallback_basic"
        )