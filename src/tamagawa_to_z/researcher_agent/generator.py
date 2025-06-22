"""
ProposalGenerator: Generate improvement proposals using LLM analysis.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import date

from openai import OpenAI

from .loader import LoadedData
from .rca import FailureCluster


@dataclass
class Proposal:
    """Container for a single improvement proposal."""
    id: str
    title: str
    changes: List[Dict[str, Any]]
    expected_effect: Dict[str, str]
    rationale: str
    risk: str
    human_effort: str
    priority: str


class ProposalGenerator:
    """Generate improvement proposals using LLM analysis and heuristics."""
    
    def __init__(self, client: OpenAI, data: LoadedData, failure_clusters: List[FailureCluster], config: Dict[str, Any]):
        """
        Initialize generator with LLM client and analysis data.
        
        Parameters
        ----------
        client : OpenAI
            OpenAI API client
        data : LoadedData
            Loaded artifacts and data
        failure_clusters : List[FailureCluster]
            Identified failure patterns
        config : Dict[str, Any]
            Configuration parameters
        """
        self.client = client
        self.data = data
        self.failure_clusters = failure_clusters
        self.config = config
    
    def generate(self, n: int = 3) -> List[Proposal]:
        """
        Generate n improvement proposals.
        
        Parameters
        ----------
        n : int
            Number of proposals to generate
            
        Returns
        -------
        List[Proposal]
            Generated proposals
        """
        proposals = []
        
        # Generate proposals using different strategies
        strategies = [
            self._generate_parameter_proposal,
            self._generate_prompt_proposal,
            self._generate_mask_proposal
        ]
        
        for i, strategy in enumerate(strategies[:n]):
            try:
                proposal = strategy(f"{chr(65+i)}")  # A, B, C
                if proposal:
                    proposals.append(proposal)
            except Exception as e:
                print(f"Warning: Failed to generate proposal with strategy {i}: {e}")
                # Generate fallback proposal
                fallback = self._generate_fallback_proposal(f"{chr(65+i)}")
                if fallback:
                    proposals.append(fallback)
        
        # If we have fewer than n proposals, fill with heuristic proposals
        while len(proposals) < n:
            fallback = self._generate_fallback_proposal(f"{chr(65+len(proposals))}")
            if fallback:
                proposals.append(fallback)
            else:
                break
        
        return proposals[:n]
    
    def _generate_parameter_proposal(self, proposal_id: str) -> Optional[Proposal]:
        """Generate a parameter-based improvement proposal."""
        # Analyze current metrics and suggest parameter changes
        baseline_metrics = getattr(self.data, 'baseline_metrics', {})
        workload = baseline_metrics.get('workload', 500)
        recall = baseline_metrics.get('recall@100', 0.3)
        
        # Build parameter adjustment context
        context = self._build_parameter_context()
        
        # Call LLM for parameter suggestions
        prompt = f"""以下の性能分析に基づいて、具体的なパラメータ調整を提案してください：

{context}

現在の性能:
- 処理負荷: {workload} 候補
- Recall@100: {recall:.3f}
- 語根多様性: {baseline_metrics.get('root_diversity', 2.0):.2f}

特定された失敗パターン: {len(self.failure_clusters)} クラスター

処理負荷を管理しながらrecallを改善する具体的なパラメータ変更を1-2個提案してください。
距離閾値、頻度閾値、または重み調整に焦点を当ててください。

以下のJSON形式で回答してください：
{{
    "title": "簡潔な説明タイトル",
    "parameters": {{
        "parameter_name": new_value
    }},
    "rationale": "この変更が役立つ理由",
    "expected_recall_change": "+X%",
    "expected_workload_change": "+/-N",
    "risk": "主なリスクや懸念",
    "effort": "★☆☆ (簡単) / ★★☆ (中程度) / ★★★ (困難)"
}}"""
        
        try:
            # Responses APIでの実行
            full_input = f"あなたは考古学遺跡検出のための地理空間パラメータ最適化の専門家です。\n\n{prompt}"
            response = self.client.responses.create(
                model="o3-pro",
                input=full_input
            )
            
            # レスポンス内容を取得
            response_text = ""
            if hasattr(response, 'output_text'):
                response_text = response.output_text
            elif hasattr(response, 'output') and hasattr(response.output, 'text'):
                response_text = response.output.text
            elif hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'choices') and response.choices:
                response_text = response.choices[0].message.content
            else:
                response_text = str(response)
            
            # Extract JSON from response (handle markdown code blocks)
            if not response_text or not hasattr(response_text, 'strip') or response_text.strip() == "":
                return []
                
            if '```json' in response_text:
                json_part = response_text.split('```json')[1].split('```')[0]
            elif '{' in response_text:
                json_part = response_text[response_text.find('{'):response_text.rfind('}')+1]
            else:
                json_part = response_text
            
            if not json_part or not hasattr(json_part, 'strip') or not json_part.strip():
                return []
                
            data = json.loads(json_part)
            
            # Build changes list
            changes = []
            for param_name, param_value in data.get('parameters', {}).items():
                changes.append({
                    'action': 'set_param',
                    'params': {param_name: param_value}
                })
            
            return Proposal(
                id=proposal_id,
                title=data.get('title', 'Parameter optimization'),
                changes=changes,
                expected_effect={
                    'recall@100': data.get('expected_recall_change', '+5%'),
                    'workload': data.get('expected_workload_change', '+100')
                },
                rationale=data.get('rationale', 'Parameter adjustment based on performance analysis'),
                risk=data.get('risk', 'Potential change in candidate distribution'),
                human_effort=data.get('effort', '★☆☆ (1h)'),
                priority='medium'
            )
            
        except Exception as e:
            print(f"Warning: LLM parameter proposal failed: {e}")
            return None
    
    def _generate_prompt_proposal(self, proposal_id: str) -> Optional[Proposal]:
        """Generate a prompt/code improvement proposal."""
        # Analyze code snippets for improvement opportunities
        code_context = self._build_code_context()
        
        prompt = f"""地名調和システムから以下のコードスニペットを分析し、改善提案を行ってください：

{code_context}

このシステムは水関連の地名を抽出し、考古学遺跡検出のために調和を図ります。

一般的な問題:
- 地名分類におけるLLMの幻覚
- 語根検出の不一致
- 多言語地名の処理不備

プロンプト、コードロジック、または処理ステップの具体的な改善を1つ提案してください。

以下のJSON形式で回答してください：
{{
    "title": "改善の簡潔な説明",
    "target": "harmonizer.llm_layer.harmonize",
    "change_type": "modify_prompt",
    "improvement": "具体的な変更の説明",
    "rationale": "この改善が結果を向上させる理由",
    "expected_map_change": "+X%",
    "expected_cost_change": "+X% APIコスト",
    "risk": "主な実装リスク",
    "effort": "★☆☆ / ★★☆ / ★★★"
}}"""
        
        try:
            # Responses APIでの実行
            full_input = f"あなたはNLPパイプライン最適化とプロンプトエンジニアリングの専門家です。\n\n{prompt}"
            response = self.client.responses.create(
                model="o3",
                input=full_input
            )
            
            # レスポンス内容を取得
            response_text = ""
            if hasattr(response, 'output_text'):
                response_text = response.output_text
            elif hasattr(response, 'output') and hasattr(response.output, 'text'):
                response_text = response.output.text
            elif hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'choices') and response.choices:
                response_text = response.choices[0].message.content
            else:
                response_text = str(response)
            
            # Extract JSON from response (handle markdown code blocks)
            if not response_text or not hasattr(response_text, 'strip') or response_text.strip() == "":
                return []
                
            if '```json' in response_text:
                json_part = response_text.split('```json')[1].split('```')[0]
            elif '{' in response_text:
                json_part = response_text[response_text.find('{'):response_text.rfind('}')+1]
            else:
                json_part = response_text
            
            if not json_part or not hasattr(json_part, 'strip') or not json_part.strip():
                return []
                
            data = json.loads(json_part)
            
            # Build changes
            changes = [{
                'action': data.get('change_type', 'modify_prompt'),
                'target_agent': data.get('target', 'harmonizer.llm_layer'),
                'description': data.get('improvement', 'Code improvement')
            }]
            
            return Proposal(
                id=proposal_id,
                title=data.get('title', 'Prompt/code improvement'),
                changes=changes,
                expected_effect={
                    'map': data.get('expected_map_change', '+2%'),
                    'api_cost': data.get('expected_cost_change', '+5%')
                },
                rationale=data.get('rationale', 'Improve prompt quality and reduce hallucination'),
                risk=data.get('risk', 'Implementation complexity'),
                human_effort=data.get('effort', '★★☆ (3h)'),
                priority='medium'
            )
            
        except Exception as e:
            print(f"Warning: LLM prompt proposal failed: {e}")
            return None
    
    def _generate_mask_proposal(self, proposal_id: str) -> Optional[Proposal]:
        """Generate a mask/filtering improvement proposal."""
        # Analyze failure clusters for mask opportunities
        spatial_issues = [c for c in self.failure_clusters if 'spatial' in c.cluster_id]
        
        if spatial_issues:
            # Suggest masks based on spatial clustering issues
            mask_types = ['cloud_shadow', 'urban_areas', 'water_bodies']
            selected_mask = mask_types[len(proposal_id) % len(mask_types)]  # Simple selection
            
            changes = [{
                'action': 'add_exclude_mask',
                'params': {
                    'mask_layers': [{
                        'type': selected_mask,
                        'src': 'satellite_data',
                        'threshold': 0.6
                    }]
                }
            }]
            
            workload_reduction = 100 + len(proposal_id) * 50  # Simple estimation
            
            return Proposal(
                id=proposal_id,
                title=f"Add {selected_mask.replace('_', ' ')} exclusion mask",
                changes=changes,
                expected_effect={
                    'workload': f"-{workload_reduction}",
                    'precision': '+3%'
                },
                rationale=f"Exclude false positives from {selected_mask.replace('_', ' ')} to improve precision",
                risk='Potential reduction in recall if mask is too aggressive',
                human_effort='★★☆ (2h)',
                priority='low'
            )
        
        # Fallback: weight adjustment proposal
        return self._generate_weight_proposal(proposal_id)
    
    def _generate_weight_proposal(self, proposal_id: str) -> Optional[Proposal]:
        """Generate a root weight adjustment proposal."""
        # Common Amazonian water-related roots
        root_weights = {
            'igarape': 1.5,  # Creek
            'lagoa': 1.3,    # Lake
            'rio': 1.2,      # River
            'parana': 1.4,   # River channel
            'igapo': 1.3     # Flooded forest
        }
        
        changes = [{
            'action': 'add_root_weight',
            'params': {
                'root_weight_table': root_weights
            }
        }]
        
        return Proposal(
            id=proposal_id,
            title="Adjust Amazonian water root weights",
            changes=changes,
            expected_effect={
                'root_diversity': '+0.3',
                'recall@100': '+3%'
            },
            rationale="Emphasize important Amazonian water-related toponyms to improve detection of relevant sites",
            risk='May bias results toward Portuguese/Spanish toponyms',
            human_effort='★☆☆ (30min)',
            priority='low'
        )
    
    def _generate_fallback_proposal(self, proposal_id: str) -> Optional[Proposal]:
        """Generate a simple fallback proposal."""
        fallback_proposals = [
            {
                'title': 'Increase distance threshold to 4km',
                'changes': [{'action': 'set_param', 'params': {'dist_threshold_km': 4.0}}],
                'effect': {'recall@100': '+10%', 'workload': '+200'},
                'rationale': 'Capture more distant candidates that may indicate ancient channels',
                'risk': 'Increased false positives',
                'effort': '★☆☆ (5min)'
            },
            {
                'title': 'Lower water frequency threshold',
                'changes': [{'action': 'set_param', 'params': {'water_freq_threshold': 0.05}}],
                'effect': {'recall@100': '+8%', 'workload': '+150'},
                'rationale': 'Include areas with lower current water presence but historical significance',
                'risk': 'More noise in results',
                'effort': '★☆☆ (5min)'
            },
            {
                'title': 'Add evaluation dataset',
                'changes': [{'action': 'add_eval_set', 'params': {'file': 'data/eval/amazon_test.csv'}}],
                'effect': {'validation': '+reliable', 'confidence': '+high'},
                'rationale': 'Improve evaluation reliability with region-specific test set',
                'risk': 'Requires manual curation',
                'effort': '★★★ (1 day)'
            }
        ]
        
        idx = ord(proposal_id) % len(fallback_proposals)
        template = fallback_proposals[idx]
        
        return Proposal(
            id=proposal_id,
            title=template['title'],
            changes=template['changes'],
            expected_effect=template['effect'],
            rationale=template['rationale'],
            risk=template['risk'],
            human_effort=template['effort'],
            priority='low'
        )
    
    def _build_parameter_context(self) -> str:
        """Build context about current parameters for LLM analysis."""
        context = "Current system parameters:\n"
        
        # Extract parameters from param_yaml
        params = self.data.param_yaml
        
        if params:
            for key, value in params.items():
                if isinstance(value, (int, float, str)) and len(str(value)) < 100:
                    context += f"- {key}: {value}\n"
        
        # Add performance context
        if hasattr(self.data, 'baseline_metrics'):
            context += f"\nCurrent performance:\n"
            for metric, value in self.data.baseline_metrics.items():
                context += f"- {metric}: {value}\n"
        
        # Add failure context
        if self.failure_clusters:
            context += f"\nIdentified issues:\n"
            for cluster in self.failure_clusters[:3]:
                context += f"- {cluster.description}\n"
        
        return context
    
    def _build_code_context(self) -> str:
        """Build context about current code for LLM analysis."""
        context = "Key code snippets:\n\n"
        
        # Extract relevant code snippets
        snippets = self.data.code_snippets
        
        for module_name, module_snippets in snippets.items():
            context += f"## {module_name}\n"
            
            for file_path, functions in module_snippets.items():
                if len(functions) > 0:
                    context += f"### {file_path}\n"
                    # Show first function only to keep context manageable
                    first_func = next(iter(functions.values()))
                    # Truncate if too long
                    if len(first_func) > 500:
                        first_func = first_func[:500] + "\n# ... (truncated)"
                    context += f"```python\n{first_func}\n```\n\n"
            
            # Limit total context length
            if len(context) > 2000:
                context += "... (additional code snippets truncated)\n"
                break
        
        return context