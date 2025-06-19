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
        prompt = f"""Based on the following performance analysis, suggest specific parameter adjustments:

{context}

Current Performance:
- Workload: {workload} candidates
- Recall@100: {recall:.3f}
- Root diversity: {baseline_metrics.get('root_diversity', 2.0):.2f}

Failure patterns identified: {len(self.failure_clusters)} clusters

Please suggest 1-2 specific parameter changes that would improve recall while managing workload.
Focus on distance thresholds, frequency thresholds, or weight adjustments.

Respond in this JSON format:
{{
    "title": "Brief descriptive title",
    "parameters": {{
        "parameter_name": new_value
    }},
    "rationale": "Why this change will help",
    "expected_recall_change": "+X%",
    "expected_workload_change": "+/-N",
    "risk": "Main risk or concern",
    "effort": "★☆☆ (easy) / ★★☆ (medium) / ★★★ (hard)"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in geospatial parameter optimization for archaeological site detection."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content
            # Extract JSON from response (handle markdown code blocks)
            if '```json' in response_text:
                json_part = response_text.split('```json')[1].split('```')[0]
            elif '{' in response_text:
                json_part = response_text[response_text.find('{'):response_text.rfind('}')+1]
            else:
                json_part = response_text
            
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
        
        prompt = f"""Analyze the following code snippets from a toponym harmonization system and suggest improvements:

{code_context}

The system extracts water-related toponyms and harmonizes them for archaeological site detection.

Common issues:
- LLM hallucination in toponym classification
- Inconsistent root detection
- Poor handling of multilingual place names

Suggest ONE specific improvement to prompts, code logic, or processing steps.

Respond in this JSON format:
{{
    "title": "Brief improvement description",
    "target": "harmonizer.llm_layer.harmonize",
    "change_type": "modify_prompt",
    "improvement": "Specific change description",
    "rationale": "Why this will improve results",
    "expected_map_change": "+X%",
    "expected_cost_change": "+X% API costs",
    "risk": "Main implementation risk",
    "effort": "★☆☆ / ★★☆ / ★★★"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in NLP pipeline optimization and prompt engineering."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
            
            response_text = response.choices[0].message.content
            if '```json' in response_text:
                json_part = response_text.split('```json')[1].split('```')[0]
            elif '{' in response_text:
                json_part = response_text[response_text.find('{'):response_text.rfind('}')+1]
            else:
                json_part = response_text
            
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