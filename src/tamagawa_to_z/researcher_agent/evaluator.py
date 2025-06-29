"""
Evaluator: Simulate IA proposals and estimate their effects on metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from .loader import LoadedData


@dataclass
class IaEffect:
    """Container for estimated effects of IA proposal."""
    baseline_metrics: Dict[str, float]
    estimated_metrics: Dict[str, float]
    confidence: float
    workload_change: int
    reasoning: str


class Evaluator:
    """Evaluate IA proposals by simulating their effects."""
    
    def __init__(self, data: LoadedData):
        """
        Initialize evaluator with loaded data.
        
        Parameters
        ----------
        data : LoadedData
            Loaded artifacts and data
        """
        self.data = data
        self.baseline_metrics = self._calculate_baseline_metrics()
    
    def quick_simulate(self) -> IaEffect:
        """
        Quickly simulate the IA proposal to estimate its effects.
        
        Returns
        -------
        IaEffect
            Estimated effects of the IA proposal
        """
        if not self.data.ia_plan:
            return IaEffect(
                baseline_metrics=self.baseline_metrics,
                estimated_metrics=self.baseline_metrics.copy(),
                confidence=0.0,
                workload_change=0,
                reasoning="No IA proposal found to evaluate"
            )
        
        # Extract action from IA plan
        action = self.data.ia_plan.get('action', '')
        params = self.data.ia_plan.get('params', {})
        
        if action == 'set_param':
            return self._simulate_param_change(params)
        elif action == 'add_exclude_mask':
            return self._simulate_mask_addition(params)
        elif action == 'add_root_weight':
            return self._simulate_root_weight(params)
        else:
            return IaEffect(
                baseline_metrics=self.baseline_metrics,
                estimated_metrics=self.baseline_metrics.copy(),
                confidence=0.0,
                workload_change=0,
                reasoning=f"Unknown action type: {action}"
            )
    
    def _calculate_baseline_metrics(self) -> Dict[str, float]:
        """Calculate baseline metrics from current data."""
        metrics = {}
        
        if not self.data.candidates.empty and not self.data.known_sites.empty:
            # Simple approximation of key metrics
            n_candidates = len(self.data.candidates)
            n_known = len(self.data.known_sites)
            
            # Workload (number of candidates to investigate)
            metrics['workload'] = float(n_candidates)
            
            # Rough estimates based on data characteristics
            if n_candidates > 0:
                # Estimated recall based on candidate density
                candidate_density = n_candidates / max(1, n_known)
                metrics['recall@100'] = min(0.8, candidate_density * 0.3)
                
                # Estimated mAP based on workload
                if n_candidates < 300:
                    metrics['map'] = 0.4
                elif n_candidates < 800:
                    metrics['map'] = 0.3
                else:
                    metrics['map'] = 0.2
                
                # Root diversity (placeholder)
                unique_roots = self._estimate_root_diversity()
                metrics['root_diversity'] = unique_roots / max(1, n_candidates) * 10
            else:
                metrics['recall@100'] = 0.0
                metrics['map'] = 0.0
                metrics['root_diversity'] = 0.0
        
        return metrics
    
    def _estimate_root_diversity(self) -> float:
        """Estimate root diversity from candidates."""
        if self.data.candidates.empty or 'name' not in self.data.candidates.columns:
            return 0.0
        
        # Simple heuristic: count unique first words in names
        first_words = set()
        for name in self.data.candidates['name'].dropna():
            words = str(name).lower().split()
            if words:
                first_words.add(words[0])
        
        return len(first_words)
    
    def _simulate_param_change(self, params: Dict[str, Any]) -> IaEffect:
        """Simulate parameter changes."""
        estimated = self.baseline_metrics.copy()
        reasoning_parts = []
        workload_change = 0
        confidence = 0.7
        
        for param_name, new_value in params.items():
            if 'threshold' in param_name.lower() or 'dist' in param_name.lower():
                # Distance threshold changes
                if isinstance(new_value, (int, float)):
                    current_workload = self.baseline_metrics.get('workload', 0)
                    
                    # Estimate workload change based on threshold
                    if new_value > 3.0:  # Increasing distance threshold
                        workload_multiplier = min(2.0, new_value / 3.0)
                        workload_change = int(current_workload * (workload_multiplier - 1))
                        estimated['workload'] = current_workload * workload_multiplier
                        
                        # Higher threshold -> more candidates -> potentially better recall
                        estimated['recall@100'] = min(0.8, estimated['recall@100'] * 1.2)
                        # But lower precision
                        estimated['map'] = max(0.1, estimated['map'] * 0.9)
                        
                        reasoning_parts.append(
                            f"Increasing {param_name} to {new_value} km: "
                            f"Expected +{workload_change} candidates, +20% recall, -10% mAP"
                        )
                    else:
                        # Decreasing distance threshold
                        workload_multiplier = max(0.5, new_value / 3.0)
                        workload_change = int(current_workload * (workload_multiplier - 1))
                        estimated['workload'] = current_workload * workload_multiplier
                        
                        # Lower threshold -> fewer candidates -> lower recall but higher precision
                        estimated['recall@100'] = max(0.1, estimated['recall@100'] * 0.8)
                        estimated['map'] = min(0.6, estimated['map'] * 1.1)
                        
                        reasoning_parts.append(
                            f"Decreasing {param_name} to {new_value} km: "
                            f"Expected {workload_change} candidates, -20% recall, +10% mAP"
                        )
            
            elif 'freq' in param_name.lower() or 'occ' in param_name.lower():
                # Water frequency threshold changes
                if isinstance(new_value, (int, float)):
                    # Lower frequency threshold = more candidates
                    if new_value < 0.1:
                        workload_change = int(self.baseline_metrics.get('workload', 0) * 0.3)
                        estimated['workload'] = estimated.get('workload', 0) + workload_change
                        reasoning_parts.append(
                            f"Lowering {param_name} to {new_value}: "
                            f"Expected +{workload_change} candidates from low-frequency areas"
                        )
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Parameter changes simulated"
        
        return IaEffect(
            baseline_metrics=self.baseline_metrics,
            estimated_metrics=estimated,
            confidence=confidence,
            workload_change=workload_change,
            reasoning=reasoning
        )
    
    def _simulate_mask_addition(self, params: Dict[str, Any]) -> IaEffect:
        """Simulate adding exclusion masks."""
        estimated = self.baseline_metrics.copy()
        
        mask_types = params.get('mask_layers', [])
        excluded_fraction = 0.0
        
        for mask in mask_types:
            mask_type = mask.get('type', '')
            if 'cloud' in mask_type.lower():
                excluded_fraction += 0.1  # Estimate 10% cloud coverage
            elif 'urban' in mask_type.lower():
                excluded_fraction += 0.05  # Estimate 5% urban areas
            elif 'water' in mask_type.lower():
                excluded_fraction += 0.15  # Estimate 15% water bodies
        
        # Reduce workload but also reduce recall
        workload_change = -int(self.baseline_metrics.get('workload', 0) * excluded_fraction)
        estimated['workload'] = max(0, estimated['workload'] + workload_change)
        estimated['recall@100'] = max(0.1, estimated['recall@100'] * (1 - excluded_fraction * 0.5))
        # Slightly improve precision by removing false positives
        estimated['map'] = min(0.6, estimated['map'] * 1.05)
        
        reasoning = (
            f"Adding exclusion masks ({len(mask_types)} types): "
            f"Expected {workload_change} candidates, "
            f"-{excluded_fraction*50:.0f}% recall, +5% mAP"
        )
        
        return IaEffect(
            baseline_metrics=self.baseline_metrics,
            estimated_metrics=estimated,
            confidence=0.6,
            workload_change=workload_change,
            reasoning=reasoning
        )
    
    def _simulate_root_weight(self, params: Dict[str, Any]) -> IaEffect:
        """Simulate root weight adjustments."""
        estimated = self.baseline_metrics.copy()
        
        root_weights = params.get('root_weight_table', {})
        
        # Estimate effect on root diversity
        high_weight_roots = [k for k, v in root_weights.items() if v > 1.0]
        low_weight_roots = [k for k, v in root_weights.items() if v < 1.0]
        
        diversity_change = len(high_weight_roots) * 0.1 - len(low_weight_roots) * 0.05
        estimated['root_diversity'] = max(0, estimated['root_diversity'] + diversity_change)
        
        # Small effect on recall (emphasizing important roots)
        estimated['recall@100'] = min(0.8, estimated['recall@100'] * 1.05)
        
        reasoning = (
            f"Adjusting weights for {len(root_weights)} roots: "
            f"Expected +{diversity_change:.1f} diversity, +5% recall"
        )
        
        return IaEffect(
            baseline_metrics=self.baseline_metrics,
            estimated_metrics=estimated,
            confidence=0.5,
            workload_change=0,
            reasoning=reasoning
        )