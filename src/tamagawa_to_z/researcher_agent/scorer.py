"""
ProposalScorer: Score and rank proposals based on improvement, cost, and diversity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any

from .loader import LoadedData
from .generator import Proposal


@dataclass
class RankedProposal:
    """Container for a scored and ranked proposal."""
    proposal: Proposal
    improvement_score: float
    diversity_score: float
    cost_score: float
    total_score: float
    rank: int


class ProposalScorer:
    """Score and rank improvement proposals."""
    
    def __init__(self, data: LoadedData, proposals: List[Proposal], weight_cfg: Dict[str, float]):
        """
        Initialize scorer with proposals and weights.
        
        Parameters
        ----------
        data : LoadedData
            Loaded artifacts and data
        proposals : List[Proposal]
            Proposals to score
        weight_cfg : Dict[str, float]
            Scoring weights configuration
        """
        self.data = data
        self.proposals = proposals
        self.weights = {
            'improvement': weight_cfg.get('improvement', 0.6),
            'diversity': weight_cfg.get('diversity', 0.2), 
            'cost': weight_cfg.get('cost', 0.2)
        }
    
    def rank(self) -> List[RankedProposal]:
        """
        Score and rank all proposals.
        
        Returns
        -------
        List[RankedProposal]
            Ranked proposals in descending order of total score
        """
        if not self.proposals:
            return []
        
        # Score each proposal
        scored_proposals = []
        
        for proposal in self.proposals:
            improvement_score = self._score_improvement(proposal)
            diversity_score = self._score_diversity(proposal, self.proposals)
            cost_score = self._score_cost(proposal)
            
            # Calculate total weighted score
            total_score = (
                improvement_score * self.weights['improvement'] +
                diversity_score * self.weights['diversity'] +
                cost_score * self.weights['cost']
            )
            
            scored_proposals.append(RankedProposal(
                proposal=proposal,
                improvement_score=improvement_score,
                diversity_score=diversity_score,
                cost_score=cost_score,
                total_score=total_score,
                rank=0  # Will be set after sorting
            ))
        
        # Sort by total score (descending)
        scored_proposals.sort(key=lambda x: x.total_score, reverse=True)
        
        # Assign ranks
        for i, scored_proposal in enumerate(scored_proposals):
            scored_proposal.rank = i + 1
        
        return scored_proposals
    
    def _score_improvement(self, proposal: Proposal) -> float:
        """
        Score proposal based on expected improvement.
        
        Parameters
        ----------
        proposal : Proposal
            Proposal to score
            
        Returns
        -------
        float
            Improvement score (0.0 to 1.0)
        """
        score = 0.0
        
        expected_effects = proposal.expected_effect
        
        # Score based on expected metric improvements
        for metric, effect_str in expected_effects.items():
            if isinstance(effect_str, str):
                numeric_improvement = self._extract_numeric_change(effect_str)
                
                if metric in ['recall@100', 'map']:
                    # Positive changes in recall/mAP are good
                    if numeric_improvement > 0:
                        score += min(0.4, numeric_improvement / 20.0)  # Max 0.4 for 20% improvement
                elif metric == 'workload':
                    # Lower workload is good (if not too extreme)
                    if numeric_improvement < 0:
                        score += min(0.2, abs(numeric_improvement) / 500.0)  # Max 0.2 for -500 workload
                    elif numeric_improvement > 0:
                        # Higher workload is bad, but small increases might be acceptable
                        score -= min(0.1, numeric_improvement / 1000.0)
                elif metric in ['root_diversity', 'precision']:
                    # Positive changes are good
                    if numeric_improvement > 0:
                        score += min(0.2, numeric_improvement / 10.0)
        
        # Bonus for addressing specific failure clusters
        if hasattr(self.data, 'failure_clusters') and len(getattr(self.data, 'failure_clusters', [])) > 0:
            # Check if proposal addresses identified issues
            proposal_text = f"{proposal.title} {proposal.rationale}".lower()
            
            for cluster in getattr(self.data, 'failure_clusters', []):
                cluster_type = getattr(cluster, 'cluster_id', '').lower()
                if any(keyword in proposal_text for keyword in [cluster_type, 'distance', 'root', 'spatial']):
                    score += 0.1  # Bonus for addressing specific issues
        
        # Priority bonus
        priority_bonus = {
            'high': 0.2,
            'medium': 0.1,
            'low': 0.0
        }
        score += priority_bonus.get(proposal.priority, 0.0)
        
        return max(0.0, min(1.0, score))
    
    def _score_diversity(self, proposal: Proposal, all_proposals: List[Proposal]) -> float:
        """
        Score proposal based on diversity from other proposals.
        
        Parameters
        ----------
        proposal : Proposal
            Proposal to score
        all_proposals : List[Proposal]
            All proposals for comparison
            
        Returns
        -------
        float
            Diversity score (0.0 to 1.0)
        """
        if len(all_proposals) <= 1:
            return 1.0
        
        # Analyze action types in this proposal
        proposal_actions = set()
        for change in proposal.changes:
            action_type = change.get('action', 'unknown')
            proposal_actions.add(action_type)
        
        # Compare with other proposals
        similarity_scores = []
        
        for other_proposal in all_proposals:
            if other_proposal.id == proposal.id:
                continue
            
            other_actions = set()
            for change in other_proposal.changes:
                action_type = change.get('action', 'unknown')
                other_actions.add(action_type)
            
            # Calculate Jaccard similarity
            intersection = len(proposal_actions & other_actions)
            union = len(proposal_actions | other_actions)
            
            if union > 0:
                similarity = intersection / union
                similarity_scores.append(similarity)
        
        # Diversity is inverse of average similarity
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            diversity_score = 1.0 - avg_similarity
        else:
            diversity_score = 1.0
        
        # Bonus for unique action types
        unique_actions = ['modify_prompt', 'add_eval_set', 'add_exclude_mask']
        proposal_action_types = [change.get('action') for change in proposal.changes]
        
        if any(action in unique_actions for action in proposal_action_types):
            diversity_score += 0.2
        
        return max(0.0, min(1.0, diversity_score))
    
    def _score_cost(self, proposal: Proposal) -> float:
        """
        Score proposal based on implementation cost (inverse - lower cost = higher score).
        
        Parameters
        ----------
        proposal : Proposal
            Proposal to score
            
        Returns
        -------
        float
            Cost score (0.0 to 1.0, higher = lower cost)
        """
        # Parse human effort
        effort_str = proposal.human_effort.lower()
        
        if '★★★' in effort_str:
            effort_score = 0.2  # High effort = low score
        elif '★★☆' in effort_str:
            effort_score = 0.6  # Medium effort = medium score
        elif '★☆☆' in effort_str:
            effort_score = 1.0  # Low effort = high score
        else:
            # Try to parse time estimates
            if any(term in effort_str for term in ['day', 'week']):
                effort_score = 0.1
            elif any(term in effort_str for term in ['hour', 'h']):
                effort_score = 0.7
            elif any(term in effort_str for term in ['min', 'minute']):
                effort_score = 1.0
            else:
                effort_score = 0.5  # Default
        
        # Analyze risk level
        risk_str = proposal.risk.lower()
        risk_penalty = 0.0
        
        if any(term in risk_str for term in ['high', 'major', 'significant']):
            risk_penalty = 0.3
        elif any(term in risk_str for term in ['medium', 'moderate']):
            risk_penalty = 0.1
        elif any(term in risk_str for term in ['low', 'minor', 'minimal']):
            risk_penalty = 0.0
        else:
            # Analyze specific risk types
            if any(term in risk_str for term in ['cost', 'expensive', 'api']):
                risk_penalty = 0.2
            elif any(term in risk_str for term in ['complexity', 'difficult', 'hard']):
                risk_penalty = 0.15
        
        # Check for API cost implications
        api_cost_penalty = 0.0
        expected_effects = proposal.expected_effect
        
        for metric, effect_str in expected_effects.items():
            if 'cost' in metric.lower() or 'api' in metric.lower():
                if isinstance(effect_str, str) and '+' in effect_str:
                    # Positive API cost change is bad
                    numeric_change = self._extract_numeric_change(effect_str)
                    if numeric_change > 0:
                        api_cost_penalty = min(0.2, numeric_change / 50.0)  # Max penalty 0.2 for +50% cost
        
        # Calculate final cost score
        cost_score = effort_score - risk_penalty - api_cost_penalty
        
        return max(0.0, min(1.0, cost_score))
    
    def _extract_numeric_change(self, effect_str: str) -> float:
        """
        Extract numeric change from effect string like '+5%', '-200', etc.
        
        Parameters
        ----------
        effect_str : str
            Effect string to parse
            
        Returns
        -------
        float
            Numeric change value
        """
        if not isinstance(effect_str, str):
            return 0.0
        
        # Remove common prefixes/suffixes
        clean_str = effect_str.strip().replace('+', '').replace('%', '')
        
        # Try to extract number
        numbers = re.findall(r'-?\d+\.?\d*', clean_str)
        
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
        
        return 0.0