"""
Formatters: Generate Markdown reports and YAML plans with schema validation.
"""
from __future__ import annotations

import json
import yaml
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict, Any

import jsonschema

from .scorer import RankedProposal
from .evaluator import IaEffect
from .rca import FailureCluster


class MdFormatter:
    """Generate Markdown research reports."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize formatter with output directory.
        
        Parameters
        ----------
        output_dir : Path
            Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_report(self, 
                    ranked_proposals: List[RankedProposal],
                    ia_eval: IaEffect,
                    failure_clusters: List[FailureCluster],
                    extra_context: Dict[str, Any] = None) -> Path:
        """
        Write comprehensive Markdown research report.
        
        Parameters
        ----------
        ranked_proposals : List[RankedProposal]
            Ranked improvement proposals
        ia_eval : IaEffect
            IA proposal evaluation
        failure_clusters : List[FailureCluster]
            Identified failure patterns
        extra_context : Dict[str, Any], optional
            Additional context including optimization analysis
            
        Returns
        -------
        Path
            Path to generated report file
        """
        # Generate report content
        content = self._generate_report_content(ranked_proposals, ia_eval, failure_clusters, extra_context or {})
        
        # Write to file
        today = date.today().isoformat()
        report_path = self.output_dir / f"research_report_{today}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return report_path
    
    def _generate_report_content(self,
                               ranked_proposals: List[RankedProposal],
                               ia_eval: IaEffect,
                               failure_clusters: List[FailureCluster],
                               extra_context: Dict[str, Any] = None) -> str:
        """Generate the full Markdown report content."""
        today = date.today().isoformat()
        timestamp = datetime.now().isoformat()
        
        # 最適化分析があるかチェック
        extra_context = extra_context or {}
        optimization_analysis = extra_context.get("optimization_analysis")
        
        content = f"""# Researcher Agent Analysis Report

**Date**: {today}  
**Generated**: {timestamp}  
**Analyst**: Researcher Agent v1.0

---

## Executive Summary

This report presents a comprehensive analysis of the Inspector Agent's proposal and provides three alternative improvement strategies. The analysis identified {len(failure_clusters)} failure patterns and generated {len(ranked_proposals)} ranked proposals for system optimization."""
        
        # 最適化分析があれば追加
        if optimization_analysis:
            content += f"""

### Optimization Analysis:
- **Optimization Health Score**: {optimization_analysis.optimization_health_score:.1%}
- **Key Issues Found**: {len(optimization_analysis.identified_issues)}
- **Analysis Confidence**: {optimization_analysis.confidence:.1%}"""

        content += f"""

### Key Findings:
- **IA Proposal Confidence**: {ia_eval.confidence:.1%}
- **Primary Issue**: {failure_clusters[0].root_cause if failure_clusters else 'No critical issues identified'}
- **Recommended Action**: {ranked_proposals[0].proposal.title if ranked_proposals else 'No proposals generated'}

---"""
        
        # 最適化分析の詳細セクション
        if optimization_analysis:
            content += f"""

## Optimization Process Analysis

### Overall Assessment
{optimization_analysis.overall_assessment}

### Key Findings
"""
            for i, finding in enumerate(optimization_analysis.key_findings, 1):
                content += f"{i}. {finding}\n"

            if optimization_analysis.identified_issues:
                content += f"""

### Identified Optimization Issues
"""
                for issue in optimization_analysis.identified_issues:
                    severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(issue.get("severity", ""), "⚪")
                    content += f"""
#### {severity_icon} {issue.get('issue_type', 'Unknown Issue')} ({issue.get('severity', 'unknown').title()})

**Description**: {issue.get('description', 'No description available')}

**Evidence**: {issue.get('evidence', 'No evidence provided')}

**Potential Causes**: {', '.join(issue.get('potential_causes', ['Unknown']))}

**Impact**: {issue.get('impact', 'Impact not specified')}
"""

            content += "\n---"

        content += """

## Inspector Agent Proposal Evaluation

### Baseline Performance
"""
        
        # Add baseline metrics table
        if ia_eval.baseline_metrics:
            content += "\n| Metric | Current Value | Status |\n|--------|---------------|--------|\n"
            for metric, value in ia_eval.baseline_metrics.items():
                status = self._get_metric_status(metric, value)
                if isinstance(value, float):
                    content += f"| {metric} | {value:.3f} | {status} |\n"
                else:
                    content += f"| {metric} | {value} | {status} |\n"
        
        # IA proposal effects
        content += f"""

### IA Proposal Impact Assessment

**Confidence Level**: {ia_eval.confidence:.1%}  
**Estimated Workload Change**: {ia_eval.workload_change:+d} candidates

**Reasoning**: {ia_eval.reasoning}

#### Projected Changes:
"""
        
        if ia_eval.estimated_metrics:
            content += "\n| Metric | Before | After | Change |\n|--------|--------|-------|--------|\n"
            for metric in ia_eval.baseline_metrics:
                before = ia_eval.baseline_metrics.get(metric, 0)
                after = ia_eval.estimated_metrics.get(metric, before)
                if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                    change = after - before
                    change_str = f"{change:+.3f}" if isinstance(change, float) else f"{change:+d}"
                    content += f"| {metric} | {before:.3f} | {after:.3f} | {change_str} |\n"
        
        # Root cause analysis
        content += "\n---\n\n## Root Cause Analysis\n\n"
        
        if failure_clusters:
            for i, cluster in enumerate(failure_clusters, 1):
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(cluster.severity, "⚪")
                
                content += f"""### {i}. {cluster.description} {severity_icon}

**Pattern**: {cluster.pattern}  
**Occurrences**: {cluster.count}  
**Severity**: {cluster.severity.upper()}

**Root Cause**: {cluster.root_cause}

**Suggested Fix**: {cluster.suggested_fix}

"""
                if cluster.examples:
                    content += "**Examples**:\n"
                    for j, example in enumerate(cluster.examples[:3], 1):
                        example_desc = example.get('description', str(example))
                        content += f"{j}. {example_desc}\n"
                    content += "\n"
        else:
            content += "No critical failure patterns identified in the current dataset.\n\n"
        
        # Improvement proposals
        content += "---\n\n## Improvement Proposals\n\n"
        
        if ranked_proposals:
            content += f"Generated {len(ranked_proposals)} proposals ranked by improvement potential, diversity, and implementation cost.\n\n"
            
            for ranked in ranked_proposals:
                proposal = ranked.proposal
                rank_icon = {1: "🥇", 2: "🥈", 3: "🥉"}.get(ranked.rank, f"{ranked.rank}.")
                
                score_text = f"{ranked.total_score:.2f}"
                improvement_text = f"{ranked.improvement_score:.2f}"
                diversity_text = f"{ranked.diversity_score:.2f}"
                cost_text = f"{ranked.cost_score:.2f}"
                
                content += f"""### {rank_icon} Proposal {proposal.id}: {proposal.title}

**Overall Score**: {score_text}/1.0  
**Rank**: #{ranked.rank}

#### Score Breakdown:
- **Improvement Potential**: {improvement_text}/1.0
- **Approach Diversity**: {diversity_text}/1.0  
- **Implementation Cost**: {cost_text}/1.0 (lower cost = higher score)

#### Proposed Changes:
"""
                
                for j, change in enumerate(proposal.changes, 1):
                    action = change.get('action', 'unknown')
                    content += f"{j}. **{action}**"
                    
                    if 'params' in change:
                        params_str = ", ".join(f"{k}={v}" for k, v in change['params'].items())
                        content += f": {params_str}"
                    
                    if 'description' in change:
                        content += f" - {change['description']}"
                    
                    content += "\n"
                
                content += f"""
#### Expected Effects:
"""
                for metric, effect in proposal.expected_effect.items():
                    content += f"- **{metric}**: {effect}\n"
                
                content += f"""
**Rationale**: {proposal.rationale}

**Risk Assessment**: {proposal.risk}

**Implementation Effort**: {proposal.human_effort}

**Priority**: {proposal.priority.upper()}

---

"""
        else:
            content += "No improvement proposals could be generated.\n\n"
        
        # Recommendations
        content += "## Recommendations\n\n"
        
        if ranked_proposals:
            top_proposal = ranked_proposals[0].proposal
            content += f"""### 1. Immediate Action: {top_proposal.title}

Based on the analysis, **Proposal {top_proposal.id}** offers the best balance of improvement potential and implementation feasibility.

**Next Steps**:
1. Review the proposal parameters in `research_plan_{today}.yaml`
2. Test the proposed changes in a development environment  
3. Evaluate impact on a subset of the data
4. Deploy if results meet expectations

### 2. Alternative Strategies

"""
            if len(ranked_proposals) > 1:
                for ranked in ranked_proposals[1:]:
                    score_val = f"{ranked.total_score:.2f}"
                    content += f"- **Proposal {ranked.proposal.id}**: {ranked.proposal.title} (Score: {score_val})\n"
            
            content += f"""
### 3. Monitoring and Evaluation

After implementing changes:
- Monitor key metrics: recall@100, mAP, workload
- Validate improvements with test dataset
- Document lessons learned for future iterations

"""
        else:
            content += """### Manual Review Required

The automated analysis could not generate reliable improvement proposals. 
Consider manual parameter tuning based on the failure analysis above.

"""
        
        # Footer
        content += f"""---

*This report was generated by Researcher Agent on {timestamp}.  
For questions or issues, please review the methodology documentation.*
"""
        
        return content
    
    def _get_metric_status(self, metric: str, value: Any) -> str:
        """Get status indicator for a metric value."""
        if not isinstance(value, (int, float)):
            return "ℹ️"
        
        if metric == 'recall@100':
            if value >= 0.5:
                return "🟢 Good"
            elif value >= 0.3:
                return "🟡 Fair"
            else:
                return "🔴 Low"
        elif metric == 'map':
            if value >= 0.4:
                return "🟢 Good"
            elif value >= 0.2:
                return "🟡 Fair"
            else:
                return "🔴 Low"
        elif metric == 'workload':
            if value < 500:
                return "🟢 Manageable"
            elif value < 1000:
                return "🟡 Moderate"
            else:
                return "🔴 High"
        else:
            return "ℹ️"


class YamlFormatter:
    """Generate YAML research plans with schema validation."""
    
    # JSON Schema for validation
    SCHEMA = {
        "type": "object",
        "required": ["experiment_id", "base_commit", "proposals"],
        "properties": {
            "experiment_id": {"type": "string"},
            "base_commit": {"type": "string"},
            "proposals": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "required": ["id", "title", "changes", "expected_effect", "rationale", "risk", "human_effort"],
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "changes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["action"],
                                "properties": {
                                    "action": {"type": "string"},
                                    "params": {"type": "object"},
                                    "target_agent": {"type": "string"},
                                    "description": {"type": "string"}
                                }
                            }
                        },
                        "expected_effect": {"type": "object"},
                        "rationale": {"type": "string"},
                        "risk": {"type": "string"},
                        "human_effort": {"type": "string"}
                    }
                }
            }
        }
    }
    
    def __init__(self, output_dir: Path):
        """
        Initialize formatter with output directory.
        
        Parameters
        ----------
        output_dir : Path
            Directory to save YAML plans
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_yaml(self, ranked_proposals: List[RankedProposal], extra_context: Dict[str, Any] = None) -> Path:
        """
        Write research plan YAML with schema validation.
        
        Parameters
        ----------
        ranked_proposals : List[RankedProposal]
            Ranked improvement proposals
        extra_context : Dict[str, Any], optional
            Additional context including optimization analysis
            
        Returns
        -------
        Path
            Path to generated YAML file
        """
        # Build YAML structure
        today = date.today().isoformat()
        yaml_data = {
            "experiment_id": f"exp_{today.replace('-', '_')}_01",
            "base_commit": "latest",  # Could be extracted from git if needed
            "proposals": []
        }
        
        # Convert proposals to YAML format
        for ranked in ranked_proposals:
            proposal = ranked.proposal
            proposal_data = {
                "id": proposal.id,
                "title": proposal.title,
                "changes": proposal.changes,
                "expected_effect": proposal.expected_effect,
                "rationale": proposal.rationale,
                "risk": proposal.risk,
                "human_effort": proposal.human_effort
            }
            yaml_data["proposals"].append(proposal_data)
        
        # Validate against schema
        try:
            jsonschema.validate(yaml_data, self.SCHEMA)
        except jsonschema.ValidationError as e:
            print(f"Warning: YAML validation failed: {e}")
            # Continue anyway, but log the error
        
        # Write to file
        yaml_path = self.output_dir / f"research_plan_{today}.yaml"
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        return yaml_path