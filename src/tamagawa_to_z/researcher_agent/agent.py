"""
ResearcherAgent: High-level orchestration of researcher agent steps.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

from openai import OpenAI

from .loader import ArtefactLoader
from .evaluator import Evaluator
from .rca import RootCauseAnalyzer
from .generator import ProposalGenerator
from .scorer import ProposalScorer
from .formatter import YamlFormatter, MdFormatter


class ResearcherAgent:
    """High-level orchestration of researcher agent steps."""

    def __init__(self, openai_client: OpenAI, cfg: dict):
        """
        Initialize researcher agent.
        
        Parameters
        ----------
        openai_client : OpenAI
            OpenAI API client
        cfg : dict
            Configuration dictionary
        """
        self.client = openai_client
        self.cfg = cfg

    def run(self, artefact_dir: Path, output_dir: Path) -> Tuple[Path, Path]:
        """
        Execute complete researcher agent pipeline.
        
        Parameters
        ----------
        artefact_dir : Path
            Directory that contains IA artefacts & CSVs.
        output_dir : Path
            Dir to save `research_report.md` and `research_plan.yaml`.

        Returns
        -------
        Tuple[Path, Path]
            (report_path, yaml_path)
        """
        print("🔬 Starting Researcher Agent analysis...")
        
        # 1. Load artifacts and data
        print("📂 Loading artifacts and data...")
        loader = ArtefactLoader(artefact_dir, config=self.cfg)
        data = loader.load()
        
        if data.candidates.empty:
            print("⚠️  Warning: No candidate data found")
        if not data.ia_plan:
            print("⚠️  Warning: No IA plan found for evaluation")
        
        print(f"✅ Loaded: {len(data.candidates)} candidates, "
              f"{len(data.known_sites)} known sites, "
              f"{len(data.code_snippets)} code modules")
        
        # 2. Evaluate IA proposal
        print("📊 Evaluating IA proposal...")
        evaluator = Evaluator(data)
        ia_eval = evaluator.quick_simulate()
        
        print(f"✅ IA evaluation complete (confidence: {ia_eval.confidence:.1%})")
        
        # 3. Root cause analysis
        print("🔍 Performing root cause analysis...")
        rca = RootCauseAnalyzer(self.client, data)
        try:
            failure_clusters = rca.analyze()
            print(f"✅ Identified {len(failure_clusters)} failure patterns")
            
            for cluster in failure_clusters:
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(cluster.severity, "⚪")
                print(f"  {severity_icon} {cluster.description} ({cluster.count} cases)")
        
        except Exception as e:
            print(f"⚠️  RCA failed: {e}")
            failure_clusters = []
        
        # 4. Generate proposals
        print("💡 Generating improvement proposals...")
        generator = ProposalGenerator(self.client, data, failure_clusters, self.cfg)
        
        try:
            proposals = generator.generate(n=3)
            print(f"✅ Generated {len(proposals)} proposals:")
            
            for proposal in proposals:
                print(f"  • {proposal.id}: {proposal.title}")
        
        except Exception as e:
            print(f"⚠️  Proposal generation failed: {e}")
            proposals = []
        
        # 5. Score and rank proposals
        print("🏆 Scoring and ranking proposals...")
        
        if proposals:
            scorer = ProposalScorer(data, proposals, weight_cfg=self.cfg.get("weights", {}))
            ranked = scorer.rank()
            
            print("📈 Final rankings:")
            for ranked_proposal in ranked:
                proposal = ranked_proposal.proposal
                print(f"  #{ranked_proposal.rank} {proposal.id}: {proposal.title} "
                      f"(Score: {ranked_proposal.total_score:.2f})")
        else:
            ranked = []
            print("⚠️  No proposals to rank")
        
        # 6. Format outputs
        print("📝 Generating reports...")
        
        # Generate Markdown report
        md_formatter = MdFormatter(output_dir)
        report_path = md_formatter.write_report(ranked, ia_eval, failure_clusters)
        print(f"✅ Research report: {report_path}")
        
        # Generate YAML plan
        yaml_formatter = YamlFormatter(output_dir)
        yaml_path = yaml_formatter.write_yaml(ranked)
        print(f"✅ Research plan: {yaml_path}")
        
        print("🎉 Researcher Agent analysis complete!")
        
        return report_path, yaml_path


def run(artefact_dir: str, 
        output_dir: str,
        config_path: str = None,
        api_key: str = None) -> Tuple[Path, Path]:
    """
    Convenience function to run Researcher Agent.
    
    Parameters
    ----------
    artefact_dir : str
        Directory containing IA artifacts
    output_dir : str
        Output directory for reports
    config_path : str, optional
        Path to configuration file
    api_key : str, optional
        OpenAI API key
        
    Returns
    -------
    Tuple[Path, Path]
        (report_path, yaml_path)
    """
    # Setup OpenAI client
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY_TIRE5"))
    
    # Load configuration
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    else:
        # Default configuration
        cfg = {
            "weights": {
                "improvement": 0.6,
                "diversity": 0.2,
                "cost": 0.2
            },
            "max_proposals": 3,
            "llm_model": "o3",
            "temperature": 0.3
        }
    
    # Run researcher agent
    agent = ResearcherAgent(client, cfg)
    return agent.run(Path(artefact_dir), Path(output_dir))