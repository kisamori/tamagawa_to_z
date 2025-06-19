"""
Prefect flow for Researcher Agent automation.
"""
from pathlib import Path
from datetime import date
import subprocess
import os

from prefect import flow, task
from prefect.logging import get_run_logger


@task
def run_researcher_agent(artefacts_dir: str, output_dir: str, config_path: str = None) -> str:
    """
    Run the researcher agent script.
    
    Parameters
    ----------
    artefacts_dir : str
        Directory containing IA artifacts
    output_dir : str
        Output directory for research reports
    config_path : str, optional
        Path to config file
        
    Returns
    -------
    str
        Output directory path
    """
    logger = get_run_logger()
    
    # Build command
    cmd = [
        "python", 
        "scripts/run_researcher.py",
        "--artefacts", artefacts_dir,
        "--output", output_dir,
        "--verbose"
    ]
    
    if config_path:
        cmd.extend(["--config", config_path])
    
    # Set working directory to project root
    project_root = Path(__file__).parent.parent
    
    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info(f"Working directory: {project_root}")
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info("Researcher agent completed successfully")
        logger.info(f"Output: {result.stdout}")
        
        if result.stderr:
            logger.warning(f"Stderr: {result.stderr}")
        
        return output_dir
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Researcher agent failed with exit code {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        raise


@task
def notify_completion(output_dir: str, notification_channel: str = None):
    """
    Notify completion of research analysis.
    
    Parameters
    ----------
    output_dir : str
        Directory where reports were saved
    notification_channel : str, optional
        Notification channel (e.g., Slack webhook)
    """
    logger = get_run_logger()
    
    # Find generated files
    output_path = Path(output_dir)
    today = date.today().isoformat()
    
    report_file = output_path / f"research_report_{today}.md"
    plan_file = output_path / f"research_plan_{today}.yaml"
    
    message = f"""🔬 Researcher Agent Analysis Complete!

📁 Output Directory: {output_dir}
📄 Report: {report_file.name if report_file.exists() else 'Not found'}
📋 Plan: {plan_file.name if plan_file.exists() else 'Not found'}

🎯 Next Steps:
1. Review the research report
2. Select an improvement proposal
3. Update parameters and re-run harmonizer
"""
    
    logger.info(message)
    
    # TODO: Add actual notification integration (Slack, email, etc.)
    if notification_channel:
        logger.info(f"Would send notification to: {notification_channel}")


@flow(name="research_flow")
def research_flow(
    artefacts_dir: str,
    output_dir: str = None,
    config_path: str = None,
    notify_channel: str = None
) -> str:
    """
    Main research flow to analyze IA outputs and generate improvement proposals.
    
    Parameters
    ----------
    artefacts_dir : str
        Directory containing Inspector Agent artifacts
    output_dir : str, optional
        Output directory for research reports (auto-generated if not provided)
    config_path : str, optional
        Path to researcher configuration file
    notify_channel : str, optional
        Notification channel for completion alerts
        
    Returns
    -------
    str
        Path to output directory
    """
    logger = get_run_logger()
    
    # Auto-generate output directory if not provided
    if not output_dir:
        today = date.today().isoformat()
        output_dir = f"data/output/research_reports/{today}"
    
    logger.info(f"Starting research flow")
    logger.info(f"Artefacts: {artefacts_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run researcher agent
    completed_output_dir = run_researcher_agent(artefacts_dir, output_dir, config_path)
    
    # Send completion notification
    notify_completion(completed_output_dir, notify_channel)
    
    logger.info(f"Research flow completed successfully")
    logger.info(f"Results saved to: {completed_output_dir}")
    
    return completed_output_dir


# Convenience functions for different trigger scenarios

def trigger_after_inspector(ia_output_dir: str) -> str:
    """
    Trigger research flow after Inspector Agent completes.
    
    Parameters
    ----------
    ia_output_dir : str
        Inspector Agent output directory
        
    Returns
    -------
    str
        Research output directory
    """
    # Auto-generate research output directory
    timestamp = date.today().isoformat()
    research_output = f"data/output/research_reports/{timestamp}"
    
    return research_flow(
        artefacts_dir=ia_output_dir,
        output_dir=research_output
    )


def trigger_manual(
    candidates_csv: str,
    known_sites: str,
    ia_reports_dir: str,
    output_dir: str = None
) -> str:
    """
    Manually trigger research flow with specific files.
    
    Parameters
    ----------
    candidates_csv : str
        Path to candidates CSV
    known_sites : str
        Path to known sites file
    ia_reports_dir : str
        Directory containing IA reports
    output_dir : str, optional
        Custom output directory
        
    Returns
    -------
    str
        Research output directory
    """
    # Create temporary artefacts directory structure
    temp_dir = Path("tmp/research_artefacts")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy/symlink files to expected structure
    import shutil
    
    if Path(candidates_csv).exists():
        shutil.copy2(candidates_csv, temp_dir / "candidates.csv")
    
    if Path(known_sites).exists():
        shutil.copy2(known_sites, temp_dir / "known_sites.gpkg")
    
    # Copy IA reports
    if Path(ia_reports_dir).exists():
        for file in Path(ia_reports_dir).glob("*.md"):
            shutil.copy2(file, temp_dir)
        for file in Path(ia_reports_dir).glob("*.yaml"):
            shutil.copy2(file, temp_dir)
    
    return research_flow(
        artefacts_dir=str(temp_dir),
        output_dir=output_dir
    )


if __name__ == "__main__":
    # Example usage - can be run directly for testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python research_flow.py <artefacts_dir> [output_dir]")
        sys.exit(1)
    
    artefacts = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = research_flow(artefacts, output)
    print(f"Research flow completed: {result}")