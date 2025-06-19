"""
ArtefactLoader: Load IA outputs and supporting data for analysis.
"""
from __future__ import annotations

import json
import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import pandas as pd
import geopandas as gpd

import sys
from pathlib import Path

# Add parent directory to path to import code_extract directly
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from code_extract import auto_extract_from_module


@dataclass
class LoadedData:
    """Container for all loaded artifacts and data."""
    candidates: pd.DataFrame
    toponym_dict: pd.DataFrame
    known_sites: gpd.GeoDataFrame
    param_yaml: Dict[str, Any]
    ia_report: str
    ia_plan: Dict[str, Any]
    code_snippets: Dict[str, Dict[str, str]]
    meta_info: Dict[str, Any]


class ArtefactLoader:
    """Load all artifacts produced by Inspector Agent and supporting data."""
    
    def __init__(self, artefact_dir: Path, config: Dict[str, Any] = None):
        """
        Initialize the loader with artifacts directory.
        
        Parameters
        ----------
        artefact_dir : Path
            Directory containing IA outputs and data files
        config : Dict[str, Any], optional
            Configuration dictionary with default paths
        """
        self.artefact_dir = Path(artefact_dir)
        self.config = config or {}
        self.meta_info = {}
        self.project_root = self._find_project_root()
    
    def load(self) -> LoadedData:
        """
        Load all required artifacts and data.
        
        Returns
        -------
        LoadedData
            Container with all loaded data
        """
        # Load CSV data
        candidates = self._load_candidates()
        toponym_dict = self._load_toponym_dict()
        
        # Load geospatial data
        known_sites = self._load_known_sites()
        
        # Load configuration
        param_yaml = self._load_param_yaml()
        
        # Load IA outputs
        ia_report, ia_plan = self._load_ia_outputs()
        
        # Extract code snippets
        code_snippets = self._extract_code_snippets()
        
        return LoadedData(
            candidates=candidates,
            toponym_dict=toponym_dict,
            known_sites=known_sites,
            param_yaml=param_yaml,
            ia_report=ia_report,
            ia_plan=ia_plan,
            code_snippets=code_snippets,
            meta_info=self.meta_info
        )
    
    def _load_candidates(self) -> pd.DataFrame:
        """Load candidates CSV file."""
        filename_patterns = [
            "candidates.csv",
            "region_candidates.csv", 
            "acre_candidates.csv",
            "*candidates*.csv"
        ]
        
        path = self._resolve_file_path('candidates', filename_patterns)
        
        if path:
            try:
                df = pd.read_csv(path)
                # Ensure geometry column is treated as string for compatibility
                if 'geometry' in df.columns:
                    df['geometry'] = df['geometry'].astype(str)
                self.meta_info['candidates_path'] = str(path)
                print(f"✅ Loaded candidates from: {path}")
                return df
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
        
        # Return empty DataFrame if no file found
        default_path = self.config.get('default_paths', {}).get('candidates', 'data/interim/region_candidates.csv')
        print(f"Warning: No candidates CSV file found. Expected: {default_path}")
        return pd.DataFrame()
    
    def _load_toponym_dict(self) -> pd.DataFrame:
        """Load toponym dictionary CSV file."""
        filename_patterns = [
            "toponym_dict.csv",
            "water_roots.csv",
            "*dict*.csv"
        ]
        
        path = self._resolve_file_path('toponym_dict', filename_patterns)
        
        if path:
            try:
                df = pd.read_csv(path)
                self.meta_info['dict_path'] = str(path)
                print(f"✅ Loaded toponym dictionary from: {path}")
                return df
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
        
        default_path = self.config.get('default_paths', {}).get('toponym_dict', 'data/dict/toponym_dict.csv')
        print(f"Warning: No toponym dictionary found. Expected: {default_path}")
        return pd.DataFrame()
    
    def _load_known_sites(self) -> gpd.GeoDataFrame:
        """Load known sites from GeoPackage or Shapefile."""
        filename_patterns = [
            "known_sites.gpkg",
            "known_sites.shp",
            "*sites*.gpkg",
            "*sites*.shp"
        ]
        
        path = self._resolve_file_path('known_sites', filename_patterns)
        
        if path:
            try:
                gdf = gpd.read_file(path)
                self.meta_info['known_sites_path'] = str(path)
                print(f"✅ Loaded known sites from: {path}")
                return gdf
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
        
        default_path = self.config.get('default_paths', {}).get('known_sites', 'data/raw/known_sites.gpkg')
        print(f"Warning: No known sites file found. Expected: {default_path}")
        return gpd.GeoDataFrame()
    
    def _load_param_yaml(self) -> Dict[str, Any]:
        """Load parameter YAML file."""
        filename_patterns = [
            "param.yaml",
            "run_meta.yaml",
            "config.yaml",
            "*param*.yaml",
            "*config*.yaml"
        ]
        
        path = self._resolve_file_path('param_yaml', filename_patterns)
        
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.meta_info['param_path'] = str(path)
                print(f"✅ Loaded parameters from: {path}")
                return config
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
        
        default_path = self.config.get('default_paths', {}).get('param_yaml', 'config/run_meta.yaml')
        print(f"Warning: No parameter YAML found. Expected: {default_path}")
        return {}
    
    def _load_ia_outputs(self) -> tuple[str, Dict[str, Any]]:
        """Load Inspector Agent report and plan."""
        ia_report = ""
        ia_plan = {}
        
        # Find most recent IA output directory
        inspector_dirs = []
        possible_base_dirs = [
            self.artefact_dir,
            self.artefact_dir / "../output/inspector_reports"
        ]
        
        for base_dir in possible_base_dirs:
            if base_dir.exists():
                # Look for timestamped directories
                for item in base_dir.iterdir():
                    if item.is_dir() and any(char.isdigit() for char in item.name):
                        inspector_dirs.append(item)
        
        if inspector_dirs:
            # Sort by modification time and take the most recent
            latest_dir = max(inspector_dirs, key=lambda p: p.stat().st_mtime)
            
            # Load report
            for report_file in latest_dir.glob("report_*.md"):
                try:
                    ia_report = report_file.read_text(encoding='utf-8')
                    break
                except Exception as e:
                    print(f"Warning: Could not load report {report_file}: {e}")
            
            # Load plan
            for plan_file in latest_dir.glob("plan_*.yaml"):
                try:
                    with open(plan_file, 'r', encoding='utf-8') as f:
                        ia_plan = yaml.safe_load(f)
                    break
                except Exception as e:
                    print(f"Warning: Could not load plan {plan_file}: {e}")
            
            self.meta_info['ia_output_dir'] = str(latest_dir)
        
        return ia_report, ia_plan
    
    def _extract_code_snippets(self) -> Dict[str, Dict[str, str]]:
        """Extract relevant code snippets from harmonizer and inspector modules."""
        code_snippets = {}
        
        if not self.project_root:
            print("Warning: Could not find project root. Skipping code extraction.")
            print("  Hint: Run from project directory or set TAMAGAWA_PROJECT_ROOT environment variable")
            return {}
        
        src_dir = self.project_root / "src" / "tamagawa_to_z"
        if not src_dir.exists():
            print(f"Warning: Source directory not found: {src_dir}")
            return {}
        
        # Extract from harmonizer module
        harmonizer_dir = src_dir / "harmonizer"
        if harmonizer_dir.exists():
            harmonizer_snippets = auto_extract_from_module(harmonizer_dir, max_files=5)
            if harmonizer_snippets:
                code_snippets['harmonizer'] = harmonizer_snippets
        
        # Extract from inspector_agent module
        inspector_dir = src_dir / "inspector_agent"
        if inspector_dir.exists():
            inspector_snippets = auto_extract_from_module(inspector_dir, max_files=3)
            if inspector_snippets:
                code_snippets['inspector_agent'] = inspector_snippets
        
        return code_snippets
    
    def _find_project_root(self) -> Optional[Path]:
        """Find project root directory."""
        # Check environment variable first
        if 'TAMAGAWA_PROJECT_ROOT' in os.environ:
            root = Path(os.environ['TAMAGAWA_PROJECT_ROOT'])
            if root.exists():
                return root
        
        # Start from artefact directory and search upwards
        current = self.artefact_dir.resolve()
        while current.parent != current:
            # Check for pyproject.toml (Poetry project marker)
            if (current / "pyproject.toml").exists():
                return current
            # Check for src/tamagawa_to_z structure
            if (current / "src" / "tamagawa_to_z").exists():
                return current
            current = current.parent
        
        # Try current working directory
        cwd = Path.cwd()
        if (cwd / "pyproject.toml").exists() or (cwd / "src" / "tamagawa_to_z").exists():
            return cwd
        
        return None
    
    def _get_ia_metadata(self) -> Dict[str, Any]:
        """Extract metadata from IA results JSON file."""
        # Look for results JSON file in artefact directory
        for json_file in self.artefact_dir.glob("results_*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                return results.get('meta_info', {})
            except Exception as e:
                print(f"Warning: Could not load IA metadata from {json_file}: {e}")
        
        return {}
    
    def _resolve_file_path(self, file_key: str, filename_patterns: List[str]) -> Optional[Path]:
        """
        Resolve file path using hierarchical search strategy.
        
        Parameters
        ----------
        file_key : str
            Key in config for default path (e.g., 'candidates', 'known_sites')
        filename_patterns : List[str]
            List of filename patterns to search for
            
        Returns
        -------
        Optional[Path]
            Resolved file path or None if not found
        """
        # 1. Check IA metadata for input files (future enhancement)
        ia_metadata = self._get_ia_metadata()
        if 'input_files' in ia_metadata and file_key in ia_metadata['input_files']:
            path = Path(ia_metadata['input_files'][file_key])
            if path.exists():
                return path
        
        # 2. Check config default paths
        default_paths = self.config.get('default_paths', {})
        if file_key in default_paths:
            # Try relative to project root
            if self.project_root:
                path = self.project_root / default_paths[file_key]
                if path.exists():
                    return path
            
            # Try as absolute path
            path = Path(default_paths[file_key])
            if path.exists():
                return path
        
        # 3. Search in project standard directories
        if self.project_root:
            search_dirs = [
                self.project_root / "data" / "interim",
                self.project_root / "data" / "raw", 
                self.project_root / "data" / "dict",
                self.project_root / "config"
            ]
            
            for search_dir in search_dirs:
                if search_dir.exists():
                    for pattern in filename_patterns:
                        for found_file in search_dir.glob(pattern):
                            if found_file.exists():
                                return found_file
        
        # 4. Search relative to artefact directory (original behavior)
        relative_searches = [
            self.artefact_dir,
            self.artefact_dir / ".." / "interim",
            self.artefact_dir / ".." / "raw",
            self.artefact_dir / ".." / "dict",
            self.artefact_dir / ".." / "config"
        ]
        
        for search_dir in relative_searches:
            if search_dir.exists():
                for pattern in filename_patterns:
                    for found_file in search_dir.glob(pattern):
                        if found_file.exists():
                            return found_file
        
        return None