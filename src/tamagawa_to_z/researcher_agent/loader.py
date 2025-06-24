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

# Import optimization schema
from .schemas.optimization_schema import OptimizationSummary, OptimizationTrial, OptimizationObjective, OptimizationSearchSpace, calculate_optimization_patterns


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
    optimization_logs: Optional[OptimizationSummary] = None  # 新規追加
    optimization_config: Optional[Dict[str, Any]] = None     # 新規追加


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
        
        # Load optimization logs and config
        optimization_logs = self.load_optimization_logs()
        optimization_config = self.load_optimization_config()
        
        return LoadedData(
            candidates=candidates,
            toponym_dict=toponym_dict,
            known_sites=known_sites,
            param_yaml=param_yaml,
            ia_report=ia_report,
            ia_plan=ia_plan,
            code_snippets=code_snippets,
            meta_info=self.meta_info,
            optimization_logs=optimization_logs,
            optimization_config=optimization_config
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
        default_path = self.config.get('default_paths', {}).get('candidates', 'data/output/candidates/region_candidates.csv')
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
        """Load known sites from GeoPackage, Shapefile, or KMZ."""
        filename_patterns = [
            "known_acre.kmz",
            "known_sites.gpkg",
            "known_sites.shp",
            "*acre*.kmz",
            "*sites*.gpkg",
            "*sites*.shp"
        ]
        
        path = self._resolve_file_path('known_sites', filename_patterns)
        
        if path:
            try:
                if str(path).endswith('.kmz'):
                    # KMZファイルの場合は特別な処理
                    gdf = self._load_kmz_file(path)
                else:
                    gdf = gpd.read_file(path)
                self.meta_info['known_sites_path'] = str(path)
                print(f"✅ Loaded known sites from: {path}")
                return gdf
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
        
        default_path = self.config.get('default_paths', {}).get('known_sites', 'data/known/known_acre.kmz')
        print(f"Warning: No known sites file found. Expected: {default_path}")
        return gpd.GeoDataFrame()
    
    def _load_kmz_file(self, kmz_path: Path) -> gpd.GeoDataFrame:
        """Load KMZ file by extracting KML and parsing XML."""
        import zipfile
        import tempfile
        from xml.etree import ElementTree as ET
        from shapely.geometry import Point
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # KMZファイルを展開
            with zipfile.ZipFile(kmz_path, 'r') as kmz:
                kml_files = [name for name in kmz.namelist() if name.endswith('.kml')]
                if not kml_files:
                    raise ValueError(f"No KML files found in KMZ: {kmz_path}")
                
                # 最初のKMLファイルを展開
                kml_filename = kml_files[0]
                kml_content = kmz.read(kml_filename).decode('utf-8')
        
        # XMLを解析
        root = ET.fromstring(kml_content)
        
        # 名前空間の定義
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        
        data = []
        # Placemarkを探して処理
        placemarks = root.findall('.//kml:Placemark', ns)
        
        for placemark in placemarks:
            # 場所名を取得
            name_element = placemark.find('kml:name', ns)
            place_name = name_element.text if name_element is not None else ''
            
            # 座標を取得
            coordinates_element = placemark.find('.//kml:coordinates', ns)
            if coordinates_element is not None:
                coords_text = coordinates_element.text.strip()
                coords_parts = coords_text.split(',')
                if len(coords_parts) >= 2:
                    try:
                        longitude = float(coords_parts[0])
                        latitude = float(coords_parts[1])
                        
                        data.append({
                            'name': place_name,
                            'geometry': Point(longitude, latitude)
                        })
                    except ValueError:
                        continue
        
        return gpd.GeoDataFrame(data, crs='EPSG:4326')
    
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
                self.project_root / "data" / "output" / "candidates",
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
    
    def load_optimization_logs(self) -> Optional[OptimizationSummary]:
        """最適化ログを汎用形式で読み込み"""
        
        # 1. 標準化済みファイルを探す
        summary_files = ["optimization_summary.yaml", "opt_summary.yaml"]
        for summary_file in summary_files:
            path = self._find_file_in_search_paths(summary_file)
            if path:
                try:
                    return OptimizationSummary.from_yaml(path)
                except Exception as e:
                    print(f"Warning: Could not load optimization summary from {path}: {e}")
        
        # 2. Optuna固有ファイルから変換
        optuna_data = self._load_optuna_logs()
        if optuna_data:
            try:
                return self._convert_optuna_to_standard(optuna_data)
            except Exception as e:
                print(f"Warning: Could not convert Optuna logs to standard format: {e}")
        
        # 3. 他の形式（将来の拡張ポイント）
        # Grid Search、Manual等のログ変換
        
        print("Warning: No optimization logs found")
        return None
    
    def load_optimization_config(self) -> Optional[Dict[str, Any]]:
        """最適化設定を読み込み"""
        
        config_files = [
            "optuna_space.yaml",
            "optimization_config.yaml", 
            "tuning_config.yaml"
        ]
        
        for config_file in config_files:
            path = self._find_file_in_search_paths(config_file)
            if path:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    print(f"✅ Loaded optimization config from: {path}")
                    return config
                except Exception as e:
                    print(f"Warning: Could not load optimization config from {path}: {e}")
        
        print("Warning: No optimization config found")
        return None
    
    def _load_optuna_logs(self) -> Optional[Dict[str, Any]]:
        """Optuna固有のログファイルを読み込み"""
        
        optuna_files = {
            "optimization_history.json": "json",
            "best_params.json": "json",
            "optuna_hybrid.log": "log",
            "optuna.db": "db"  # SQLiteデータベース（将来対応）
        }
        
        loaded_data = {}
        
        for filename, file_type in optuna_files.items():
            path = self._find_file_in_search_paths(filename)
            if path:
                try:
                    if file_type == "json":
                        with open(path, 'r', encoding='utf-8') as f:
                            loaded_data[filename] = json.load(f)
                    elif file_type == "log":
                        loaded_data[filename] = self._parse_optuna_log(path)
                    elif file_type == "db":
                        # SQLiteデータベース読み込み（将来実装）
                        loaded_data[filename] = {"type": "sqlite", "path": str(path)}
                    
                    print(f"✅ Loaded Optuna file: {path}")
                    
                except Exception as e:
                    print(f"Warning: Could not load {path}: {e}")
        
        return loaded_data if loaded_data else None
    
    def _parse_optuna_log(self, log_path: Path) -> Dict[str, Any]:
        """Optunaログファイルを解析"""
        
        log_data = {
            "trials_info": [],
            "errors": [],
            "warnings": [],
            "config_info": {}
        }
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # 試行完了ログの解析
                    if "Trial" in line and "finished with value" in line:
                        # 例: [I 2025-06-23 23:07:36,744] Trial 0 finished with value: -0.04515449934959718
                        parts = line.split()
                        if len(parts) >= 8:
                            try:
                                trial_num = int(parts[4])
                                score_str = parts[8].rstrip('.')
                                score = float(score_str)
                                
                                log_data["trials_info"].append({
                                    "trial_number": trial_num,
                                    "score": score,
                                    "line_number": line_num,
                                    "raw_line": line
                                })
                            except (ValueError, IndexError):
                                pass
                    
                    # エラーログの解析
                    elif "ERROR" in line:
                        log_data["errors"].append({
                            "line_number": line_num,
                            "message": line
                        })
                    
                    # 警告ログの解析
                    elif "WARNING" in line or "Warning" in line:
                        log_data["warnings"].append({
                            "line_number": line_num,
                            "message": line
                        })
                    
                    # 設定情報の解析
                    elif "distance_km" in line or "occ_pct" in line:
                        log_data["config_info"]["params_logged"] = True
        
        except Exception as e:
            print(f"Warning: Error parsing log file {log_path}: {e}")
        
        return log_data
    
    def _convert_optuna_to_standard(self, optuna_data: Dict[str, Any]) -> OptimizationSummary:
        """Optunaログを標準形式に変換"""
        
        # 最適化履歴から試行データを抽出
        history_data = optuna_data.get("optimization_history.json", {})
        trials_list = history_data.get("optimization_history", [])
        
        # 最良パラメータ情報
        best_params_data = optuna_data.get("best_params.json", {})
        
        # ログファイル情報
        log_data = optuna_data.get("optuna_hybrid.log", {})
        
        # 試行データの変換
        trials = []
        for i, trial_data in enumerate(trials_list):
            trials.append(OptimizationTrial(
                trial_id=trial_data.get("trial_number", i),
                score=trial_data.get("score"),
                candidates_count=trial_data.get("candidates_count", 0),
                params=trial_data.get("params", {}),
                timestamp=trial_data.get("datetime"),
                status="completed" if trial_data.get("score") is not None else "failed"
            ))
        
        # 目的関数設定（デフォルト値を使用、実際の設定は optimization_config から取得）
        objective = OptimizationObjective(
            direction="maximize",  # Optunaのデフォルト
            function_design="weighted_composite",
            weights={
                "recall": 0.6,
                "map": 0.2, 
                "workload": -0.2
            }
        )
        
        # 探索空間（デフォルト値、実際の設定は optimization_config から取得）
        search_space = OptimizationSearchSpace(parameters={
            "distance_km": {"type": "float", "low": 1.0, "high": 10.0, "log": False},
            "occ_pct": {"type": "float", "low": 1.0, "high": 10.0, "log": False}
        })
        
        # パターン分析
        patterns = calculate_optimization_patterns(trials)
        
        # サマリー作成
        summary = OptimizationSummary(
            method="optuna_tpe",
            status="completed",
            total_trials=len(trials),
            successful_trials=len([t for t in trials if t.status == "completed"]),
            objective=objective,
            search_space=search_space,
            trials=trials,
            patterns=patterns,
            study_name=best_params_data.get("study_name")
        )
        
        return summary
    
    def _find_file_in_search_paths(self, filename: str) -> Optional[Path]:
        """複数の検索パスでファイルを探す"""
        
        search_paths = []
        
        # 1. アーティファクトディレクトリ
        search_paths.append(self.artefact_dir)
        
        # 2. 最適化関連の標準ディレクトリ
        search_paths.extend([
            self.artefact_dir / "optuna",
            self.artefact_dir / "optimization",
            self.artefact_dir / ".." / "optuna",
            self.artefact_dir / ".." / "optimization"
        ])
        
        # 3. プロジェクトルートベースの検索
        if self.project_root:
            search_paths.extend([
                self.project_root / "data" / "output" / "optuna",
                self.project_root / "data" / "output" / "optimization",
                self.project_root / "configs",
                self.project_root / "config"
            ])
        
        # 4. 設定で指定されたパス
        default_paths = self.config.get('optimization_paths', {})
        if filename in default_paths:
            search_paths.append(Path(default_paths[filename]))
        
        # 各パスで検索
        for search_path in search_paths:
            if search_path.exists():
                # 直接ファイル名で検索
                target_file = search_path / filename
                if target_file.exists():
                    return target_file
                
                # グロブパターンで検索
                for found_file in search_path.glob(f"*{filename}*"):
                    if found_file.is_file():
                        return found_file
        
        return None