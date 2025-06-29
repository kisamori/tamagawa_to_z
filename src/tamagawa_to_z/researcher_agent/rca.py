"""
RootCauseAnalyzer: Analyze failure cases and identify patterns using LLM.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import defaultdict

import pandas as pd
import numpy as np
from openai import OpenAI

from .loader import LoadedData


@dataclass
class FailureCluster:
    """Container for a cluster of similar failure cases."""
    cluster_id: str
    description: str
    pattern: str
    examples: List[Dict[str, Any]]
    count: int
    severity: str  # 'high', 'medium', 'low'
    root_cause: str
    suggested_fix: str


class RootCauseAnalyzer:
    """Analyze failure patterns and root causes using clustering and LLM."""
    
    def __init__(self, client: OpenAI, data: LoadedData):
        """
        Initialize analyzer with OpenAI client and loaded data.
        
        Parameters
        ----------
        client : OpenAI
            OpenAI API client
        data : LoadedData
            Loaded artifacts and data
        """
        self.client = client
        self.data = data
    
    def analyze(self) -> List[FailureCluster]:
        """
        Analyze failure cases and identify clusters.
        
        Returns
        -------
        List[FailureCluster]
            List of identified failure clusters
        """
        # Identify failure cases
        failure_cases = self._identify_failures()
        
        if not failure_cases:
            return []
        
        # Cluster failures by patterns
        clusters = self._cluster_failures(failure_cases)
        
        # Analyze each cluster with LLM
        analyzed_clusters = []
        for cluster in clusters:
            analyzed = self._analyze_cluster_with_llm(cluster)
            if analyzed:
                analyzed_clusters.append(analyzed)
        
        return analyzed_clusters
    
    def _identify_failures(self) -> List[Dict[str, Any]]:
        """Identify failure cases from the data."""
        failures = []
        
        if self.data.candidates.empty or self.data.known_sites.empty:
            return failures
        
        # Case 1: Spatial clustering failures
        # Identify areas with high candidate density but no known sites
        spatial_failures = self._find_spatial_failures()
        failures.extend(spatial_failures)
        
        # Case 2: Root diversity failures  
        # Identify repeated patterns that might indicate missing roots
        root_failures = self._find_root_failures()
        failures.extend(root_failures)
        
        # Case 3: Distance-based failures
        # Identify candidates that are very far from rivers but still selected
        distance_failures = self._find_distance_failures()
        failures.extend(distance_failures)
        
        return failures
    
    def _find_spatial_failures(self) -> List[Dict[str, Any]]:
        """Find spatial clustering issues."""
        failures = []
        
        if 'geometry' not in self.data.candidates.columns:
            return failures
        
        # Simple heuristic: high candidate density areas
        if len(self.data.candidates) > 100:
            # Group candidates by approximate location
            try:
                # Extract coordinates from geometry strings (simplified)
                coords = []
                for geom_str in self.data.candidates['geometry'].dropna():
                    if 'POINT' in str(geom_str):
                        # Extract coordinates from WKT string
                        coord_part = str(geom_str).replace('POINT (', '').replace(')', '')
                        if ' ' in coord_part:
                            try:
                                lon, lat = map(float, coord_part.split())
                                coords.append((lon, lat))
                            except:
                                continue
                
                if len(coords) > 10:
                    # Find areas with many candidates
                    from collections import Counter
                    # Round to 0.1 degree for clustering
                    rounded_coords = [(round(lon, 1), round(lat, 1)) for lon, lat in coords]
                    coord_counts = Counter(rounded_coords)
                    
                    for (lon, lat), count in coord_counts.items():
                        if count >= 5:  # High density threshold
                            failures.append({
                                'type': 'spatial_clustering',
                                'location': f"{lat:.1f}, {lon:.1f}",
                                'count': count,
                                'description': f"High candidate density ({count} candidates) at {lat:.1f}, {lon:.1f}",
                                'severity': 'medium' if count < 10 else 'high'
                            })
            
            except Exception as e:
                # Skip spatial analysis if geometry parsing fails
                pass
        
        return failures
    
    def _find_root_failures(self) -> List[Dict[str, Any]]:
        """Find root diversity and pattern issues."""
        failures = []
        
        if 'name' not in self.data.candidates.columns:
            return failures
        
        # Analyze name patterns
        name_patterns = defaultdict(list)
        
        for idx, name in enumerate(self.data.candidates['name'].dropna()):
            name_str = str(name).lower()
            words = name_str.split()
            
            # Group by first word (common root pattern)
            if words:
                first_word = words[0]
                name_patterns[first_word].append({
                    'index': idx,
                    'name': name,
                    'words': words
                })
        
        # Find overrepresented patterns
        total_candidates = len(self.data.candidates)
        for pattern, examples in name_patterns.items():
            if len(examples) >= max(3, total_candidates * 0.1):  # At least 3 or 10% of total
                failures.append({
                    'type': 'root_overrepresentation',
                    'pattern': pattern,
                    'count': len(examples),
                    'examples': [ex['name'] for ex in examples[:5]],
                    'description': f"Pattern '{pattern}' appears {len(examples)} times",
                    'severity': 'medium' if len(examples) < 10 else 'high'
                })
        
        # Find single-occurrence patterns (might be misclassified)
        single_patterns = [p for p, examples in name_patterns.items() if len(examples) == 1]
        if len(single_patterns) > total_candidates * 0.3:  # More than 30% are unique
            failures.append({
                'type': 'root_fragmentation',
                'pattern': 'unique_patterns',
                'count': len(single_patterns),
                'examples': single_patterns[:10],
                'description': f"Too many unique patterns ({len(single_patterns)}), possible root fragmentation",
                'severity': 'medium'
            })
        
        return failures
    
    def _find_distance_failures(self) -> List[Dict[str, Any]]:
        """Find distance-related issues."""
        failures = []
        
        if 'dist_km' not in self.data.candidates.columns:
            return failures
        
        distances = self.data.candidates['dist_km'].dropna()
        
        if len(distances) > 0:
            # Find very distant candidates
            far_threshold = distances.quantile(0.9)  # Top 10% of distances
            
            if far_threshold > 5.0:  # More than 5km from rivers
                far_candidates = self.data.candidates[self.data.candidates['dist_km'] > far_threshold]
                
                failures.append({
                    'type': 'excessive_distance',
                    'pattern': 'far_from_rivers',
                    'count': len(far_candidates),
                    'threshold': far_threshold,
                    'examples': far_candidates['name'].head(5).tolist() if 'name' in far_candidates.columns else [],
                    'description': f"{len(far_candidates)} candidates >{far_threshold:.1f}km from rivers",
                    'severity': 'medium' if far_threshold < 10 else 'high'
                })
        
        return failures
    
    def _cluster_failures(self, failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group similar failures into clusters."""
        if not failures:
            return []
        
        # Simple clustering by failure type
        clusters = defaultdict(list)
        
        for failure in failures:
            failure_type = failure.get('type', 'unknown')
            clusters[failure_type].append(failure)
        
        # Convert to cluster format
        result_clusters = []
        for cluster_type, cluster_failures in clusters.items():
            total_count = sum(f.get('count', 1) for f in cluster_failures)
            max_severity = max((f.get('severity', 'low') for f in cluster_failures), 
                             key=lambda x: {'low': 1, 'medium': 2, 'high': 3}[x])
            
            result_clusters.append({
                'cluster_id': cluster_type,
                'type': cluster_type,
                'failures': cluster_failures,
                'total_count': total_count,
                'severity': max_severity,
                'description': f"Cluster of {len(cluster_failures)} {cluster_type} issues"
            })
        
        return result_clusters
    
    def _analyze_cluster_with_llm(self, cluster: Dict[str, Any]) -> Optional[FailureCluster]:
        """Analyze a failure cluster using LLM."""
        try:
            # Prepare context for LLM
            cluster_type = cluster.get('type', 'unknown')
            failures = cluster.get('failures', [])
            
            # Build analysis prompt
            prompt = self._build_analysis_prompt(cluster_type, failures)
            
            # Responses APIでの実行
            full_input = f"{self._get_system_prompt()}\n\n{prompt}"
            response = self.client.responses.create(
                model="o3-pro",
                input=full_input
            )
            
            # レスポンス内容を取得
            if hasattr(response, 'output_text'):
                response_content = response.output_text
            elif hasattr(response, 'output') and hasattr(response.output, 'text'):
                response_content = response.output.text
            elif hasattr(response, 'content'):
                response_content = response.content
            else:
                response_content = ""
            
            # Parse response
            analysis = response_content
            
            # Extract structured information
            root_cause, suggested_fix = self._parse_llm_response(analysis)
            
            return FailureCluster(
                cluster_id=cluster['cluster_id'],
                description=cluster['description'],
                pattern=cluster_type,
                examples=[f for f in failures[:3]],  # Top 3 examples
                count=cluster.get('total_count', len(failures)),
                severity=cluster.get('severity', 'medium'),
                root_cause=root_cause,
                suggested_fix=suggested_fix
            )
        
        except Exception as e:
            print(f"Warning: LLM analysis failed for cluster {cluster.get('cluster_id', 'unknown')}: {e}")
            return None
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM analysis."""
        return """あなたは地理空間解析と考古学遺跡検出システムの専門家です。
        
あなたのタスクは、地名ベースの考古学遺跡検出パイプラインにおける失敗パターンの分析です。
このシステムは水関連の地名を抽出し、古代集落遺跡の可能性を特定します。

以下を提供してください：
1. 失敗パターンの根本原因分析
2. 具体的な修正提案または改善案
3. 影響度に基づく優先度評価

実用的で実現可能な解決策に焦点を当ててください。"""
    
    def _build_analysis_prompt(self, cluster_type: str, failures: List[Dict[str, Any]]) -> str:
        """Build analysis prompt for a specific cluster."""
        prompt = f"""## 失敗クラスター分析

**クラスタータイプ**: {cluster_type}
**問題数**: {len(failures)}

**失敗詳細**:
"""
        
        for i, failure in enumerate(failures[:5], 1):  # Show top 5 failures
            prompt += f"\n{i}. {failure.get('description', 'No description')}"
            if 'examples' in failure and failure['examples']:
                examples = failure['examples'][:3]  # Top 3 examples
                prompt += f"\n   Examples: {', '.join(map(str, examples))}"
            if 'count' in failure:
                prompt += f"\n   Count: {failure['count']}"
        
        prompt += f"""

**コンテキスト**: これは以下の処理を行うパイプラインです：
1. OpenStreetMapから水関連の地名を抽出
2. 現在の河川までの距離を計算
3. 水域出現頻度を解析
4. 古代河川跡と集落の可能性を特定

**質問**: この失敗パターンの根本原因は何で、どのように修正すべきでしょうか？

分析して以下を提供してください：
1. **根本原因**: なぜこのパターンが発生しているのか？
2. **修正提案**: 具体的なパラメータ調整またはプロセス改善
3. **優先度**: 影響度に基づくHigh/Medium/Low"""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> tuple[str, str]:
        """Parse LLM response to extract root cause and fix."""
        lines = response.split('\n')
        
        root_cause = ""
        suggested_fix = ""
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if 'root cause' in line.lower():
                current_section = 'root_cause'
                if ':' in line:
                    root_cause = line.split(':', 1)[1].strip()
            elif 'suggested fix' in line.lower() or 'fix' in line.lower():
                current_section = 'suggested_fix'
                if ':' in line:
                    suggested_fix = line.split(':', 1)[1].strip()
            elif current_section == 'root_cause' and line:
                root_cause += " " + line
            elif current_section == 'suggested_fix' and line:
                suggested_fix += " " + line
        
        # Fallback: use entire response
        if not root_cause and not suggested_fix:
            parts = response.split('\n\n')
            if len(parts) >= 2:
                root_cause = parts[0].strip()
                suggested_fix = parts[1].strip()
            else:
                root_cause = response.strip()
                suggested_fix = "パイプラインパラメータの見直しと調整"
        
        return root_cause.strip(), suggested_fix.strip()