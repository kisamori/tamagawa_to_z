"""Inspector-Validator Agent メイン処理モジュール

このモジュールは、多言語トポニム解析の結果を評価し、
改善提案を行うメイン処理を提供します。
"""

import json
import os
import pathlib
import uuid
from datetime import datetime
from typing import Dict, Optional, Any

import pandas as pd
import geopandas as gpd
import yaml
from jinja2 import Template
from openai import OpenAI

from .agent_schema import PROPOSE_SCHEMA, DIAGNOSE_SCHEMA, validate_action_params
from .metrics import calculate_all_metrics, analyze_spatial_distribution


class InspectorValidatorAgent:
    """Inspector-Validator Agent クラス
    
    候補データと既知遺跡データを分析し、改善提案を行います。
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """エージェントの初期化
        
        Parameters
        ----------
        api_key : Optional[str]
            OpenAI API キー（環境変数OPENAI_API_KEYからも読み込み可能）
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "o3"
        self.tools = [
            {"type": "function", "function": PROPOSE_SCHEMA},
            {"type": "function", "function": DIAGNOSE_SCHEMA},
        ]
        self.instructions = """あなたは多言語トポニム解析システムのInspector-Validator Agentです。

あなたの役割：
1. 候補抽出結果のメトリクス（Recall@K, mAP, workload等）を分析
2. 問題点を特定し、具体的な改善アクションを提案
3. アマゾン地域の地理的・言語的特性を考慮した専門的な評価

重要な観点：
- Recall@100が0.3未満の場合は「低リコール」として問題視
- Workloadが1000以上の場合は「高負荷」として効率化が必要
- 語根の多様性が低い場合は辞書の拡充が必要
- 空間的偏りがある場合は地理的バランスの調整が必要

提案可能なアクション：
1. set_param: パラメータ調整（距離閾値、水域頻度閾値等）
2. add_exclude_mask: 除外マスクの追加（都市部、保護区域等）
3. add_root_weight: 語根重み調整（igarapé、lagoa等の重要語彙）

必ず具体的で実行可能な改善案を1件提案してください。"""
    
    
    def _build_analysis_prompt(self, metrics: Dict[str, float], 
                              spatial_stats: Dict[str, float],
                              meta_info: Dict[str, Any]) -> str:
        """分析用プロンプトを構築する
        
        Parameters
        ----------
        metrics : Dict[str, float]
            計算されたメトリクス
        spatial_stats : Dict[str, float]
            空間分布統計
        meta_info : Dict[str, Any]
            メタ情報
            
        Returns
        -------
        str
            分析用プロンプト
        """
        prompt = f"""## 分析対象データ
実行ID: {meta_info.get('run_id', 'unknown')}
処理日時: {meta_info.get('timestamp', 'unknown')}
対象地域: {meta_info.get('region', 'アマゾン流域')}

## 評価メトリクス
"""
        
        # メトリクスの追加
        for key, value in metrics.items():
            if isinstance(value, float):
                prompt += f"- {key}: {value:.3f}\n"
            else:
                prompt += f"- {key}: {value}\n"
        
        # 空間統計の追加
        if spatial_stats:
            prompt += "\n## 空間分布統計\n"
            for key, value in spatial_stats.items():
                prompt += f"- {key}: {value:.4f}\n"
        
        prompt += """
## 分析タスク
上記のメトリクスを分析し、以下を実行してください：

1. diagnose_results関数で問題点を特定・分析
2. propose_action関数で具体的な改善アクションを1件提案

アマゾン地域の地理的・言語的特性を考慮し、実用的で効果的な改善提案を行ってください。
"""
        
        return prompt
    
    def analyze_and_propose(self, 
                           candidates_path: str,
                           known_sites_path: str,
                           meta_path: Optional[str] = None) -> Dict[str, Any]:
        """候補データを分析し、改善提案を行う
        
        Parameters
        ----------
        candidates_path : str
            候補データのCSVファイルパス
        known_sites_path : str
            既知遺跡データのファイルパス
        meta_path : Optional[str]
            メタ情報のYAMLファイルパス
            
        Returns
        -------
        Dict[str, Any]
            分析結果と改善提案
        """
        # データの読み込み
        candidates = pd.read_csv(candidates_path)
        # geometry列を文字列として明示的に扱う
        if 'geometry' in candidates.columns:
            candidates['geometry'] = candidates['geometry'].astype(str)
        known_sites = gpd.read_file(known_sites_path)
        
        # メタ情報の読み込み
        meta_info = {}
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_info = yaml.safe_load(f)
        
        # 実行IDの生成
        run_id = meta_info.get('run_id', str(uuid.uuid4())[:8])
        meta_info['run_id'] = run_id
        meta_info['timestamp'] = datetime.now().isoformat()
        
        # メトリクス計算
        metrics = calculate_all_metrics(candidates, known_sites)
        spatial_stats = analyze_spatial_distribution(candidates)
        
        # 分析プロンプトの構築
        prompt = self._build_analysis_prompt(metrics, spatial_stats, meta_info)
        
        # Responses API による分析
        thread = self.client.responses.threads.create()
        self.client.responses.threads.messages.create(
            thread_id=thread.id,
            role="system",
            content=self.instructions,
        )
        self.client.responses.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt,
        )

        # 実行と結果取得
        run = self.client.responses.threads.runs.create_and_poll(
            thread_id=thread.id,
            model=self.model,
            tools=self.tools,
            tool_choice="auto"
        )
        
        # 結果の処理
        diagnosis = None
        proposal = None
        
        if run.status == "requires_action":
            # Function callの結果を処理
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                if tool_call.function.name == "diagnose_results":
                    diagnosis = json.loads(tool_call.function.arguments)
                elif tool_call.function.name == "propose_action":
                    proposal = json.loads(tool_call.function.arguments)
        
        # 結果の統合
        results = {
            "meta_info": meta_info,
            "metrics": metrics,
            "spatial_stats": spatial_stats,
            "diagnosis": diagnosis,
            "proposal": proposal,
            "run_id": run_id,
            "timestamp": meta_info['timestamp']
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """分析結果を保存する
        
        Parameters
        ----------
        results : Dict[str, Any]
            分析結果
        output_dir : str
            出力ディレクトリ
        """
        # タイムスタンプ付きディレクトリの作成
        timestamp = results["timestamp"]
        # ISO形式のタイムスタンプをファイル名に適した形式に変換
        timestamp_str = timestamp.replace(":", "-").replace(".", "-").split("T")[0] + "_" + timestamp.split("T")[1].replace(":", "-").split(".")[0]
        
        timestamped_dir = os.path.join(output_dir, timestamp_str)
        pathlib.Path(timestamped_dir).mkdir(parents=True, exist_ok=True)
        
        run_id = results["run_id"]
        
        # YAML計画の保存
        if results.get("proposal"):
            plan_path = os.path.join(timestamped_dir, f"plan_{run_id}.yaml")
            with open(plan_path, 'w', encoding='utf-8') as f:
                yaml.dump(results["proposal"], f, default_flow_style=False, 
                         allow_unicode=True)
            print(f"改善計画を保存しました: {plan_path}")
        
        # Markdownレポートの生成と保存
        report_path = os.path.join(timestamped_dir, f"report_{run_id}.md")
        self._generate_markdown_report(results, report_path)
        print(f"分析レポートを保存しました: {report_path}")
        
        # JSON結果の保存（詳細データ）
        json_path = os.path.join(timestamped_dir, f"results_{run_id}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"詳細結果を保存しました: {json_path}")
        
        print(f"全ての結果を保存しました: {timestamped_dir}/")
    
    def _generate_markdown_report(self, results: Dict[str, Any], output_path: str):
        """Markdownレポートを生成する
        
        Parameters
        ----------
        results : Dict[str, Any]
            分析結果
        output_path : str
            出力ファイルパス
        """
        template_path = os.path.join(
            os.path.dirname(__file__), 
            "templates", 
            "report_md.jinja"
        )
        
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                template = Template(f.read())
            
            markdown_content = template.render(**results)
        else:
            # テンプレートが存在しない場合の簡易版
            markdown_content = self._generate_simple_report(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    
    def _generate_simple_report(self, results: Dict[str, Any]) -> str:
        """簡易的なMarkdownレポートを生成する
        
        Parameters
        ----------
        results : Dict[str, Any]
            分析結果
            
        Returns
        -------
        str
            Markdownレポート
        """
        report = f"""# Inspector-Validator Analysis Report

**実行ID**: {results['run_id']}  
**実行日時**: {results['timestamp']}

## 評価メトリクス

| 指標 | 値 |
|------|-----|
"""
        
        for key, value in results["metrics"].items():
            if isinstance(value, float):
                report += f"| {key} | {value:.3f} |\n"
            else:
                report += f"| {key} | {value} |\n"
        
        if results.get("diagnosis"):
            report += "\n## 診断結果\n\n"
            diagnosis = results["diagnosis"]
            
            if diagnosis.get("overall_assessment"):
                report += f"**総合評価**: {diagnosis['overall_assessment']}\n\n"
            
            if diagnosis.get("issues"):
                report += "### 特定された問題点\n\n"
                for issue in diagnosis["issues"]:
                    report += f"- **{issue['issue_type']}** ({issue['severity']}): {issue['description']}\n"
        
        if results.get("proposal"):
            report += "\n## 改善提案\n\n"
            proposal = results["proposal"]
            
            report += f"**アクション**: {proposal['action']}\n\n"
            report += f"**理由**: {proposal['rationale']}\n\n"
            
            if proposal.get("expected_improvement"):
                report += f"**期待効果**: {proposal['expected_improvement']}\n\n"
            
            report += "**パラメータ**:\n```yaml\n"
            report += yaml.dump(proposal.get("params", {}), default_flow_style=False)
            report += "```\n"
        
        return report


def run(candidates_path: str, 
        known_sites_path: str, 
        output_dir: str,
        meta_path: Optional[str] = None,
        dict_path: Optional[str] = None) -> Dict[str, Any]:
    """Inspector-Validator Agent を実行する
    
    Parameters
    ----------
    candidates_path : str
        候補データのCSVファイルパス
    known_sites_path : str
        既知遺跡データのファイルパス
    output_dir : str
        出力ディレクトリ
    meta_path : Optional[str]
        メタ情報のYAMLファイルパス
    dict_path : Optional[str]
        辞書データのCSVファイルパス（将来の拡張用）
        
    Returns
    -------
    Dict[str, Any]
        分析結果
    """
    # エージェントの作成
    agent = InspectorValidatorAgent()
    
    # 分析の実行
    results = agent.analyze_and_propose(
        candidates_path=candidates_path,
        known_sites_path=known_sites_path,
        meta_path=meta_path
    )
    
    # 結果の保存
    agent.save_results(results, output_dir)
    
    return results