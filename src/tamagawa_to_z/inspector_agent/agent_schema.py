"""Inspector-Validator Agent スキーマ定義

OpenAI Function Callingで使用するスキーマを定義します。
"""

from typing import Dict, Any

# 改善アクション提案用のスキーマ
PROPOSE_SCHEMA = {
    "name": "propose_action",
    "description": "分析結果に基づいて改善アクションを1件提案する",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["set_param", "add_exclude_mask", "add_root_weight"],
                "description": "実行するアクションの種類"
            },
            "params": {
                "type": "object",
                "description": "アクションに必要なパラメータ",
                "properties": {
                    # set_param用
                    "param_name": {
                        "type": "string",
                        "description": "設定するパラメータ名"
                    },
                    "param_value": {
                        "description": "設定する値（数値、文字列、オブジェクト等）"
                    },
                    # add_exclude_mask用
                    "mask_type": {
                        "type": "string",
                        "description": "除外マスクの種類（urban, protected_area等）"
                    },
                    "mask_source": {
                        "type": "string", 
                        "description": "マスクデータのソース"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "閾値"
                    },
                    # add_root_weight用
                    "root": {
                        "type": "string",
                        "description": "重み付けする語根"
                    },
                    "weight": {
                        "type": "number",
                        "description": "設定する重み"
                    }
                }
            },
            "rationale": {
                "type": "string",
                "description": "提案理由の詳細説明"
            },
            "expected_improvement": {
                "type": "string",
                "description": "期待される改善効果"
            },
            "priority": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "改善の優先度"
            }
        },
        "required": ["action", "params", "rationale"]
    }
}

# 診断結果分析用のスキーマ
DIAGNOSE_SCHEMA = {
    "name": "diagnose_results",
    "description": "メトリクス結果を分析して問題点を診断する",
    "parameters": {
        "type": "object",
        "properties": {
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "issue_type": {
                            "type": "string",
                            "enum": ["low_recall", "high_workload", "spatial_bias", "root_imbalance"],
                            "description": "問題の種類"
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["critical", "major", "minor"],
                            "description": "問題の深刻度"
                        },
                        "description": {
                            "type": "string",
                            "description": "問題の詳細説明"
                        },
                        "metrics_evidence": {
                            "type": "object",
                            "description": "問題を示すメトリクス値"
                        }
                    },
                    "required": ["issue_type", "severity", "description"]
                },
                "description": "特定された問題点のリスト"
            },
            "overall_assessment": {
                "type": "string",
                "description": "全体的な評価とサマリー"
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "診断の信頼度"
            }
        },
        "required": ["issues", "overall_assessment"]
    }
}


def create_tools_config() -> list[Dict[str, Any]]:
    """OpenAI Assistants API用のツール設定を作成する
    
    Returns
    -------
    list[Dict[str, Any]]
        ツール設定のリスト
    """
    return [
        {
            "type": "function",
            "function": PROPOSE_SCHEMA
        },
        {
            "type": "function", 
            "function": DIAGNOSE_SCHEMA
        }
    ]


def validate_action_params(action: str, params: Dict[str, Any]) -> bool:
    """アクションパラメータの妥当性を検証する
    
    Parameters
    ----------
    action : str
        アクションの種類
    params : Dict[str, Any]
        パラメータ辞書
        
    Returns
    -------
    bool
        妥当性チェック結果
    """
    if action == "set_param":
        return "param_name" in params and "param_value" in params
    
    elif action == "add_exclude_mask":
        required = ["mask_type", "mask_source", "threshold"]
        return all(key in params for key in required)
    
    elif action == "add_root_weight":
        return "root" in params and "weight" in params
    
    return False


# プリセット改善アクション例
PRESET_ACTIONS = {
    "increase_distance_threshold": {
        "action": "set_param",
        "params": {
            "param_name": "dist_threshold_km",
            "param_value": 5.0
        },
        "rationale": "河川距離閾値を緩和して候補数を増加させる",
        "expected_improvement": "Recall向上、ただしWorkload増加のトレードオフ",
        "priority": "medium"
    },
    
    "add_urban_mask": {
        "action": "add_exclude_mask",
        "params": {
            "mask_type": "urban",
            "mask_source": "GHSL",
            "threshold": 0.7
        },
        "rationale": "都市部を除外して遺跡候補の精度を向上させる",
        "expected_improvement": "偽陽性の削減、Workload減少",
        "priority": "high"
    },
    
    "boost_igarape_weight": {
        "action": "add_root_weight",
        "params": {
            "root": "igarapé",
            "weight": 1.5
        },
        "rationale": "アマゾン地域で重要な水系語彙「igarapé」の重みを強化",
        "expected_improvement": "地域特有の水系地名の検出精度向上",
        "priority": "high"
    }
}