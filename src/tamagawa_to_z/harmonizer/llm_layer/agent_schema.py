"""
OpenAI Function Calling Schema for Toponym Harmonization

多言語トポニム正規化のためのOpenAI Function Callingスキーマ定義
"""

from typing import Dict, Any, List
import json

# トポニム判定関数のスキーマ
DECIDE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "decide_toponym_relation",
        "description": """
        与えられた未知のトポニムと類似候補リストを比較分析し、最適な正規化判定を行います。
        
        分析観点：
        1. 語音的類似性（音韻変化、方言差異）
        2. 語根の共通性（igarapé, lagoa, rio等の水関連語根）
        3. 地理的妥当性（同一地域での表記揺れ）
        4. 言語系統（ポルトガル語、土着語、混在形）
        5. 歴史的変遷（植民地時代の表記変化）
        
        判定基準：
        - same: 同一地名の表記揺れ（confidence > 0.8）
        - similar: 語根は共通だが別地名の可能性（0.3 < confidence < 0.8）  
        - different: 明確に異なる地名（confidence < 0.3）
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "canonical_id": {
                    "type": "string",
                    "description": "正規化ID（類似候補から選択、または新規生成）"
                },
                "canonical_name": {
                    "type": "string", 
                    "description": "正規化された地名（最も標準的な表記）"
                },
                "relation": {
                    "type": "string",
                    "enum": ["same", "similar", "different"],
                    "description": "未知トポニムと類似候補との関係性"
                },
                "root": {
                    "type": "string",
                    "description": "水関連語根（igarape, lagoa, rio, paranã, etc.）"
                },
                "lang": {
                    "type": "string",
                    "enum": ["por", "tup", "mixed", "unknown"],
                    "description": "言語系統（por=ポルトガル語、tup=トゥピ語系、mixed=混在、unknown=不明）"
                },
                "meaning_en": {
                    "type": "string",
                    "description": "英語での意味・語源説明"
                },
                "meaning_ja": {
                    "type": "string", 
                    "description": "日本語での意味・語源説明"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "判定確信度（0.0-1.0）"
                },
                "reasoning": {
                    "type": "string",
                    "description": "判定理由の詳細説明（音韻変化、地理的根拠、言語学的分析等）"
                }
            },
            "required": [
                "canonical_id",
                "canonical_name", 
                "relation",
                "confidence",
                "reasoning"
            ]
        }
    }
}

# トポニム分析システムプロンプト
SYSTEM_PROMPT = """
あなたはアマゾン流域の多言語トポニム（地名）解析の専門家です。

専門知識：
- ポルトガル語とトゥピ語系言語の地名学
- アマゾン流域の水文地理学
- ブラジル植民地時代の地名変遷史
- 音韻変化と方言差異のパターン

主要水関連語根の知識：
- igarapé/ygarapé: 小川、細流（トゥピ語 y-garapé）
- rio: 川（ポルトガル語）
- lagoa: 湖、湿地（ポルトガル語）
- paranã/paraná: 大河（トゥピ語）
- igarité: 運河状水路（トゥピ語）
- ressaca: 氾濫原、後背湿地（ポルトガル語）

分析時の注意事項：
1. 音韻変化：b/p, d/t, g/k の交替、鼻音化、母音変化
2. 表記揺れ：ç/s, lh/li, nh/ni, アクセント有無
3. 語順変化：形容詞+名詞 ⇔ 名詞+形容詞
4. 縮約形：de+o→do, em+o→no 等
5. 地域方言：北部と南部の発音差異

false positiveを避けるため、確信が持てない場合は保守的に「different」判定してください。
"""

# ユーザープロンプトテンプレート
USER_PROMPT_TEMPLATE = """
未知トポニム: {raw_name}

類似候補（embedding類似度順）:
{candidates_text}

この未知トポニムについて、上記類似候補との関係性を分析し、decide_toponym_relation関数を呼び出して判定結果を返してください。

分析観点：
1. 音韻的類似性の検討
2. 語根・語幹の共通性
3. 地理的妥当性（アマゾン流域の地名として）
4. 言語系統の識別
5. 歴史的変遷の可能性

特に水関連語根（igarapé系、rio系、lagoa系等）に注目して分析してください。
"""


def format_candidates_for_prompt(candidates: List[Dict[str, Any]]) -> str:
    """
    類似候補リストをプロンプト用テキストに整形
    
    Args:
        candidates: 類似候補のリスト
        
    Returns:
        プロンプト用整形テキスト
    """
    if not candidates:
        return "（類似候補なし）"
    
    lines = []
    for i, cand in enumerate(candidates, 1):
        canonical_name = cand.get('canonical_name', 'N/A')
        variant_name = cand.get('variant_name', 'N/A') 
        root = cand.get('root', 'N/A')
        lang = cand.get('lang', 'N/A')
        confidence = cand.get('confidence', 'N/A')
        meaning_en = cand.get('meaning_en', 'N/A')
        
        line = f"{i}. {canonical_name} (variant: {variant_name})"
        line += f" | root: {root} | lang: {lang} | conf: {confidence}"
        if meaning_en != 'N/A':
            line += f" | meaning: {meaning_en}"
        
        lines.append(line)
    
    return "\n".join(lines)


def create_user_prompt(raw_name: str, candidates: List[Dict[str, Any]]) -> str:
    """
    ユーザープロンプトを生成
    
    Args:
        raw_name: 未知トポニム名
        candidates: 類似候補のリスト
        
    Returns:
        ユーザープロンプト
    """
    candidates_text = format_candidates_for_prompt(candidates)
    
    return USER_PROMPT_TEMPLATE.format(
        raw_name=raw_name,
        candidates_text=candidates_text
    )


def validate_response(response: Dict[str, Any]) -> Dict[str, str]:
    """
    OpenAI関数呼び出しレスポンスを検証
    
    Args:
        response: OpenAI関数呼び出しレスポンス
        
    Returns:
        エラー辞書（エラーがない場合は空辞書）
    """
    errors = {}
    
    required_fields = ["canonical_id", "canonical_name", "relation", "confidence", "reasoning"]
    for field in required_fields:
        if field not in response:
            errors[field] = f"Required field '{field}' is missing"
    
    # relation値の検証
    if "relation" in response:
        valid_relations = ["same", "similar", "different"]
        if response["relation"] not in valid_relations:
            errors["relation"] = f"Invalid relation: {response['relation']}. Must be one of {valid_relations}"
    
    # confidence値の検証
    if "confidence" in response:
        try:
            conf = float(response["confidence"])
            if not (0.0 <= conf <= 1.0):
                errors["confidence"] = f"Confidence must be between 0.0 and 1.0, got {conf}"
        except (ValueError, TypeError):
            errors["confidence"] = f"Confidence must be a number, got {type(response['confidence'])}"
    
    # lang値の検証（存在する場合）
    if "lang" in response and response["lang"]:
        valid_langs = ["por", "tup", "mixed", "unknown"]
        if response["lang"] not in valid_langs:
            errors["lang"] = f"Invalid language: {response['lang']}. Must be one of {valid_langs}"
    
    return errors


# エクスポート用のスキーマ情報
SCHEMA_INFO = {
    "function_schema": DECIDE_SCHEMA,
    "system_prompt": SYSTEM_PROMPT,
    "user_prompt_template": USER_PROMPT_TEMPLATE,
    "required_fields": ["canonical_id", "canonical_name", "relation", "confidence", "reasoning"],
    "optional_fields": ["root", "lang", "meaning_en", "meaning_ja"],
    "valid_relations": ["same", "similar", "different"],
    "valid_languages": ["por", "tup", "mixed", "unknown"]
}