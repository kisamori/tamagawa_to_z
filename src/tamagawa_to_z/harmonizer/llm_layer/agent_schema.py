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
                    "enum": ["por", "tup", "araw", "mixed", "unknown"],
                    "description": "言語系統（por=ポルトガル語、tup=トゥピ語系、araw=アラワク語、mixed=混在、unknown=不明）"
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

# 新語根提案関数のスキーマ
PROPOSE_ROOT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "propose_new_water_root",
        "description": """
        未知の水関連語根を分析し、新しい水語彙エントリーを提案します。
        
        分析観点：
        1. 語音的パターンの識別（音韻構造、音節数）
        2. 語源分析（ポルトガル語、トゥピ語、アラワク語等の系統）
        3. 語意推定（地形・水文特徴との関連）
        4. 地理的分布（同一語根の地域的広がり）
        5. 言語学的妥当性（既存語根との整合性）
        
        提案基準：
        - 複数の地名で同一語根が確認される（frequency >= 2）
        - 水関連の地形・地理的文脈で使用される
        - 既存語根と明確に区別される音韻的特徴を持つ
        - 言語学的に妥当な語源を持つ
        """,
        "parameters": {
            "type": "object", 
            "properties": {
                "root": {
                    "type": "string",
                    "description": "提案する新語根（例: camaa, yukaru, etc.）"
                },
                "lang": {
                    "type": "string",
                    "enum": ["por", "tup", "araw", "macro-je", "mixed", "unknown"],
                    "description": "推定言語系統"
                },
                "meaning_en": {
                    "type": "string",
                    "description": "推定意味（英語）"
                },
                "meaning_ja": {
                    "type": "string",
                    "description": "推定意味（日本語）"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "提案確信度（0.0-1.0）"
                },
                "frequency": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "対象データセット内での出現頻度"
                },
                "example_toponyms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "この語根を含む地名例"
                },
                "phonetic_pattern": {
                    "type": "string", 
                    "description": "音韻パターン（正規表現形式）"
                },
                "reasoning": {
                    "type": "string",
                    "description": "提案理由の詳細説明（語源分析、地理的文脈、言語学的根拠等）"
                }
            },
            "required": [
                "root",
                "lang", 
                "meaning_en",
                "meaning_ja",
                "confidence",
                "frequency",
                "example_toponyms",
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

重要：rootフィールドについて
- 水関連語根を文字列で返してください（例: "igarape", "lagoa", "rio"）
- 新しい語根を発見した場合も、その語根名を文字列で返してください
"""

# 新語根提案用プロンプトテンプレート
PROPOSE_ROOT_PROMPT_TEMPLATE = """
候補語根パターン分析対象:

{pattern_analysis}

上記の地名パターンから、新しい水関連語根の提案を行ってください。

分析データ:
- 共通パターン: {common_pattern}
- 出現頻度: {frequency}
- 地名例: {example_toponyms}
- 既存語根との比較: {existing_roots_comparison}

propose_new_water_root関数を呼び出して、以下を考慮した提案を返してください：

1. 音韻的一貫性: 同じ語根が複数の地名で使用されているか
2. 地理的文脈: 水関連の地形・地物との関連性
3. 言語系統: ポルトガル語・トゥピ語・その他言語系統の特徴
4. 既存語根との区別: 既知の語根と明確に異なるか
5. 語源的妥当性: 言語学的に合理的な語源を持つか

重要な判定基準:
- frequency >= 2 (複数地名で確認)
- 水関連の地理的文脈での使用
- 既存語根リストにない新規性
- 音韻的に一貫したパターン
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


def create_propose_root_prompt(
    pattern_analysis: str,
    common_pattern: str, 
    frequency: int,
    example_toponyms: List[str],
    existing_roots_comparison: str
) -> str:
    """
    新語根提案用のプロンプトを生成
    
    Args:
        pattern_analysis: パターン分析結果
        common_pattern: 共通パターン
        frequency: 出現頻度
        example_toponyms: 地名例
        existing_roots_comparison: 既存語根との比較
        
    Returns:
        新語根提案用プロンプト
    """
    return PROPOSE_ROOT_PROMPT_TEMPLATE.format(
        pattern_analysis=pattern_analysis,
        common_pattern=common_pattern,
        frequency=frequency,
        example_toponyms=", ".join(example_toponyms),
        existing_roots_comparison=existing_roots_comparison
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


def validate_propose_root_response(response: Dict[str, Any]) -> Dict[str, str]:
    """
    新語根提案のOpenAI関数呼び出しレスポンスを検証
    
    Args:
        response: OpenAI関数呼び出しレスポンス
        
    Returns:
        エラー辞書（エラーがない場合は空辞書）
    """
    errors = {}
    
    required_fields = ["root", "lang", "meaning_en", "meaning_ja", "confidence", "frequency", "example_toponyms", "reasoning"]
    for field in required_fields:
        if field not in response:
            errors[field] = f"Required field '{field}' is missing"
    
    # lang値の検証
    if "lang" in response:
        valid_langs = ["por", "tup", "araw", "macro-je", "mixed", "unknown"]
        if response["lang"] not in valid_langs:
            errors["lang"] = f"Invalid language: {response['lang']}. Must be one of {valid_langs}"
    
    # confidence値の検証
    if "confidence" in response:
        try:
            conf = float(response["confidence"])
            if not (0.0 <= conf <= 1.0):
                errors["confidence"] = f"Confidence must be between 0.0 and 1.0, got {conf}"
        except (ValueError, TypeError):
            errors["confidence"] = f"Confidence must be a number, got {type(response['confidence'])}"
    
    # frequency値の検証
    if "frequency" in response:
        try:
            freq = int(response["frequency"])
            if freq < 1:
                errors["frequency"] = f"Frequency must be >= 1, got {freq}"
        except (ValueError, TypeError):
            errors["frequency"] = f"Frequency must be an integer, got {type(response['frequency'])}"
    
    # example_toponyms値の検証
    if "example_toponyms" in response:
        if not isinstance(response["example_toponyms"], list):
            errors["example_toponyms"] = f"example_toponyms must be a list, got {type(response['example_toponyms'])}"
        elif len(response["example_toponyms"]) == 0:
            errors["example_toponyms"] = "example_toponyms must not be empty"
    
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

# 新語根提案用のスキーマ情報
PROPOSE_ROOT_SCHEMA_INFO = {
    "function_schema": PROPOSE_ROOT_SCHEMA,
    "system_prompt": SYSTEM_PROMPT,
    "user_prompt_template": PROPOSE_ROOT_PROMPT_TEMPLATE,
    "required_fields": ["root", "lang", "meaning_en", "meaning_ja", "confidence", "frequency", "example_toponyms", "reasoning"],
    "optional_fields": ["phonetic_pattern"],
    "valid_languages": ["por", "tup", "araw", "macro-je", "mixed", "unknown"]
}