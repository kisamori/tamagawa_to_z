"""LLM語根重み推論機能 - OpenAI APIを使用して語根の重みを推論する."""

from __future__ import annotations

import json
import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

import diskcache
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

# グローバルキャッシュインスタンス
_CACHE: Optional[diskcache.Cache] = None


def _get_cache(cache_dir: str = ".llm_root_cache") -> diskcache.Cache:
    """キャッシュインスタンスを取得または作成する."""
    global _CACHE
    if _CACHE is None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(exist_ok=True)
        _CACHE = diskcache.Cache(str(cache_path))
        logger.info(f"Initialized LLM cache at {cache_path}")
    return _CACHE


def get_root_weights(
    context: Dict[str, Any],
    model: str = "o3",
    temperature: float = 0.0,
    max_tokens: int = 1000,
    cache_dir: str = ".llm_root_cache",
    system_prompt: Optional[str] = None,
    user_prompt_template: Optional[str] = None
) -> Dict[str, float]:
    """
    LLMから語根ウェイトを取得し、キャッシュする.
    
    Args:
        context: コンテキスト情報辞書
            - distance_km: 距離しきい値
            - occ_pct: 水域出現率しきい値  
            - toponym_stats: 地名統計
            - false_pos: 偽陽性例
        model: OpenAIモデル名
        temperature: 温度パラメータ
        max_tokens: 最大トークン数
        cache_dir: キャッシュディレクトリ
        system_prompt: システムプロンプト（Noneの場合はデフォルト）
        user_prompt_template: ユーザープロンプトテンプレート（Noneの場合はデフォルト）
        
    Returns:
        語根ウェイト辞書 {"root_name": weight, ...}
        
    Raises:
        ValueError: APIレスポンスが不正な場合
        openai.OpenAIError: API呼び出しエラー
    """
    # キャッシュキーを生成（決定論的）
    cache_key = _generate_cache_key(context, model, temperature)
    
    # キャッシュから取得を試行
    cache = _get_cache(cache_dir)
    if cache_key in cache:
        logger.debug(f"Cache hit for key: {cache_key[:16]}...")
        return cache[cache_key]
    
    logger.info(f"Querying LLM for root weights (model: {model})")
    
    # プロンプト作成
    sys_prompt = system_prompt or _get_default_system_prompt()
    user_prompt = _format_user_prompt(context, user_prompt_template)
    
    try:
        # OpenAI API呼び出し
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_TIRE5"))
        
        # Responses APIでの実行
        full_input = f"{sys_prompt}\n\n{user_prompt}"
        response = client.responses.create(
            model=model,
            input=full_input
        )
        
        # レスポンス解析 - Response APIのレスポンス処理
        content = ""
        if hasattr(response, 'output_text'):
            content = response.output_text
        elif hasattr(response, 'output') and hasattr(response.output, 'text'):
            content = response.output.text
        elif hasattr(response, 'content'):
            content = response.content
        elif hasattr(response, 'text'):
            content = response.text
        elif hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
        else:
            content = str(response)
        weights = _parse_llm_response(content)
        
        # キャッシュに保存
        cache[cache_key] = weights
        logger.info(f"Cached LLM response with {len(weights)} root weights")
        
        return weights
        
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in LLM query: {e}")
        raise


def clear_cache(cache_dir: str = ".llm_root_cache") -> int:
    """
    キャッシュをクリアする.
    
    Args:
        cache_dir: キャッシュディレクトリ
        
    Returns:
        削除されたエントリ数
    """
    global _CACHE
    
    cache = _get_cache(cache_dir)
    count = len(cache)
    cache.clear()
    
    # グローバルキャッシュもリセット
    _CACHE = None
    
    logger.info(f"Cleared {count} cache entries")
    return count


def get_cache_info(cache_dir: str = ".llm_root_cache") -> Dict[str, Any]:
    """
    キャッシュ情報を取得する.
    
    Args:
        cache_dir: キャッシュディレクトリ
        
    Returns:
        キャッシュ情報辞書
    """
    cache = _get_cache(cache_dir)
    
    info = {
        "cache_dir": cache_dir,
        "entry_count": len(cache),
        "disk_size_bytes": cache.volume(),
        "hit_rate": getattr(cache, 'stats', {}).get('hit_rate', 'N/A')
    }
    
    return info


def _generate_cache_key(
    context: Dict[str, Any], 
    model: str, 
    temperature: float
) -> str:
    """コンテキストからキャッシュキーを生成する."""
    # 履歴ハッシュを含む主要な要素でキーを生成
    key_components = {
        "D": context.get("distance_km"),
        "O": context.get("occ_pct"),
        "H": context.get("history_hash"),
        "model": model,
        "temperature": temperature
    }
    
    serialized = json.dumps(key_components, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def _get_default_system_prompt() -> str:
    """デフォルトのシステムプロンプトを返す."""
    return """You are a geo-linguistics expert for Amazonian archaeology. 
Your task is to analyze toponyms and assign appropriate weights to linguistic roots 
based on their archaeological significance.

You understand that:
- Water-related toponyms may indicate ancient riverbeds or settlements
- Different linguistic roots have varying archaeological relevance
- Distance from current rivers and water occurrence frequency are key factors
- Recent false positives should inform weight adjustments

Respond only with valid JSON."""


def _format_user_prompt(
    context: Dict[str, Any], 
    template: Optional[str] = None
) -> str:
    """ユーザープロンプトをフォーマットする."""
    if template is None:
        template = """Distance threshold = {distance_km} km, 
Water occurrence = {occ_pct} %.
Toponym stats: {toponym_stats}
Recent false positives: {false_pos}

Return JSON: {{"weights": {{"root": float, ...}}}} with values 0.1-1.0"""

    try:
        return template.format(**context)
    except KeyError as e:
        logger.warning(f"Missing context key {e}, using partial formatting")
        # 利用可能なキーのみでフォーマット
        available_context = {k: v for k, v in context.items() 
                           if f"{{{k}}}" in template}
        return template.format(**available_context)


def _parse_llm_response(content: str) -> Dict[str, float]:
    """
    LLMレスポンスをパースして語根ウェイト辞書を抽出する.
    
    Args:
        content: LLMの生レスポンス
        
    Returns:
        語根ウェイト辞書
        
    Raises:
        ValueError: JSONパースエラーまたは形式エラー
    """
    try:
        # JSONを抽出（マークダウンコードブロック等を除去）
        content_clean = content.strip()
        
        # ```json ... ``` の場合の処理
        if "```json" in content_clean:
            # ```jsonブロックから JSON部分を抽出
            json_start = content_clean.find("```json")
            if json_start != -1:
                json_start = content_clean.find("\n", json_start) + 1
                json_end = content_clean.find("```", json_start)
                if json_end != -1:
                    content_clean = content_clean[json_start:json_end].strip()
        elif "{" in content_clean and "}" in content_clean:
            # JSONブロックのみを抽出
            start = content_clean.find("{")
            end = content_clean.rfind("}") + 1
            content_clean = content_clean[start:end]
        
        # JSONパース
        response_data = json.loads(content_clean)
        
        # weights キーの存在確認
        if "weights" not in response_data:
            raise ValueError("Response missing 'weights' key")
            
        weights = response_data["weights"]
        
        # 値の型と範囲チェック
        validated_weights = {}
        for root, weight in weights.items():
            if not isinstance(weight, (int, float)):
                logger.warning(f"Invalid weight type for {root}: {type(weight)}")
                continue
                
            weight = float(weight)
            if not (0.1 <= weight <= 1.0):
                logger.warning(f"Weight out of range for {root}: {weight}")
                weight = max(0.1, min(1.0, weight))  # クランプ
                
            validated_weights[str(root)] = weight
            
        if not validated_weights:
            raise ValueError("No valid weights found in response")
            
        logger.debug(f"Parsed {len(validated_weights)} root weights")
        return validated_weights
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Raw content: {content}")
        raise ValueError(f"Invalid JSON response: {e}")
    except Exception as e:
        logger.error(f"Response parsing error: {e}")
        raise ValueError(f"Failed to parse LLM response: {e}")


def create_mock_weights(n_roots: int = 10) -> Dict[str, float]:
    """
    テスト用のモック重みを生成する.
    
    Args:
        n_roots: 生成する語根数
        
    Returns:
        モック語根ウェイト辞書
    """
    import random
    
    # 一般的なアマゾン地域の水関連語根
    sample_roots = [
        "igarape", "paranã", "paraná", "açu", "mirim", 
        "guaçu", "tinga", "preta", "cocha", "quebrada",
        "caño", "lago", "pozo", "agua", "rio"
    ]
    
    # ランダム選択と重み生成
    selected_roots = random.sample(sample_roots, min(n_roots, len(sample_roots)))
    weights = {root: round(random.uniform(0.1, 1.0), 2) for root in selected_roots}
    
    logger.info(f"Generated mock weights for {len(weights)} roots")
    return weights


if __name__ == "__main__":
    # テスト実行
    import os
    
    # テスト用コンテキスト
    test_context = {
        "distance_km": 2.5,
        "occ_pct": 5.0,
        "toponym_stats": {"igarape": 15, "paraná": 8, "lago": 3},
        "false_pos": ["Rio Falso", "Lago Seco", "Paranã Antiga"]
    }
    
    print("=== LLM Root Weights Test ===")
    
    # モック重みのテスト
    mock_weights = create_mock_weights(5)
    print(f"Mock weights: {mock_weights}")
    
    # APIキーの確認
    if os.getenv("OPENAI_API_KEY"):
        try:
            weights = get_root_weights(test_context)
            print(f"LLM weights: {weights}")
        except Exception as e:
            print(f"LLM query failed: {e}")
    else:
        print("OPENAI_API_KEY not set, skipping LLM test")
    
    # キャッシュ情報
    cache_info = get_cache_info()
    print(f"Cache info: {cache_info}")
    
    print("Test completed.")