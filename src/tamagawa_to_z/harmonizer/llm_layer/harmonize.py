"""
Toponym Harmonization with LLM Integration

LLMを使用した多言語トポニム正規化とタグ付けの統合システム
"""

import pandas as pd
import numpy as np
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
    load_dotenv()  # .envファイルから環境変数を読み込む
except ImportError:
    pass  # python-dotenvが利用できない場合はスキップ

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .dictionary_io import load_dict, append_entries, get_dict_stats
from .embedding import ToponymEmbedding
from .agent_schema import (
    DECIDE_SCHEMA, 
    SYSTEM_PROMPT, 
    create_user_prompt, 
    validate_response
)

logger = logging.getLogger(__name__)


class ToponymHarmonizer:
    """
    多言語トポニム正規化のメインクラス
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        embedding_model: str = "sentence-transformers/distiluse-base-multilingual-cased-v2",
        max_retries: int = 3,
        timeout: float = 30.0,
        cache_dir: Optional[Path] = None
    ):
        """
        初期化
        
        Args:
            openai_api_key: OpenAI APIキー（NoneならOPENAI_API_KEY環境変数を使用）
            model: 使用するOpenAIモデル
            embedding_model: Embedding用モデル
            max_retries: APIリトライ回数
            timeout: APIタイムアウト（秒）
            cache_dir: キャッシュディレクトリ
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install: openai>=1.0.0")
        
        # OpenAI client初期化
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. "
                "Set OPENAI_API_KEY environment variable or copy .env.example to .env and set your API key there"
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Embedding初期化
        self.embedding = ToponymEmbedding(
            model_name=embedding_model,
            cache_dir=cache_dir
        )
        
        # 統計情報
        self.stats = {
            'total_processed': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'api_call_duration': []
        }
        
        logger.info(f"ToponymHarmonizer initialized with model: {model}")
    
    def prime_index(
        self, 
        dict_df: Optional[pd.DataFrame] = None,
        rebuild_index: bool = False
    ) -> None:
        """
        既存辞書からEmbeddingインデックスを構築する
        
        Args:
            dict_df: 辞書DataFrame（Noneなら既存辞書を読み込み）
            rebuild_index: インデックスを強制再構築するかどうか
        """
        if dict_df is None:
            dict_df = load_dict()
        
        if dict_df.empty:
            logger.warning("Empty dictionary - no index to build")
            return
        
        # バリアント名を抽出
        variants = dict_df["variant_name"].dropna().unique().tolist()
        
        if not variants:
            logger.warning("No variant names found in dictionary")
            return
        
        logger.info(f"Building embedding index from {len(variants)} variants")
        self.embedding.build_index(variants)
        
        # 辞書統計情報をログ出力
        stats = get_dict_stats()
        logger.info(f"Dictionary stats: {stats}")
    
    def _call_openai_function(
        self, 
        raw_name: str, 
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        OpenAI Function Callingを実行する
        
        Args:
            raw_name: 未知トポニム名
            candidates: 類似候補リスト
            
        Returns:
            LLM応答結果
        """
        user_prompt = create_user_prompt(raw_name, candidates)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=[DECIDE_SCHEMA],
                    tool_choice={"type": "function", "function": {"name": "decide_toponym_relation"}},
                    timeout=self.timeout,
                    temperature=0.1  # 一貫性を重視
                )
                
                duration = time.time() - start_time
                self.stats['api_call_duration'].append(duration)
                
                # Tool call結果を取得
                if response.choices[0].message.tool_calls:
                    function_result = json.loads(
                        response.choices[0].message.tool_calls[0].function.arguments
                    )
                else:
                    raise ValueError("No tool call in response")
                
                # レスポンス検証
                errors = validate_response(function_result)
                if errors:
                    logger.warning(f"Validation errors in LLM response: {errors}")
                    # エラーがあっても処理続行（必須フィールドがあれば）
                    if not all(field in function_result for field in ["canonical_id", "canonical_name", "relation", "confidence"]):
                        raise ValueError(f"Missing required fields: {errors}")
                
                # 成功統計
                self.stats['successful_calls'] += 1
                
                logger.debug(f"LLM analysis for '{raw_name}': {function_result.get('relation', 'unknown')} (conf: {function_result.get('confidence', 0)})")
                
                return function_result
                    
            except Exception as e:
                logger.warning(f"OpenAI API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt == self.max_retries - 1:
                    self.stats['failed_calls'] += 1
                    logger.error(f"Failed to get LLM response for '{raw_name}' after {self.max_retries} attempts")
                    raise
                
                # 指数バックオフで待機
                wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                time.sleep(wait_time)
        
        raise RuntimeError("Unexpected end of retry loop")
    
    def _create_fallback_entry(
        self, 
        raw_name: str, 
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        LLM呼び出し失敗時のフォールバックエントリーを作成
        
        Args:
            raw_name: 未知トポニム名
            candidates: 類似候補
            
        Returns:
            フォールバックエントリー
        """
        # 一意IDを生成
        import uuid
        canonical_id = f"fallback_{uuid.uuid4().hex[:8]}"
        
        # 最も類似度の高い候補から語根を推定
        root = "unknown"
        lang = "unknown"
        
        if candidates:
            best_candidate = candidates[0]
            root = best_candidate.get('root', 'unknown')
            lang = best_candidate.get('lang', 'unknown')
        
        return {
            'canonical_id': canonical_id,
            'canonical_name': raw_name,  # 正規化できないのでそのまま
            'relation': 'different',
            'root': root,
            'lang': lang,
            'meaning_en': 'Unknown (fallback entry)',
            'meaning_ja': '不明（フォールバックエントリー）',
            'confidence': 0.1,  # 低い確信度
            'reasoning': 'LLM analysis failed - created fallback entry'
        }
    
    def harmonize_single(
        self, 
        raw_name: str,
        k_candidates: int = 5,
        use_fallback: bool = True
    ) -> Dict[str, Any]:
        """
        単一トポニムの正規化を実行
        
        Args:
            raw_name: 未知トポニム名
            k_candidates: 取得する類似候補数
            use_fallback: LLM失敗時にフォールバックエントリーを作成するか
            
        Returns:
            正規化結果
        """
        logger.debug(f"Harmonizing single toponym: {raw_name}")
        
        # 類似候補を検索
        similar_variants = self.embedding.find_similar(
            raw_name, 
            k=k_candidates,
            min_similarity=0.2
        )
        
        # 候補情報を辞書から取得
        candidates = []
        if similar_variants:
            dict_df = load_dict()
            for variant_name, similarity in similar_variants:
                candidate_rows = dict_df[dict_df["variant_name"] == variant_name]
                if not candidate_rows.empty:
                    candidate = candidate_rows.iloc[0].to_dict()
                    candidate['similarity_score'] = similarity
                    candidates.append(candidate)
        
        logger.debug(f"Found {len(candidates)} candidates for '{raw_name}'")
        
        # LLM分析実行
        try:
            result = self._call_openai_function(raw_name, candidates)
            result['variant_name'] = raw_name
            result['similarity_candidates'] = len(candidates)
            
        except Exception as e:
            if use_fallback:
                logger.warning(f"Using fallback for '{raw_name}': {e}")
                result = self._create_fallback_entry(raw_name, candidates)
                result['variant_name'] = raw_name
                result['similarity_candidates'] = len(candidates)
                result['error'] = str(e)
            else:
                raise
        
        self.stats['total_processed'] += 1
        
        return result
    
    def attach_llm_tags(
        self, 
        gdf: pd.DataFrame,
        name_column: str = "name",
        batch_size: int = 10,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        GeoDataFrameにLLMタグを付与し、辞書を更新する
        
        Args:
            gdf: 入力GeoDataFrame
            name_column: 地名列の名前
            batch_size: バッチサイズ（中間保存用）
            save_intermediate: 中間結果を保存するかどうか
            
        Returns:
            LLMタグ付きGeoDataFrame
        """
        if name_column not in gdf.columns:
            raise ValueError(f"Column '{name_column}' not found in GeoDataFrame")
        
        logger.info(f"Starting LLM tagging for {len(gdf)} toponyms")
        
        # 一意な地名を取得
        unique_names = gdf[name_column].dropna().unique()
        logger.info(f"Processing {len(unique_names)} unique toponyms")
        
        # 結果格納用
        harmonization_results = []
        
        # バッチ処理
        for i, name in enumerate(unique_names):
            try:
                result = self.harmonize_single(name)
                harmonization_results.append(result)
                
                # 進捗ログ
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(unique_names)} toponyms")
                
                # 中間保存
                if save_intermediate and (i + 1) % batch_size == 0:
                    self._save_intermediate_results(harmonization_results)
                
            except Exception as e:
                logger.error(f"Failed to process '{name}': {e}")
                continue
        
        if not harmonization_results:
            logger.warning("No toponyms were successfully processed")
            return gdf
        
        # 結果をDataFrameに変換
        results_df = pd.DataFrame(harmonization_results)
        
        # 辞書に追加
        logger.info("Updating toponym dictionary...")
        append_entries(results_df)
        
        # 元のGeoDataFrameにマージ
        gdf_tagged = gdf.merge(
            results_df,
            left_on=name_column,
            right_on="variant_name",
            how="left"
        )
        
        # 統計情報をログ出力
        self._log_processing_stats()
        
        logger.info(f"LLM tagging completed. Tagged {len(results_df)} unique toponyms")
        
        return gdf_tagged
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]]) -> None:
        """中間結果を保存"""
        if not results:
            return
        
        results_df = pd.DataFrame(results)
        append_entries(results_df)
        logger.info(f"Saved intermediate results: {len(results)} entries")
    
    def _log_processing_stats(self) -> None:
        """処理統計情報をログ出力"""
        stats = self.stats.copy()
        
        if stats['api_call_duration']:
            stats['avg_api_duration'] = np.mean(stats['api_call_duration'])
            stats['total_api_duration'] = np.sum(stats['api_call_duration'])
        
        success_rate = stats['successful_calls'] / max(stats['total_processed'], 1) * 100
        stats['success_rate_pct'] = success_rate
        
        logger.info(f"Processing statistics: {stats}")
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        stats = self.stats.copy()
        stats['embedding_stats'] = self.embedding.get_stats()
        stats['dictionary_stats'] = get_dict_stats()
        
        return stats