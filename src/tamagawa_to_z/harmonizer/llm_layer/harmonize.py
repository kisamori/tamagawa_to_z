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
    PROPOSE_ROOT_SCHEMA,
    SYSTEM_PROMPT, 
    create_user_prompt,
    create_propose_root_prompt,
    validate_response,
    validate_propose_root_response
)
from .root_io import append_roots, format_root_for_csv, validate_root_entry

logger = logging.getLogger(__name__)


class ToponymHarmonizer:
    """
    多言語トポニム正規化のメインクラス
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "o3",
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
        logger.info(f"🤖 LLM Analysis: decide_toponym for '{raw_name}'")
        logger.info(f"   📋 Similar candidates: {len(candidates)}")
        if candidates:
            for i, cand in enumerate(candidates[:3], 1):  # 上位3件のみ表示
                logger.info(f"      {i}. {cand.get('canonical_name', 'N/A')} (similarity: {cand.get('similarity_score', 'N/A'):.3f})")
        
        user_prompt = create_user_prompt(raw_name, candidates)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                logger.info(f"   🌐 Calling OpenAI API (attempt {attempt + 1}/{self.max_retries})...")
                
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
                    raw_arguments = response.choices[0].message.tool_calls[0].function.arguments
                    logger.debug(f"   🔍 Raw LLM arguments: {raw_arguments}")
                    
                    try:
                        function_result = json.loads(raw_arguments)
                    except json.JSONDecodeError as e:
                        logger.error(f"   ❌ JSON parse error: {e}")
                        logger.error(f"   📄 Raw response: {raw_arguments}")
                        raise ValueError(f"Invalid JSON in LLM response: {e}")
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
                
                # 詳細結果ログ
                relation = function_result.get('relation', 'unknown')
                confidence = function_result.get('confidence', 0)
                logger.info(f"   ✅ LLM Decision: {relation} (confidence: {confidence:.3f})")
                
                # root情報の表示
                if function_result.get("root"):
                    logger.info(f"   🔍 Root detected: {function_result['root']}")
                
                logger.info(f"   ⏱️  API call completed in {duration:.2f}s")
                
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
        save_intermediate: bool = True,
        include_candidates: bool = True
    ) -> pd.DataFrame:
        """
        GeoDataFrameにLLMタグを付与し、辞書を更新する
        
        Args:
            gdf: 入力GeoDataFrame
            name_column: 地名列の名前
            batch_size: バッチサイズ（中間保存用）
            save_intermediate: 中間結果を保存するかどうか
            include_candidates: 候補地名を類似度検索対象に含めるかどうか
            
        Returns:
            LLMタグ付きGeoDataFrame
        """
        if name_column not in gdf.columns:
            raise ValueError(f"Column '{name_column}' not found in GeoDataFrame")
        
        logger.info(f"Starting LLM tagging for {len(gdf)} toponyms")
        
        # 一意な地名を取得
        unique_names = gdf[name_column].dropna().unique()
        logger.info(f"🔍 LLM同一判定の進捗: {len(unique_names)}件の候補があります")
        
        # 候補地名を類似度検索対象に含める
        if include_candidates:
            logger.info("🔧 候補地名を類似度検索インデックスに追加中...")
            candidate_names = unique_names.tolist()
            self.embedding.add_candidates_to_index(candidate_names)
            logger.info("✅ 候補地名をインデックスに追加完了")
        
        # 結果格納用
        harmonization_results = []
        new_root_entries = []
        
        # バッチ処理
        for i, name in enumerate(unique_names):
            try:
                logger.info(f"🔍 LLM同一判定の進捗: {i + 1}/{len(unique_names)}件進行中 - 処理中: '{name}'")
                result = self.harmonize_single(name)
                harmonization_results.append(result)
                
                # 新語根発見機能は一時的に無効化（スキーマを単純化したため）
                # 将来的には root が文字列として返された場合の新語根発見ロジックを実装
                if isinstance(result.get("root"), dict):
                    logger.info(f"🔍 Root object detected (simplified schema does not support this)")
                elif result.get("root") and isinstance(result.get("root"), str):
                    logger.info(f"🔍 Root string detected: {result['root']}")
                
                # 進捗ログ (10件ごと)
                if (i + 1) % 10 == 0:
                    logger.info(f"🔍 LLM同一判定の進捗: {i + 1}/{len(unique_names)}件完了")
                
                # 中間保存
                if save_intermediate and (i + 1) % batch_size == 0:
                    self._save_intermediate_results(harmonization_results, new_root_entries)
                    new_root_entries = []  # リセット
                
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
        
        # 語根辞書に追加
        if new_root_entries:
            logger.info(f"📝 Adding {len(new_root_entries)} new root entries to water_roots.csv...")
            for entry in new_root_entries:
                logger.info(f"   Adding: {entry['root']} ({entry.get('lang', 'N/A')}) -> {entry.get('regex_token', 'N/A')}")
            
            roots_df = pd.DataFrame(new_root_entries)
            append_roots(roots_df)
            logger.info(f"   ✅ Successfully appended new roots to CSV")
            logger.info(f"   🔄 Next run will use updated regex patterns automatically")
        
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
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]], root_entries: List[Dict[str, Any]] = None) -> None:
        """中間結果を保存"""
        if not results:
            return
        
        results_df = pd.DataFrame(results)
        
        # variant_id列が必要な場合は追加（新規エントリー用のダミーID）
        if 'variant_id' not in results_df.columns:
            results_df['variant_id'] = ''  # 空文字、append_entriesで正しいIDが割り当てられる
        
        append_entries(results_df)
        logger.info(f"Saved intermediate results: {len(results)} entries")
        
        # 語根エントリーも保存
        if root_entries:
            roots_df = pd.DataFrame(root_entries)
            append_roots(roots_df)
            logger.info(f"Saved intermediate root entries: {len(root_entries)} entries")
    
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
    
    def propose_new_root(
        self,
        candidate_toponyms: List[str],
        existing_roots: Optional[List[str]] = None,
        min_frequency: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        新しい水関連語根を提案する
        
        Args:
            candidate_toponyms: 候補地名のリスト
            existing_roots: 既存語根リスト（Noneなら自動取得）
            min_frequency: 最小出現頻度
            
        Returns:
            提案された新語根情報、または提案なしの場合はNone
        """
        if len(candidate_toponyms) < min_frequency:
            logger.warning(f"候補地名が最小頻度({min_frequency})を満たしません: {len(candidate_toponyms)}")
            return None
        
        logger.info(f"🔍 新語根提案分析: {len(candidate_toponyms)}件の地名を解析中...")
        
        # 既存語根の取得
        if existing_roots is None:
            try:
                from .root_io import load_water_roots
                water_roots_df = load_water_roots()
                existing_roots = water_roots_df['root'].tolist() if not water_roots_df.empty else []
            except Exception as e:
                logger.warning(f"既存語根の読み込みに失敗: {e}")
                existing_roots = []
        
        # パターン分析の実行
        pattern_analysis = self._analyze_toponym_patterns(candidate_toponyms)
        common_pattern = self._extract_common_pattern(candidate_toponyms)
        
        # 既存語根との比較
        existing_roots_comparison = self._compare_with_existing_roots(
            common_pattern, existing_roots
        )
        
        # LLM呼び出し用プロンプト作成
        prompt = create_propose_root_prompt(
            pattern_analysis=pattern_analysis,
            common_pattern=common_pattern,
            frequency=len(candidate_toponyms),
            example_toponyms=candidate_toponyms[:5],  # 上位5件のみ
            existing_roots_comparison=existing_roots_comparison
        )
        
        # OpenAI API呼び出し
        try:
            response = self._call_openai_propose_root(prompt)
            
            # 基本的な品質チェック
            if response.get('confidence', 0) < 0.3:
                logger.warning(f"提案確信度が低すぎます: {response.get('confidence')}")
                return None
            
            if response.get('frequency', 0) < min_frequency:
                logger.warning(f"頻度が不足: {response.get('frequency')} < {min_frequency}")
                return None
            
            logger.info(f"✅ 新語根提案成功: {response.get('root')} (confidence: {response.get('confidence'):.3f})")
            return response
            
        except Exception as e:
            logger.error(f"新語根提案で失敗: {e}")
            return None
    
    def _analyze_toponym_patterns(self, toponyms: List[str]) -> str:
        """地名パターンを分析"""
        analysis_lines = []
        analysis_lines.append(f"対象地名数: {len(toponyms)}")
        analysis_lines.append(f"地名例: {', '.join(toponyms[:3])}")
        
        # 共通プレフィックス・サフィックス分析
        if len(toponyms) >= 2:
            prefixes = set()
            suffixes = set()
            
            for toponym in toponyms:
                words = toponym.lower().split()
                if words:
                    prefixes.add(words[0])
                    suffixes.add(words[-1])
            
            common_prefixes = [p for p in prefixes if sum(1 for t in toponyms if t.lower().startswith(p)) >= 2]
            common_suffixes = [s for s in suffixes if sum(1 for t in toponyms if t.lower().endswith(s)) >= 2]
            
            if common_prefixes:
                analysis_lines.append(f"共通プレフィックス: {', '.join(common_prefixes)}")
            if common_suffixes:
                analysis_lines.append(f"共通サフィックス: {', '.join(common_suffixes)}")
        
        return "\n".join(analysis_lines)
    
    def _extract_common_pattern(self, toponyms: List[str]) -> str:
        """共通パターンを抽出"""
        if not toponyms:
            return "パターンなし"
        
        # 最も頻繁に現れる単語を探す
        word_counts = {}
        for toponym in toponyms:
            words = toponym.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # 2回以上現れる単語を頻度順にソート
        common_words = [(word, count) for word, count in word_counts.items() if count >= 2]
        common_words.sort(key=lambda x: x[1], reverse=True)
        
        if common_words:
            return common_words[0][0]  # 最も頻度の高い単語
        
        return "共通パターン未検出"
    
    def _compare_with_existing_roots(self, pattern: str, existing_roots: List[str]) -> str:
        """既存語根との比較"""
        if not existing_roots:
            return f"'{pattern}' は既存語根リストにありません（新規語根の可能性）"
        
        # 完全一致チェック
        if pattern in existing_roots:
            return f"'{pattern}' は既存語根と完全一致します"
        
        # 部分一致チェック
        partial_matches = [root for root in existing_roots if pattern in root or root in pattern]
        
        if partial_matches:
            return f"'{pattern}' は以下の既存語根と部分一致: {', '.join(partial_matches)}"
        
        return f"'{pattern}' は既存語根と一致せず（新規語根の可能性）"
    
    def _call_openai_propose_root(self, prompt: str) -> Dict[str, Any]:
        """
        新語根提案用のOpenAI Function Callingを実行
        
        Args:
            prompt: 新語根提案用プロンプト
            
        Returns:
            LLM応答結果
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                logger.info(f"   🌐 Calling OpenAI API for root proposal (attempt {attempt + 1}/{self.max_retries})...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=[PROPOSE_ROOT_SCHEMA],
                    tool_choice={"type": "function", "function": {"name": "propose_new_water_root"}},
                    timeout=self.timeout,
                    temperature=0.1
                )
                
                duration = time.time() - start_time
                self.stats['api_call_duration'].append(duration)
                
                # Tool call結果を取得
                if response.choices[0].message.tool_calls:
                    raw_arguments = response.choices[0].message.tool_calls[0].function.arguments
                    logger.debug(f"   🔍 Raw LLM arguments: {raw_arguments}")
                    
                    try:
                        function_result = json.loads(raw_arguments)
                    except json.JSONDecodeError as e:
                        logger.error(f"   ❌ JSON parse error: {e}")
                        raise ValueError(f"Invalid JSON in LLM response: {e}")
                else:
                    raise ValueError("No tool call in response")
                
                # レスポンス検証
                errors = validate_propose_root_response(function_result)
                if errors:
                    logger.warning(f"Validation errors in LLM response: {errors}")
                    if any(field in errors for field in ["root", "confidence", "frequency"]):
                        raise ValueError(f"Missing critical fields: {errors}")
                
                self.stats['successful_calls'] += 1
                logger.info(f"   ⏱️  Root proposal API call completed in {duration:.2f}s")
                
                return function_result
                    
            except Exception as e:
                logger.warning(f"OpenAI API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt == self.max_retries - 1:
                    self.stats['failed_calls'] += 1
                    logger.error(f"Failed to get root proposal after {self.max_retries} attempts")
                    raise
                
                wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                time.sleep(wait_time)
        
        raise RuntimeError("Unexpected end of retry loop")