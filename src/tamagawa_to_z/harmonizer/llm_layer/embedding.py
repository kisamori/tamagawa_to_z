"""
Toponym Embedding and Similarity Search

Sentence-TransformersとFAISSを使用した多言語トポニムの
ベクトル化と類似検索機能を提供します。
"""

import numpy as np
import pandas as pd
import logging
import pickle
import pathlib
from typing import List, Optional, Tuple, Dict
import unidecode
import re

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

DEPS_AVAILABLE = SENTENCE_TRANSFORMERS_AVAILABLE and SKLEARN_AVAILABLE

logger = logging.getLogger(__name__)


class ToponymEmbedding:
    """
    多言語トポニムのEmbeddingと類似検索を行うクラス
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/distiluse-base-multilingual-cased-v2",
        use_faiss: bool = True,
        cache_dir: Optional[pathlib.Path] = None
    ):
        """
        初期化
        
        Args:
            model_name: Sentence-Transformersモデル名
            use_faiss: FAISSを使用するかどうか
            cache_dir: キャッシュディレクトリ
        """
        if not DEPS_AVAILABLE:
            missing = []
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                missing.append("sentence-transformers")
            if not SKLEARN_AVAILABLE:
                missing.append("scikit-learn")
            raise ImportError(f"Required dependencies not available. Install: {', '.join(missing)}")
        
        self.model_name = model_name
        self.use_faiss = use_faiss
        self.cache_dir = cache_dir or pathlib.Path(__file__).parent.parent.parent.parent.parent / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # モデル初期化
        self.model = None
        self.index = None
        self.variants = []
        self.embeddings = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Sentence-Transformersモデルを読み込む"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def normalize_text(self, text: str) -> str:
        """
        テキストを正規化する
        
        Args:
            text: 正規化するテキスト
            
        Returns:
            正規化されたテキスト
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Unicode正規化とアクセント除去
        normalized = unidecode.unidecode(text.lower())
        
        # 特殊文字を空白に置換
        normalized = re.sub(r'[^a-z0-9\s\-]', ' ', normalized)
        
        # 複数の空白を単一に
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def build_index(self, variants: List[str]) -> None:
        """
        バリアントリストからEmbeddingインデックスを構築する
        
        Args:
            variants: バリアント名のリスト
        """
        if not variants:
            logger.warning("No variants provided for index building")
            return
        
        logger.info(f"Building embedding index for {len(variants)} variants")
        
        # テキスト正規化
        normalized_variants = [self.normalize_text(v) for v in variants]
        
        # 空のバリアントを除去
        valid_pairs = [(v, n) for v, n in zip(variants, normalized_variants) if n]
        if not valid_pairs:
            logger.warning("No valid variants after normalization")
            return
        
        self.variants, normalized_variants = zip(*valid_pairs)
        self.variants = list(self.variants)
        
        # Embedding生成
        try:
            logger.info("Generating embeddings...")
            self.embeddings = self.model.encode(
                normalized_variants,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            logger.info(f"Generated embeddings shape: {self.embeddings.shape}")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
        
        # インデックス構築
        self._build_search_index()
    
    def _build_search_index(self) -> None:
        """検索インデックスを構築する"""
        if self.embeddings is None:
            logger.error("No embeddings available for index building")
            return
        
        try:
            if self.use_faiss and FAISS_AVAILABLE and len(self.variants) > 100:  # FAISSは大量データで有効
                logger.info("Building FAISS index")
                dimension = self.embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
                
                # L2正規化（コサイン類似度用）
                embeddings_normalized = self.embeddings / np.linalg.norm(
                    self.embeddings, axis=1, keepdims=True
                )
                self.index.add(embeddings_normalized.astype(np.float32))
                
            else:
                logger.info("Building scikit-learn NearestNeighbors index")
                n_neighbors = min(10, len(self.variants))
                self.index = NearestNeighbors(
                    n_neighbors=n_neighbors,
                    metric='cosine'
                ).fit(self.embeddings)
            
            logger.info("Search index built successfully")
            
        except Exception as e:
            logger.error(f"Failed to build search index: {e}")
            raise
    
    def find_similar(
        self, 
        query: str, 
        k: int = 5,
        min_similarity: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        クエリに類似するバリアントを検索する
        
        Args:
            query: 検索クエリ
            k: 返す結果数
            min_similarity: 最小類似度閾値
            
        Returns:
            (バリアント名, 類似度スコア)のリスト
        """
        if not self.variants or self.index is None:
            logger.warning("No index available for similarity search")
            return []
        
        # クエリの正規化とEmbedding
        normalized_query = self.normalize_text(query)
        if not normalized_query:
            logger.warning(f"Query normalized to empty string: {query}")
            return []
        
        try:
            query_embedding = self.model.encode([normalized_query])
            
            if FAISS_AVAILABLE and isinstance(self.index, faiss.Index):
                # FAISS検索
                query_normalized = query_embedding / np.linalg.norm(query_embedding)
                similarities, indices = self.index.search(
                    query_normalized.astype(np.float32), k
                )
                
                results = []
                for sim, idx in zip(similarities[0], indices[0]):
                    if idx < len(self.variants) and sim >= min_similarity:
                        results.append((self.variants[idx], float(sim)))
                
            else:
                # scikit-learn検索
                distances, indices = self.index.kneighbors(query_embedding)
                
                results = []
                for dist, idx in zip(distances[0], indices[0]):
                    similarity = 1 - dist  # distance to similarity
                    if similarity >= min_similarity:
                        results.append((self.variants[idx], similarity))
            
            # 類似度でソート（降順）
            results.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Found {len(results)} similar variants for query: {query}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar variants: {e}")
            return []
    
    def save_index(self, filepath: pathlib.Path) -> None:
        """
        インデックスをファイルに保存する
        
        Args:
            filepath: 保存先ファイルパス
        """
        try:
            data = {
                'model_name': self.model_name,
                'variants': self.variants,
                'embeddings': self.embeddings,
                'use_faiss': self.use_faiss
            }
            
            if FAISS_AVAILABLE and isinstance(self.index, faiss.Index):
                # FAISSインデックスは別途保存
                faiss_path = filepath.with_suffix('.faiss')
                faiss.write_index(self.index, str(faiss_path))
                data['faiss_path'] = str(faiss_path)
            else:
                data['sklearn_index'] = self.index
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Index saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def load_index(self, filepath: pathlib.Path) -> None:
        """
        保存されたインデックスを読み込む
        
        Args:
            filepath: インデックスファイルパス
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # モデル名が異なる場合は警告
            if data['model_name'] != self.model_name:
                logger.warning(
                    f"Loaded index uses different model: {data['model_name']} "
                    f"vs current: {self.model_name}"
                )
            
            self.variants = data['variants']
            self.embeddings = data['embeddings']
            
            if 'faiss_path' in data:
                # FAISSインデックス読み込み
                self.index = faiss.read_index(data['faiss_path'])
            elif 'sklearn_index' in data:
                # scikit-learnインデックス読み込み
                self.index = data['sklearn_index']
            
            logger.info(f"Index loaded from: {filepath}")
            logger.info(f"Loaded {len(self.variants)} variants")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """
        インデックスの統計情報を取得する
        
        Returns:
            統計情報辞書
        """
        # インデックスタイプの判定
        if FAISS_AVAILABLE and hasattr(self, 'index') and self.index is not None:
            try:
                import faiss
                index_type = 'FAISS' if isinstance(self.index, faiss.Index) else 'scikit-learn'
            except ImportError:
                index_type = 'scikit-learn'
        else:
            index_type = 'scikit-learn' if hasattr(self, 'index') and self.index is not None else 'none'
        
        return {
            'model_name': self.model_name,
            'num_variants': len(self.variants),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'index_type': index_type,
            'use_faiss': self.use_faiss,
            'faiss_available': FAISS_AVAILABLE
        }