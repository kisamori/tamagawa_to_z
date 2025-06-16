# DEMO FILE: トポニム埋め込みモジュール

"""
embed: 地名（トポニム）の埋め込みモジュール

このモジュールは、地名テキストを埋め込みベクトルに変換するための機能を提供します。
実際の実装では、Sentence Transformersなどの多言語モデルを使用します。
"""

import numpy as np
from typing import List, Optional, Union, Dict, Any


def mock_embeddings(texts: List[str], dim: int = 512, seed: Optional[int] = 42) -> np.ndarray:
    """デモ用の埋め込みベクトルを生成する
    
    実際の実装では、Sentence Transformersなどの多言語モデルを使用します。
    このデモ関数は、テキストの特徴に基づいて決定論的な埋め込みを生成します。
    
    Parameters
    ----------
    texts : List[str]
        埋め込むテキストのリスト
    dim : int, optional
        埋め込みベクトルの次元数
    seed : Optional[int], optional
        乱数シード
        
    Returns
    -------
    np.ndarray
        埋め込みベクトル（shape: (len(texts), dim)）
    """
    # 乱数シードを固定（再現性のため）
    np.random.seed(seed)
    
    # 埋め込みベクトルの初期化
    embeddings = np.zeros((len(texts), dim))
    
    # 各テキストに対して埋め込みを生成
    for i, text in enumerate(texts):
        # テキストの特徴に基づく決定論的な埋め込み
        
        # 1. テキストの長さに基づく基本ベクトル
        base_vector = np.random.normal(0, 1, dim)
        base_vector = base_vector / np.linalg.norm(base_vector)  # 正規化
        
        # 2. 単語の特徴に基づく修正
        words = text.split()
        
        # 水関連語彙のリスト
        water_words = ["rio", "igarape", "lago", "parana", "cachoeira", 
                      "corrego", "lagoa", "canal", "baia", "represa"]
        
        # 地形関連語彙のリスト
        terrain_words = ["serra", "monte", "pico", "vale", "planalto"]
        
        # 集落関連語彙のリスト
        settlement_words = ["cidade", "vila", "povoado", "aldeia", "comunidade"]
        
        # 特徴ベクトル
        feature_vector = np.zeros(dim)
        
        # 水関連語彙の影響
        for word in words:
            if word in water_words:
                # 水関連の特徴ベクトル
                water_vector = np.random.normal(0, 1, dim)
                water_vector = water_vector / np.linalg.norm(water_vector)  # 正規化
                feature_vector += 0.5 * water_vector
            
            elif word in terrain_words:
                # 地形関連の特徴ベクトル
                terrain_vector = np.random.normal(0, 1, dim)
                terrain_vector = terrain_vector / np.linalg.norm(terrain_vector)  # 正規化
                feature_vector += 0.3 * terrain_vector
            
            elif word in settlement_words:
                # 集落関連の特徴ベクトル
                settlement_vector = np.random.normal(0, 1, dim)
                settlement_vector = settlement_vector / np.linalg.norm(settlement_vector)  # 正規化
                feature_vector += 0.2 * settlement_vector
        
        # 特徴ベクトルの正規化
        if np.linalg.norm(feature_vector) > 0:
            feature_vector = feature_vector / np.linalg.norm(feature_vector)
        
        # 基本ベクトルと特徴ベクトルの組み合わせ
        combined_vector = 0.7 * base_vector + 0.3 * feature_vector
        
        # 正規化
        combined_vector = combined_vector / np.linalg.norm(combined_vector)
        
        # 埋め込みベクトルに設定
        embeddings[i] = combined_vector
    
    return embeddings


def load_embedding_model(model_name: str, cache_dir: Optional[str] = None) -> Any:
    """埋め込みモデルをロードする
    
    実際の実装では、Sentence Transformersなどの多言語モデルをロードします。
    
    Parameters
    ----------
    model_name : str
        モデル名
    cache_dir : Optional[str], optional
        キャッシュディレクトリ
        
    Returns
    -------
    Any
        埋め込みモデル
    """
    # 実際の実装では、以下のようにモデルをロードします
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer(model_name, cache_folder=cache_dir)
    # return model
    
    # デモ用のダミーモデル
    class DummyModel:
        def __init__(self, model_name: str):
            self.model_name = model_name
            self.embedding_dim = 512 if "base" in model_name else 768
        
        def encode(self, texts: List[str], **kwargs) -> np.ndarray:
            return mock_embeddings(texts, dim=self.embedding_dim)
    
    return DummyModel(model_name)


def embed_toponyms(texts: List[str], model: Any) -> np.ndarray:
    """地名テキストを埋め込みベクトルに変換する
    
    Parameters
    ----------
    texts : List[str]
        埋め込むテキストのリスト
    model : Any
        埋め込みモデル
        
    Returns
    -------
    np.ndarray
        埋め込みベクトル
    """
    # 実際の実装では、以下のようにモデルを使用します
    # embeddings = model.encode(texts, convert_to_numpy=True)
    # return embeddings
    
    # デモ用の埋め込み生成
    if hasattr(model, 'encode'):
        return model.encode(texts)
    else:
        return mock_embeddings(texts)


def save_embeddings(embeddings: np.ndarray, output_path: str) -> None:
    """埋め込みベクトルを保存する
    
    Parameters
    ----------
    embeddings : np.ndarray
        埋め込みベクトル
    output_path : str
        出力ファイルパス
    """
    np.save(output_path, embeddings)
    print(f"Embeddings saved to {output_path}")


def load_embeddings(input_path: str) -> np.ndarray:
    """埋め込みベクトルを読み込む
    
    Parameters
    ----------
    input_path : str
        入力ファイルパス
        
    Returns
    -------
    np.ndarray
        埋め込みベクトル
    """
    embeddings = np.load(input_path)
    print(f"Embeddings loaded from {input_path}")
    return embeddings
