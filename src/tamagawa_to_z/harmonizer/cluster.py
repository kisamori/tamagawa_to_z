# DEMO FILE: トポニムクラスタリングモジュール

"""
cluster: 地名（トポニム）のクラスタリングモジュール

このモジュールは、埋め込みベクトルに基づいて地名をクラスタリングし、
水関連度を計算するための機能を提供します。
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Tuple, Optional, Union, Any


def calculate_water_scores(toponyms: List[str], 
                          embeddings: np.ndarray,
                          water_seeds: List[str],
                          seed_embeddings: Optional[np.ndarray] = None,
                          k: int = 3) -> np.ndarray:
    """水関連度スコアを計算する
    
    Parameters
    ----------
    toponyms : List[str]
        地名のリスト
    embeddings : np.ndarray
        地名の埋め込みベクトル
    water_seeds : List[str]
        水関連語彙のシードリスト
    seed_embeddings : Optional[np.ndarray], optional
        シード語彙の埋め込みベクトル
    k : int, optional
        最近傍の数
        
    Returns
    -------
    np.ndarray
        水関連度スコア
    """
    # シード語彙の埋め込みがない場合は生成
    if seed_embeddings is None:
        # 実際の実装では、モデルを使用して埋め込みを生成
        # from .embed import load_embedding_model, embed_toponyms
        # model = load_embedding_model("sentence-transformers/distiluse-base-multilingual-v2")
        # seed_embeddings = model.encode(water_seeds)
        
        # デモ用の簡易実装
        from .embed import mock_embeddings
        seed_embeddings = mock_embeddings(water_seeds, dim=embeddings.shape[1])
    
    # 最近傍探索器の初期化
    nn = NearestNeighbors(n_neighbors=min(k, len(seed_embeddings)), metric='cosine')
    nn.fit(seed_embeddings)
    
    # 各地名埋め込みに対して最近傍のシード語彙を探索
    distances, indices = nn.kneighbors(embeddings)
    
    # 類似度スコアの計算（コサイン距離を類似度に変換）
    similarity_scores = 1 - distances.mean(axis=1)
    
    return similarity_scores


def cluster_toponyms(embeddings: np.ndarray, 
                     n_clusters: Optional[int] = None,
                     min_clusters: int = 2,
                     max_clusters: int = 10) -> Tuple[np.ndarray, int]:
    """地名をクラスタリングする
    
    Parameters
    ----------
    embeddings : np.ndarray
        地名の埋め込みベクトル
    n_clusters : Optional[int], optional
        クラスタ数（指定しない場合は自動決定）
    min_clusters : int, optional
        最小クラスタ数
    max_clusters : int, optional
        最大クラスタ数
        
    Returns
    -------
    Tuple[np.ndarray, int]
        クラスタラベルとクラスタ数
    """
    # クラスタ数が指定されていない場合は自動決定
    if n_clusters is None:
        # シルエットスコアに基づいてクラスタ数を決定
        best_score = -1
        best_n_clusters = min_clusters
        
        for n in range(min_clusters, min(max_clusters + 1, len(embeddings))):
            # KMeansクラスタリング
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # シルエットスコアの計算
            if len(np.unique(labels)) > 1:  # 複数のクラスタがある場合のみ
                score = silhouette_score(embeddings, labels)
                
                if score > best_score:
                    best_score = score
                    best_n_clusters = n
        
        n_clusters = best_n_clusters
    
    # KMeansクラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    return labels, n_clusters


def assign_cluster_types(toponyms: List[str], 
                        labels: np.ndarray,
                        water_scores: np.ndarray,
                        threshold: float = 0.5) -> Dict[int, str]:
    """クラスタにタイプを割り当てる
    
    Parameters
    ----------
    toponyms : List[str]
        地名のリスト
    labels : np.ndarray
        クラスタラベル
    water_scores : np.ndarray
        水関連度スコア
    threshold : float, optional
        水関連度の閾値
        
    Returns
    -------
    Dict[int, str]
        クラスタIDとタイプのマッピング
    """
    # クラスタごとの水関連度の平均を計算
    cluster_water_scores = {}
    for i in range(max(labels) + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_water_scores[i] = water_scores[cluster_indices].mean()
    
    # クラスタにタイプを割り当て
    cluster_types = {}
    for cluster_id, score in cluster_water_scores.items():
        if score >= threshold:
            cluster_types[cluster_id] = "water"
        else:
            cluster_types[cluster_id] = "non-water"
    
    return cluster_types


def create_cluster_summary(toponyms: List[str], 
                          labels: np.ndarray,
                          cluster_types: Dict[int, str]) -> Dict[int, Dict[str, Any]]:
    """クラスタの要約を作成する
    
    Parameters
    ----------
    toponyms : List[str]
        地名のリスト
    labels : np.ndarray
        クラスタラベル
    cluster_types : Dict[int, str]
        クラスタIDとタイプのマッピング
        
    Returns
    -------
    Dict[int, Dict[str, Any]]
        クラスタの要約
    """
    # クラスタの要約
    cluster_summary = {}
    
    for i in range(max(labels) + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_toponyms = [toponyms[idx] for idx in cluster_indices]
        
        cluster_summary[i] = {
            "type": cluster_types.get(i, "unknown"),
            "size": len(cluster_indices),
            "examples": cluster_toponyms[:5],  # 最初の5つの例
            "common_prefix": find_common_prefix(cluster_toponyms)
        }
    
    return cluster_summary


def find_common_prefix(toponyms: List[str]) -> str:
    """地名リストの共通接頭辞を見つける
    
    Parameters
    ----------
    toponyms : List[str]
        地名のリスト
        
    Returns
    -------
    str
        共通接頭辞
    """
    if not toponyms:
        return ""
    
    # 各地名の最初の単語を抽出
    prefixes = [name.split()[0] if name and ' ' in name else name for name in toponyms]
    
    # 最も頻度の高い接頭辞を返す
    from collections import Counter
    prefix_counts = Counter(prefixes)
    
    if not prefix_counts:
        return ""
    
    most_common = prefix_counts.most_common(1)[0]
    return most_common[0]
