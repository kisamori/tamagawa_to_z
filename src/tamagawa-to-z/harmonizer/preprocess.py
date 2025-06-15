# DEMO FILE: トポニム前処理モジュール

"""
preprocess: 地名（トポニム）の前処理モジュール

このモジュールは、地名データの正規化や前処理のための機能を提供します。
"""

import pandas as pd
import geopandas as gpd
import unidecode
import re
from typing import List, Union


def normalize_text(text: str) -> str:
    """テキストを正規化する
    
    1. 小文字化
    2. アクセント除去
    3. 特殊文字を空白に置換
    4. 前後の空白を削除
    
    Parameters
    ----------
    text : str
        正規化するテキスト
        
    Returns
    -------
    str
        正規化されたテキスト
    """
    # 小文字化
    text = text.lower()
    
    # アクセント除去
    text = unidecode.unidecode(text)
    
    # 英数字とスペース、ハイフン以外を空白に置換
    text = re.sub(r'[^a-z0-9\s-]', ' ', text)
    
    # 余分な空白を削除
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def normalize_toponyms(toponyms: Union[pd.DataFrame, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """地名データを正規化する
    
    Parameters
    ----------
    toponyms : Union[pd.DataFrame, gpd.GeoDataFrame]
        地名データ
        
    Returns
    -------
    gpd.GeoDataFrame
        正規化された地名データ
    """
    # コピーを作成
    processed = toponyms.copy()
    
    # 'name' カラムがない場合はエラー
    if 'name' not in processed.columns:
        raise ValueError("DataFrame must have a 'name' column")
    
    # 正規化を適用
    processed['normalized_name'] = processed['name'].apply(normalize_text)
    
    # 言語情報の追加（存在しない場合）
    if 'language' not in processed.columns:
        # 簡易的な言語推定（実際の実装ではより高度な方法を使用）
        processed['language'] = 'unknown'
    
    return processed


def extract_name_parts(toponyms: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """地名から構成要素を抽出する
    
    例: "Rio Amazonas" -> ["rio", "amazonas"]
    
    Parameters
    ----------
    toponyms : gpd.GeoDataFrame
        正規化された地名データ
        
    Returns
    -------
    gpd.GeoDataFrame
        構成要素が追加された地名データ
    """
    # コピーを作成
    processed = toponyms.copy()
    
    # 'normalized_name' カラムがない場合はエラー
    if 'normalized_name' not in processed.columns:
        raise ValueError("DataFrame must have a 'normalized_name' column")
    
    # 構成要素の抽出
    processed['name_parts'] = processed['normalized_name'].apply(lambda x: x.split())
    
    # 先頭要素（通常は地物タイプ）
    processed['prefix'] = processed['name_parts'].apply(lambda x: x[0] if len(x) > 0 else "")
    
    # 残りの要素（通常は固有名詞）
    processed['suffix'] = processed['name_parts'].apply(lambda x: " ".join(x[1:]) if len(x) > 1 else "")
    
    return processed


def filter_water_related(toponyms: gpd.GeoDataFrame, 
                         water_prefixes: List[str],
                         threshold: float = 0.5) -> gpd.GeoDataFrame:
    """水関連の地名をフィルタリングする
    
    Parameters
    ----------
    toponyms : gpd.GeoDataFrame
        地名データ
    water_prefixes : List[str]
        水関連の接頭辞リスト
    threshold : float, optional
        水関連度の閾値
        
    Returns
    -------
    gpd.GeoDataFrame
        水関連の地名のみを含むデータ
    """
    # 'prefix' カラムがない場合は抽出
    if 'prefix' not in toponyms.columns:
        toponyms = extract_name_parts(toponyms)
    
    # 水関連度の計算
    def calculate_water_score(row):
        # 接頭辞が水関連リストに含まれる場合
        if row['prefix'] in water_prefixes:
            return 1.0
        
        # 特徴タイプによる判定
        if 'feature_type' in row and row['feature_type'] in ['river', 'stream', 'lake', 'channel', 'waterfall']:
            return 0.9
        
        # デフォルト（水関連度低）
        return 0.1
    
    # 水関連度の計算
    toponyms['water_score'] = toponyms.apply(calculate_water_score, axis=1)
    
    # 閾値以上の水関連度を持つ地名をフィルタリング
    water_related = toponyms[toponyms['water_score'] >= threshold].copy()
    
    return water_related
