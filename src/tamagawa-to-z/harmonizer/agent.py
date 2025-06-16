"""
agent: LLM Agentによる候補地点評価モジュール

このモジュールは、OpenAI Agents SDKを使用して候補地点を評価するための機能を提供します。
S-5後半: "川が無いのに川名が残る"ポイント抽出（LLM Agent判定）
"""

import os
import json
import pandas as pd
import geopandas as gpd
from typing import List, Dict, Any, Optional, Union
from openai_agents import Agent, tool


@tool()
def decide_keep(name: str, dist_km: float, occ_pct: float) -> bool:
    """
    Decide if a toponym should be kept as 'ancient river trace' candidate.
    Heuristics: dist_km>3 & occ_pct<5 ⇒ likely; else unlikely.
    
    Parameters
    ----------
    name : str
        The name of the toponym
    dist_km : float
        Distance to nearest current river in kilometers
    occ_pct : float
        Water occurrence percentage from GSW
        
    Returns
    -------
    bool
        True if the toponym should be kept, False otherwise
    """
    return (dist_km > 3) and (occ_pct < 5)


def create_review_agent() -> Agent:
    """LLM Agentを作成する
    
    Returns
    -------
    Agent
        OpenAI Agents SDKのAgentオブジェクト
    """
    system_prompt = """
    You are a field archaeologist agent. Accept JSON records with keys
    {name, dist_km, occ_pct}. Return JSON {"keep": true|false}.
    Criteria: keep if dist_km>3 AND occ_pct<5. Be strict.
    """
    
    return Agent(
        llm="gpt-4o-mini",
        system_message=system_prompt,
        tools=[decide_keep],
    )


def filter_with_agent(df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """LLM Agentを使用して候補地点をフィルタリングする
    
    Parameters
    ----------
    df : Union[pd.DataFrame, gpd.GeoDataFrame]
        候補地点データ
        
    Returns
    -------
    gpd.GeoDataFrame
        フィルタリングされた候補地点データ
    """
    # Agentの作成
    review_agent = create_review_agent()
    
    # 結果を格納するリスト
    results = []
    
    # 各行を処理
    for _, row in df.iterrows():
        # 入力データの作成
        input_data = {
            "name": row["name"],
            "dist_km": float(row["dist_km"]),
            "occ_pct": float(row["occ_pct"])
        }
        
        # Agentの実行
        output = review_agent.run(json.dumps(input_data))
        
        # 結果の処理
        if output.get("keep", False):
            results.append(row)
    
    # 結果がない場合は空のGeoDataFrameを返す
    if not results:
        return gpd.GeoDataFrame([], columns=df.columns, crs=df.crs)
    
    # 結果をGeoDataFrameに変換
    if isinstance(df, gpd.GeoDataFrame):
        return gpd.GeoDataFrame(results, crs=df.crs)
    else:
        return gpd.GeoDataFrame(results)


def batch_filter_with_agent(df: Union[pd.DataFrame, gpd.GeoDataFrame], 
                            batch_size: int = 100) -> gpd.GeoDataFrame:
    """大規模データセットをバッチ処理する
    
    Parameters
    ----------
    df : Union[pd.DataFrame, gpd.GeoDataFrame]
        候補地点データ
    batch_size : int, optional
        バッチサイズ
        
    Returns
    -------
    gpd.GeoDataFrame
        フィルタリングされた候補地点データ
    """
    # データが空の場合は空のGeoDataFrameを返す
    if df.empty:
        return gpd.GeoDataFrame([], columns=df.columns, crs=getattr(df, "crs", None))
    
    # バッチ処理
    results = []
    total_rows = len(df)
    
    for i in range(0, total_rows, batch_size):
        # バッチの取得
        batch = df.iloc[i:min(i+batch_size, total_rows)]
        
        # バッチの処理
        batch_results = filter_with_agent(batch)
        
        # 結果の追加
        if not batch_results.empty:
            results.append(batch_results)
    
    # 結果がない場合は空のGeoDataFrameを返す
    if not results:
        return gpd.GeoDataFrame([], columns=df.columns, crs=getattr(df, "crs", None))
    
    # 結果の結合
    combined = pd.concat(results, ignore_index=True)
    
    # GeoDataFrameに変換
    if isinstance(df, gpd.GeoDataFrame):
        return gpd.GeoDataFrame(combined, crs=df.crs)
    else:
        return gpd.GeoDataFrame(combined)


def score_candidates(df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """候補地点にスコアを付ける
    
    Parameters
    ----------
    df : Union[pd.DataFrame, gpd.GeoDataFrame]
        候補地点データ
        
    Returns
    -------
    gpd.GeoDataFrame
        スコア付きの候補地点データ
    """
    # データのコピーを作成
    result_df = df.copy()
    
    # 必要なカラムの存在確認
    required_cols = ["dist_km", "occ_pct"]
    for col in required_cols:
        if col not in result_df.columns:
            raise ValueError(f"DataFrame must have a '{col}' column")
    
    # スコアの計算
    # 1. 距離スコア: 距離が遠いほど高スコア（最大1.0）
    result_df["dist_score"] = result_df["dist_km"].clip(upper=10) / 10
    
    # 2. 水域頻度スコア: 頻度が低いほど高スコア（最大1.0）
    result_df["occ_score"] = 1 - (result_df["occ_pct"].clip(upper=100) / 100)
    
    # 3. 総合スコア: 距離スコアと水域頻度スコアの加重平均
    result_df["total_score"] = 0.6 * result_df["dist_score"] + 0.4 * result_df["occ_score"]
    
    # スコアでソート
    result_df = result_df.sort_values("total_score", ascending=False)
    
    return result_df
