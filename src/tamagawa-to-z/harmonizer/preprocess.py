"""
preprocess: 地名（トポニム）の前処理モジュール

このモジュールは、地名データの正規化や前処理のための機能を提供します。
S-1: 対象地域のBBox定義
S-2: 水場系トポニムの抽出
S-3: クレンジング & タイプ付け
"""

import re
import requests
import unidecode
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point
from typing import List, Union, Dict, Any, Optional


# S-1: 対象地域のBBox定義
ACRE_BBOX = box(-70.5, -11.5, -66.5, -8.5)   # lon_min, lat_min, lon_max, lat_max

def make_bbox_gdf():
    """アクレ州マデイラ川上流西部のBBoxをGeoDataFrameとして返す
    
    Returns
    -------
    gpd.GeoDataFrame
        BBoxを含むGeoDataFrame
    """
    return gpd.GeoDataFrame({"id": [1]}, geometry=[ACRE_BBOX], crs="EPSG:4326")


# S-2: 水場系トポニムの抽出
# 水関連キーワードの正規表現パターン
KW = re.compile(r'(?i)igarap[eé]|igap[oó]|lagoa|baixio|porto|furo|paran[aá]')

def bngeb_fetch(bbox, page=1):
    """BNGB APIから地名データを取得する
    
    Parameters
    ----------
    bbox : shapely.geometry.box
        検索範囲のバウンディングボックス
    page : int, optional
        ページ番号
        
    Returns
    -------
    list
        地名データのリスト
    """
    url = ("https://servicodados.ibge.gov.br/api/v3/bcgn/nomes?"
           f"latitude={bbox.bounds[1]}&longitude={bbox.bounds[0]}"
           f"&latitude2={bbox.bounds[3]}&longitude2={bbox.bounds[2]}"
           "&categoria=hidrografia&page=" + str(page))
    return requests.get(url, timeout=30).json()

def collect_names(bbox):
    """BNGBから水関連の地名を収集する
    
    Parameters
    ----------
    bbox : shapely.geometry.box
        検索範囲のバウンディングボックス
        
    Returns
    -------
    gpd.GeoDataFrame
        収集された地名データ
    """
    rec, page = [], 1
    while True:
        data = bngeb_fetch(bbox, page)
        if not data: 
            break
        for d in data:
            if KW.search(d["nome"].lower()):
                rec.append({
                    "name": d["nome"],
                    "geometry": Point(float(d["longitude"]), float(d["latitude"])),
                    "source": "bngb"
                })
        page += 1
    return gpd.GeoDataFrame(rec, crs="EPSG:4326")

def collect_osm_names(bbox):
    """OpenStreetMapから水関連の地名を収集する
    
    Parameters
    ----------
    bbox : shapely.geometry.box
        検索範囲のバウンディングボックス
        
    Returns
    -------
    gpd.GeoDataFrame
        収集された地名データ
    """
    # Overpass APIのクエリ
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # バウンディングボックスの座標を取得
    south, west, north, east = bbox.bounds
    
    # 水関連の名前を持つノードを検索するクエリ
    query = f"""
    [out:json];
    (
      node["name"~"(?i)igarap[eé]|igap[oó]|lagoa|baixio|porto|furo|paran[aá]"]({south},{west},{north},{east});
      way["name"~"(?i)igarap[eé]|igap[oó]|lagoa|baixio|porto|furo|paran[aá]"]({south},{west},{north},{east});
      relation["name"~"(?i)igarap[eé]|igap[oó]|lagoa|baixio|porto|furo|paran[aá]"]({south},{west},{north},{east});
    );
    out center;
    """
    
    # APIリクエスト
    response = requests.post(overpass_url, data={"data": query}, timeout=60)
    data = response.json()
    
    # 結果の処理
    records = []
    for element in data.get("elements", []):
        # ノードの場合
        if element["type"] == "node":
            lat, lon = element["lat"], element["lon"]
        # ウェイまたはリレーションの場合（中心点を使用）
        else:
            lat, lon = element.get("center", {}).get("lat"), element.get("center", {}).get("lon")
        
        # 座標が取得できた場合のみ追加
        if lat and lon:
            records.append({
                "name": element.get("tags", {}).get("name", "Unknown"),
                "geometry": Point(lon, lat),
                "source": "osm"
            })
    
    # GeoDataFrameに変換
    if records:
        return gpd.GeoDataFrame(records, crs="EPSG:4326")
    else:
        # 空のGeoDataFrameを返す
        return gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")


# S-3: クレンジング & タイプ付け
def normalize_name(s: str) -> str:
    """地名を正規化する
    
    1. 小文字化
    2. アクセント除去
    3. 特殊文字を空白に置換
    4. 前後の空白を削除
    
    Parameters
    ----------
    s : str
        正規化する地名
        
    Returns
    -------
    str
        正規化された地名
    """
    # 小文字化
    s = s.lower()
    
    # アクセント除去
    s = unidecode.unidecode(s)
    
    # 英数字とスペース、ハイフン以外を空白に置換
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    
    # 余分な空白を削除
    s = re.sub(r"\s+", " ", s).strip()
    
    return s

def infer_type(name: str) -> Optional[str]:
    """地名から水系タイプを推定する
    
    Parameters
    ----------
    name : str
        正規化された地名
        
    Returns
    -------
    str or None
        推定された水系タイプ
    """
    for kw in ["igarape", "igapo", "lagoa", "baixio", "porto", "furo", "parana"]:
        if kw in name:
            return kw
    return None

def process_toponyms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """地名データを処理する
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        地名データ
        
    Returns
    -------
    gpd.GeoDataFrame
        処理された地名データ
    """
    # コピーを作成
    processed = gdf.copy()
    
    # 正規化
    processed["normalized_name"] = processed["name"].apply(normalize_name)
    
    # タイプ推定
    processed["type"] = processed["normalized_name"].apply(infer_type)
    
    return processed

def merge_toponyms(bngb_gdf: gpd.GeoDataFrame, osm_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """BNGBとOSMの地名データをマージする
    
    Parameters
    ----------
    bngb_gdf : gpd.GeoDataFrame
        BNGBから収集された地名データ
    osm_gdf : gpd.GeoDataFrame
        OSMから収集された地名データ
        
    Returns
    -------
    gpd.GeoDataFrame
        マージされた地名データ
    """
    # 空のDataFrameチェック
    if bngb_gdf.empty and osm_gdf.empty:
        return gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")
    elif bngb_gdf.empty:
        return osm_gdf
    elif osm_gdf.empty:
        return bngb_gdf
    
    # マージ
    merged = pd.concat([bngb_gdf, osm_gdf], ignore_index=True)
    
    # GeoDataFrameに変換
    if not isinstance(merged, gpd.GeoDataFrame):
        merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
    
    return merged
