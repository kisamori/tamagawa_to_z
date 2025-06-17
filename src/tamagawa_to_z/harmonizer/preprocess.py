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
from pathlib import Path


# S-1: 対象地域のBBox定義
ACRE_BBOX = box(-70.5, -11.5, -66.5, -8.5)   # lon_min, lat_min, lon_max, lat_max

def make_bbox_gdf():
    """Returns the BBox of Western Upper Madeira River, Acre State as a GeoDataFrame
    
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the BBox
    """
    return gpd.GeoDataFrame({"id": [1]}, geometry=[ACRE_BBOX], crs="EPSG:4326")


# S-2: 水場系トポニムの抽出
# 水関連キーワードの正規表現パターン（既存のOverpass API用）
KW = re.compile(r'(?i)igarap[eé]|igap[oó]|lagoa|baixio|porto|furo|paran[aá]')

# 拡張された水語彙パターン（Pyrosm用）
WATER_TOKENS_EXTENDED = re.compile(r'(?i)\b(' \
    r'igarap[eé]|igap[oó]|lagoa|baixio|furo|paran[aá]|yaku|aku' \
    r'|ygarapé|yaru|cam[aã]|charco|swamp|marsh|porto' \
    r')\b')

# 除外する水域タグ
EXCLUDE_WATER_TAGS = ["waterway", "natural", "water", "wetland", "riverbank"]

# 取得対象のname系カラム
NAME_COLS = ["name", "alt_name", "old_name", "loc_name"]

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
        地名データのリスト。エラー時は空リストを返す。
    """
    url = ("https://servicodados.ibge.gov.br/api/v3/bcgn/nomes?"
           f"latitude={bbox.bounds[1]}&longitude={bbox.bounds[0]}"
           f"&latitude2={bbox.bounds[3]}&longitude2={bbox.bounds[2]}"
           "&categoria=hidrografia&page=" + str(page))
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # HTTPエラーをチェック
        
        # レスポンスが空でないかチェック
        if not response.text.strip():
            print(f"警告: BNGB APIからの空のレスポンス（ページ {page}）")
            return []
        
        # JSONパースを試行
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"ネットワークエラー: {e}")
        return []
    except ValueError as e:  # JSON decode error
        print(f"JSONデコードエラー: {e}")
        print(f"レスポンス内容: {response.text[:200]}...")  # 最初の200文字だけ表示
        return []

def collect_names(bbox, use_pyrosm=False, pbf_path=None):
    """BNGBから水関連の地名を収集する
    
    Parameters
    ----------
    bbox : shapely.geometry.box
        検索範囲のバウンディングボックス
    use_pyrosm : bool, optional
        Pyrosmを使用してローカルPBFファイルから追加データを取得するかどうか
    pbf_path : str, optional
        PBFファイルのパス（use_pyrosm=Trueの場合のみ使用）
        
    Returns
    -------
    gpd.GeoDataFrame
        収集された地名データ。APIエラー時は空のGeoDataFrameを返す。
    """
    rec, page = [], 1
    max_retries = 3
    retry_count = 0
    
    try:
        while retry_count < max_retries:
            data = bngeb_fetch(bbox, page)
            if not data:
                # データがない場合はリトライするか終了
                if retry_count < max_retries - 1:
                    print(f"BNGB APIからデータを取得できませんでした。リトライ中... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    import time
                    time.sleep(2)  # 2秒待機してからリトライ
                else:
                    print("BNGB APIからのデータ取得を諦めます。")
                    break
            else:
                # データが取得できたらリトライカウントをリセット
                retry_count = 0
                for d in data:
                    if KW.search(d["nome"].lower()):
                        rec.append({
                            "name": d["nome"],
                            "geometry": Point(float(d["longitude"]), float(d["latitude"])),
                            "source": "bngb"
                        })
                page += 1
    except Exception as e:
        print(f"地名収集中にエラーが発生しました: {e}")
    
    # BNGBの結果をGeoDataFrameに変換
    bngb_gdf = gpd.GeoDataFrame(rec, crs="EPSG:4326") if rec else gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")
    
    # Pyrosmを使用する場合は追加データを取得
    if use_pyrosm:
        print("Pyrosmを使用してローカルPBFから追加データを取得中...")
        pyrosm_gdf = extract_acre_toponyms_pyrosm(bbox, pbf_path)
        
        # BNGBとPyrosmのデータをマージ
        if not bngb_gdf.empty and not pyrosm_gdf.empty:
            combined_gdf = pd.concat([bngb_gdf, pyrosm_gdf], ignore_index=True)
            combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry="geometry", crs="EPSG:4326")
            print(f"統合結果: BNGB({len(bngb_gdf)}件) + Pyrosm({len(pyrosm_gdf)}件) = {len(combined_gdf)}件")
            return combined_gdf
        elif not pyrosm_gdf.empty:
            print(f"Pyrosmのみから{len(pyrosm_gdf)}件のデータを取得")
            return pyrosm_gdf
        else:
            print("Pyrosmからはデータを取得できませんでした。")
    
    # 結果が空でも有効なGeoDataFrameを返す
    if bngb_gdf.empty:
        print("BNGBからの地名データは取得できませんでした。空のデータセットを返します。")
        return gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")
    
    return bngb_gdf

def collect_osm_names(bbox):
    """OpenStreetMapから水関連の地名を収集する
    
    Parameters
    ----------
    bbox : shapely.geometry.box
        検索範囲のバウンディングボックス
        
    Returns
    -------
    gpd.GeoDataFrame
        収集された地名データ。エラー時は空のGeoDataFrameを返す。
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
    
    try:
        # APIリクエスト
        response = requests.post(overpass_url, data={"data": query}, timeout=60)
        response.raise_for_status()  # HTTPエラーをチェック
        
        # レスポンスが空でないかチェック
        if not response.text.strip():
            print(f"警告: Overpass APIからの空のレスポンス")
            return gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")
        
        # JSONパースを試行
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
            
    except requests.exceptions.RequestException as e:
        print(f"Overpass APIネットワークエラー: {e}")
        return gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")
    except ValueError as e:  # JSON decode error
        print(f"Overpass API JSONデコードエラー: {e}")
        print(f"レスポンス内容: {response.text[:200]}...")  # 最初の200文字だけ表示
        return gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")
    except Exception as e:
        print(f"Overpass API予期しないエラー: {e}")
        return gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")


def _has_water_toponym(row, pattern):
    """行が水語彙を含むかチェックする
    
    Parameters
    ----------
    row : pd.Series
        データ行
    pattern : re.Pattern
        水語彙の正規表現パターン
        
    Returns
    -------
    bool
        水語彙を含む場合True
    """
    return any(pattern.search(str(row.get(col, ''))) for col in NAME_COLS if row.get(col))


def _filter_non_water_features(gdf):
    """現行水域タグを持つ地物を除外する
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        フィルタリング対象のGeoDataFrame
        
    Returns
    -------
    gpd.GeoDataFrame
        水域タグを持たない地物のみのGeoDataFrame
    """
    # 除外対象のタグが存在するかチェック
    existing_exclude_cols = [col for col in EXCLUDE_WATER_TAGS if col in gdf.columns]
    
    if not existing_exclude_cols:
        return gdf
    
    # いずれかの水域タグを持つ行を除外
    mask = ~gdf[existing_exclude_cols].notna().any(axis=1)
    return gdf[mask]


def extract_acre_toponyms_pyrosm(bbox, pbf_path=None):
    """PyrosmでローカルPBFファイルから水語彙地名を抽出する
    
    Parameters
    ----------
    bbox : shapely.geometry.box
        検索範囲のバウンディングボックス
    pbf_path : str, optional
        PBFファイルのパス。デフォルトは data/raw/osm/norte-latest.osm.pbf
        
    Returns
    -------
    gpd.GeoDataFrame
        収集された地名データ。エラー時は空のGeoDataFrameを返す。
    """
    try:
        # pyrosmのインポート（オプショナル）
        try:
            from pyrosm import OSM
        except ImportError:
            print("警告: pyrosmがインストールされていません。pip install pyrosm>=0.6.0 を実行してください。")
            return gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")
        
        # PBFファイルパスの設定
        if pbf_path is None:
            # プロジェクトルートからの相対パス
            current_file = Path(__file__).resolve()
            project_root = current_file.parents[4]  # src/tamagawa_to_z/harmonizer/preprocess.py から4階層上
            pbf_path = project_root / "data/raw/osm/norte-latest.osm.pbf"
        
        pbf_path = Path(pbf_path)
        if not pbf_path.exists():
            print(f"警告: PBFファイルが見つかりません: {pbf_path}")
            return gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")
        
        print(f"PBFファイルを読み込み中: {pbf_path}")
        
        # OSMデータの読み込み（PyrosmはListフォーマットのbounding_boxを期待）
        west, south, east, north = bbox.bounds
        osm = OSM(str(pbf_path), bounding_box=[west, south, east, north])
        print("OSMデータを解析中...")
        
        # 複数のカテゴリから地物を取得して統合
        gdfs = []
        
        # place (村、集落等)
        try:
            place_gdf = osm.get_data_by_custom_criteria(
                custom_filter={"place": True}
            )
            if not place_gdf.empty:
                gdfs.append(place_gdf)
                print(f"place: {len(place_gdf)}件")
        except Exception as e:
            print(f"place取得エラー: {e}")
        
        # landuse (土地利用)
        try:
            landuse_gdf = osm.get_data_by_custom_criteria(
                custom_filter={"landuse": True}
            )
            if not landuse_gdf.empty:
                gdfs.append(landuse_gdf)
                print(f"landuse: {len(landuse_gdf)}件")
        except Exception as e:
            print(f"landuse取得エラー: {e}")
        
        # man_made
        try:
            man_made_gdf = osm.get_data_by_custom_criteria(
                custom_filter={"man_made": True}
            )
            if not man_made_gdf.empty:
                gdfs.append(man_made_gdf)
                print(f"man_made: {len(man_made_gdf)}件")
        except Exception as e:
            print(f"man_made取得エラー: {e}")
        
        # highway
        try:
            highway_gdf = osm.get_data_by_custom_criteria(
                custom_filter={"highway": True}
            )
            if not highway_gdf.empty:
                gdfs.append(highway_gdf)
                print(f"highway: {len(highway_gdf)}件")
        except Exception as e:
            print(f"highway取得エラー: {e}")
        
        # 全部のデータを統合
        if not gdfs:
            print("どのカテゴリからもデータを取得できませんでした。")
            return gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")
        
        gdf = pd.concat(gdfs, ignore_index=True)
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
        
        if gdf.empty:
            print("指定した範囲から地物が見つかりません。")
            return gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")
        
        print(f"初期取得: {len(gdf)}件の地物")
        
        # 水語彙を含む地物のフィルタリング
        gdf = gdf[gdf.apply(lambda row: _has_water_toponym(row, WATER_TOKENS_EXTENDED), axis=1)]
        print(f"水語彙フィルタ後: {len(gdf)}件")
        
        if gdf.empty:
            print("水語彙を含む地物が見つかりません。")
            return gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")
        
        # 水域タグを持つ地物を除外
        gdf = _filter_non_water_features(gdf)
        print(f"非水域フィルタ後: {len(gdf)}件")
        
        if gdf.empty:
            print("水域以外の地物が見つかりません。")
            return gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")
        
        # 結果の整理
        records = []
        for _, row in gdf.iterrows():
            # 利用可能な名前を優先順位で取得
            name = None
            for col in NAME_COLS:
                if row.get(col) and pd.notna(row.get(col)):
                    name = str(row[col])
                    break
            
            if name:
                # ジオメトリの処理
                geom = row['geometry']
                if hasattr(geom, 'centroid'):
                    # Polygonの場合は中心点を使用
                    point_geom = geom.centroid
                else:
                    # Pointの場合はそのまま使用
                    point_geom = geom
                
                records.append({
                    "name": name,
                    "geometry": point_geom,
                    "source": "osm_pyrosm"
                })
        
        if records:
            result_gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
            print(f"最終結果: {len(result_gdf)}件の地名を取得")
            return result_gdf
        else:
            print("有効な地名が見つかりません。")
            return gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")
            
    except Exception as e:
        print(f"Pyrosm処理中にエラーが発生しました: {e}")
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
