"""
preprocess: 地名（トポニム）の前処理モジュール

このモジュールは、地名データの正規化や前処理のための機能を提供します。
S-1: 対象地域のBBox定義
S-2: 水場系トポニムの抽出
S-3: クレンジング & タイプ付け
"""

import re
import unidecode
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from typing import Optional
from pathlib import Path


# S-1: 対象地域のBBox定義
DEFAULT_BBOX = box(-70.5, -11.5, -66.5, -8.5)   # lon_min, lat_min, lon_max, lat_max

def make_bbox_gdf(
    lon_min: float = DEFAULT_BBOX.bounds[0],
    lat_min: float = DEFAULT_BBOX.bounds[1],
    lon_max: float = DEFAULT_BBOX.bounds[2],
    lat_max: float = DEFAULT_BBOX.bounds[3],
) -> gpd.GeoDataFrame:
    """指定した範囲のBBoxをGeoDataFrameとして返す

    Parameters
    ----------
    lon_min, lat_min, lon_max, lat_max : float, optional
        バウンディングボックスの四隅の座標。
        省略時はデフォルト地域 (``DEFAULT_BBOX``) を使用する。

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the BBox
    """

    bbox = box(lon_min, lat_min, lon_max, lat_max)
    return gpd.GeoDataFrame({"id": [1]}, geometry=[bbox], crs="EPSG:4326")


# S-2: 水場系トポニムの抽出

# 水語彙パターンは water_roots.csv から動的に生成されます

# 除外する水域タグ
EXCLUDE_WATER_TAGS = ["waterway", "natural", "water", "wetland", "riverbank"]

# 取得対象のname系カラム
NAME_COLS = ["name", "alt_name", "old_name", "loc_name"]





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


def extract_toponyms_pyrosm(bbox, pbf_path=None, regex=None, include_water_features=False, osm_keys=None):
    """PyrosmでローカルPBFファイルから水語彙地名を抽出する
    
    Parameters
    ----------
    bbox : shapely.geometry.box
        検索範囲のバウンディングボックス
    pbf_path : str, optional
        PBFファイルのパス。デフォルトは data/raw/osm/norte-latest.osm.pbf
    regex : re.Pattern, optional
        水語彙フィルタリング用の正規表現。デフォルトはWATER_TOKENS_EXTENDED
    include_water_features : bool, optional
        水域タグを持つ地物も含めるかどうか。デフォルトはFalse（除外）
    osm_keys : List[str], optional
        抽出対象のOSMキーのリスト。デフォルトは['place', 'landuse', 'man_made', 'highway']
        
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
        
        # OSMキーの設定（デフォルトは従来の4つのキー）
        if osm_keys is None:
            osm_keys = ['place', 'landuse', 'man_made', 'highway']
        
        # 複数のカテゴリから地物を取得して統合
        gdfs = []
        
        # 設定されたOSMキーから地物を動的に取得
        for key in osm_keys:
            try:
                key_gdf = osm.get_data_by_custom_criteria(
                    custom_filter={key: True}
                )
                if not key_gdf.empty:
                    gdfs.append(key_gdf)
                    print(f"{key}: {len(key_gdf)}件")
            except Exception as e:
                print(f"{key}取得エラー: {e}")
        
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
        
        # 水語彙フィルタリング用の正規表現を決定
        if regex is None:
            raise ValueError("水語彙Regexパターンが提供されていません。water_roots.csvから生成してください。")
        
        # 水語彙を含む地物のフィルタリング
        gdf = gdf[gdf.apply(lambda row: _has_water_toponym(row, regex), axis=1)]
        print(f"水語彙フィルタ後: {len(gdf)}件")
        
        if gdf.empty:
            print("水語彙を含む地物が見つかりません。")
            return gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")
        
        # 水域タグを持つ地物を除外（オプション）
        if not include_water_features:
            gdf = _filter_non_water_features(gdf)
            print(f"非水域フィルタ後: {len(gdf)}件")
            
            if gdf.empty:
                print("水域以外の地物が見つかりません。")
                return gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")
        else:
            print(f"水域タグ除外をスキップ: {len(gdf)}件")
        
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
    
    water_roots.csvから語根を動的に読み込んで判定を行う
    
    Parameters
    ----------
    name : str
        正規化された地名
        
    Returns
    -------
    str or None
        推定された水系タイプ
    """
    try:
        # 相対インポートで語根読み込み機能を使用
        from .llm_layer.root_io import load_roots
        
        # 水語彙語根を動的に取得
        roots_df = load_roots()
        root_list = roots_df["root"].dropna().tolist()
        
        # 各語根でマッチング確認
        for root in root_list:
            if root in name:
                return root
                
    except Exception as e:
        # CSVが読み込めない場合はログ出力してNoneを返す
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load water roots for type inference: {e}")
    
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

