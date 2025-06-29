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





def _has_toponym_by_category(row, regex_dict):
    """行がカテゴリ別の語彙を含むかチェックし、マッチしたカテゴリを返す
    
    Parameters
    ----------
    row : pd.Series
        データ行
    regex_dict : dict
        カテゴリ名をキー、正規表現パターンを値とする辞書
        
    Returns
    -------
    list
        マッチしたカテゴリのリスト
    """
    matched_categories = []
    for category, pattern in regex_dict.items():
        if any(pattern.search(str(row.get(col, ''))) for col in NAME_COLS if row.get(col)):
            matched_categories.append(category)
    return matched_categories


def _has_water_toponym(row, pattern):
    """行が水語彙を含むかチェックする（後方互換性のため保持）
    
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


def extract_toponyms_pyrosm(bbox, pbf_path=None, regex_dict=None, regex=None, include_water_features=False, osm_keys=None):
    """PyrosmでローカルPBFファイルからカテゴリ別語彙地名を抽出する
    
    Parameters
    ----------
    bbox : shapely.geometry.box
        検索範囲のバウンディングボックス
    pbf_path : str, optional
        PBFファイルのパス。デフォルトは data/raw/osm/norte-latest.osm.pbf
    regex_dict : dict, optional
        カテゴリ別正規表現辞書 {category: re.Pattern}
    regex : re.Pattern, optional
        水語彙フィルタリング用の正規表現（後方互換性のため保持）
    include_water_features : bool, optional
        水域タグを持つ地物も含めるかどうか。デフォルトはFalse（除外）
    osm_keys : List[str], optional
        抽出対象のOSMキーのリスト。デフォルトは['place', 'landuse', 'man_made', 'highway']
        
    Returns
    -------
    gpd.GeoDataFrame
        収集された地名データ（root_categoryカラム付き）。エラー時は空のGeoDataFrameを返す。
    """
    try:
        # pyrosmのインポート（オプショナル）
        try:
            from pyrosm import OSM
        except ImportError:
            print("警告: pyrosmがインストールされていません。pip install pyrosm>=0.6.0 を実行してください。")
            return gpd.GeoDataFrame([], columns=["name", "geometry", "source", "root_category"], crs="EPSG:4326")
        
        # PBFファイルパスの設定
        if pbf_path is None:
            # プロジェクトルートからの相対パス
            current_file = Path(__file__).resolve()
            project_root = current_file.parents[4]  # src/tamagawa_to_z/harmonizer/preprocess.py から4階層上
            pbf_path = project_root / "data/raw/osm/norte-latest.osm.pbf"
        
        pbf_path = Path(pbf_path)
        if not pbf_path.exists():
            print(f"警告: PBFファイルが見つかりません: {pbf_path}")
            return gpd.GeoDataFrame([], columns=["name", "geometry", "source", "root_category"], crs="EPSG:4326")
        
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
                if key_gdf is not None and not key_gdf.empty:
                    gdfs.append(key_gdf)
                    print(f"{key}: {len(key_gdf)}件")
            except Exception as e:
                print(f"{key}取得エラー: {e}")
        
        # 全部のデータを統合
        if not gdfs:
            print("どのカテゴリからもデータを取得できませんでした。")
            return gpd.GeoDataFrame([], columns=["name", "geometry", "source", "root_category"], crs="EPSG:4326")
        
        gdf = pd.concat(gdfs, ignore_index=True)
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
        
        if gdf.empty:
            print("指定した範囲から地物が見つかりません。")
            return gpd.GeoDataFrame([], columns=["name", "geometry", "source", "root_category"], crs="EPSG:4326")
        
        print(f"初期取得: {len(gdf)}件の地物")
        
        # デバッグ: データの構造を確認
        if not gdf.empty:
            available_name_cols = [col for col in NAME_COLS if col in gdf.columns]
            print(f"利用可能な名前列: {available_name_cols}")
            print(f"全ての列: {list(gdf.columns)}")
            
            # 名前データがある行の数を確認
            name_data_count = 0
            for col in available_name_cols:
                non_null_count = gdf[col].notna().sum()
                if non_null_count > 0:
                    name_data_count += non_null_count
                    print(f"  {col}: {non_null_count}件の非NULL値")
            print(f"名前データを持つ総エントリ数: {name_data_count}")
            
            # tagsカラムの調査
            if 'tags' in gdf.columns:
                print("tagsカラムの内容をサンプル調査:")
                for i, row in gdf.head(3).iterrows():
                    tags = row.get('tags', {})
                    if isinstance(tags, dict):
                        tag_keys = list(tags.keys())
                        print(f"  行{i}: tags keys = {tag_keys}")
                        # nameに関連するキーを探す
                        name_related = [k for k in tag_keys if 'name' in k.lower()]
                        if name_related:
                            print(f"    name関連キー: {name_related}")
                            for nk in name_related:
                                print(f"      {nk}: {tags[nk]}")
                    else:
                        print(f"  行{i}: tags = {tags} (型: {type(tags)})")
        
        # 語彙フィルタリング用の正規表現を決定
        if regex_dict is not None:
            # 新形式：カテゴリ別辞書
            print(f"カテゴリ別フィルタリングを実行: {list(regex_dict.keys())}")
            
            # 特別なケース: 'all'カテゴリが含まれる場合はフィルタリングをスキップ
            if 'all' in regex_dict:
                print("'all'カテゴリが検出されたため、フィルタリングをスキップします")
                gdf['root_category'] = 'all'
                print(f"フィルタリングスキップ後: {len(gdf)}件")
            else:
                # カテゴリ別マッチング
                gdf['matched_categories'] = gdf.apply(lambda row: _has_toponym_by_category(row, regex_dict), axis=1)
                
                # いずれかのカテゴリにマッチした地物のみを保持
                gdf = gdf[gdf['matched_categories'].apply(lambda x: len(x) > 0)]
                
                # プライマリカテゴリを設定（複数マッチの場合は最初のもの）
                gdf['root_category'] = gdf['matched_categories'].apply(lambda x: x[0] if x else None)
                
                print(f"カテゴリ別フィルタ後: {len(gdf)}件")
                
                # カテゴリ別統計
                if not gdf.empty:
                    category_counts = gdf['root_category'].value_counts()
                    print(f"カテゴリ別件数: {category_counts.to_dict()}")
        
        elif regex is not None:
            # 旧形式：単一パターン（後方互換性）
            print("水語彙フィルタリングを実行（後方互換性モード）")
            gdf = gdf[gdf.apply(lambda row: _has_water_toponym(row, regex), axis=1)]
            gdf['root_category'] = 'water'  # 水系カテゴリを設定
            print(f"水語彙フィルタ後: {len(gdf)}件")
        
        else:
            raise ValueError("regex_dict または regex パラメータが必要です。")
        
        if gdf.empty:
            print("語彙を含む地物が見つかりません。")
            return gpd.GeoDataFrame([], columns=["name", "geometry", "source", "root_category"], crs="EPSG:4326")
        
        # 水域タグを持つ地物を除外（オプション）
        if not include_water_features:
            gdf = _filter_non_water_features(gdf)
            print(f"非水域フィルタ後: {len(gdf)}件")
            
            if gdf.empty:
                print("水域以外の地物が見つかりません。")
                return gpd.GeoDataFrame([], columns=["name", "geometry", "source", "root_category"], crs="EPSG:4326")
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
            
            # nameが見つからない場合の代替手段（意味のある名前のみ）
            if not name:
                # 1. 通常の代替カラムをチェック
                meaningful_sources = {
                    'ref': lambda x: x if x and len(x) <= 15 and not x.lower() in ['yes', 'no', 'true', 'false'] else None,  # 道路番号など
                    'addr:street': lambda x: x,  # 住所の通り名
                    'addr:city': lambda x: x,    # 住所の市名
                    'official_name': lambda x: x, # 公式名
                    'short_name': lambda x: x,   # 短縮名
                    'local_name': lambda x: x,   # 地域名
                }
                
                for alt_col, validator in meaningful_sources.items():
                    if row.get(alt_col) and pd.notna(row.get(alt_col)):
                        alt_value = str(row[alt_col]).strip()
                        validated_name = validator(alt_value)
                        if validated_name and len(validated_name) > 1:
                            name = validated_name
                            break
                
                # 2. tagsカラムから名前を抽出
                if not name and row.get('tags') and isinstance(row['tags'], dict):
                    tags = row['tags']
                    # tagsの中から名前関連のキーを探す
                    tag_name_sources = ['name', 'name:ja', 'name:pt', 'name:en', 'alt_name', 'old_name', 'loc_name', 'ref']
                    for tag_key in tag_name_sources:
                        if tag_key in tags and tags[tag_key]:
                            tag_value = str(tags[tag_key]).strip()
                            if len(tag_value) > 1 and tag_value.lower() not in ['yes', 'no', 'true', 'false']:
                                name = tag_value
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
                    "source": "osm_pyrosm",
                    "root_category": row.get('root_category', 'unknown')
                })
        
        if records:
            result_gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
            print(f"最終結果: {len(result_gdf)}件の地名を取得")
            return result_gdf
        else:
            print("有効な地名が見つかりません。")
            return gpd.GeoDataFrame([], columns=["name", "geometry", "source", "root_category"], crs="EPSG:4326")
            
    except Exception as e:
        print(f"Pyrosm処理中にエラーが発生しました: {e}")
        return gpd.GeoDataFrame([], columns=["name", "geometry", "source", "root_category"], crs="EPSG:4326")


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
    """地名から語根タイプを推定する
    
    all_roots.csv優先で語根を動的に読み込んで判定を行う
    
    Parameters
    ----------
    name : str
        正規化された地名
        
    Returns
    -------
    str or None
        推定された語根タイプ（root名またはcategory名）
    """
    try:
        # 相対インポートで語根読み込み機能を使用
        from .llm_layer.root_io import load_roots
        
        # まずall_roots.csvから試行
        try:
            roots_df = load_roots(use_all_roots=True)
            has_category = 'category' in roots_df.columns
        except Exception:
            # all_roots.csvがない場合はwater_roots.csvにフォールバック
            roots_df = load_roots(use_all_roots=False)
            has_category = False
        
        root_list = roots_df["root"].dropna().tolist()
        
        # 各語根でマッチング確認
        for i, root in enumerate(root_list):
            if root in name:
                if has_category:
                    # all_roots.csvの場合はcategoryを返す
                    category = roots_df.iloc[i].get('category', root)
                    return category
                else:
                    # water_roots.csvの場合はrootを返す（従来通り）
                    return root
                
    except Exception as e:
        # CSVが読み込めない場合はログ出力してNoneを返す
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load roots for type inference: {e}")
    
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

