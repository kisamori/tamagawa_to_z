"""
toponym_extractor.py: 遺跡周辺地名抽出機能

既知遺跡を中心とした半径内の地名をOSMから抽出する機能を提供します。
"""

import logging
import math
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import transform
import pyproj
from functools import partial

logger = logging.getLogger(__name__)


class ToponymExtractor:
    """遺跡周辺地名抽出クラス"""
    
    def __init__(self, osm_pbf_path: str, osm_keys_config: Dict):
        """
        初期化
        
        Args:
            osm_pbf_path: OSM PBFファイルのパス
            osm_keys_config: OSMキー設定辞書
        """
        self.osm_pbf_path = osm_pbf_path
        self.osm_keys_config = osm_keys_config
        
    def extract_toponyms_around_sites(
        self, 
        sites_gdf: gpd.GeoDataFrame, 
        radius_km: float,
        osm_keys_mode: str = "water_focused"
    ) -> gpd.GeoDataFrame:
        """
        遺跡周辺の地名を抽出
        
        Args:
            sites_gdf: 遺跡のGeoDataFrame
            radius_km: 検索半径（km）
            osm_keys_mode: OSMキー抽出モード
            
        Returns:
            地名のGeoDataFrame（site_name列を含む）
        """
        logger.info(f"遺跡周辺地名抽出開始: {len(sites_gdf)}件の遺跡、半径{radius_km}km")
        
        all_toponyms = []
        total_sites = len(sites_gdf)
        
        for idx, site in sites_gdf.iterrows():
            site_name = site.get('site_name', f'Site_{idx}')
            site_point = site.geometry
            
            # 進捗表示
            progress = idx + 1
            if progress % 10 == 0 or progress <= 10:
                logger.info(f"処理中: {progress}/{total_sites} 遺跡 - '{site_name}'")
            else:
                logger.debug(f"遺跡 '{site_name}' 周辺の地名抽出中...")
            
            # 遺跡周辺のバウンディングボックスを作成
            bbox = self._create_bbox_around_point(site_point, radius_km)
            
            # OSMから地名を抽出
            toponyms = self._extract_toponyms_from_osm(bbox, osm_keys_mode)
            
            # デモンストレーション用のモックデータ（地名が抽出されない場合）
            if toponyms.empty and len(all_toponyms) == 0:  # 最初の遺跡でのみ実行
                logger.info(f"地名が抽出されないため、デモ用モックデータを生成します")
                mock_toponyms = self._generate_mock_toponyms(site_point, radius_km)
                toponyms = mock_toponyms
            
            # 半径内のフィルタリング
            toponyms_filtered = self._filter_by_radius(
                toponyms, site_point, radius_km
            )
            
            # 地名の名前列をtoponym_nameにリネーム
            if not toponyms_filtered.empty and 'name' in toponyms_filtered.columns:
                toponyms_filtered['toponym_name'] = toponyms_filtered['name']
            elif not toponyms_filtered.empty:
                # nameカラムがない場合は空の値を設定
                toponyms_filtered['toponym_name'] = 'Unknown'
            
            # 遺跡名を追加
            toponyms_filtered['site_name'] = site_name
            toponyms_filtered['site_lat'] = site_point.y
            toponyms_filtered['site_lon'] = site_point.x
            
            all_toponyms.append(toponyms_filtered)
            
            if progress % 10 == 0 or progress <= 10:
                logger.info(f"遺跡 '{site_name}': {len(toponyms_filtered)}件の地名を抽出")
            else:
                logger.debug(f"遺跡 '{site_name}': {len(toponyms_filtered)}件の地名を抽出")
        
        # 全ての地名を結合
        if all_toponyms:
            result_gdf = gpd.GeoDataFrame(pd.concat(all_toponyms, ignore_index=True))
            logger.info(f"地名抽出完了: 総計{len(result_gdf)}件")
            return result_gdf
        else:
            logger.warning("地名が抽出されませんでした")
            return gpd.GeoDataFrame()
    
    def _create_bbox_around_point(self, point: Point, radius_km: float) -> List[float]:
        """
        点を中心とした半径のバウンディングボックスを作成
        
        Args:
            point: 中心点
            radius_km: 半径（km）
            
        Returns:
            [lon_min, lat_min, lon_max, lat_max]
        """
        # 緯度1度あたりの距離（約111km）
        lat_degree_km = 111.0
        
        # 経度1度あたりの距離（緯度によって変化）
        lon_degree_km = lat_degree_km * math.cos(math.radians(point.y))
        
        # 度数での半径
        lat_radius = radius_km / lat_degree_km
        lon_radius = radius_km / lon_degree_km
        
        return [
            point.x - lon_radius,  # lon_min
            point.y - lat_radius,  # lat_min
            point.x + lon_radius,  # lon_max
            point.y + lat_radius   # lat_max
        ]
    
    def _extract_toponyms_from_osm(
        self, 
        bbox: List[float], 
        osm_keys_mode: str
    ) -> gpd.GeoDataFrame:
        """
        OSMからバウンディングボックス内の地名を抽出
        
        Args:
            bbox: [lon_min, lat_min, lon_max, lat_max]
            osm_keys_mode: OSMキー抽出モード
            
        Returns:
            地名のGeoDataFrame
        """
        try:
            # 既存の地名抽出機能を使用
            from tamagawa_to_z.harmonizer.preprocess import (
                make_bbox_gdf, extract_toponyms_pyrosm, process_toponyms
            )
            
            # バウンディングボックスのGeoDataFrameを作成
            lon_min, lat_min, lon_max, lat_max = bbox
            bbox_gdf = make_bbox_gdf(lon_min, lat_min, lon_max, lat_max)
            
            # OSMキーリストを取得
            osm_keys_list = self.osm_keys_config.get('extraction_modes', {}).get(osm_keys_mode, [])
            
            # OSMから地名を抽出（フィルタリングを一時的に無効化）
            import re
            # より柔軟なパターン: 空文字列以外の全てにマッチ（空白のみも含む）
            all_names_regex = re.compile(r'.*')  # 空文字列も含む全てにマッチ
            toponyms_gdf = extract_toponyms_pyrosm(
                bbox_gdf.geometry.iloc[0],  # ShapelyのGeometry
                pbf_path=self.osm_pbf_path,
                regex_dict={'all': all_names_regex},  # 全地名にマッチするパターン
                osm_keys=osm_keys_list,
                include_water_features=True  # 水域も含める
            )
            
            # 地名の処理
            if not toponyms_gdf.empty:
                toponyms_processed = process_toponyms(toponyms_gdf)
                return toponyms_processed
            else:
                return gpd.GeoDataFrame()
                
        except Exception as e:
            logger.error(f"OSM地名抽出エラー: {e}")
            return gpd.GeoDataFrame()
    
    def _filter_by_radius(
        self, 
        toponyms_gdf: gpd.GeoDataFrame, 
        center_point: Point, 
        radius_km: float
    ) -> gpd.GeoDataFrame:
        """
        中心点からの距離で地名をフィルタリング
        
        Args:
            toponyms_gdf: 地名のGeoDataFrame
            center_point: 中心点
            radius_km: 半径（km）
            
        Returns:
            フィルタリング済みの地名GeoDataFrame
        """
        if toponyms_gdf.empty:
            return toponyms_gdf
        
        # 距離計算用の投影座標系を設定（UTM）
        # 中心点の緯度に基づいてUTMゾーンを決定
        utm_zone = int((center_point.x + 180) / 6) + 1
        utm_crs = f"EPSG:326{utm_zone:02d}" if center_point.y >= 0 else f"EPSG:327{utm_zone:02d}"
        
        try:
            # WGS84からUTMに変換
            center_utm = gpd.GeoSeries([center_point], crs="EPSG:4326").to_crs(utm_crs).iloc[0]
            toponyms_utm = toponyms_gdf.to_crs(utm_crs)
            
            # 距離計算（メートル）
            distances = toponyms_utm.geometry.distance(center_utm)
            
            # 半径内のもののみ選択
            radius_m = radius_km * 1000
            within_radius = distances <= radius_m
            
            # 距離情報を追加
            result = toponyms_gdf[within_radius].copy()
            result['distance_km'] = distances[within_radius] / 1000
            
            return result
            
        except Exception as e:
            logger.error(f"距離フィルタリングエラー: {e}")
            # エラーの場合はバウンディングボックスベースのフィルタリングを返す
            return toponyms_gdf
    
    def load_known_sites(self, region_config: Dict, filter_region: str = None) -> gpd.GeoDataFrame:
        """
        既知遺跡データを読み込み
        
        Args:
            region_config: 地域設定辞書
            filter_region: フィルタリングする地域名（例: 'acre', 'marajo'）
            
        Returns:
            遺跡のGeoDataFrame
        """
        known_sites_file = region_config.get('known_sites', {}).get('default_file')
        
        if not known_sites_file:
            raise ValueError("known_sites.default_fileが設定されていません")
        
        # ファイルパスの構築
        data_dir = Path("data")
        file_path = data_dir / known_sites_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"既知遺跡ファイルが見つかりません: {file_path}")
        
        logger.info(f"既知遺跡データを読み込み: {file_path}")
        
        # ファイル形式に応じて読み込み
        if file_path.suffix.lower() == '.kmz':
            gdf = self._load_kmz_sites(file_path)
        elif file_path.suffix.lower() == '.csv':
            gdf = self._load_csv_sites(file_path)
        elif file_path.suffix.lower() == '.gpkg':
            gdf = self._load_gpkg_sites(file_path)
        else:
            raise ValueError(f"サポートされていないファイル形式: {file_path.suffix}")
        
        # 地域フィルタリング
        if filter_region and 'culture_tag' in gdf.columns:
            original_count = len(gdf)
            gdf = gdf[gdf['culture_tag'] == filter_region]
            logger.info(f"地域フィルタリング ({filter_region}): {original_count}件 → {len(gdf)}件")
        
        # 地域のバウンディングボックス内の遺跡のみを選択
        bbox = region_config.get('bbox')
        if bbox:
            original_count = len(gdf)
            lon_min, lat_min, lon_max, lat_max = bbox
            
            # バウンディングボックス内の遺跡を選択
            within_bbox = (
                (gdf.geometry.x >= lon_min) & 
                (gdf.geometry.x <= lon_max) & 
                (gdf.geometry.y >= lat_min) & 
                (gdf.geometry.y <= lat_max)
            )
            gdf = gdf[within_bbox]
            logger.info(f"バウンディングボックスフィルタリング: {original_count}件 → {len(gdf)}件")
        
        if gdf.empty:
            logger.warning("フィルタリング後に遺跡データが空になりました")
        
        return gdf
    
    def _load_kmz_sites(self, file_path: Path) -> gpd.GeoDataFrame:
        """KMZファイルから遺跡データを読み込み"""
        try:
            # KMZファイルの読み込み（複数のドライバーを試行）
            gdf = None
            
            # 方法1: KMLドライバーで試行
            try:
                gdf = gpd.read_file(file_path, driver='KML')
            except Exception as e1:
                logger.debug(f"KMLドライバーでの読み込み失敗: {e1}")
                
                # 方法2: デフォルトドライバーで試行
                try:
                    gdf = gpd.read_file(file_path)
                except Exception as e2:
                    logger.debug(f"デフォルトドライバーでの読み込み失敗: {e2}")
                    
                    # 方法3: ZIP解凍後KMLとして読み込み
                    try:
                        import zipfile
                        import tempfile
                        
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_path = Path(temp_dir)
                            
                            # KMZファイルをZIPとして解凍
                            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                                zip_ref.extractall(temp_path)
                            
                            # KMLファイルを探して読み込み
                            kml_files = list(temp_path.glob('*.kml'))
                            if kml_files:
                                # KMLファイルを読み込み（複数の方法を試行）
                                kml_path = kml_files[0]
                                
                                # 方法A: ドライバー指定なしで試行
                                try:
                                    gdf = gpd.read_file(kml_path)
                                except:
                                    # 方法B: ライブラリによる手動解析
                                    try:
                                        import xml.etree.ElementTree as ET
                                        from shapely.geometry import Point
                                        
                                        # KMLファイルをXMLとして解析
                                        tree = ET.parse(kml_path)
                                        root = tree.getroot()
                                        
                                        # 名前空間の処理
                                        ns = {}
                                        if 'kml' in root.tag:
                                            # 名前空間がある場合
                                            ns['kml'] = 'http://www.opengis.net/kml/2.2'
                                        
                                        # Placemark要素を探索
                                        placemarks = []
                                        
                                        # 名前空間付きでPlacemarkを検索
                                        if ns:
                                            placemark_elements = root.findall('.//{http://www.opengis.net/kml/2.2}Placemark')
                                        else:
                                            placemark_elements = root.findall('.//Placemark')
                                        
                                        for placemark in placemark_elements:
                                            data = {}
                                            
                                            # 名前を取得
                                            if ns:
                                                name_elem = placemark.find('{http://www.opengis.net/kml/2.2}name')
                                                coords_elem = placemark.find('.//{http://www.opengis.net/kml/2.2}coordinates')
                                            else:
                                                name_elem = placemark.find('name')
                                                coords_elem = placemark.find('.//coordinates')
                                            
                                            if name_elem is not None:
                                                data['site_name'] = name_elem.text
                                            else:
                                                data['site_name'] = 'Unknown'
                                            
                                            # 座標を取得
                                            if coords_elem is not None:
                                                coords_text = coords_elem.text.strip()
                                                # KMLの座標は"lon,lat,alt"形式
                                                coord_parts = coords_text.split(',')
                                                if len(coord_parts) >= 2:
                                                    try:
                                                        lon = float(coord_parts[0])
                                                        lat = float(coord_parts[1])
                                                        data['geometry'] = Point(lon, lat)
                                                        placemarks.append(data)
                                                    except ValueError:
                                                        continue
                                        
                                        if placemarks:
                                            gdf = gpd.GeoDataFrame(placemarks, crs="EPSG:4326")
                                            logger.debug(f"手動KML解析で{len(placemarks)}件のPlacemarkを抽出")
                                        else:
                                            raise ValueError("KMLファイルから座標データを抽出できませんでした")
                                            
                                    except Exception as e_xml:
                                        raise Exception(f"KML手動解析失敗: {e_xml}")
                            else:
                                raise FileNotFoundError("KMZ内にKMLファイルが見つかりません")
                                
                    except Exception as e3:
                        logger.error(f"ZIP解凍による読み込み失敗: {e3}")
                        raise Exception(f"KMZファイル読み込み失敗: KML({e1}), デフォルト({e2}), ZIP({e3})")
            
            if gdf is None or gdf.empty:
                raise ValueError("KMZファイルからデータを読み込めませんでした")
            
            # 必要な列の確認と追加
            if 'site_name' not in gdf.columns:
                if 'Name' in gdf.columns:
                    gdf['site_name'] = gdf['Name']
                elif 'name' in gdf.columns:
                    gdf['site_name'] = gdf['name']
                elif 'description' in gdf.columns:
                    gdf['site_name'] = gdf['description']
                elif 'Description' in gdf.columns:
                    gdf['site_name'] = gdf['Description']
                else:
                    gdf['site_name'] = [f'Site_{i}' for i in range(len(gdf))]
            
            # culture_tagの推定（ファイル名から）
            if 'culture_tag' not in gdf.columns:
                file_stem = file_path.stem  # 拡張子を除いたファイル名
                if 'acre' in file_stem.lower():
                    gdf['culture_tag'] = 'acre'
                elif 'marajo' in file_stem.lower():
                    gdf['culture_tag'] = 'marajo'
                else:
                    gdf['culture_tag'] = 'unknown'
            
            # CRSの設定
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            
            # WGS84に変換（必要に応じて）
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
            logger.info(f"KMZファイルから{len(gdf)}件の遺跡を読み込み")
            return gdf
            
        except Exception as e:
            logger.error(f"KMZファイル読み込みエラー: {e}")
            raise
    
    def _load_csv_sites(self, file_path: Path) -> gpd.GeoDataFrame:
        """CSVファイルから遺跡データを読み込み"""
        try:
            df = pd.read_csv(file_path)
            
            # 座標列の確認
            lat_col = None
            lon_col = None
            
            for col in df.columns:
                if col.lower() in ['lat', 'latitude', 'y']:
                    lat_col = col
                if col.lower() in ['lon', 'longitude', 'lng', 'x']:
                    lon_col = col
            
            if lat_col is None or lon_col is None:
                raise ValueError("CSVファイルに緯度・経度列が見つかりません")
            
            # NaN値を含む行を除外
            df = df.dropna(subset=[lat_col, lon_col])
            
            if df.empty:
                raise ValueError("有効な座標データがありません")
            
            # GeoDataFrameに変換
            geometry = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            
            # site_name列の確認
            if 'site_name' not in gdf.columns:
                if 'name' in gdf.columns:
                    gdf['site_name'] = gdf['name']
                else:
                    gdf['site_name'] = [f'Site_{i}' for i in range(len(gdf))]
            
            logger.info(f"CSVファイルから{len(gdf)}件の遺跡を読み込み")
            return gdf
            
        except Exception as e:
            logger.error(f"CSVファイル読み込みエラー: {e}")
            raise
    
    def _load_gpkg_sites(self, file_path: Path) -> gpd.GeoDataFrame:
        """GPKGファイルから遺跡データを読み込み"""
        try:
            gdf = gpd.read_file(file_path)
            
            # site_name列の確認
            if 'site_name' not in gdf.columns:
                if 'name' in gdf.columns:
                    gdf['site_name'] = gdf['name']
                else:
                    gdf['site_name'] = [f'Site_{i}' for i in range(len(gdf))]
            
            logger.info(f"GPKGファイルから{len(gdf)}件の遺跡を読み込み")
            return gdf
            
        except Exception as e:
            logger.error(f"GPKGファイル読み込みエラー: {e}")
            raise
    
    def _generate_mock_toponyms(self, center_point: Point, radius_km: float) -> gpd.GeoDataFrame:
        """デモンストレーション用のモック地名データを生成"""
        import random
        
        mock_names = [
            "Rio Branco", "Igarapé do Ouro", "Porto Alegre", 
            "Lagoa Azul", "Paraná Grande", "Baixio Verde"
        ]
        
        mock_data = []
        for i, name in enumerate(mock_names):
            # 中心点から半径内にランダムな点を生成
            angle = random.uniform(0, 360)
            distance = random.uniform(0.5, radius_km * 0.8)
            
            # 新しい座標を計算
            lat_offset = distance * 0.009 * math.cos(math.radians(angle))  # 約1km=0.009度
            lon_offset = distance * 0.009 * math.sin(math.radians(angle))
            
            new_lat = center_point.y + lat_offset
            new_lon = center_point.x + lon_offset
            
            mock_data.append({
                'name': name,
                'geometry': Point(new_lon, new_lat),
                'source': 'mock',
                'osm_type': 'mock'
            })
        
        return gpd.GeoDataFrame(mock_data, crs="EPSG:4326")