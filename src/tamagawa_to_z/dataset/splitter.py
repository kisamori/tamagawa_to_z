"""データ分割機能 - 遺跡データをTrain/Val/Test-time/Test-regionに分割する."""

from __future__ import annotations

import logging
import zipfile
import tempfile
import xml.etree.ElementTree as ET
import re
from pathlib import Path
from typing import Dict, Union, Optional

import pandas as pd
import geopandas as gpd
import yaml
from shapely.geometry import Point
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    遺跡データの分割を管理するクラス.
    
    機能:
    - Train/Val/Test-time/Test-region の分割
    - 発見年代によるTest-time自動振分
    - 文化圏別のTest-region作成
    - KMZ/CSV/GPKG形式の入力ファイル対応
    """
    
    def __init__(self, cfg_path: Path, sites_path: Path):
        """
        初期化.
        
        Args:
            cfg_path: 設定ファイルパス (dataset_split.yaml)
            sites_path: 遺跡ファイルパス (.kmz/.csv/.gpkg)
        """
        self.cfg_path = cfg_path
        self.sites_path = sites_path
        
        # 設定読み込み
        with open(cfg_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
            
        # 遺跡データ読み込み（フォーマット自動検出）
        self.sites = self._load_sites_data(sites_path)
        
        # 必要カラムの確認
        required_cols = ['site_name', 'lat', 'lon', 'culture_tag', 'discovery_year']
        missing_cols = [col for col in required_cols if col not in self.sites.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # GeoDataFrame化
        self._make_gdf()
        
        logger.info(f"Loaded {len(self.sites)} sites from {sites_path}")
        logger.info(f"Culture distribution: {self.sites['culture_tag'].value_counts().to_dict()}")
        
    def split(self) -> Dict[str, Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]]]:
        """
        データを分割する.
        
        Returns:
            分割結果辞書:
            - "train": TrainデータのGeoDataFrame
            - "val": ValidationデータのGeoDataFrame  
            - "test": TestデータのGeoDataFrame
            - "test_region": 文化圏別Test-regionデータの辞書（オプション）
        """
        logger.info("Starting data splitting...")
        
        # 新しい比率ベースの分割を使用
        if "split_ratios" in self.cfg:
            return self._simple_ratio_split()
        else:
            # 旧ロジック（後方互換性のため）
            return self._legacy_split()
        
    def get_stats(self) -> Dict:
        """
        データセットの統計情報を取得.
        
        Returns:
            統計情報辞書
        """
        stats = {
            "total_sites": len(self.sites),
            "culture_distribution": self.sites["culture_tag"].value_counts().to_dict(),
            "discovery_year_range": {
                "min": int(self.sites["discovery_year"].min()),
                "max": int(self.sites["discovery_year"].max()),
                "mean": float(self.sites["discovery_year"].mean())
            },
            "geographic_bounds": {
                "north": float(self.sites["lat"].max()),
                "south": float(self.sites["lat"].min()),
                "east": float(self.sites["lon"].max()),
                "west": float(self.sites["lon"].min())
            }
        }
        return stats
        
    # ---------- Internal Methods ----------
    
    def _simple_ratio_split(self) -> Dict[str, Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]]]:
        """
        シンプルな比率ベースの分割.
        
        Returns:
            分割結果辞書
        """
        ratios = self.cfg["split_ratios"]
        train_ratio = ratios["train"]
        val_ratio = ratios["val"]
        test_ratio = ratios["test"]
        
        # 比率の合計が1.0になるか確認
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            logger.warning(f"Split ratios sum to {total_ratio:.3f}, not 1.0. Normalizing...")
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio
        
        # 全データをシャッフル
        shuffled_sites = self.sites.sample(frac=1, random_state=self.cfg.get("random_state", 42)).reset_index(drop=True)
        
        total_sites = len(shuffled_sites)
        train_size = int(total_sites * train_ratio)
        val_size = int(total_sites * val_ratio)
        # test_size = total_sites - train_size - val_size  # 余りをtestに
        
        # 分割
        result = {}
        
        if train_ratio > 0:
            train_gdf = shuffled_sites[:train_size].copy()
            result["train"] = train_gdf
        else:
            train_size = 0  # trainが0%の場合
            
        val_gdf = shuffled_sites[train_size:train_size + val_size].copy()
        test_gdf = shuffled_sites[train_size + val_size:].copy()
        
        result["val"] = val_gdf
        result["test"] = test_gdf
        
        # ログ出力（trainがある場合とない場合を分ける）
        if train_ratio > 0:
            logger.info(f"Simple ratio split: train={len(result['train'])}, val={len(val_gdf)}, test={len(test_gdf)}")
        else:
            logger.info(f"Simple ratio split: val={len(val_gdf)}, test={len(test_gdf)} (no train set)")
        
        # Test-region分割（オプション）
        if self.cfg.get("culture_blocks"):
            test_region_dict = self._test_region_split()
            if test_region_dict:
                result["test_region"] = test_region_dict
                logger.info(f"Test-region: {[(k, len(v)) for k, v in test_region_dict.items()]}")
        
        return result
    
    def _legacy_split(self) -> Dict[str, Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]]]:
        """
        旧ロジックの分割（後方互換性のため）.
        
        Returns:
            分割結果辞書
        """
        result = {}
        
        # Train/Val分割
        if self.cfg.get("use_train_split", True):
            train_gdf, val_gdf = self._train_val_split()
            result["train"] = train_gdf
            result["val"] = val_gdf
            logger.info(f"Train/Val split: train={len(train_gdf)}, val={len(val_gdf)}")
        else:
            val_gdf = self._val_only()
            result["val"] = val_gdf
            logger.info(f"Validation only: val={len(val_gdf)}")
            
        # Test-time分割
        test_time_gdf = self._test_time_split()
        result["test_time"] = test_time_gdf
        logger.info(f"Test-time: {len(test_time_gdf)} sites")
        
        # Test-region分割
        test_region_dict = self._test_region_split()
        result["test_region"] = test_region_dict
        logger.info(f"Test-region: {[(k, len(v)) for k, v in test_region_dict.items()]}")
        
        return result
    
    def _load_sites_data(self, sites_path: Path) -> pd.DataFrame:
        """
        サイトデータを読み込む（KMZ/CSV/GPKG対応）.
        
        Args:
            sites_path: 入力ファイルパス
            
        Returns:
            サイトデータのDataFrame
        """
        file_ext = sites_path.suffix.lower()
        
        if file_ext == '.kmz':
            return self._load_kmz(sites_path)
        elif file_ext == '.csv':
            return pd.read_csv(sites_path, encoding='utf-8')
        elif file_ext == '.gpkg':
            gdf = gpd.read_file(sites_path)
            # 座標カラムを追加
            gdf['lat'] = gdf.geometry.y
            gdf['lon'] = gdf.geometry.x
            return pd.DataFrame(gdf.drop(columns='geometry'))
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported: .kmz, .csv, .gpkg")
    
    def _load_kmz(self, kmz_path: Path) -> pd.DataFrame:
        """
        KMZファイルを読み込む.
        
        Args:
            kmz_path: KMZファイルパス
            
        Returns:
            サイトデータのDataFrame
        """
        logger.info(f"Loading KMZ file: {kmz_path}")
        
        # KMZを展開してKMLを取得
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # ZIPとしてKMZを展開
            with zipfile.ZipFile(kmz_path, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
            
            # KMLファイルを探す
            kml_files = list(temp_path.glob('*.kml'))
            if not kml_files:
                raise ValueError(f"No KML file found in {kmz_path}")
            
            kml_path = kml_files[0]  # 最初のKMLファイルを使用
            logger.info(f"Found KML: {kml_path.name}")
            
            # KMLファイルを直接パースする
            try:
                gdf = self._parse_kml_file(kml_path)
            except Exception as e:
                raise ValueError(f"Failed to read KML from {kmz_path}: {e}")
        
        # 必要なカラムの確認と作成
        if 'Name' in gdf.columns:
            gdf['site_name'] = gdf['Name']
        elif 'name' in gdf.columns:
            gdf['site_name'] = gdf['name']
        else:
            # サイト名がない場合はインデックスベースで作成
            gdf['site_name'] = [f"site_{i:03d}" for i in range(len(gdf))]
            logger.warning("No name field found, generated site names")
        
        # 座標を取得
        gdf['lat'] = gdf.geometry.y
        gdf['lon'] = gdf.geometry.x
        
        # culture_tagとdiscovery_yearのデフォルト値
        if 'culture_tag' not in gdf.columns:
            # ファイル名から文化圏を推定
            culture_tag = self._extract_culture_from_filename(kmz_path)
            gdf['culture_tag'] = culture_tag
            logger.info(f"Set culture_tag to: {culture_tag}")
        
        if 'discovery_year' not in gdf.columns:
            # デフォルト値として2015を設定
            gdf['discovery_year'] = 2015
            logger.warning("No discovery_year field, set default to 2015")
        
        # DataFrameとして返す
        return pd.DataFrame(gdf.drop(columns='geometry'))
    
    def _parse_kml_file(self, kml_path: Path) -> gpd.GeoDataFrame:
        """
        KMLファイルを直接パースしてGeoDataFrameを作成する.
        
        Args:
            kml_path: KMLファイルパス
            
        Returns:
            GeoDataFrame
        """
        logger.info(f"Parsing KML file: {kml_path.name}")
        
        # XMLファイルを読み込み
        tree = ET.parse(kml_path)
        root = tree.getroot()
        
        # 名前空間を取得
        namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
        if root.tag.startswith('{'):
            # 名前空間が明示的にある場合
            ns_match = re.match(r'\{([^}]+)\}', root.tag)
            if ns_match:
                namespace['kml'] = ns_match.group(1)
        
        # Placemarkを探す
        placemarks = root.findall('.//kml:Placemark', namespace)
        if not placemarks:
            # 名前空間なしで試行
            placemarks = root.findall('.//Placemark')
        
        if not placemarks:
            raise ValueError("No Placemark elements found in KML")
        
        sites_data = []
        
        for placemark in placemarks:
            site_data = {}
            
            # 名前を取得
            name_elem = placemark.find('.//kml:name', namespace)
            if name_elem is None:
                name_elem = placemark.find('.//name')
            
            if name_elem is not None and name_elem.text:
                site_data['Name'] = name_elem.text.strip()
            else:
                site_data['Name'] = f"site_{len(sites_data):03d}"
            
            # 座標を取得
            coord_elem = placemark.find('.//kml:coordinates', namespace)
            if coord_elem is None:
                coord_elem = placemark.find('.//coordinates')
            
            if coord_elem is not None and coord_elem.text:
                coords_text = coord_elem.text.strip()
                # 座標形式: "longitude,latitude[,altitude]"
                coords = coords_text.split(',')
                if len(coords) >= 2:
                    try:
                        lon = float(coords[0])
                        lat = float(coords[1])
                        site_data['geometry'] = Point(lon, lat)
                    except ValueError:
                        logger.warning(f"Invalid coordinates for {site_data['Name']}: {coords_text}")
                        continue
                else:
                    logger.warning(f"Invalid coordinate format for {site_data['Name']}: {coords_text}")
                    continue
            else:
                logger.warning(f"No coordinates found for {site_data['Name']}")
                continue
            
            # 説明やその他の情報を取得
            desc_elem = placemark.find('.//kml:description', namespace)
            if desc_elem is None:
                desc_elem = placemark.find('.//description')
            
            if desc_elem is not None and desc_elem.text:
                site_data['Description'] = desc_elem.text.strip()
            
            sites_data.append(site_data)
        
        if not sites_data:
            raise ValueError("No valid sites found in KML")
        
        # GeoDataFrameを作成
        gdf = gpd.GeoDataFrame(sites_data, crs='EPSG:4326')
        logger.info(f"Parsed {len(gdf)} sites from KML")
        
        return gdf
    
    def _extract_culture_from_filename(self, file_path: Path) -> str:
        """
        ファイル名から文化圏タグを抽出する.
        
        Args:
            file_path: ファイルパス
            
        Returns:
            文化圏タグ
        """
        filename = file_path.stem.lower()
        
        # 設定済み文化圏との照合
        culture_blocks = self.cfg.get('culture_blocks', {})
        for culture_tag in culture_blocks.keys():
            if culture_tag in filename:
                return culture_tag
        
        # 既知のパターンとの照合
        culture_patterns = {
            'acre': ['acre'],
            'santarem': ['santarem', 'santarém'],
            'marajo': ['marajo', 'marajó'],
            'casarabe': ['casarabe'],
            'kuhikugu': ['kuhikugu', 'xingu'],
            'upano': ['upano']
        }
        
        for culture_tag, patterns in culture_patterns.items():
            if any(pattern in filename for pattern in patterns):
                return culture_tag
        
        # デフォルト値
        logger.warning(f"Could not determine culture from filename: {filename}, using 'unknown'")
        return 'unknown'
    
    def _make_gdf(self):
        """サイトデータをGeoDataFrameに変換."""
        self.sites = gpd.GeoDataFrame(
            self.sites,
            geometry=gpd.points_from_xy(self.sites.lon, self.sites.lat),
            crs=self.cfg.get("target_crs", "EPSG:4326")
        )
        
    def _train_val_split(self) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Train/Validation分割.
        
        発見年しきい値未満のサイトをTrain/Valに分割.
        """
        threshold = self.cfg["discovery_year_threshold"]
        early_sites = self.sites[self.sites["discovery_year"] < threshold].copy()
        
        if len(early_sites) == 0:
            raise ValueError(f"No sites with discovery_year < {threshold}")
            
        # 層化サンプリング用の設定
        stratify = None
        if self.cfg.get("stratify_by_culture", True):
            # 文化圏別に最低2サイトあるかチェック
            culture_counts = early_sites["culture_tag"].value_counts()
            valid_cultures = culture_counts[culture_counts >= 2].index
            
            if len(valid_cultures) > 0:
                # 有効な文化圏のサイトのみで層化
                early_sites_filtered = early_sites[
                    early_sites["culture_tag"].isin(valid_cultures)
                ].copy()
                
                if len(early_sites_filtered) >= 4:  # 最低4サイト必要
                    stratify = early_sites_filtered["culture_tag"]
                    early_sites = early_sites_filtered
                    
        train_gdf, val_gdf = train_test_split(
            early_sites,
            test_size=self.cfg["val_ratio"],
            stratify=stratify,
            random_state=self.cfg.get("random_state", 42)
        )
        
        return train_gdf.reset_index(drop=True), val_gdf.reset_index(drop=True)
        
    def _val_only(self) -> gpd.GeoDataFrame:
        """
        Validation のみ（Train を作らない場合）.
        
        発見年しきい値未満のサイトをすべてValidationとして使用.
        """
        threshold = self.cfg["discovery_year_threshold"]
        val_sites = self.sites[self.sites["discovery_year"] < threshold].copy()
        
        if len(val_sites) == 0:
            raise ValueError(f"No sites with discovery_year < {threshold}")
            
        return val_sites.reset_index(drop=True)
        
    def _test_time_split(self) -> gpd.GeoDataFrame:
        """
        Test-time分割.
        
        発見年しきい値以降のサイトをTest-timeとして抽出.
        """
        threshold = self.cfg["discovery_year_threshold"]
        test_time_sites = self.sites[self.sites["discovery_year"] >= threshold].copy()
        
        return test_time_sites.reset_index(drop=True)
        
    def _test_region_split(self) -> Dict[str, gpd.GeoDataFrame]:
        """
        Test-region分割.
        
        文化圏別にサイトを分割.
        """
        test_region_dict = {}
        
        culture_blocks = self.cfg.get("culture_blocks", {})
        
        for culture_tag, culture_info in culture_blocks.items():
            # 指定された文化圏のサイトを抽出
            culture_sites = self.sites[self.sites["culture_tag"] == culture_tag].copy()
            
            # さらに特定サイト名で絞り込み（設定されている場合）
            if "sites" in culture_info and culture_info["sites"]:
                specified_sites = culture_info["sites"]
                culture_sites = culture_sites[
                    culture_sites["site_name"].isin(specified_sites)
                ].copy()
                
            if len(culture_sites) > 0:
                test_region_dict[culture_tag] = culture_sites.reset_index(drop=True)
            else:
                logger.warning(f"No sites found for culture: {culture_tag}")
                
        return test_region_dict


def create_sample_master_csv(output_path: Path) -> None:
    """
    サンプルのknown_sites_master.csvを作成する（テスト用）.
    
    Args:
        output_path: 出力ファイルパス
    """
    sample_sites = [
        # Acre
        {"site_name": "Fazenda Atlântica", "lat": -9.0, "lon": -68.0, 
         "culture_tag": "acre", "discovery_year": 2015},
        {"site_name": "Fazenda Colorada", "lat": -9.2, "lon": -68.2, 
         "culture_tag": "acre", "discovery_year": 2018},
        {"site_name": "Fazenda Brasil", "lat": -9.4, "lon": -68.4, 
         "culture_tag": "acre", "discovery_year": 2021},
         
        # Casarabe
        {"site_name": "Cotoca", "lat": -17.8, "lon": -63.0, 
         "culture_tag": "casarabe", "discovery_year": 2019},
        {"site_name": "Landívar", "lat": -17.6, "lon": -62.8, 
         "culture_tag": "casarabe", "discovery_year": 2020},
        {"site_name": "Salvatierra", "lat": -17.4, "lon": -62.6, 
         "culture_tag": "casarabe", "discovery_year": 2022},
         
        # Santarem
        {"site_name": "Alter do Chão", "lat": -2.5, "lon": -54.9, 
         "culture_tag": "santarem", "discovery_year": 2010},
        {"site_name": "Santarém (urban core)", "lat": -2.4, "lon": -54.7, 
         "culture_tag": "santarem", "discovery_year": 2016},
         
        # Marajo
        {"site_name": "Teso dos Bichos", "lat": -0.3, "lon": -49.6, 
         "culture_tag": "marajo", "discovery_year": 2012},
        {"site_name": "Teso dos Caldeirões", "lat": -0.1, "lon": -49.4, 
         "culture_tag": "marajo", "discovery_year": 2021}
    ]
    
    df = pd.DataFrame(sample_sites)
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Sample master CSV created: {output_path}")


if __name__ == "__main__":
    # テスト実行
    import tempfile
    
    # サンプルデータ作成
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_path = Path(f.name)
        
    create_sample_master_csv(sample_path)
    
    # 設定ファイルパス
    cfg_path = Path("configs/dataset_split.yaml")
    
    # 分割テスト
    try:
        splitter = DataSplitter(cfg_path, sample_path)
        splits = splitter.split()
        stats = splitter.get_stats()
        
        print("=== Split Results ===")
        for name, data in splits.items():
            if isinstance(data, dict):
                print(f"{name}: {[(k, len(v)) for k, v in data.items()]}")
            else:
                print(f"{name}: {len(data)} sites")
                
        print("\n=== Dataset Stats ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    finally:
        sample_path.unlink()  # クリーンアップ