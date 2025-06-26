#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
region_config.py: 地域別設定管理ユーティリティ

地域（acre, marajo等）に応じた設定を読み込み、各スクリプトで使用する
パスやBBOX情報を提供する。
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class RegionConfig:
    """地域別設定管理クラス"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Args:
            config_path: regions.yamlのパス（Noneの場合はデフォルトパスを使用）
        """
        if config_path is None:
            # プロジェクトルートからの相対パス
            project_root = Path(__file__).resolve().parents[3]
            config_path = project_root / "config" / "regions.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"地域設定を読み込み: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"地域設定の読み込みに失敗: {e}")
            raise
    
    def get_available_regions(self) -> List[str]:
        """利用可能な地域のリストを取得"""
        return list(self.config.get('regions', {}).keys())
    
    def get_default_region(self) -> str:
        """デフォルト地域を取得"""
        return self.config.get('default_region', 'acre')
    
    def get_region_config(self, region: str) -> Dict[str, Any]:
        """指定された地域の設定を取得"""
        regions = self.config.get('regions', {})
        if region not in regions:
            available = self.get_available_regions()
            raise ValueError(f"未知の地域: {region}. 利用可能: {available}")
        
        return regions[region]
    
    def get_bbox(self, region: str) -> List[float]:
        """指定された地域のBBOXを取得"""
        region_config = self.get_region_config(region)
        bbox = region_config.get('bbox')
        if not bbox or len(bbox) != 4:
            raise ValueError(f"地域 {region} の不正なBBOX: {bbox}")
        return bbox
    
    def get_osm_pbf_path(self, region: str, data_root: Path) -> Path:
        """指定された地域のOSM PBFファイルパスを取得"""
        region_config = self.get_region_config(region)
        pbf_file = region_config.get('osm', {}).get('pbf_file')
        if not pbf_file:
            raise ValueError(f"地域 {region} のOSM PBFファイルが設定されていません")
        
        return data_root / "osm" / pbf_file
    
    def get_hydrorivers_path(self, region: str, data_root: Path) -> Path:
        """指定された地域のHydroRIVERSシェープファイルパスを取得"""
        region_config = self.get_region_config(region)
        shapefile = region_config.get('hydrorivers', {}).get('shapefile')
        if not shapefile:
            raise ValueError(f"地域 {region} のHydroRIVERSファイルが設定されていません")
        
        return data_root / shapefile
    
    def get_gsw_occurrence_path(self, region: str, data_root: Path) -> Path:
        """指定された地域のGSW occurrenceファイルパスを取得"""
        region_config = self.get_region_config(region)
        occurrence_file = region_config.get('gsw', {}).get('occurrence_file')
        if not occurrence_file:
            raise ValueError(f"地域 {region} のGSW occurrenceファイルが設定されていません")
        
        return data_root / occurrence_file
    
    def get_gsw_occurrence_paths(self, region: str, data_root: Path) -> List[Path]:
        """指定された地域のGSW occurrenceファイルパス（複数対応）を取得"""
        region_config = self.get_region_config(region)
        gsw_config = region_config.get('gsw', {})
        
        # 複数ファイルがある場合
        occurrence_files = gsw_config.get('occurrence_files')
        if occurrence_files:
            return [data_root / f for f in occurrence_files]
        
        # 単一ファイルの場合
        occurrence_file = gsw_config.get('occurrence_file')
        if occurrence_file:
            return [data_root / occurrence_file]
        
        raise ValueError(f"地域 {region} のGSW occurrenceファイルが設定されていません")
    
    def get_known_sites_path(self, region: str, data_root: Path) -> Path:
        """指定された地域の既知遺跡ファイルパスを取得"""
        region_config = self.get_region_config(region)
        known_file = region_config.get('known_sites', {}).get('default_file')
        if not known_file:
            raise ValueError(f"地域 {region} の既知遺跡ファイルが設定されていません")
        
        return data_root.parent / known_file  # data/rawではなくdataディレクトリ直下
    
    def validate_region_config(self, region: str, data_root: Path) -> Dict[str, bool]:
        """指定された地域の設定とファイル存在を検証"""
        results = {}
        
        try:
            # 設定の存在確認
            region_config = self.get_region_config(region)
            results['config_exists'] = True
            
            # BBOX検証
            try:
                bbox = self.get_bbox(region)
                results['bbox_valid'] = (
                    len(bbox) == 4 and
                    -180 <= bbox[0] <= 180 and  # lon_min
                    -90 <= bbox[1] <= 90 and    # lat_min
                    -180 <= bbox[2] <= 180 and  # lon_max
                    -90 <= bbox[3] <= 90 and    # lat_max
                    bbox[0] < bbox[2] and       # lon_min < lon_max
                    bbox[1] < bbox[3]           # lat_min < lat_max
                )
            except Exception:
                results['bbox_valid'] = False
            
            # ファイル存在確認
            try:
                osm_path = self.get_osm_pbf_path(region, data_root)
                results['osm_file_exists'] = osm_path.exists()
            except Exception:
                results['osm_file_exists'] = False
            
            try:
                rivers_path = self.get_hydrorivers_path(region, data_root)
                results['hydrorivers_file_exists'] = rivers_path.exists()
            except Exception:
                results['hydrorivers_file_exists'] = False
            
            try:
                gsw_path = self.get_gsw_occurrence_path(region, data_root)
                results['gsw_file_exists'] = gsw_path.exists()
            except Exception:
                results['gsw_file_exists'] = False
            
            try:
                sites_path = self.get_known_sites_path(region, data_root)
                results['known_sites_file_exists'] = sites_path.exists()
            except Exception:
                results['known_sites_file_exists'] = False
                
        except Exception:
            results['config_exists'] = False
        
        return results
    
    def print_region_info(self, region: str):
        """指定された地域の情報を表示"""
        try:
            region_config = self.get_region_config(region)
            print(f"\n=== 地域設定: {region} ===")
            print(f"名前: {region_config.get('name', 'N/A')}")
            print(f"説明: {region_config.get('description', 'N/A')}")
            print(f"BBOX: {region_config.get('bbox', 'N/A')}")
            print(f"OSM PBF: {region_config.get('osm', {}).get('pbf_file', 'N/A')}")
            print(f"HydroRIVERS: {region_config.get('hydrorivers', {}).get('shapefile', 'N/A')}")
            print(f"GSW: {region_config.get('gsw', {}).get('occurrence_file', 'N/A')}")
            print(f"既知遺跡: {region_config.get('known_sites', {}).get('default_file', 'N/A')}")
            
        except Exception as e:
            print(f"地域 {region} の情報取得に失敗: {e}")


def get_region_config(region: str = None, config_path: Path = None) -> RegionConfig:
    """地域設定インスタンスを取得するヘルパー関数"""
    return RegionConfig(config_path)


def add_region_argument(parser):
    """argparseパーサーに地域引数を追加するヘルパー関数"""
    config = RegionConfig()
    available_regions = config.get_available_regions()
    default_region = config.get_default_region()
    
    parser.add_argument(
        '--region', '-r',
        type=str,
        choices=available_regions,
        default=default_region,
        help=f'対象地域 ({", ".join(available_regions)}) (デフォルト: {default_region})'
    )
    return parser