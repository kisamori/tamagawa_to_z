"""
polar_converter.py: 極座標変換機能

地名の遺跡からの角度・距離を極座標形式で計算する機能を提供します。
"""

import logging
import math
from typing import List, Dict, Tuple

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

logger = logging.getLogger(__name__)


class PolarConverter:
    """極座標変換クラス"""
    
    def __init__(self, angle_reference: str = "north", angle_direction: str = "clockwise"):
        """
        初期化
        
        Args:
            angle_reference: 角度の基準（"north" or "east"）
            angle_direction: 角度の方向（"clockwise" or "counterclockwise"）
        """
        self.angle_reference = angle_reference
        self.angle_direction = angle_direction
        
        logger.info(f"極座標変換設定: 基準={angle_reference}, 方向={angle_direction}")
    
    def convert_to_polar(self, toponyms_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        地名データを極座標に変換
        
        Args:
            toponyms_gdf: 地名のGeoDataFrame（site_lat, site_lonを含む）
            
        Returns:
            極座標情報を追加したGeoDataFrame
        """
        if toponyms_gdf.empty:
            logger.warning("変換対象の地名データが空です")
            return toponyms_gdf
        
        logger.info(f"極座標変換開始: {len(toponyms_gdf)}件の地名")
        
        result_gdf = toponyms_gdf.copy()
        
        # 角度と距離を計算
        angles = []
        radii = []
        
        for idx, row in toponyms_gdf.iterrows():
            site_point = Point(row['site_lon'], row['site_lat'])
            toponym_point = row.geometry
            
            angle, radius = self._calculate_polar_coordinates(site_point, toponym_point)
            angles.append(angle)
            radii.append(radius)
        
        # 結果を追加
        result_gdf['angle'] = angles
        result_gdf['radius'] = radii
        
        logger.info(f"極座標変換完了: 角度範囲 {min(angles):.2f}°-{max(angles):.2f}°, "
                   f"距離範囲 {min(radii):.3f}-{max(radii):.3f}km")
        
        return result_gdf
    
    def _calculate_polar_coordinates(self, center: Point, target: Point) -> Tuple[float, float]:
        """
        中心点から対象点への極座標を計算
        
        Args:
            center: 中心点（遺跡）
            target: 対象点（地名）
            
        Returns:
            (角度, 距離) のタプル
        """
        # 座標差分を計算
        dx = target.x - center.x
        dy = target.y - center.y
        
        # 距離を計算（Haversine公式を使用してより正確に）
        radius_km = self._haversine_distance(center.y, center.x, target.y, target.x)
        
        # 角度を計算（atan2を使用）
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # 基準角度と方向に応じて調整
        angle_adjusted = self._adjust_angle(angle_deg)
        
        return angle_adjusted, radius_km
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Haversine公式による距離計算（km）
        
        Args:
            lat1, lon1: 点1の緯度・経度
            lat2, lon2: 点2の緯度・経度
            
        Returns:
            距離（km）
        """
        # 地球の半径（km）
        R = 6371.0
        
        # 度をラジアンに変換
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # 差分
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Haversine公式
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _adjust_angle(self, angle_deg: float) -> float:
        """
        角度を設定に応じて調整
        
        Args:
            angle_deg: atan2から得られた角度（度）
            
        Returns:
            調整後の角度（度）
        """
        # atan2は東を0度、反時計回りで返すので調整が必要
        
        if self.angle_reference == "north":
            # 北を0度にする場合
            # atan2の東(0度)を北(0度)に変換: 90度回転
            adjusted = 90 - angle_deg
        else:  # east
            # 東を0度のまま
            adjusted = angle_deg
        
        if self.angle_direction == "clockwise":
            # 時計回りにする場合（デフォルトは反時計回り）
            adjusted = -adjusted
        
        # 0-360度の範囲に正規化
        while adjusted < 0:
            adjusted += 360
        while adjusted >= 360:
            adjusted -= 360
        
        return adjusted
    
    def create_polar_summary(self, polar_gdf: gpd.GeoDataFrame) -> Dict:
        """
        極座標データの統計サマリを作成
        
        Args:
            polar_gdf: 極座標変換済みのGeoDataFrame
            
        Returns:
            統計情報の辞書
        """
        if polar_gdf.empty or 'angle' not in polar_gdf.columns:
            return {}
        
        angles = polar_gdf['angle'].dropna()
        radii = polar_gdf['radius'].dropna()
        
        summary = {
            'total_toponyms': len(polar_gdf),
            'angle_stats': {
                'min': float(angles.min()) if len(angles) > 0 else None,
                'max': float(angles.max()) if len(angles) > 0 else None,
                'mean': float(angles.mean()) if len(angles) > 0 else None,
                'std': float(angles.std()) if len(angles) > 0 else None
            },
            'radius_stats': {
                'min': float(radii.min()) if len(radii) > 0 else None,
                'max': float(radii.max()) if len(radii) > 0 else None,
                'mean': float(radii.mean()) if len(radii) > 0 else None,
                'std': float(radii.std()) if len(radii) > 0 else None
            },
            'angle_distribution': self._calculate_angle_distribution(angles),
            'radius_distribution': self._calculate_radius_distribution(radii)
        }
        
        return summary
    
    def _calculate_angle_distribution(self, angles: pd.Series) -> Dict:
        """角度分布の計算"""
        if len(angles) == 0:
            return {}
        
        # 8方位での分布
        bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        
        counts, _ = np.histogram(angles, bins=bins)
        
        distribution = {}
        for i, label in enumerate(labels):
            distribution[label] = int(counts[i])
        
        return distribution
    
    def _calculate_radius_distribution(self, radii: pd.Series) -> Dict:
        """距離分布の計算"""
        if len(radii) == 0:
            return {}
        
        # 距離帯での分布
        max_radius = radii.max()
        bins = np.linspace(0, max_radius, 6)  # 5つの区間
        
        counts, bin_edges = np.histogram(radii, bins=bins)
        
        distribution = {}
        for i in range(len(counts)):
            range_label = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}km"
            distribution[range_label] = int(counts[i])
        
        return distribution