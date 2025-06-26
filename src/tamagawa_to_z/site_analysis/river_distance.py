"""
river_distance.py: 川からの距離計算機能

地名から最寄りの川への距離と角度を計算する機能を提供します。
"""

import logging
import math
from typing import List, Dict, Tuple, Optional

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

logger = logging.getLogger(__name__)


class RiverDistanceCalculator:
    """川距離計算クラス"""
    
    def __init__(self, rivers_shapefile: str, max_search_distance: float = 50.0):
        """
        初期化
        
        Args:
            rivers_shapefile: HydroRIVERSシェープファイルのパス
            max_search_distance: 最大検索距離（km）
        """
        self.rivers_shapefile = rivers_shapefile
        self.max_search_distance = max_search_distance
        self.rivers_gdf = None
        
        logger.info(f"川距離計算器初期化: {rivers_shapefile}")
    
    def load_rivers(self, bbox: Optional[List[float]] = None) -> gpd.GeoDataFrame:
        """
        河川データを読み込み
        
        Args:
            bbox: [lon_min, lat_min, lon_max, lat_max]の境界ボックス
            
        Returns:
            河川のGeoDataFrame
        """
        try:
            logger.info(f"河川データを読み込み中: {self.rivers_shapefile}")
            
            if bbox:
                # バウンディングボックスでフィルタリング
                rivers_gdf = gpd.read_file(self.rivers_shapefile, bbox=bbox)
            else:
                rivers_gdf = gpd.read_file(self.rivers_shapefile)
            
            # CRSの確認
            if rivers_gdf.crs is None:
                logger.warning("河川データにCRSが設定されていません。EPSG:4326を仮定します。")
                rivers_gdf = rivers_gdf.set_crs("EPSG:4326")
            
            # WGS84に変換
            if rivers_gdf.crs != "EPSG:4326":
                rivers_gdf = rivers_gdf.to_crs("EPSG:4326")
            
            logger.info(f"河川データ読み込み完了: {len(rivers_gdf)}件")
            self.rivers_gdf = rivers_gdf
            
            return rivers_gdf
            
        except Exception as e:
            logger.error(f"河川データ読み込みエラー: {e}")
            raise
    
    def calculate_river_distances(self, toponyms_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        地名から最寄り河川への距離を計算
        
        Args:
            toponyms_gdf: 地名のGeoDataFrame
            
        Returns:
            河川距離情報を追加したGeoDataFrame
        """
        if toponyms_gdf.empty:
            logger.warning("地名データが空です")
            return toponyms_gdf
        
        if self.rivers_gdf is None:
            raise ValueError("河川データが読み込まれていません。load_rivers()を先に実行してください")
        
        logger.info(f"川距離計算開始: {len(toponyms_gdf)}件の地名")
        
        result_gdf = toponyms_gdf.copy()
        
        # 川距離と角度を計算
        river_angles = []
        river_radii = []
        river_names = []
        
        for idx, row in toponyms_gdf.iterrows():
            toponym_point = row.geometry
            
            river_angle, river_radius, river_name = self._find_nearest_river(toponym_point)
            
            river_angles.append(river_angle)
            river_radii.append(river_radius)
            river_names.append(river_name)
        
        # 結果を追加
        result_gdf['river_angle'] = river_angles
        result_gdf['river_radius'] = river_radii
        result_gdf['nearest_river'] = river_names
        
        # 統計情報をログ出力
        valid_distances = [r for r in river_radii if r is not None and r <= self.max_search_distance]
        if valid_distances:
            logger.info(f"川距離計算完了: 有効距離 {len(valid_distances)}/{len(river_radii)}件, "
                       f"距離範囲 {min(valid_distances):.3f}-{max(valid_distances):.3f}km")
        else:
            logger.warning("有効な川距離が計算できませんでした")
        
        return result_gdf
    
    def _find_nearest_river(self, point: Point) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        点から最寄りの川を検索
        
        Args:
            point: 検索対象の点
            
        Returns:
            (角度, 距離, 川名) のタプル
        """
        try:
            # 検索用のバッファを作成（概算）
            buffer_deg = self.max_search_distance / 111.0  # 1度≈111km
            search_area = point.buffer(buffer_deg)
            
            # 検索エリア内の河川を抽出
            nearby_rivers = self.rivers_gdf[self.rivers_gdf.intersects(search_area)]
            
            if nearby_rivers.empty:
                return None, None, None
            
            # 各河川への距離を計算
            min_distance = float('inf')
            nearest_river_point = None
            nearest_river_name = None
            
            for idx, river in nearby_rivers.iterrows():
                river_geom = river.geometry
                
                # 最寄り点を見つける
                nearest_geom = nearest_points(point, river_geom)
                distance_km = self._calculate_distance_km(point, nearest_geom[1])
                
                if distance_km < min_distance and distance_km <= self.max_search_distance:
                    min_distance = distance_km
                    nearest_river_point = nearest_geom[1]
                    nearest_river_name = river.get('MAIN_RIV', river.get('name', f'River_{idx}'))
            
            if nearest_river_point is None:
                return None, None, None
            
            # 角度を計算
            angle = self._calculate_angle_to_point(point, nearest_river_point)
            
            return angle, min_distance, nearest_river_name
            
        except Exception as e:
            logger.error(f"最寄り川検索エラー: {e}")
            return None, None, None
    
    def _calculate_distance_km(self, point1: Point, point2: Point) -> float:
        """
        2点間の距離を計算（km）
        
        Args:
            point1: 点1
            point2: 点2
            
        Returns:
            距離（km）
        """
        # Haversine公式を使用
        R = 6371.0  # 地球の半径（km）
        
        lat1_rad = math.radians(point1.y)
        lon1_rad = math.radians(point1.x)
        lat2_rad = math.radians(point2.y)
        lon2_rad = math.radians(point2.x)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _calculate_angle_to_point(self, from_point: Point, to_point: Point) -> float:
        """
        点から点への角度を計算（北を0度、時計回り）
        
        Args:
            from_point: 起点
            to_point: 終点
            
        Returns:
            角度（度）
        """
        dx = to_point.x - from_point.x
        dy = to_point.y - from_point.y
        
        # atan2による角度計算（東を0度、反時計回り）
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # 北を0度、時計回りに変換
        adjusted_angle = 90 - angle_deg
        
        # 0-360度の範囲に正規化
        while adjusted_angle < 0:
            adjusted_angle += 360
        while adjusted_angle >= 360:
            adjusted_angle -= 360
        
        return adjusted_angle
    
    def create_river_distance_summary(self, river_distance_gdf: gpd.GeoDataFrame) -> Dict:
        """
        川距離データの統計サマリを作成
        
        Args:
            river_distance_gdf: 川距離計算済みのGeoDataFrame
            
        Returns:
            統計情報の辞書
        """
        if (river_distance_gdf.empty or 
            'river_angle' not in river_distance_gdf.columns or
            'river_radius' not in river_distance_gdf.columns):
            return {}
        
        # 有効なデータのみを抽出
        valid_data = river_distance_gdf[
            (river_distance_gdf['river_radius'].notna()) & 
            (river_distance_gdf['river_radius'] <= self.max_search_distance)
        ]
        
        if valid_data.empty:
            return {'total_toponyms': len(river_distance_gdf), 'valid_river_distances': 0}
        
        river_angles = valid_data['river_angle'].dropna()
        river_radii = valid_data['river_radius'].dropna()
        
        summary = {
            'total_toponyms': len(river_distance_gdf),
            'valid_river_distances': len(valid_data),
            'coverage_rate': len(valid_data) / len(river_distance_gdf),
            'river_angle_stats': {
                'min': float(river_angles.min()) if len(river_angles) > 0 else None,
                'max': float(river_angles.max()) if len(river_angles) > 0 else None,
                'mean': float(river_angles.mean()) if len(river_angles) > 0 else None,
                'std': float(river_angles.std()) if len(river_angles) > 0 else None
            },
            'river_radius_stats': {
                'min': float(river_radii.min()) if len(river_radii) > 0 else None,
                'max': float(river_radii.max()) if len(river_radii) > 0 else None,
                'mean': float(river_radii.mean()) if len(river_radii) > 0 else None,
                'std': float(river_radii.std()) if len(river_radii) > 0 else None
            },
            'distance_distribution': self._calculate_distance_distribution(river_radii),
            'river_names': list(valid_data['nearest_river'].value_counts().head(10).to_dict().items())
        }
        
        return summary
    
    def _calculate_distance_distribution(self, distances: pd.Series) -> Dict:
        """距離分布の計算"""
        if len(distances) == 0:
            return {}
        
        # 距離帯での分布
        bins = [0, 1, 5, 10, 20, self.max_search_distance]
        labels = ['0-1km', '1-5km', '5-10km', '10-20km', f'20-{self.max_search_distance}km']
        
        counts, _ = np.histogram(distances, bins=bins)
        
        distribution = {}
        for i, label in enumerate(labels):
            if i < len(counts):
                distribution[label] = int(counts[i])
        
        return distribution