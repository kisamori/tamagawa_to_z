#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
site_clusterer.py: 考古学遺跡の空間クラスタリングモジュール

近接する遺跡（200m以内）を同一遺跡として統合する機能を提供します。
連鎖的な近接性（A-B-C のようなチェーン）も考慮します。

例:
    地点A(0m) - 地点B(150m) - 地点C(250m)
    → A-B間: 150m < 200m (統合)
    → B-C間: 100m < 200m (統合)  
    → 結果: A, B, C すべて同じクラスターとして統合
"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Dict, Set, Tuple
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic

logger = logging.getLogger(__name__)


class SiteClusterer:
    """考古学遺跡の空間クラスタリングクラス"""
    
    def __init__(self, max_distance_m: float = 200.0):
        """
        初期化
        
        Args:
            max_distance_m (float): クラスタリングの最大距離（メートル）
        """
        self.max_distance_m = max_distance_m
        logger.info(f"SiteClusterer初期化: 最大距離 {max_distance_m}m")
    
    def cluster_sites(self, sites_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        遺跡の空間クラスタリングを実行
        
        Args:
            sites_gdf (gpd.GeoDataFrame): 遺跡データ
            
        Returns:
            gpd.GeoDataFrame: クラスタリング済み遺跡データ
        """
        if len(sites_gdf) == 0:
            logger.warning("遺跡データが空です")
            return sites_gdf
        
        logger.info(f"遺跡クラスタリング開始: {len(sites_gdf)}件の遺跡")
        
        # 1. 距離行列を計算
        distance_matrix = self._calculate_distance_matrix(sites_gdf)
        
        # 2. DBSCANによるクラスタリング
        clusters = self._perform_clustering(distance_matrix)
        
        # 3. クラスタリング結果を適用
        clustered_gdf = self._apply_clustering_results(sites_gdf, clusters)
        
        # 4. 統計情報の出力
        self._log_clustering_statistics(sites_gdf, clustered_gdf)
        
        return clustered_gdf
    
    def _calculate_distance_matrix(self, sites_gdf: gpd.GeoDataFrame) -> np.ndarray:
        """
        遺跡間の距離行列を計算
        
        Args:
            sites_gdf (gpd.GeoDataFrame): 遺跡データ
            
        Returns:
            np.ndarray: 距離行列（メートル単位）
        """
        n_sites = len(sites_gdf)
        distance_matrix = np.zeros((n_sites, n_sites))
        
        logger.debug(f"距離行列計算中: {n_sites} x {n_sites}")
        
        # 座標を抽出
        coords = [(point.y, point.x) for point in sites_gdf.geometry]
        
        # 全ペアの距離を計算
        for i in range(n_sites):
            for j in range(i + 1, n_sites):
                # geodesic距離（測地線距離）で正確な距離を計算
                distance_m = geodesic(coords[i], coords[j]).meters
                distance_matrix[i, j] = distance_m
                distance_matrix[j, i] = distance_m  # 対称行列
        
        logger.debug(f"距離行列計算完了: 最小距離 {distance_matrix[distance_matrix > 0].min():.1f}m, "
                    f"最大距離 {distance_matrix.max():.1f}m")
        
        return distance_matrix
    
    def _perform_clustering(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        DBSCANによるクラスタリングを実行
        
        Args:
            distance_matrix (np.ndarray): 距離行列
            
        Returns:
            np.ndarray: クラスター番号配列
        """
        # DBSCANパラメータ
        # eps: 最大距離（km単位でDBSCANに渡す）
        eps_km = self.max_distance_m / 1000.0  
        min_samples = 1  # 1つだけでもクラスターとして認める
        
        logger.debug(f"DBSCAN実行: eps={eps_km:.3f}km, min_samples={min_samples}")
        
        # 距離行列をkm単位に変換
        distance_matrix_km = distance_matrix / 1000.0
        
        # DBSCANクラスタリング（距離行列を直接使用）
        clustering = DBSCAN(
            eps=eps_km,
            min_samples=min_samples,
            metric='precomputed'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix_km)
        
        # 統計情報
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"クラスタリング完了: {n_clusters}個のクラスター, {n_noise}個のノイズ点")
        
        return cluster_labels
    
    def _apply_clustering_results(
        self, 
        sites_gdf: gpd.GeoDataFrame, 
        cluster_labels: np.ndarray
    ) -> gpd.GeoDataFrame:
        """
        クラスタリング結果を遺跡データに適用
        
        Args:
            sites_gdf (gpd.GeoDataFrame): 元の遺跡データ
            cluster_labels (np.ndarray): クラスター番号配列
            
        Returns:
            gpd.GeoDataFrame: 統合済み遺跡データ
        """
        logger.debug("クラスタリング結果適用中...")
        
        # クラスター情報を追加
        sites_with_clusters = sites_gdf.copy()
        sites_with_clusters['cluster_id'] = cluster_labels
        
        # クラスターごとに遺跡を統合
        clustered_sites = []
        
        # 各クラスターについて処理
        unique_clusters = set(cluster_labels)
        
        for cluster_id in unique_clusters:
            cluster_sites = sites_with_clusters[sites_with_clusters['cluster_id'] == cluster_id]
            
            if cluster_id == -1:
                # ノイズ点（孤立遺跡）は個別に保持
                for _, site in cluster_sites.iterrows():
                    clustered_sites.append(self._create_individual_site(site))
            else:
                # クラスター内の遺跡を統合
                merged_site = self._merge_cluster_sites(cluster_sites, cluster_id)
                clustered_sites.append(merged_site)
        
        # 新しいGeoDataFrameを作成
        if clustered_sites:
            result_gdf = gpd.GeoDataFrame(clustered_sites, crs=sites_gdf.crs)
        else:
            # 空の場合は元のスキーマを保持
            result_gdf = sites_gdf.iloc[0:0].copy()
        
        logger.debug(f"統合完了: {len(sites_gdf)}件 → {len(result_gdf)}件")
        
        return result_gdf
    
    def _create_individual_site(self, site: pd.Series) -> Dict:
        """
        個別遺跡（クラスター化されない遺跡）の情報を作成
        
        Args:
            site (pd.Series): 遺跡情報
            
        Returns:
            Dict: 統合された遺跡情報
        """
        return {
            'site_name': site.get('site_name', f"Site_{site.name}"),
            'original_names': [site.get('site_name', f"Site_{site.name}")],
            'n_merged_sites': 1,
            'geometry': site.geometry,
            'cluster_id': site.get('cluster_id', -1),
            'merged_coordinates': [(site.geometry.y, site.geometry.x)],
            'region': site.get('region', ''),
            'culture_tag': site.get('culture_tag', '')
        }
    
    def _merge_cluster_sites(self, cluster_sites: gpd.GeoDataFrame, cluster_id: int) -> Dict:
        """
        クラスター内の複数遺跡を統合
        
        Args:
            cluster_sites (gpd.GeoDataFrame): クラスター内の遺跡群
            cluster_id (int): クラスターID
            
        Returns:
            Dict: 統合された遺跡情報
        """
        # 統合後の遺跡名を生成
        original_names = cluster_sites['site_name'].tolist()
        primary_name = original_names[0]  # 最初の遺跡名を主名とする
        
        # 統合後の名前
        if len(original_names) == 1:
            merged_name = primary_name
        else:
            merged_name = f"{primary_name}_cluster{cluster_id}"
        
        # 中心座標を計算（重心）
        coords = [(point.y, point.x) for point in cluster_sites.geometry]
        center_lat = np.mean([coord[0] for coord in coords])
        center_lon = np.mean([coord[1] for coord in coords])
        center_point = Point(center_lon, center_lat)
        
        # 統合情報
        merged_site = {
            'site_name': merged_name,
            'original_names': original_names,
            'n_merged_sites': len(cluster_sites),
            'geometry': center_point,
            'cluster_id': cluster_id,
            'merged_coordinates': coords,
            'region': cluster_sites.iloc[0].get('region', ''),
            'culture_tag': cluster_sites.iloc[0].get('culture_tag', '')
        }
        
        logger.debug(f"クラスター{cluster_id}: {len(original_names)}件統合 → '{merged_name}'")
        
        return merged_site
    
    def _log_clustering_statistics(
        self, 
        original_gdf: gpd.GeoDataFrame, 
        clustered_gdf: gpd.GeoDataFrame
    ) -> None:
        """
        クラスタリング統計情報をログ出力
        
        Args:
            original_gdf (gpd.GeoDataFrame): 元の遺跡データ
            clustered_gdf (gpd.GeoDataFrame): クラスタリング後のデータ
        """
        original_count = len(original_gdf)
        clustered_count = len(clustered_gdf)
        reduction = original_count - clustered_count
        reduction_pct = (reduction / original_count * 100) if original_count > 0 else 0
        
        logger.info(f"=== クラスタリング統計 ===")
        logger.info(f"元の遺跡数: {original_count}件")
        logger.info(f"統合後遺跡数: {clustered_count}件")
        logger.info(f"削減数: {reduction}件 ({reduction_pct:.1f}%)")
        
        # 統合詳細
        if 'n_merged_sites' in clustered_gdf.columns:
            merged_sites = clustered_gdf[clustered_gdf['n_merged_sites'] > 1]
            if len(merged_sites) > 0:
                logger.info(f"統合クラスター数: {len(merged_sites)}個")
                for _, site in merged_sites.iterrows():
                    original_names = site.get('original_names', [])
                    logger.info(f"  '{site['site_name']}': {site['n_merged_sites']}件統合 "
                               f"({', '.join(original_names[:3])}{'...' if len(original_names) > 3 else ''})")
    
    def get_clustering_summary(self, clustered_gdf: gpd.GeoDataFrame) -> Dict:
        """
        クラスタリング結果のサマリーを取得
        
        Args:
            clustered_gdf (gpd.GeoDataFrame): クラスタリング済みデータ
            
        Returns:
            Dict: サマリー情報
        """
        if 'n_merged_sites' not in clustered_gdf.columns:
            return {'total_sites': len(clustered_gdf), 'merged_clusters': 0}
        
        total_sites = len(clustered_gdf)
        merged_clusters = len(clustered_gdf[clustered_gdf['n_merged_sites'] > 1])
        individual_sites = len(clustered_gdf[clustered_gdf['n_merged_sites'] == 1])
        total_original_sites = clustered_gdf['n_merged_sites'].sum()
        
        return {
            'total_sites': total_sites,
            'merged_clusters': merged_clusters,
            'individual_sites': individual_sites,
            'total_original_sites': int(total_original_sites),
            'reduction_count': int(total_original_sites - total_sites),
            'reduction_percentage': float((total_original_sites - total_sites) / total_original_sites * 100) if total_original_sites > 0 else 0.0
        }