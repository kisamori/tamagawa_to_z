#!/usr/bin/env python3
"""最良パラメータ再実行CLI - 最適化で得られた最良パラメータで再実行."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# 地図背景用
try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    logger = logging.getLogger(__name__)

# パッケージのインポートパス追加
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tamagawa_to_z.dataset.splitter import DataSplitter
from tamagawa_to_z.tuning.pipeline_runner import run_pipeline_with_params

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def visualize_candidates(candidates_file: Path, experiment_id: str, output_dir: Path):
    """候補地点の可視化を実行"""
    if not candidates_file.exists():
        logger.warning(f"候補ファイルが見つかりません: {candidates_file}")
        return
    
    logger.info(f"=== 📊 可視化実行: {candidates_file} ===")
    
    try:
        # 候補データ読み込み
        candidates = pd.read_csv(candidates_file)
        
        if len(candidates) == 0:
            logger.warning("候補データが空です")
            return
        
        # 空間データに変換
        if 'lat' in candidates.columns and 'lon' in candidates.columns:
            gdf = gpd.GeoDataFrame(
                candidates, 
                geometry=gpd.points_from_xy(candidates['lon'], candidates['lat']),
                crs='EPSG:4326'
            )
        else:
            logger.warning("座標情報が見つかりません")
            return
        
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 可視化1: 候補地点分布
        _plot_candidate_distribution(gdf, experiment_id, output_dir)
        
        # 可視化2: 距離分析
        if 'dist_km' in gdf.columns:
            _plot_distance_analysis(gdf, experiment_id, output_dir)
        
        # 可視化3: 水域頻度分析
        if 'occ_pct' in gdf.columns:
            _plot_water_frequency_analysis(gdf, experiment_id, output_dir)
        
        # 可視化4: 最終結果マップ
        if 'total_score' in gdf.columns:
            _plot_final_results_map(gdf, experiment_id, output_dir)
        
        logger.info(f"✅ 可視化完了: {output_dir}")
        
    except Exception as e:
        logger.error(f"可視化中にエラーが発生しました: {e}")


def _plot_candidate_distribution(gdf: gpd.GeoDataFrame, experiment_id: str, output_dir: Path):
    """候補地点分布の可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 座標系をWeb Mercatorに変換（地図背景用）
    gdf_web = gdf.to_crs(epsg=3857)
    
    # 1. 地理的分布
    ax1 = axes[0, 0]
    gdf_web.plot(ax=ax1, color='red', alpha=0.6, markersize=30)
    
    # 地図背景を追加
    if HAS_CONTEXTILY:
        try:
            ctx.add_basemap(ax1, crs=gdf_web.crs.to_string(), 
                           source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            logger.warning(f"地図背景の追加に失敗: {e}")
    
    ax1.set_title('Geographic Distribution of Candidates', fontweight='bold')
    ax1.set_xlabel('X Coordinate (Web Mercator)')
    ax1.set_ylabel('Y Coordinate (Web Mercator)')
    
    # 2. タイプ別分布
    ax2 = axes[0, 1]
    if 'type' in gdf.columns:
        type_counts = gdf['type'].value_counts()
        type_counts.plot(kind='bar', ax=ax2)
        ax2.set_title('Distribution by Water Type')
        ax2.set_xlabel('Water Type')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. スコア分布
    ax3 = axes[1, 0]
    if 'total_score' in gdf.columns:
        ax3.hist(gdf['total_score'], bins=20, alpha=0.7, color='blue')
        ax3.set_title('Score Distribution')
        ax3.set_xlabel('Total Score')
        ax3.set_ylabel('Frequency')
    
    # 4. 統計サマリー
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"Candidate Statistics\n\n"
    stats_text += f"Total Candidates: {len(gdf)}\n"
    if 'total_score' in gdf.columns:
        stats_text += f"Max Score: {gdf['total_score'].max():.3f}\n"
        stats_text += f"Mean Score: {gdf['total_score'].mean():.3f}\n"
        stats_text += f"Min Score: {gdf['total_score'].min():.3f}\n"
    if 'type' in gdf.columns:
        stats_text += f"Unique Types: {gdf['type'].nunique()}\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    output_path = output_dir / f'{experiment_id}_candidate_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"📊 候補分布図を保存: {output_path}")


def _plot_distance_analysis(gdf: gpd.GeoDataFrame, experiment_id: str, output_dir: Path):
    """距離分析の可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 距離分布
    ax1 = axes[0, 0]
    ax1.hist(gdf['dist_km'], bins=20, alpha=0.7, color='green')
    ax1.set_title('Distance from Current Rivers Distribution')
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Frequency')
    
    # 2. 距離 vs スコア
    ax2 = axes[0, 1]
    if 'total_score' in gdf.columns:
        ax2.scatter(gdf['dist_km'], gdf['total_score'], alpha=0.6)
        ax2.set_title('Distance vs Total Score')
        ax2.set_xlabel('Distance (km)')
        ax2.set_ylabel('Total Score')
    
    # 3. 距離範囲別統計
    ax3 = axes[1, 0]
    distance_bins = [0, 1, 2, 3, 4, 5, float('inf')]
    distance_labels = ['0-1km', '1-2km', '2-3km', '3-4km', '4-5km', '5km+']
    gdf['dist_range'] = pd.cut(gdf['dist_km'], bins=distance_bins, labels=distance_labels)
    dist_counts = gdf['dist_range'].value_counts().sort_index()
    dist_counts.plot(kind='bar', ax=ax3)
    ax3.set_title('Candidates by Distance Range')
    ax3.set_xlabel('Distance Range')
    ax3.set_ylabel('Count')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 統計サマリー
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"Distance Statistics\n\n"
    stats_text += f"Mean Distance: {gdf['dist_km'].mean():.2f} km\n"
    stats_text += f"Median Distance: {gdf['dist_km'].median():.2f} km\n"
    stats_text += f"Max Distance: {gdf['dist_km'].max():.2f} km\n"
    stats_text += f"Min Distance: {gdf['dist_km'].min():.2f} km\n"
    stats_text += f"Std Distance: {gdf['dist_km'].std():.2f} km\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    output_path = output_dir / f'{experiment_id}_distance_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"📊 距離分析図を保存: {output_path}")


def _plot_water_frequency_analysis(gdf: gpd.GeoDataFrame, experiment_id: str, output_dir: Path):
    """水域頻度分析の可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 水域頻度分布
    ax1 = axes[0, 0]
    ax1.hist(gdf['occ_pct'], bins=20, alpha=0.7, color='cyan')
    ax1.set_title('Water Occurrence Frequency Distribution')
    ax1.set_xlabel('Water Occurrence (%)')
    ax1.set_ylabel('Frequency')
    
    # 2. 水域頻度 vs スコア
    ax2 = axes[0, 1]
    if 'total_score' in gdf.columns:
        ax2.scatter(gdf['occ_pct'], gdf['total_score'], alpha=0.6)
        ax2.set_title('Water Occurrence vs Total Score')
        ax2.set_xlabel('Water Occurrence (%)')
        ax2.set_ylabel('Total Score')
    
    # 3. 距離 vs 水域頻度
    ax3 = axes[1, 0]
    if 'dist_km' in gdf.columns:
        ax3.scatter(gdf['dist_km'], gdf['occ_pct'], alpha=0.6, color='orange')
        ax3.set_title('Distance vs Water Occurrence')
        ax3.set_xlabel('Distance (km)')
        ax3.set_ylabel('Water Occurrence (%)')
    
    # 4. 統計サマリー
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"Water Occurrence Statistics\n\n"
    stats_text += f"Mean Occurrence: {gdf['occ_pct'].mean():.2f}%\n"
    stats_text += f"Median Occurrence: {gdf['occ_pct'].median():.2f}%\n"
    stats_text += f"Max Occurrence: {gdf['occ_pct'].max():.2f}%\n"
    stats_text += f"Min Occurrence: {gdf['occ_pct'].min():.2f}%\n"
    stats_text += f"Std Occurrence: {gdf['occ_pct'].std():.2f}%\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    output_path = output_dir / f'{experiment_id}_water_frequency_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"📊 水域頻度分析図を保存: {output_path}")


def _plot_final_results_map(gdf: gpd.GeoDataFrame, experiment_id: str, output_dir: Path):
    """最終結果マップの可視化"""
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # 座標系をWeb Mercatorに変換（地図背景用）
    gdf_web = gdf.to_crs(epsg=3857)
    
    # トップ候補を抽出（例：スコア上位30%）
    top_threshold = gdf['total_score'].quantile(0.7)
    top_candidates = gdf_web[gdf['total_score'] >= top_threshold]
    other_candidates = gdf_web[gdf['total_score'] < top_threshold]
    
    # 背景の候補地点を薄いグレーで描画
    if len(other_candidates) > 0:
        other_candidates.plot(ax=ax, color='lightgray', alpha=0.3, markersize=15)
    
    # トップ候補をスコア別にカラーコードで描画
    if len(top_candidates) > 0:
        top_candidates.plot(ax=ax, column='total_score', cmap='plasma', 
                           markersize=80, alpha=0.8, legend=True, 
                           edgecolors='white', linewidth=1)
    
    # 地図の範囲を設定
    if len(gdf_web) > 0:
        bounds = gdf_web.total_bounds
        margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
        ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
        ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
    
    # 地図背景を追加
    if HAS_CONTEXTILY:
        try:
            ctx.add_basemap(ax, crs=gdf_web.crs.to_string(), 
                           source=ctx.providers.OpenStreetMap.Mapnik)
            logger.info("🗺️ OpenStreetMap背景を追加しました")
        except Exception as e:
            logger.warning(f"地図背景の追加に失敗: {e}")
    
    ax.set_title('Paleochannel Candidate Sites (Color-coded by Score)', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('X Coordinate (Web Mercator)')
    ax.set_ylabel('Y Coordinate (Web Mercator)')
    
    # カラーバーのラベル設定
    legend = ax.get_legend()
    if legend:
        legend.set_title('Total Score', prop={'size': 12, 'weight': 'bold'})
        legend.set_bbox_to_anchor((1.05, 1))
        legend.set_loc('upper left')
    
    # 統計情報をテキストで追加
    stats_text = f"Candidate Sites: {len(gdf)}\n"
    stats_text += f"Top Candidates: {len(top_candidates)}\n"
    if len(gdf) > 0:
        stats_text += f"Max Score: {gdf['total_score'].max():.3f}\n"
        stats_text += f"Mean Score: {gdf['total_score'].mean():.3f}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='white', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / f'{experiment_id}_final_results_map.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"📊 最終結果マップを保存: {output_path}")


def create_step_by_step_visualizations(intermediate_data: Dict, experiment_id: str, output_dir: Path):
    """S1-S5のステップバイステップ可視化を作成"""
    logger.info("=== 📊 S1-S5ステップ可視化作成開始 ===")
    
    # 必要なデータを抽出
    bbox_gdf = intermediate_data.get('bbox_gdf')
    toponyms = intermediate_data.get('toponyms')
    processed_toponyms = intermediate_data.get('processed_toponyms')
    with_distance = intermediate_data.get('with_distance')
    candidates = intermediate_data.get('candidates')
    
    # S-1: 対象地域BBox可視化
    if bbox_gdf is not None:
        _plot_step1_bbox(bbox_gdf, experiment_id, output_dir)
    
    # S-2: トポニム分布可視化
    if toponyms is not None and len(toponyms) > 0:
        _plot_step2_toponyms(toponyms, bbox_gdf, experiment_id, output_dir)
    
    # S-3: タイプ別分布可視化
    if processed_toponyms is not None and len(processed_toponyms) > 0:
        _plot_step3_types(processed_toponyms, experiment_id, output_dir)
    
    # S-4: 距離分析可視化
    if with_distance is not None and len(with_distance) > 0:
        _plot_step4_distances(with_distance, experiment_id, output_dir)
    
    # S-5: 最終候補可視化
    if candidates is not None and len(candidates) > 0:
        _plot_step5_candidates(candidates, with_distance, experiment_id, output_dir)
    
    logger.info(f"✅ S1-S5ステップ可視化完了: {output_dir}")


def _plot_step1_bbox(bbox_gdf: gpd.GeoDataFrame, experiment_id: str, output_dir: Path):
    """S-1: 対象地域BBox可視化"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 座標系をWeb Mercatorに変換（地図背景用）
    bbox_gdf_web = bbox_gdf.to_crs(epsg=3857)
    bbox_gdf_web.plot(ax=ax, color='none', edgecolor='red', linewidth=3)
    
    # 地図背景を追加
    if HAS_CONTEXTILY:
        try:
            ctx.add_basemap(ax, crs=bbox_gdf_web.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
            logger.info("🗺️ OpenStreetMap背景を追加しました")
        except Exception as e:
            logger.warning(f"地図背景の追加に失敗: {e}")
    
    ax.set_title('S-1: Target Region Definition', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Coordinate (Web Mercator)')
    ax.set_ylabel('Y Coordinate (Web Mercator)')
    plt.tight_layout()
    
    output_path = output_dir / f'{experiment_id}_step1_bbox.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"📊 S-1可視化保存: {output_path}")


def _plot_step2_toponyms(toponyms, bbox_gdf, experiment_id: str, output_dir: Path):
    """S-2: トポニム分布可視化"""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # GeoDataFrameかどうかチェック
    if not hasattr(toponyms, 'to_crs'):
        logger.warning("トポニムデータがGeoDataFrameではありません。lon/latから変換します。")
        if 'lon' in toponyms.columns and 'lat' in toponyms.columns:
            toponyms = gpd.GeoDataFrame(
                toponyms, 
                geometry=gpd.points_from_xy(toponyms['lon'], toponyms['lat']),
                crs='EPSG:4326'
            )
        else:
            logger.error("トポニムデータにlon/lat列がありません")
            return
    
    # 座標系をWeb Mercatorに変換（地図背景用）
    bbox_gdf_web = bbox_gdf.to_crs(epsg=3857) if bbox_gdf is not None else None
    toponyms_web = toponyms.to_crs(epsg=3857)
    
    # 対象領域の境界を描画
    if bbox_gdf_web is not None:
        bbox_gdf_web.plot(ax=ax, color='none', edgecolor='red', linewidth=3, alpha=0.8)
    
    # トポニムを描画
    if 'source' in toponyms_web.columns:
        toponyms_web.plot(ax=ax, column='source', cmap='Set1', markersize=50, 
                         legend=True, alpha=0.8, edgecolors='white', linewidth=0.5)
    else:
        toponyms_web.plot(ax=ax, color='blue', markersize=50, alpha=0.8, 
                         edgecolors='white', linewidth=0.5)
    
    # 地図の範囲を設定
    if bbox_gdf_web is not None:
        bounds = bbox_gdf_web.total_bounds
        margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
        ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
        ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
    
    # 地図背景を追加
    if HAS_CONTEXTILY:
        try:
            ctx.add_basemap(ax, crs=toponyms_web.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
            logger.info("🗺️ OpenStreetMap背景を追加しました")
        except Exception as e:
            logger.warning(f"地図背景の追加に失敗: {e}")
    
    ax.set_title('S-2: Water Toponyms Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('X Coordinate (Web Mercator)')
    ax.set_ylabel('Y Coordinate (Web Mercator)')
    
    # 凡例の調整
    if 'source' in toponyms_web.columns:
        legend = ax.get_legend()
        if legend:
            legend.set_bbox_to_anchor((1.05, 1))
            legend.set_loc('upper left')
    
    plt.tight_layout()
    
    output_path = output_dir / f'{experiment_id}_step2_toponyms.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"📊 S-2可視化保存: {output_path}")


def _plot_step3_types(processed_toponyms, experiment_id: str, output_dir: Path):
    """S-3: タイプ別分布可視化"""
    # GeoDataFrameかどうかチェック
    if not hasattr(processed_toponyms, 'to_crs'):
        logger.warning("処理済みトポニムデータがGeoDataFrameではありません。")
        if 'lon' in processed_toponyms.columns and 'lat' in processed_toponyms.columns:
            processed_toponyms = gpd.GeoDataFrame(
                processed_toponyms, 
                geometry=gpd.points_from_xy(processed_toponyms['lon'], processed_toponyms['lat']),
                crs='EPSG:4326'
            )
        else:
            logger.warning("処理済みトポニムデータに座標情報がありません")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'type' in processed_toponyms.columns:
        type_counts = processed_toponyms['type'].value_counts()
        type_counts.plot(kind='bar', ax=ax)
        ax.set_title('S-3: Distribution by Water System Type')
        ax.set_xlabel('Water System Type')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, 'No type information available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('S-3: Water System Type Analysis')
    
    plt.tight_layout()
    
    output_path = output_dir / f'{experiment_id}_step3_types.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"📊 S-3可視化保存: {output_path}")


def _plot_step4_distances(with_distance, experiment_id: str, output_dir: Path):
    """S-4: 距離分析可視化"""
    # GeoDataFrameかどうかチェック
    if not hasattr(with_distance, 'to_crs'):
        logger.warning("距離データがGeoDataFrameではありません。")
        if 'lon' in with_distance.columns and 'lat' in with_distance.columns:
            with_distance = gpd.GeoDataFrame(
                with_distance, 
                geometry=gpd.points_from_xy(with_distance['lon'], with_distance['lat']),
                crs='EPSG:4326'
            )
        else:
            logger.warning("距離データに座標情報がありません")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'dist_km' in with_distance.columns:
        ax.hist(with_distance['dist_km'], bins=20, alpha=0.7, color='green')
        ax.set_title('S-4: Distance from Current Rivers Distribution')
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Frequency')
        ax.axvline(x=3, color='red', linestyle='--', label='Threshold (3km)')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No distance information available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('S-4: Distance Analysis')
    
    plt.tight_layout()
    
    output_path = output_dir / f'{experiment_id}_step4_distances.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"📊 S-4可視化保存: {output_path}")


def _plot_step5_candidates(candidates, all_toponyms, experiment_id: str, output_dir: Path):
    """S-5: 最終候補可視化"""
    # GeoDataFrameかどうかチェック
    if not hasattr(candidates, 'to_crs'):
        logger.warning("候補データがGeoDataFrameではありません。")
        if 'lon' in candidates.columns and 'lat' in candidates.columns:
            candidates = gpd.GeoDataFrame(
                candidates, 
                geometry=gpd.points_from_xy(candidates['lon'], candidates['lat']),
                crs='EPSG:4326'
            )
        else:
            logger.warning("候補データに座標情報がありません")
            return
    
    if all_toponyms is not None and not hasattr(all_toponyms, 'to_crs'):
        logger.warning("全トポニムデータがGeoDataFrameではありません。")
        if 'lon' in all_toponyms.columns and 'lat' in all_toponyms.columns:
            all_toponyms = gpd.GeoDataFrame(
                all_toponyms, 
                geometry=gpd.points_from_xy(all_toponyms['lon'], all_toponyms['lat']),
                crs='EPSG:4326'
            )
        else:
            logger.warning("全トポニムデータに座標情報がありません")
            all_toponyms = None
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # 座標系をWeb Mercatorに変換（地図背景用）
    all_toponyms_web = all_toponyms.to_crs(epsg=3857) if all_toponyms is not None else None
    candidates_web = candidates.to_crs(epsg=3857)
    
    # 全地名を薄いグレーで背景として描画
    if all_toponyms_web is not None:
        all_toponyms_web.plot(ax=ax, color='lightgray', alpha=0.3, markersize=15)
    
    # 候補地点をスコア別にカラーコードで描画
    if 'total_score' in candidates.columns:
        candidates_web.plot(ax=ax, column='total_score', cmap='plasma', 
                           markersize=80, alpha=0.8, legend=True, 
                           edgecolors='white', linewidth=1)
    else:
        candidates_web.plot(ax=ax, color='red', markersize=80, alpha=0.8, 
                           edgecolors='white', linewidth=1)
    
    # 地図の範囲を設定
    if len(candidates_web) > 0:
        bounds = candidates_web.total_bounds
        margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.2
        ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
        ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
    
    # 地図背景を追加
    if HAS_CONTEXTILY:
        try:
            ctx.add_basemap(ax, crs=candidates_web.crs.to_string(), 
                           source=ctx.providers.OpenStreetMap.Mapnik)
            logger.info("🗺️ OpenStreetMap背景を追加しました")
        except Exception as e:
            logger.warning(f"地図背景の追加に失敗: {e}")
    
    ax.set_title('S-5: Paleochannel Candidate Sites (Color-coded by Score)', fontsize=16, fontweight='bold')
    ax.set_xlabel('X Coordinate (Web Mercator)')
    ax.set_ylabel('Y Coordinate (Web Mercator)')
    
    # カラーバーのラベル設定
    legend = ax.get_legend()
    if legend:
        legend.set_title('Total Score', prop={'size': 12, 'weight': 'bold'})
        legend.set_bbox_to_anchor((1.05, 1))
        legend.set_loc('upper left')
    
    # 統計情報をテキストで追加
    stats_text = f"Candidate Sites: {len(candidates)}\n"
    if len(candidates) > 0 and 'total_score' in candidates.columns:
        stats_text += f"Max Score: {candidates['total_score'].max():.2f}\n"
        stats_text += f"Mean Score: {candidates['total_score'].mean():.2f}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='white', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    
    output_path = output_dir / f'{experiment_id}_step5_candidates.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"📊 S-5可視化保存: {output_path}")


def create_comprehensive_dashboard(candidates_files: Dict[str, Path], output_dir: Path, experiment_id: str):
    """複数データセットの包括的ダッシュボード作成"""
    logger.info("=== 📊 包括的ダッシュボード作成開始 ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # データセット別の統計情報を収集
    dataset_stats = {}
    all_data = {}
    
    for dataset_name, file_path in candidates_files.items():
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                dataset_stats[dataset_name] = {
                    'count': len(df),
                    'mean_score': df['total_score'].mean() if 'total_score' in df.columns else 0,
                    'max_score': df['total_score'].max() if 'total_score' in df.columns else 0,
                    'mean_distance': df['dist_km'].mean() if 'dist_km' in df.columns else 0,
                    'mean_water_freq': df['occ_pct'].mean() if 'occ_pct' in df.columns else 0
                }
                all_data[dataset_name] = df
        except Exception as e:
            logger.warning(f"データセット {dataset_name} の読み込みに失敗: {e}")
    
    if not dataset_stats:
        logger.warning("有効なデータセットが見つかりません")
        return
    
    # 1. 候補数比較
    ax1 = axes[0, 0]
    datasets = list(dataset_stats.keys())
    counts = [dataset_stats[ds]['count'] for ds in datasets]
    bars = ax1.bar(datasets, counts, color=['blue', 'green', 'orange', 'red', 'purple'][:len(datasets)])
    ax1.set_title('Candidate Count by Dataset')
    ax1.set_ylabel('Number of Candidates')
    ax1.tick_params(axis='x', rotation=45)
    
    # バーの上に数値を表示
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    # 2. 平均スコア比較
    ax2 = axes[0, 1]
    mean_scores = [dataset_stats[ds]['mean_score'] for ds in datasets]
    bars = ax2.bar(datasets, mean_scores, color=['blue', 'green', 'orange', 'red', 'purple'][:len(datasets)])
    ax2.set_title('Mean Score by Dataset')
    ax2.set_ylabel('Mean Total Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. スコア分布比較
    ax3 = axes[0, 2]
    for i, (dataset_name, df) in enumerate(all_data.items()):
        if 'total_score' in df.columns:
            ax3.hist(df['total_score'], alpha=0.6, label=dataset_name, bins=15)
    ax3.set_title('Score Distribution Comparison')
    ax3.set_xlabel('Total Score')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # 4. 距離分布比較
    ax4 = axes[1, 0]
    for i, (dataset_name, df) in enumerate(all_data.items()):
        if 'dist_km' in df.columns:
            ax4.hist(df['dist_km'], alpha=0.6, label=dataset_name, bins=15)
    ax4.set_title('Distance Distribution Comparison')
    ax4.set_xlabel('Distance (km)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    
    # 5. 水域頻度分布比較
    ax5 = axes[1, 1]
    for i, (dataset_name, df) in enumerate(all_data.items()):
        if 'occ_pct' in df.columns:
            ax5.hist(df['occ_pct'], alpha=0.6, label=dataset_name, bins=15)
    ax5.set_title('Water Occurrence Distribution Comparison')
    ax5.set_xlabel('Water Occurrence (%)')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    
    # 6. 統計サマリーテーブル
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # テーブルデータ準備
    table_data = []
    headers = ['Dataset', 'Count', 'Mean Score', 'Max Score', 'Mean Dist(km)', 'Mean Water(%)']
    
    for dataset_name in datasets:
        stats = dataset_stats[dataset_name]
        table_data.append([
            dataset_name[:10],  # 名前を短縮
            f"{stats['count']}",
            f"{stats['mean_score']:.3f}",
            f"{stats['max_score']:.3f}",
            f"{stats['mean_distance']:.1f}",
            f"{stats['mean_water_freq']:.1f}"
        ])
    
    table = ax6.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax6.set_title('Dataset Comparison Summary')
    
    plt.tight_layout()
    output_path = output_dir / f'{experiment_id}_comprehensive_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"📊 包括的ダッシュボードを保存: {output_path}")


def main():
    """最良パラメータでパイプラインを再実行し、詳細な評価を行う."""
    parser = argparse.ArgumentParser(
        description='最適化で得られた最良パラメータでパイプラインを再実行',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--params', '-p',
        required=True,
        help='最良パラメータJSONファイルパス'
    )
    parser.add_argument(
        '--dataset-config', '-d',
        default='configs/dataset_split.yaml',
        help='データセット分割設定ファイル (デフォルト: configs/dataset_split.yaml)'
    )
    parser.add_argument(
        '--sites', '-s',
        default='data/known/known_acre.kmz',
        help='遺跡ファイルパス (.kmz/.csv/.gpkg) (デフォルト: data/known/known_acre.kmz)'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/output/best_run',
        help='出力ディレクトリ (デフォルト: data/output/best_run)'
    )
    parser.add_argument(
        '--experiment-id', '-e',
        help='実験ID（指定しない場合は自動生成）'
    )
    parser.add_argument(
        '--validation-only',
        action='store_true',
        help='Validationデータのみで評価（Testデータは使用しない）'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='詳細ログを表示'
    )
    parser.add_argument(
        '--run-analysis',
        action='store_true',
        help='実行後にinspectorとresearcherを自動実行'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='処理結果を可視化する'
    )
    parser.add_argument(
        '--viz-output-dir',
        type=str,
        default='data/plots',
        help='可視化画像の出力ディレクトリ (デフォルト: data/plots)'
    )
    
    args = parser.parse_args()
    
    # パスオブジェクトに変換
    params_json = Path(args.params)
    dataset_config = Path(args.dataset_config)
    sites_file = Path(args.sites)
    output_dir = Path(args.output)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=== 最良パラメータ再実行開始 ===")
    logger.info(f"パラメータファイル: {params_json}")
    logger.info(f"出力ディレクトリ: {output_dir}")
    
    try:
        # 入力ファイルの存在確認
        required_files = [params_json, dataset_config, sites_file]
        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"ファイルが見つかりません: {file_path}")
                sys.exit(1)
        
        # パラメータ読み込み
        logger.info("最良パラメータを読み込み中...")
        with open(params_json, 'r', encoding='utf-8') as f:
            best_params = json.load(f)
        
        logger.info("🏆 最良パラメータ:")
        logger.info(f"  距離しきい値: {best_params['distance_km']:.2f} km")
        logger.info(f"  水域出現率: {best_params['occ_pct']:.2f} %")
        logger.info(f"  語根ウェイト数: {len(best_params['root_weights'])}")
        logger.info(f"  最適化スコア: {best_params.get('score', 'N/A')}")
        
        # データ分割器初期化
        logger.info("データ分割器を初期化中...")
        splitter = DataSplitter(dataset_config, sites_file)
        splits = splitter.split()
        
        # 実験ID設定
        if args.experiment_id is None:
            from datetime import datetime
            experiment_id = f"best_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            experiment_id = args.experiment_id
        
        logger.info(f"実験ID: {experiment_id}")
        
        # 出力ディレクトリ作成
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 可視化準備（早い段階でディレクトリ作成）
        viz_output_dir = None
        if args.visualize:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_output_dir = Path(args.viz_output_dir) / f"best_params_{timestamp}"
            viz_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 可視化出力ディレクトリ: {viz_output_dir}")
        
        # 評価結果格納
        evaluation_results = {}
        
        # Validation セットで評価（可視化用に中間データも取得）
        logger.info("=== Validation セット評価 ===")
        val_score, val_fp, intermediate_data = run_pipeline_with_params(
            distance_km=best_params['distance_km'],
            occ_pct=best_params['occ_pct'],
            root_weights=best_params['root_weights'],
            validation_set=splits['val'],
            return_fp=True,
            return_intermediate=True,
            experiment_id=f"{experiment_id}_val"
        )
        
        evaluation_results['validation'] = {
            'score': val_score,
            'n_sites': len(splits['val']),
            'false_positives': val_fp
        }
        
        logger.info(f"✅ Validation スコア: {val_score:.4f} ({len(splits['val'])} sites)")
        
        # 可視化実行（validation直後）
        if args.visualize and intermediate_data:
            logger.info("=== S1-S5ステップ可視化実行 ===")
            try:
                create_step_by_step_visualizations(intermediate_data, experiment_id, viz_output_dir)
                logger.info("✅ S1-S5ステップ可視化完了")
            except Exception as e:
                logger.error(f"S1-S5ステップ可視化エラー: {e}")
                import traceback
                traceback.print_exc()
        
        if not args.validation_only:
            # Test-time セットで評価
            if 'test_time' in splits and len(splits['test_time']) > 0:
                logger.info("=== Test-time セット評価 ===")
                test_time_score, test_time_fp = run_pipeline_with_params(
                    distance_km=best_params['distance_km'],
                    occ_pct=best_params['occ_pct'],
                    root_weights=best_params['root_weights'],
                    validation_set=splits['test_time'],
                    return_fp=True,
                    experiment_id=f"{experiment_id}_test_time"
                )
                
                evaluation_results['test_time'] = {
                    'score': test_time_score,
                    'n_sites': len(splits['test_time']),
                    'false_positives': test_time_fp
                }
                
                logger.info(f"✅ Test-time スコア: {test_time_score:.4f} ({len(splits['test_time'])} sites)")
            else:
                logger.info("⚠️  Test-time データが見つかりません")
            
            # Test-region セットで評価
            if 'test_region' in splits:
                logger.info("=== Test-region セット評価 ===")
                evaluation_results['test_region'] = {}
                
                for region_name, region_data in splits['test_region'].items():
                    if len(region_data) > 0:
                        logger.info(f"  📍 {region_name} region...")
                        region_score, region_fp = run_pipeline_with_params(
                            distance_km=best_params['distance_km'],
                            occ_pct=best_params['occ_pct'],
                            root_weights=best_params['root_weights'],
                            validation_set=region_data,
                            return_fp=True,
                            experiment_id=f"{experiment_id}_region_{region_name}"
                        )
                        
                        evaluation_results['test_region'][region_name] = {
                            'score': region_score,
                            'n_sites': len(region_data),
                            'false_positives': region_fp
                        }
                        
                        logger.info(f"  ✅ {region_name}: {region_score:.4f} ({len(region_data)} sites)")
                    else:
                        logger.info(f"  ⚠️  {region_name}: データなし")
        
        # 詳細結果の作成
        detailed_results = {
            'experiment_id': experiment_id,
            'best_parameters': best_params,
            'evaluation_results': evaluation_results,
            'dataset_info': splitter.get_stats()
        }
        
        # 結果保存
        results_file = output_dir / f"{experiment_id}_evaluation.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 詳細結果を保存: {results_file}")
        
        # サマリー表示
        logger.info("=== 評価サマリー ===")
        for eval_name, eval_data in evaluation_results.items():
            if isinstance(eval_data, dict) and 'score' in eval_data:
                logger.info(f"  {eval_name}: {eval_data['score']:.4f}")
            elif isinstance(eval_data, dict):
                # test_region の場合
                logger.info(f"  {eval_name}:")
                for sub_name, sub_data in eval_data.items():
                    logger.info(f"    {sub_name}: {sub_data['score']:.4f}")
        
        # パフォーマンス比較
        if 'test_time' in evaluation_results and 'validation' in evaluation_results:
            val_score = evaluation_results['validation']['score']
            test_score = evaluation_results['test_time']['score']
            performance_diff = test_score - val_score
            
            logger.info(f"📊 パフォーマンス変化: {performance_diff:+.4f} (Val→Test-time)")
            if abs(performance_diff) > 0.1:
                logger.warning("⚠️  大きなパフォーマンス差が検出されました（過学習の可能性）")
        
        # --run-analysis フラグが設定されている場合、inspector と researcher を自動実行
        if args.run_analysis:
            logger.info("=== 自動分析開始 ===" )
            
            # 最新の候補ファイルを探す
            candidates_file = _find_latest_candidates_file(experiment_id)
            if candidates_file and candidates_file.exists():
                logger.info(f"候補ファイル: {candidates_file}")
                
                # Inspector実行
                inspector_success = _run_inspector_analysis(candidates_file, args.sites, args.verbose)
                
                if inspector_success:
                    # Researcher実行
                    _run_researcher_analysis(args.verbose)
                    logger.info("✅ 自動分析が完了しました")
                else:
                    logger.warning("⚠️ Inspector実行に失敗しました。Researcher実行をスキップします。")
            else:
                logger.warning("⚠️ 候補ファイルが見つかりません。自動分析をスキップします。")
        
        # 既存候補ファイルベースの可視化実行
        if args.visualize and viz_output_dir:
            logger.info("=== 候補ファイルベース可視化開始 ===")
            
            # 候補ファイルを収集
            candidates_files = {}
            optuna_dir = Path("data/output/optuna")
            
            if optuna_dir.exists():
                for timestamp_dir in optuna_dir.iterdir():
                    if timestamp_dir.is_dir():
                        # 各データセットの候補ファイルを探す
                        val_file = timestamp_dir / f"{experiment_id}_val_candidates.csv"
                        if val_file.exists():
                            candidates_files['validation'] = val_file
                            
                        test_time_file = timestamp_dir / f"{experiment_id}_test_time_candidates.csv"
                        if test_time_file.exists():
                            candidates_files['test_time'] = test_time_file
                            
                        # 地域別ファイルも探す
                        for region_file in timestamp_dir.glob(f"{experiment_id}_region_*_candidates.csv"):
                            region_name = region_file.stem.replace(f"{experiment_id}_region_", "").replace("_candidates", "")
                            candidates_files[f'region_{region_name}'] = region_file
            
            if candidates_files:
                logger.info(f"可視化対象ファイル数: {len(candidates_files)}")
                
                # メインの候補ファイルを選択（validationを優先）
                main_candidates_file = None
                if 'validation' in candidates_files:
                    main_candidates_file = candidates_files['validation']
                elif candidates_files:
                    main_candidates_file = list(candidates_files.values())[0]
                
                if main_candidates_file:
                    logger.info(f"メイン可視化ファイル: {main_candidates_file}")
                    
                    # 統合可視化を実行
                    visualize_candidates(main_candidates_file, experiment_id, viz_output_dir)
                
                # 複数データセットがある場合は包括的ダッシュボードも作成
                if len(candidates_files) > 1:
                    try:
                        create_comprehensive_dashboard(candidates_files, viz_output_dir, experiment_id)
                        logger.info("✅ 包括的ダッシュボード作成完了")
                    except Exception as e:
                        logger.error(f"包括的ダッシュボード作成エラー: {e}")
                
                logger.info(f"📊 可視化結果保存先: {viz_output_dir}")
                logger.info("=== 可視化完了 ===")
            else:
                logger.warning("⚠️ 可視化対象の候補ファイルが見つかりません")
        
        logger.info("=== 再実行完了 ===")
        
    except Exception as e:
        logger.error(f"再実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _find_latest_candidates_file(experiment_id: str) -> Optional[Path]:
    """最新の候補ファイルを探す."""
    # タイムスタンプディレクトリ内を検索
    possible_paths = []
    
    optuna_dir = Path("data/output/optuna")
    if optuna_dir.exists():
        for timestamp_dir in optuna_dir.iterdir():
            if timestamp_dir.is_dir():
                possible_paths.extend([
                    timestamp_dir / f"{experiment_id}_val_candidates.csv",
                    timestamp_dir / f"{experiment_id}_test_time_candidates.csv",
                ])
    
    # 存在するファイルの中から最新のものを返す
    existing_files = [p for p in possible_paths if p.exists()]
    if existing_files:
        return max(existing_files, key=lambda p: p.stat().st_mtime)
    
    return None


def _run_inspector_analysis(candidates_file: Path, known_sites: Path, verbose: bool) -> bool:
    """Inspector Agentを実行する."""
    try:
        import subprocess
        
        cmd = [
            sys.executable, 
            "scripts/run_inspector.py",
            "--candidates", str(candidates_file),
            "--known", str(known_sites)
        ]
        
        if verbose:
            cmd.append("--verbose")
        
        logger.info(f"Inspector実行: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Inspector実行成功")
            return True
        else:
            logger.error(f"❌ Inspector実行失敗 (return code: {result.returncode})")
            return False
            
    except Exception as e:
        logger.error(f"Inspector実行中にエラー: {e}")
        return False


def _run_researcher_analysis(verbose: bool) -> bool:
    """Researcher Agentを実行する."""
    try:
        import subprocess
        
        cmd = [sys.executable, "scripts/run_researcher.py"]
        
        if verbose:
            cmd.append("--verbose")
        
        logger.info(f"Researcher実行: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Researcher実行成功")
            return True
        else:
            logger.error(f"❌ Researcher実行失敗 (return code: {result.returncode})")
            return False
            
    except Exception as e:
        logger.error(f"Researcher実行中にエラー: {e}")
        return False


def visualize_candidate_distribution(candidates_file: Path, output_dir: Path, experiment_id: str):
    """候補地点の地理的分布を可視化"""
    try:
        import seaborn as sns
        from shapely import wkt
        
        # 候補データを読み込み
        candidates = pd.read_csv(candidates_file)
        if len(candidates) == 0:
            logger.warning("候補データが空です")
            return
        
        # GeometryをWKTから復元
        if 'geometry' in candidates.columns:
            candidates['geometry'] = candidates['geometry'].apply(wkt.loads)
            candidates_gdf = gpd.GeoDataFrame(candidates, crs="EPSG:4326")
        else:
            # lat/lonから geometry を作成
            from shapely.geometry import Point
            candidates_gdf = gpd.GeoDataFrame(
                candidates,
                geometry=[Point(lon, lat) for lon, lat in zip(candidates['lon'], candidates['lat'])],
                crs="EPSG:4326"
            )
        
        logger.info(f"可視化対象候補数: {len(candidates_gdf)}")
        
        # Web Mercatorに変換
        candidates_web = candidates_gdf.to_crs(epsg=3857)
        
        # 図の作成
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Candidate Sites Analysis - {experiment_id}', fontsize=16, fontweight='bold')
        
        # 1. スコア別分布地図
        ax1 = axes[0, 0]
        candidates_web.plot(ax=ax1, column='total_score', cmap='plasma', 
                           markersize=60, alpha=0.8, legend=True,
                           edgecolors='white', linewidth=0.5)
        
        # 地図背景追加
        if HAS_CONTEXTILY:
            try:
                ctx.add_basemap(ax1, crs=candidates_web.crs.to_string(), 
                               source=ctx.providers.OpenStreetMap.Mapnik)
            except Exception as e:
                logger.warning(f"地図背景追加失敗: {e}")
        
        ax1.set_title('Geographic Distribution by Score')
        ax1.set_xlabel('X Coordinate (Web Mercator)')
        ax1.set_ylabel('Y Coordinate (Web Mercator)')
        
        # 2. 語根タイプ別分布
        ax2 = axes[0, 1]
        if 'root' in candidates.columns:
            candidates_web.plot(ax=ax2, column='root', cmap='Set3', 
                               markersize=60, alpha=0.8, legend=True,
                               edgecolors='white', linewidth=0.5)
            ax2.set_title('Distribution by Root Type')
        else:
            ax2.text(0.5, 0.5, 'Root type data not available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Root Type Distribution (N/A)')
        
        if HAS_CONTEXTILY:
            try:
                ctx.add_basemap(ax2, crs=candidates_web.crs.to_string(),
                               source=ctx.providers.OpenStreetMap.Mapnik)
            except Exception as e:
                pass
        
        ax2.set_xlabel('X Coordinate (Web Mercator)')
        ax2.set_ylabel('Y Coordinate (Web Mercator)')
        
        # 3. スコア分布ヒストグラム
        ax3 = axes[1, 0]
        ax3.hist(candidates['total_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(candidates['total_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {candidates["total_score"].mean():.3f}')
        ax3.axvline(candidates['total_score'].median(), color='orange', linestyle='--', 
                   label=f'Median: {candidates["total_score"].median():.3f}')
        ax3.set_title('Score Distribution')
        ax3.set_xlabel('Total Score')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 語根タイプ別統計
        ax4 = axes[1, 1]
        if 'root' in candidates.columns:
            root_stats = candidates.groupby('root')['total_score'].agg(['count', 'mean', 'std']).fillna(0)
            root_stats['count'].plot(kind='bar', ax=ax4, color='lightcoral')
            ax4.set_title('Candidate Count by Root Type')
            ax4.set_xlabel('Root Type')
            ax4.set_ylabel('Count')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'Root type data not available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Root Type Statistics (N/A)')
        
        plt.tight_layout()
        
        # 保存
        output_path = output_dir / f'{experiment_id}_candidate_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"候補分布可視化を保存: {output_path}")
        
    except Exception as e:
        logger.error(f"候補分布可視化エラー: {e}")


def visualize_distance_analysis(candidates_file: Path, output_dir: Path, experiment_id: str):
    """距離分析の可視化"""
    try:
        candidates = pd.read_csv(candidates_file)
        if len(candidates) == 0 or 'dist_km' not in candidates.columns:
            logger.warning("距離データが利用できません")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Distance Analysis - {experiment_id}', fontsize=16, fontweight='bold')
        
        # 1. 距離分布ヒストグラム
        ax1 = axes[0, 0]
        ax1.hist(candidates['dist_km'], bins=25, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.axvline(candidates['dist_km'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {candidates["dist_km"].mean():.2f} km')
        ax1.axvline(candidates['dist_km'].median(), color='orange', linestyle='--', 
                   label=f'Median: {candidates["dist_km"].median():.2f} km')
        ax1.set_title('Distance from Current Rivers Distribution')
        ax1.set_xlabel('Distance (km)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 距離 vs スコアの散布図
        ax2 = axes[0, 1]
        scatter = ax2.scatter(candidates['dist_km'], candidates['total_score'], 
                             alpha=0.6, c=candidates['total_score'], cmap='viridis')
        ax2.set_title('Distance vs Score Relationship')
        ax2.set_xlabel('Distance from Rivers (km)')
        ax2.set_ylabel('Total Score')
        plt.colorbar(scatter, ax=ax2, label='Total Score')
        ax2.grid(True, alpha=0.3)
        
        # 相関係数を計算
        correlation = candidates['dist_km'].corr(candidates['total_score'])
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. 距離範囲別の候補数
        ax3 = axes[1, 0]
        distance_bins = [0, 1, 2, 3, 5, 10, float('inf')]
        distance_labels = ['0-1km', '1-2km', '2-3km', '3-5km', '5-10km', '>10km']
        candidates['distance_range'] = pd.cut(candidates['dist_km'], bins=distance_bins, labels=distance_labels)
        distance_counts = candidates['distance_range'].value_counts().sort_index()
        distance_counts.plot(kind='bar', ax=ax3, color='lightgreen')
        ax3.set_title('Candidate Count by Distance Range')
        ax3.set_xlabel('Distance Range')
        ax3.set_ylabel('Count')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 語根タイプ別距離統計
        ax4 = axes[1, 1]
        if 'root' in candidates.columns:
            import seaborn as sns
            sns.boxplot(data=candidates, x='root', y='dist_km', ax=ax4)
            ax4.set_title('Distance Distribution by Root Type')
            ax4.set_xlabel('Root Type')
            ax4.set_ylabel('Distance (km)')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'Root type data not available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Distance by Root Type (N/A)')
        
        plt.tight_layout()
        
        # 保存
        output_path = output_dir / f'{experiment_id}_distance_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"距離分析可視化を保存: {output_path}")
        
    except Exception as e:
        logger.error(f"距離分析可視化エラー: {e}")


def visualize_water_frequency_analysis(candidates_file: Path, output_dir: Path, experiment_id: str):
    """水域頻度分析の可視化"""
    try:
        candidates = pd.read_csv(candidates_file)
        if len(candidates) == 0 or 'occ_pct' not in candidates.columns:
            logger.warning("水域頻度データが利用できません")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Water Frequency Analysis - {experiment_id}', fontsize=16, fontweight='bold')
        
        # 1. 水域頻度分布ヒストグラム
        ax1 = axes[0, 0]
        ax1.hist(candidates['occ_pct'], bins=25, alpha=0.7, color='lightcoral', edgecolor='black')
        ax1.axvline(candidates['occ_pct'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {candidates["occ_pct"].mean():.2f}%')
        ax1.axvline(candidates['occ_pct'].median(), color='orange', linestyle='--', 
                   label=f'Median: {candidates["occ_pct"].median():.2f}%')
        ax1.set_title('Water Occurrence Percentage Distribution')
        ax1.set_xlabel('Water Occurrence (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 水域頻度 vs スコアの散布図
        ax2 = axes[0, 1]
        scatter = ax2.scatter(candidates['occ_pct'], candidates['total_score'], 
                             alpha=0.6, c=candidates['total_score'], cmap='plasma')
        ax2.set_title('Water Frequency vs Score Relationship')
        ax2.set_xlabel('Water Occurrence (%)')
        ax2.set_ylabel('Total Score')
        plt.colorbar(scatter, ax=ax2, label='Total Score')
        ax2.grid(True, alpha=0.3)
        
        # 相関係数を計算
        correlation = candidates['occ_pct'].corr(candidates['total_score'])
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. 距離 vs 水域頻度の関係
        ax3 = axes[1, 0]
        if 'dist_km' in candidates.columns:
            scatter = ax3.scatter(candidates['dist_km'], candidates['occ_pct'], 
                                 alpha=0.6, c=candidates['total_score'], cmap='viridis')
            ax3.set_title('Distance vs Water Frequency')
            ax3.set_xlabel('Distance from Rivers (km)')
            ax3.set_ylabel('Water Occurrence (%)')
            plt.colorbar(scatter, ax=ax3, label='Total Score')
            ax3.grid(True, alpha=0.3)
            
            # 相関係数
            correlation = candidates['dist_km'].corr(candidates['occ_pct'])
            ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax3.text(0.5, 0.5, 'Distance data not available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Distance vs Water Frequency (N/A)')
        
        # 4. 水域頻度範囲別の候補数
        ax4 = axes[1, 1]
        frequency_bins = [0, 1, 5, 10, 20, 50, float('inf')]
        frequency_labels = ['0-1%', '1-5%', '5-10%', '10-20%', '20-50%', '>50%']
        candidates['frequency_range'] = pd.cut(candidates['occ_pct'], bins=frequency_bins, labels=frequency_labels)
        frequency_counts = candidates['frequency_range'].value_counts().sort_index()
        frequency_counts.plot(kind='bar', ax=ax4, color='lightseagreen')
        ax4.set_title('Candidate Count by Water Frequency Range')
        ax4.set_xlabel('Water Frequency Range')
        ax4.set_ylabel('Count')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存
        output_path = output_dir / f'{experiment_id}_water_frequency_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"水域頻度分析可視化を保存: {output_path}")
        
    except Exception as e:
        logger.error(f"水域頻度分析可視化エラー: {e}")


def visualize_final_results(candidates_file: Path, output_dir: Path, experiment_id: str, top_k: int = 20):
    """最終結果の包括的可視化"""
    try:
        from shapely import wkt
        
        candidates = pd.read_csv(candidates_file)
        if len(candidates) == 0:
            logger.warning("候補データが空です")
            return
        
        # 上位候補を抽出
        top_candidates = candidates.nlargest(top_k, 'total_score')
        
        # GeometryをWKTから復元
        if 'geometry' in candidates.columns:
            candidates['geometry'] = candidates['geometry'].apply(wkt.loads)
            candidates_gdf = gpd.GeoDataFrame(candidates, crs="EPSG:4326")
            top_candidates_gdf = candidates_gdf.nlargest(top_k, 'total_score')
        else:
            from shapely.geometry import Point
            candidates_gdf = gpd.GeoDataFrame(
                candidates,
                geometry=[Point(lon, lat) for lon, lat in zip(candidates['lon'], candidates['lat'])],
                crs="EPSG:4326"
            )
            top_candidates_gdf = candidates_gdf.nlargest(top_k, 'total_score')
        
        # Web Mercatorに変換
        candidates_web = candidates_gdf.to_crs(epsg=3857)
        top_candidates_web = top_candidates_gdf.to_crs(epsg=3857)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Final Results Summary - {experiment_id}', fontsize=18, fontweight='bold')
        
        # 1. トップ候補地点の地図
        ax1 = axes[0, 0]
        
        # 全候補を薄く表示
        candidates_web.plot(ax=ax1, color='lightgray', alpha=0.3, markersize=20)
        
        # トップ候補をスコア別カラーで強調表示
        scatter = top_candidates_web.plot(ax=ax1, column='total_score', cmap='plasma',
                                         markersize=100, alpha=0.9, legend=True,
                                         edgecolors='white', linewidth=1)
        
        # 地図背景追加
        if HAS_CONTEXTILY:
            try:
                ctx.add_basemap(ax1, crs=candidates_web.crs.to_string(),
                               source=ctx.providers.OpenStreetMap.Mapnik)
            except Exception as e:
                logger.warning(f"地図背景追加失敗: {e}")
        
        ax1.set_title(f'Top {top_k} Candidate Sites')
        ax1.set_xlabel('X Coordinate (Web Mercator)')
        ax1.set_ylabel('Y Coordinate (Web Mercator)')
        
        # 統計情報をテキストで追加
        stats_text = f"Total Candidates: {len(candidates)}\n"
        stats_text += f"Top {top_k} shown\n"
        stats_text += f"Max Score: {candidates['total_score'].max():.3f}\n"
        stats_text += f"Mean Score: {candidates['total_score'].mean():.3f}"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='white', alpha=0.8), fontsize=10)
        
        # 2. スコア構成要素の分析
        ax2 = axes[0, 1]
        if all(col in candidates.columns for col in ['dist_km', 'occ_pct']):
            # 距離とスコアの関係を色分けで表示
            scatter = ax2.scatter(top_candidates['dist_km'], top_candidates['occ_pct'],
                                 s=top_candidates['total_score'] * 200,
                                 c=top_candidates['total_score'], cmap='plasma',
                                 alpha=0.7, edgecolors='black', linewidth=0.5)
            ax2.set_title(f'Top {top_k}: Distance vs Water Frequency\n(Size = Score)')
            ax2.set_xlabel('Distance from Rivers (km)')
            ax2.set_ylabel('Water Occurrence (%)')
            plt.colorbar(scatter, ax=ax2, label='Total Score')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Distance/frequency data not available',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Score Components Analysis (N/A)')
        
        # 3. トップ候補一覧表
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        # トップ10の表を作成
        display_cols = ['name', 'total_score']
        if 'dist_km' in top_candidates.columns:
            display_cols.append('dist_km')
        if 'occ_pct' in top_candidates.columns:
            display_cols.append('occ_pct')
        if 'root' in top_candidates.columns:
            display_cols.append('root')
        
        table_data = top_candidates.head(10)[display_cols].round(3)
        
        # 列名を整理
        column_labels = []
        for col in display_cols:
            if col == 'name':
                column_labels.append('Name')
            elif col == 'total_score':
                column_labels.append('Score')
            elif col == 'dist_km':
                column_labels.append('Dist(km)')
            elif col == 'occ_pct':
                column_labels.append('Water%')
            elif col == 'root':
                column_labels.append('Root')
            else:
                column_labels.append(col)
        
        table = ax3.table(cellText=table_data.values,
                         colLabels=column_labels,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # ヘッダーのスタイル設定
        for i in range(len(column_labels)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax3.set_title(f'Top 10 Candidates', fontsize=14, fontweight='bold', pad=20)
        
        # 4. 全体統計サマリー
        ax4 = axes[1, 1]
        
        # 統計情報を計算
        stats_info = {
            'Total Candidates': len(candidates),
            'Mean Score': f"{candidates['total_score'].mean():.3f}",
            'Std Score': f"{candidates['total_score'].std():.3f}",
            'Max Score': f"{candidates['total_score'].max():.3f}",
            'Min Score': f"{candidates['total_score'].min():.3f}"
        }
        
        if 'dist_km' in candidates.columns:
            stats_info.update({
                'Mean Distance': f"{candidates['dist_km'].mean():.2f} km",
                'Max Distance': f"{candidates['dist_km'].max():.2f} km"
            })
        
        if 'occ_pct' in candidates.columns:
            stats_info.update({
                'Mean Water%': f"{candidates['occ_pct'].mean():.2f}%",
                'Max Water%': f"{candidates['occ_pct'].max():.2f}%"
            })
        
        if 'root' in candidates.columns:
            root_counts = candidates['root'].value_counts()
            stats_info.update({
                'Most Common Root': f"{root_counts.index[0]} ({root_counts.iloc[0]})",
                'Root Types': len(root_counts)
            })
        
        # 統計表として表示
        stats_table = [[k, v] for k, v in stats_info.items()]
        
        table = ax4.table(cellText=stats_table,
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # ヘッダーのスタイル設定
        table[(0, 0)].set_facecolor('#2196F3')
        table[(0, 0)].set_text_props(weight='bold', color='white')
        table[(0, 1)].set_facecolor('#2196F3')
        table[(0, 1)].set_text_props(weight='bold', color='white')
        
        ax4.axis('off')
        ax4.set_title('Overall Statistics', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # 保存
        output_path = output_dir / f'{experiment_id}_final_results.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"最終結果可視化を保存: {output_path}")
        
    except Exception as e:
        logger.error(f"最終結果可視化エラー: {e}")


def create_comprehensive_dashboard(candidates_files: Dict[str, Path], output_dir: Path, experiment_id: str):
    """複数データセットの包括的ダッシュボード作成"""
    try:
        fig, axes = plt.subplots(3, 2, figsize=(24, 18))
        fig.suptitle(f'Comprehensive Analysis Dashboard - {experiment_id}', fontsize=20, fontweight='bold')
        
        all_data = {}
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # 各データセットを読み込み
        for i, (dataset_name, file_path) in enumerate(candidates_files.items()):
            if file_path.exists():
                data = pd.read_csv(file_path)
                if len(data) > 0:
                    all_data[dataset_name] = data
                    logger.info(f"{dataset_name}: {len(data)} candidates loaded")
        
        if not all_data:
            logger.warning("読み込み可能なデータがありません")
            return
        
        # 1. データセット別候補数比較
        ax1 = axes[0, 0]
        dataset_counts = {name: len(data) for name, data in all_data.items()}
        bars = ax1.bar(dataset_counts.keys(), dataset_counts.values(), 
                      color=colors[:len(dataset_counts)])
        ax1.set_title('Candidate Count by Dataset')
        ax1.set_ylabel('Number of Candidates')
        ax1.tick_params(axis='x', rotation=45)
        
        # 値をバーの上に表示
        for bar, count in zip(bars, dataset_counts.values()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(dataset_counts.values())*0.01,
                    str(count), ha='center', va='bottom')
        
        # 2. データセット別スコア分布比較
        ax2 = axes[0, 1]
        for i, (dataset_name, data) in enumerate(all_data.items()):
            if 'total_score' in data.columns:
                ax2.hist(data['total_score'], bins=20, alpha=0.6, 
                        label=dataset_name, color=colors[i % len(colors)])
        ax2.set_title('Score Distribution Comparison')
        ax2.set_xlabel('Total Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. データセット別平均スコア比較
        ax3 = axes[1, 0]
        mean_scores = {}
        std_scores = {}
        for dataset_name, data in all_data.items():
            if 'total_score' in data.columns and len(data) > 0:
                mean_scores[dataset_name] = data['total_score'].mean()
                std_scores[dataset_name] = data['total_score'].std()
        
        if mean_scores:
            bars = ax3.bar(mean_scores.keys(), mean_scores.values(),
                          yerr=[std_scores.get(k, 0) for k in mean_scores.keys()],
                          color=colors[:len(mean_scores)], capsize=5)
            ax3.set_title('Mean Score by Dataset (±1 std)')
            ax3.set_ylabel('Mean Total Score')
            ax3.tick_params(axis='x', rotation=45)
            
            # 値をバーの上に表示
            for bar, (name, score) in zip(bars, mean_scores.items()):
                ax3.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + std_scores.get(name, 0) + max(mean_scores.values())*0.02,
                        f'{score:.3f}', ha='center', va='bottom')
        
        # 4. 距離分析比較
        ax4 = axes[1, 1]
        for i, (dataset_name, data) in enumerate(all_data.items()):
            if 'dist_km' in data.columns and len(data) > 0:
                ax4.hist(data['dist_km'], bins=20, alpha=0.6,
                        label=dataset_name, color=colors[i % len(colors)])
        ax4.set_title('Distance Distribution Comparison')
        ax4.set_xlabel('Distance from Rivers (km)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 水域頻度分析比較
        ax5 = axes[2, 0]
        for i, (dataset_name, data) in enumerate(all_data.items()):
            if 'occ_pct' in data.columns and len(data) > 0:
                ax5.hist(data['occ_pct'], bins=20, alpha=0.6,
                        label=dataset_name, color=colors[i % len(colors)])
        ax5.set_title('Water Frequency Distribution Comparison')
        ax5.set_xlabel('Water Occurrence (%)')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 統計サマリー表
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        # 統計表のデータを準備
        summary_data = []
        for dataset_name, data in all_data.items():
            row = [dataset_name, len(data)]
            
            if 'total_score' in data.columns and len(data) > 0:
                row.extend([f"{data['total_score'].mean():.3f}",
                           f"{data['total_score'].max():.3f}"])
            else:
                row.extend(['N/A', 'N/A'])
            
            if 'dist_km' in data.columns and len(data) > 0:
                row.append(f"{data['dist_km'].mean():.2f}")
            else:
                row.append('N/A')
            
            if 'occ_pct' in data.columns and len(data) > 0:
                row.append(f"{data['occ_pct'].mean():.2f}")
            else:
                row.append('N/A')
            
            summary_data.append(row)
        
        # 表の作成
        if summary_data:
            headers = ['Dataset', 'Count', 'Mean Score', 'Max Score', 'Mean Dist(km)', 'Mean Water%']
            table = ax6.table(cellText=summary_data,
                             colLabels=headers,
                             cellLoc='center',
                             loc='center',
                             bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # ヘッダーのスタイル設定
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#FF9800')
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax6.set_title('Dataset Comparison Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # 保存
        output_path = output_dir / f'{experiment_id}_comprehensive_dashboard.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"包括的ダッシュボードを保存: {output_path}")
        
    except Exception as e:
        logger.error(f"包括的ダッシュボード作成エラー: {e}")


if __name__ == "__main__":
    main()