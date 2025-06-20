#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_site_identification.py: 遺跡候補地特定専用スクリプト

このスクリプトは、run_harmonizer.pyのTask ii)を独立実行するためのものです。
- 地理空間分析パイプライン（S-1～S-5）
- 現河道との距離計算
- 水域頻度計算
- 古河道候補地の抽出とスコアリング

使用方法:
    python run_site_identification.py [--rivers-path PATH] [--gsw-path PATH] [--output-path PATH]

オプション:
    --bbox COORDS         対象領域のBBOX (lon_min lat_min lon_max lat_max)
    --rivers-path PATH    HydroRIVERSのシェープファイルパス
    --gsw-path PATH       GSW occurrenceのTIFFファイルパス
    --pbf-path PATH       PyrosmのPBFファイルパス
    --output-path PATH    出力ファイルパス
    --skip-water-freq     水域頻度計算をスキップする（距離のみで候補抽出）
    --visualize           処理結果を可視化する
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
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
    logger.warning("contextily がインストールされていません。地図背景は表示されません。")

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# プロジェクトのルートディレクトリをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# 自作パッケージのインポート
from tamagawa_to_z.harmonizer import (
    make_bbox_gdf, process_toponyms,
    attach_distance, water_occurrence, filter_candidates, score_candidates
)
from tamagawa_to_z.harmonizer.preprocess import DEFAULT_BBOX, extract_toponyms_pyrosm
from tamagawa_to_z.harmonizer.llm_layer.root_io import build_water_regex


def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(
        description='古河道候補地特定専用スクリプト（地理空間分析パイプライン）'
    )
    
    # BBOX オプション
    parser.add_argument(
        '--bbox',
        type=float,
        nargs=4,
        metavar=('LON_MIN', 'LAT_MIN', 'LON_MAX', 'LAT_MAX'),
        default=list(DEFAULT_BBOX.bounds),
        help='対象領域のBBOX (lon_min lat_min lon_max lat_max)'
    )
    
    # データパスの設定
    parser.add_argument(
        '--rivers-path', 
        type=str, 
        default=str(PROJECT_ROOT / 'data/raw/hydrorivers_sahydrorivers_sa/HydroRIVERS_v10_sa.shp'),
        help='HydroRIVERSのシェープファイルパス'
    )
    parser.add_argument(
        '--gsw-path', 
        type=str, 
        default=str(PROJECT_ROOT / 'data/raw/GSW_occurrence/occurrence_70W_10Sv1_4_2021.tif'),
        help='GSW occurrenceのTIFFファイルパス'
    )
    parser.add_argument(
        '--pbf-path',
        type=str,
        default=str(PROJECT_ROOT / 'data/raw/osm/norte-latest.osm.pbf'),
        help='PBFファイルのパス'
    )
    parser.add_argument(
        '--output-path', 
        type=str, 
        default=str(PROJECT_ROOT / 'data/output/candidates/paleochannel_candidates.csv'),
        help='出力ファイルパス'
    )
    
    # 処理オプション
    parser.add_argument(
        '--skip-water-freq',
        action='store_true',
        help='水域頻度計算をスキップする（距離のみで候補抽出）'
    )
    
    # 可視化オプション
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='処理結果を可視化する'
    )
    parser.add_argument(
        '--viz-output-dir',
        type=str,
        default=str(PROJECT_ROOT / 'data/plots'),
        help='可視化画像の出力ディレクトリ'
    )
    
    return parser.parse_args()


def check_data_files(rivers_path, gsw_path, skip_water_freq=False):
    """入力データファイルの存在を確認する"""
    missing_files = []
    
    if not os.path.exists(rivers_path):
        missing_files.append(rivers_path)
        logger.warning(f"警告: {rivers_path} が見つかりません。")
        logger.warning("HydroRIVERS_SA.shp を data/raw/ ディレクトリに配置してください。")
        logger.warning("ダウンロード先: https://www.hydrosheds.org/products/hydrorivers")
    
    if not skip_water_freq and not os.path.exists(gsw_path):
        missing_files.append(gsw_path)
        logger.warning(f"警告: {gsw_path} が見つかりません。")
        logger.warning("GSW_occurrence.tif を data/raw/ ディレクトリに配置してください。")
        logger.warning("ダウンロード先: https://global-surface-water.appspot.com/download")
    
    return missing_files


def step1_define_bbox(bbox_coords, visualize=False, output_dir=None):
    """S-1: 対象地域のBBox定義"""
    logger.info("=== 🌍 S-1: 対象地域のBBox定義 ===")

    # BBoxの作成
    bbox_gdf = make_bbox_gdf(*bbox_coords)
    bbox = bbox_gdf.geometry.iloc[0]
    logger.info(f"対象領域の境界: {bbox.bounds}")
    
    # 可視化
    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)
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
        
        ax.set_title('Target Region', fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.tight_layout()
        
        output_path = Path(output_dir) / 'step1_bbox.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 対象領域の可視化を保存: {output_path}")
    
    return bbox_gdf


def step2_extract_toponyms(bbox_gdf, pbf_path, visualize=False, output_dir=None):
    """S-2: 水場系トポニムの抽出（LLMなし）"""
    logger.info("=== 🏷️  S-2: 水場系トポニムの抽出 ===")
    bbox = bbox_gdf.geometry.iloc[0]
    
    # 最新のRegexパターンを構築
    try:
        logger.info("=== 🔧 Water Vocabulary Regex Construction ===")
        water_regex = build_water_regex()
        logger.info("=== ✅ Regex Construction Completed ===")
    except Exception as e:
        logger.error(f"❌ roots.csvからのRegex構築に失敗: {e}")
        logger.error("❌ 水語彙フィルタリングを実行できません。処理を中止します。")
        raise RuntimeError(f"water_roots.csvが読み込めません: {e}")
    
    # Pyrosmを使用してローカルPBFファイルから水語彙地名を抽出
    logger.info("PyrosmでローカルPBFから水語彙地名を抽出しています...")
    try:
        names = extract_toponyms_pyrosm(bbox, pbf_path, regex=water_regex)
        if names.empty:
            logger.warning("ローカルPBFからのデータ取得に失敗しました。")
        else:
            logger.info(f"ローカルPBFから{len(names)}件のトポニムを収集しました")
    except Exception as e:
        logger.error(f"ローカルPBFデータ収集中にエラーが発生しました: {e}")
        logger.warning("空のデータセットで処理を続行します")
        names = gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")
    
    if names.empty:
        logger.warning("どのソースからもトポニムを収集できませんでした。処理を続行できない可能性があります。")
        return names
    
    logger.info(f"合計{len(names)}件のトポニムを収集しました")
    
    # 可視化
    if visualize and output_dir and not names.empty:
        os.makedirs(output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # 座標系をWeb Mercatorに変換（地図背景用）
        bbox_gdf_web = bbox_gdf.to_crs(epsg=3857)
        names_web = names.to_crs(epsg=3857)
        
        # まず対象領域の境界を描画（座標軸の範囲を設定）
        bbox_gdf_web.plot(ax=ax, color='none', edgecolor='red', linewidth=3, alpha=0.8)
        
        # トポニムを描画
        if 'source' in names_web.columns:
            names_web.plot(ax=ax, column='source', cmap='Set1', markersize=50, 
                          legend=True, alpha=0.8, edgecolors='white', linewidth=0.5)
        else:
            names_web.plot(ax=ax, color='blue', markersize=50, alpha=0.8, 
                          edgecolors='white', linewidth=0.5)
        
        # 地図の範囲を設定（bbox_gdf_webの範囲にマージンを追加）
        bounds = bbox_gdf_web.total_bounds
        margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
        ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
        ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
        
        # 地図背景を追加（座標軸の範囲設定後）
        if HAS_CONTEXTILY:
            try:
                ctx.add_basemap(ax, crs=bbox_gdf_web.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
                logger.info("🗺️ OpenStreetMap背景を追加しました")
            except Exception as e:
                logger.warning(f"地図背景の追加に失敗: {e}")
                logger.debug(f"エラー詳細: {str(e)}")
        
        ax.set_title('Distribution of Collected Toponyms', fontsize=16, fontweight='bold')
        ax.set_xlabel('X Coordinate (Web Mercator)')
        ax.set_ylabel('Y Coordinate (Web Mercator)')
        
        # 凡例の調整
        if 'source' in names_web.columns:
            legend = ax.get_legend()
            if legend:
                legend.set_bbox_to_anchor((1.05, 1))
                legend.set_loc('upper left')
        
        plt.tight_layout()
        
        output_path = Path(output_dir) / 'step2_toponyms_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 トポニム分布の可視化を保存: {output_path}")
    
    return names


def step3_process_toponyms(names, visualize=False, output_dir=None):
    """S-3: クレンジング & タイプ付け（LLMハーモナイゼーションなし）"""
    logger.info("=== 🧹 S-3: クレンジング & タイプ付け ===")
    
    # トポニムの処理
    names = process_toponyms(names)
    logger.info(f"{len(names)}件のトポニムを処理しました")
    
    # 結果の確認
    if logger.level <= logging.INFO:
        sample_data = pd.DataFrame({
            'name': names['name'],
            'normalized_name': names['normalized_name'],
            'type': names['type']
        }).head(5)
        logger.info(f"処理結果サンプル:\n{sample_data}")
    
    # タイプ別の集計
    type_counts = names['type'].value_counts()
    logger.info(f"水系タイプ別の件数:\n{type_counts}")
    
    # 可視化
    if visualize and output_dir and len(type_counts) > 0:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        type_counts.plot(kind='bar')
        plt.title('Count by Water System Type')
        plt.xlabel('Water System Type')
        plt.ylabel('Count')
        plt.tight_layout()
        
        output_path = Path(output_dir) / 'step3_type_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 水系タイプ分布の可視化を保存: {output_path}")
    
    return names


def step4_calculate_distance(names, rivers_path, visualize=False, output_dir=None):
    """S-4: 現河道との距離計算"""
    logger.info("=== 📏 S-4: 現河道との距離計算 ===")
    
    # HydroRIVERSファイルの存在確認
    if not os.path.exists(rivers_path):
        logger.error(f"エラー: {rivers_path} が見つかりません。")
        logger.error("このステップはスキップします。")
        return names
    
    # 距離計算
    logger.info("現河道との距離を計算しています...")
    names = attach_distance(names, rivers_path)
    logger.info(f"{len(names)}件のトポニムに距離情報を追加しました")
    
    # 結果の確認
    if logger.level <= logging.INFO:
        sample_data = names.sort_values('dist_km', ascending=False).head(5)[['name', 'type', 'dist_km']]
        logger.info(f"距離計算結果サンプル:\n{sample_data}")
    
    # 可視化
    if visualize and output_dir and 'dist_km' in names.columns:
        os.makedirs(output_dir, exist_ok=True)
        # 距離の分布を可視化
        plt.figure(figsize=(10, 6))
        plt.hist(names['dist_km'], bins=20)
        plt.title('Distribution of Distance from Current Rivers')
        plt.xlabel('Distance (km)')
        plt.ylabel('Frequency')
        plt.axvline(x=3, color='red', linestyle='--', label='Threshold (3km)')
        plt.legend()
        plt.tight_layout()
        
        output_path = Path(output_dir) / 'step4_distance_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 距離分布の可視化を保存: {output_path}")
    
    return names


def step5_extract_candidates(names, gsw_path, visualize=False, skip_water_freq=False, output_dir=None):
    """S-5: "川が無いのに川名が残る"ポイント抽出"""
    logger.info("=== 🎯 S-5: 候補地点抽出 ===")
    
    if skip_water_freq:
        logger.info("水域頻度計算をスキップします（距離のみで候補抽出）")
        # 水域頻度を0に設定
        names = names.copy()
        names["occ_pct"] = 0
        logger.info(f"{len(names)}件のトポニムに仮の水域頻度情報（0%）を設定しました")
    else:
        # GSWファイルの存在確認
        if not os.path.exists(gsw_path):
            logger.error(f"エラー: {gsw_path} が見つかりません。")
            logger.error("このステップはスキップします。")
            return None
        
        # 水域頻度の計算
        logger.info("水域頻度を計算しています...")
        names = water_occurrence(names, gsw_path)
        logger.info(f"{len(names)}件のトポニムに水域頻度情報を追加しました")
    
    # 結果の確認
    if logger.level <= logging.INFO and 'occ_pct' in names.columns:
        sample_data = names.sort_values('occ_pct').head(5)[['name', 'type', 'dist_km', 'occ_pct']]
        logger.info(f"水域頻度計算結果サンプル:\n{sample_data}")
    
    # 候補地点の抽出
    if 'dist_km' in names.columns and 'occ_pct' in names.columns:
        # 閾値ベースでフィルタリング
        candidates = filter_candidates(names)
        logger.info(f"{len(candidates)}件の候補地点を抽出しました")
        
        # スコアリング
        candidates = score_candidates(candidates)
        logger.info(f"候補地点にスコアを付けました")
        
        # 結果の確認
        if logger.level <= logging.INFO:
            sample_data = candidates.sort_values('total_score', ascending=False).head(5)[
                ['name', 'type', 'dist_km', 'occ_pct', 'dist_score', 'occ_score', 'total_score']
            ]
            logger.info(f"候補地点スコアリング結果サンプル:\n{sample_data}")
        
        # 可視化
        if visualize and output_dir and len(candidates) > 0:
            os.makedirs(output_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(16, 12))
            
            # 座標系をWeb Mercatorに変換（地図背景用）
            names_web = names.to_crs(epsg=3857)
            candidates_web = candidates.to_crs(epsg=3857)
            
            # 全地名を薄いグレーで背景として描画
            names_web.plot(ax=ax, color='lightgray', alpha=0.3, markersize=15)
            
            # 候補地点をスコア別にカラーコードで描画
            candidates_web.plot(ax=ax, column='total_score', cmap='plasma', 
                               markersize=80, alpha=0.8, legend=True, 
                               edgecolors='white', linewidth=1)
            
            # 地図の範囲を設定（全データの範囲にマージンを追加）
            if len(candidates_web) > 0:
                # 候補地点を中心とした範囲設定
                bounds = candidates_web.total_bounds
                margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.2
                ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
                ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
            else:
                # 候補がない場合は全データの範囲
                bounds = names_web.total_bounds
                margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
                ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
                ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
            
            # 地図背景を追加（座標軸の範囲設定後）
            if HAS_CONTEXTILY:
                try:
                    ctx.add_basemap(ax, crs=candidates_web.crs.to_string(), 
                                   source=ctx.providers.OpenStreetMap.Mapnik)
                    logger.info("🗺️ OpenStreetMap背景を追加しました")
                except Exception as e:
                    logger.warning(f"地図背景の追加に失敗: {e}")
                    logger.debug(f"エラー詳細: {str(e)}")
            
            ax.set_title('Paleochannel Candidate Sites (Color-coded by Score)', fontsize=16, fontweight='bold')
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
            if len(candidates) > 0:
                stats_text += f"Max Score: {candidates['total_score'].max():.2f}\n"
                stats_text += f"Mean Score: {candidates['total_score'].mean():.2f}"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8), fontsize=10)
            
            plt.tight_layout()
            
            output_path = Path(output_dir) / 'step5_paleochannel_candidates.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"📊 古河道候補地点の可視化を保存: {output_path}")
        
        return candidates
    else:
        logger.warning("距離情報または水域頻度情報が不足しているため、候補地点を抽出できません。")
        return None


def backup_existing_file(output_path):
    """既存ファイルのバックアップを作成"""
    if os.path.exists(output_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{output_path}.backup_{timestamp}"
        shutil.copy2(output_path, backup_path)
        logger.info(f"既存ファイルをバックアップしました: {backup_path}")
        return backup_path
    return None


def save_results(candidates, output_path):
    """結果の保存"""
    if candidates is None or len(candidates) == 0:
        logger.warning("保存する候補地点がありません。")
        return
    
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 既存ファイルのバックアップ
    backup_path = backup_existing_file(output_path)
    
    # CSVとして保存
    candidates.to_csv(output_path, index=False)
    logger.info(f"{len(candidates)}件の候補地点を {output_path} に保存しました")
    
    # 上位候補の要約
    if 'total_score' in candidates.columns:
        top_candidates = candidates.sort_values('total_score', ascending=False).head(10)
        logger.info(f"🏆 上位候補地点（トップ10）:")
        for idx, row in top_candidates.iterrows():
            logger.info(f"   {row['name']} ({row['type']}) - スコア: {row['total_score']:.2f}, 距離: {row['dist_km']:.1f}km, 水域頻度: {row['occ_pct']:.1f}%")


def main():
    """メイン処理"""
    args = parse_args()
    
    logger.info("=== 🎯 古河道候補地特定専用スクリプト開始 ===")
    
    # 入力データファイルの確認
    missing_files = check_data_files(args.rivers_path, args.gsw_path, skip_water_freq=args.skip_water_freq)
    if missing_files:
        logger.warning("一部のデータファイルが見つかりません。可能な処理のみ実行します。")
    
    # 可視化出力ディレクトリの設定
    viz_output_dir = None
    if args.visualize:
        # タイムスタンプ付きディレクトリを作成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_output_dir = Path(args.viz_output_dir) / f"site_identification_{timestamp}"
        logger.info(f"📁 可視化出力ディレクトリ: {viz_output_dir}")
    
    # S-1: 対象地域のBBox定義
    bbox_gdf = step1_define_bbox(args.bbox, visualize=args.visualize, output_dir=viz_output_dir)
    
    # S-2: 水場系トポニムの抽出
    names = step2_extract_toponyms(bbox_gdf, args.pbf_path, visualize=args.visualize, output_dir=viz_output_dir)
    if names.empty:
        logger.error("地名収集に失敗しました。処理を終了します。")
        return
    
    # S-3: クレンジング & タイプ付け
    names = step3_process_toponyms(names, visualize=args.visualize, output_dir=viz_output_dir)
    
    # S-4: 現河道との距離計算
    names = step4_calculate_distance(names, args.rivers_path, visualize=args.visualize, output_dir=viz_output_dir)
    
    # S-5: "川が無いのに川名が残る"ポイント抽出
    candidates = step5_extract_candidates(names, args.gsw_path, visualize=args.visualize, skip_water_freq=args.skip_water_freq, output_dir=viz_output_dir)
    
    # 結果の保存
    save_results(candidates, args.output_path)
    
    logger.info("=== ✅ 古河道候補地特定専用スクリプト完了 ===")


if __name__ == "__main__":
    main()