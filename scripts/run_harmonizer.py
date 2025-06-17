#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_harmonizer.py: アクレ州マデイラ川上流西部のS-1〜S-5パイプライン実行スクリプト

このスクリプトは、notebooks/01_harmonizer.ipynbの処理を通常のPythonスクリプトとして
実行できるようにしたものです。

使用方法:
    python run_harmonizer.py [--rivers_path PATH] [--gsw_path PATH] [--output_path PATH] [--visualize]

オプション:
    --rivers_path PATH    HydroRIVERSのシェープファイルパス
    --gsw_path PATH       GSW occurrenceのTIFFファイルパス
    --output_path PATH    出力ファイルパス
    --visualize           処理結果を可視化する（デフォルト: False）
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, box

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


def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(
        description='アクレ州マデイラ川上流西部のS-1〜S-5パイプライン実行スクリプト'
    )
    
    # データパスの設定
    parser.add_argument(
        '--rivers_path', 
        type=str, 
        default=str(PROJECT_ROOT / 'data/raw/hydrorivers_sahydrorivers_sa/HydroRIVERS_v10_sa.shp'),
        help='HydroRIVERSのシェープファイルパス'
    )
    parser.add_argument(
        '--gsw_path', 
        type=str, 
        default=str(PROJECT_ROOT / 'data/raw/GSW_occurrence/occurrence_70W_10Sv1_4_2021.tif'),
        help='GSW occurrenceのTIFFファイルパス'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default=str(PROJECT_ROOT / 'data/interim/acre_candidates.parquet'),
        help='出力ファイルパス'
    )
    
    # 可視化オプション
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='処理結果を可視化する'
    )
    
    # Pyrosmオプション
    parser.add_argument(
        '--use-pyrosm',
        action='store_true',
        help='PyrosmでローカルPBFファイルからデータを取得する'
    )
    parser.add_argument(
        '--pbf-path',
        type=str,
        default=str(PROJECT_ROOT / 'data/raw/osm/norte-latest.osm.pbf'),
        help='PBFファイルのパス'
    )
    
    return parser.parse_args()


def check_data_files(rivers_path, gsw_path):
    """入力データファイルの存在を確認する"""
    missing_files = []
    
    if not os.path.exists(rivers_path):
        missing_files.append(rivers_path)
        logger.warning(f"警告: {rivers_path} が見つかりません。")
        logger.warning("HydroRIVERS_SA.shp を data/raw/ ディレクトリに配置してください。")
        logger.warning("ダウンロード先: https://www.hydrosheds.org/products/hydrorivers")
    
    if not os.path.exists(gsw_path):
        missing_files.append(gsw_path)
        logger.warning(f"警告: {gsw_path} が見つかりません。")
        logger.warning("GSW_occurrence.tif を data/raw/ ディレクトリに配置してください。")
        logger.warning("ダウンロード先: https://global-surface-water.appspot.com/download")
    
    return missing_files


def ensure_data_dirs():
    """データディレクトリの存在を確認し、なければ作成する"""
    os.makedirs(PROJECT_ROOT / "data/raw", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "data/interim", exist_ok=True)
    logger.info("データディレクトリを確認しました")


def step1_define_bbox(visualize=False):
    """S-1: 対象地域のBBox定義"""
    logger.info("S-1: 対象地域のBBox定義を実行中...")
    
    # BBoxの作成
    bbox_gdf = make_bbox_gdf()
    bbox = bbox_gdf.geometry.iloc[0]
    logger.info(f"対象領域の境界: {bbox.bounds}")
    
    # 可視化
    if visualize:
        fig, ax = plt.subplots(figsize=(10, 8))
        bbox_gdf.plot(ax=ax, color='none', edgecolor='red')
        ax.set_title('アクレ州マデイラ川上流西部の対象領域')
        plt.tight_layout()
        plt.show()
    
    return bbox_gdf


def step2_extract_toponyms(bbox_gdf, visualize=False, use_pyrosm=False, pbf_path=None):
    """S-2: 水場系トポニムの抽出"""
    logger.info("S-2: 水場系トポニムの抽出を実行中...")
    bbox = bbox_gdf.geometry.iloc[0]
    
    # Pyrosmを使用してローカルPBFファイルから水語彙地名を抽出
    logger.info("PyrosmでローカルPBFから水語彙地名を抽出しています...")
    try:
        from tamagawa_to_z.harmonizer.preprocess import extract_acre_toponyms_pyrosm
        names = extract_acre_toponyms_pyrosm(bbox, pbf_path)
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
    else:
        logger.info(f"合計{len(names)}件のトポニムを収集しました")
    
    # 可視化
    if visualize:
        fig, ax = plt.subplots(figsize=(12, 10))
        bbox_gdf.plot(ax=ax, color='none', edgecolor='red')
        names.plot(ax=ax, column='source', cmap='Set1', markersize=20, legend=True)
        ax.set_title('収集したトポニムの分布')
        plt.tight_layout()
        plt.show()
    
    return names


def step3_process_toponyms(names, visualize=False):
    """S-3: クレンジング & タイプ付け"""
    logger.info("S-3: クレンジング & タイプ付けを実行中...")
    
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
    if visualize:
        plt.figure(figsize=(10, 6))
        type_counts.plot(kind='bar')
        plt.title('水系タイプ別の件数')
        plt.xlabel('水系タイプ')
        plt.ylabel('件数')
        plt.tight_layout()
        plt.show()
    
    return names


def step4_calculate_distance(names, rivers_path, visualize=False):
    """S-4: 現河道との距離計算"""
    logger.info("S-4: 現河道との距離計算を実行中...")
    
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
    if visualize and 'dist_km' in names.columns:
        # 距離の分布を可視化
        plt.figure(figsize=(10, 6))
        plt.hist(names['dist_km'], bins=20)
        plt.title('現河道からの距離の分布')
        plt.xlabel('距離 (km)')
        plt.ylabel('頻度')
        plt.axvline(x=3, color='red', linestyle='--', label='閾値 (3km)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return names


def step5_extract_candidates(names, gsw_path, visualize=False):
    """S-5: "川が無いのに川名が残る"ポイント抽出"""
    logger.info("S-5: 候補地点抽出を実行中...")
    
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
        if visualize:
            fig, ax = plt.subplots(figsize=(12, 10))
            names.plot(ax=ax, color='gray', alpha=0.5, markersize=10)
            candidates.plot(ax=ax, column='total_score', cmap='plasma', markersize=50, alpha=0.7, legend=True)
            ax.set_title('古河道候補地点')
            plt.tight_layout()
            plt.show()
        
        return candidates
    else:
        logger.warning("距離情報または水域頻度情報が不足しているため、候補地点を抽出できません。")
        return None


def save_results(candidates, output_path):
    """結果の保存"""
    if candidates is None or len(candidates) == 0:
        logger.warning("保存する候補地点がありません。")
        return
    
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # CSVとして保存（より安全）
    csv_path = output_path.replace('.parquet', '.csv')
    candidates.to_csv(csv_path, index=False)
    logger.info(f"{len(candidates)}件の候補地点を {csv_path} に保存しました")


def main():
    """メイン処理"""
    # コマンドライン引数のパース
    args = parse_args()
    
    # データディレクトリの確認
    ensure_data_dirs()
    
    # 入力データファイルの確認
    missing_files = check_data_files(args.rivers_path, args.gsw_path)
    if missing_files:
        logger.warning("一部のデータファイルが見つかりません。可能な処理のみ実行します。")
    
    # S-1: 対象地域のBBox定義
    bbox_gdf = step1_define_bbox(visualize=args.visualize)
    
    # S-2: 水場系トポニムの抽出
    names = step2_extract_toponyms(bbox_gdf, visualize=args.visualize, use_pyrosm=True, pbf_path=args.pbf_path)
    
    # S-3: クレンジング & タイプ付け
    names = step3_process_toponyms(names, visualize=args.visualize)
    
    # S-4: 現河道との距離計算
    names = step4_calculate_distance(names, args.rivers_path, visualize=args.visualize)
    
    # S-5: "川が無いのに川名が残る"ポイント抽出
    candidates = step5_extract_candidates(names, args.gsw_path, visualize=args.visualize)
    
    # 結果の保存
    save_results(candidates, args.output_path)
    
    logger.info("処理が完了しました。")


if __name__ == "__main__":
    main()
