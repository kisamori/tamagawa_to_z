#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_harmonizer.py: 統合ハーモナイザー実行スクリプト

このスクリプトは、水語彙辞書管理と地理的サイト特定の両方のタスクを
統合して実行できるようにしたものです。

使用方法:
    # 両方のタスクを実行
    python run_harmonizer.py --mode both
    
    # 辞書管理のみ実行
    python run_harmonizer.py --mode root-extraction
    
    # サイト特定のみ実行
    python run_harmonizer.py --mode site-identification
    
    # 従来通りの一体実行（レガシー）
    python run_harmonizer.py --mode legacy

オプション:
    --mode MODE           実行モード (both, root-extraction, site-identification, legacy)
    --rivers_path PATH    HydroRIVERSのシェープファイルパス
    --gsw_path PATH       GSW occurrenceのTIFFファイルパス
    --output_path PATH    出力ファイルパス
    --visualize           処理結果を可視化する（デフォルト: False）
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

import pandas as pd

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
from tamagawa_to_z.harmonizer.preprocess import DEFAULT_BBOX


def run_root_extraction_script(args_dict):
    """
    run_root_extraction.pyスクリプトを実行する
    
    Args:
        args_dict: 引数辞書
        
    Returns:
        int: 終了コード
    """
    script_path = PROJECT_ROOT / "scripts" / "run_root_extraction.py"
    
    # 引数の構築
    cmd = [sys.executable, str(script_path)]
    
    # BBOXの追加
    if 'bbox' in args_dict and args_dict['bbox']:
        cmd.extend(['--bbox'] + [str(x) for x in args_dict['bbox']])
    
    # その他の引数
    if args_dict.get('sample_size'):
        cmd.extend(['--sample-size', str(args_dict['sample_size'])])
    
    if args_dict.get('pbf_path'):
        cmd.extend(['--pbf-path', args_dict['pbf_path']])
    
    if args_dict.get('dry_run'):
        cmd.append('--dry-run')
    
    if args_dict.get('root_output_path'):
        cmd.extend(['--output-dir', args_dict['root_output_path']])
    
    if args_dict.get('include_water_features'):
        cmd.append('--include-water-features')
    
    logger.info(f"辞書管理スクリプトを実行中: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("辞書管理スクリプトが正常に完了しました")
        return 0
    except subprocess.CalledProcessError as e:
        logger.error(f"辞書管理スクリプトの実行に失敗しました: {e}")
        logger.error(f"標準出力: {e.stdout}")
        logger.error(f"標準エラー: {e.stderr}")
        return e.returncode


def run_site_identification_script(args_dict):
    """
    run_site_identification.pyスクリプトを実行する
    
    Args:
        args_dict: 引数辞書
        
    Returns:
        int: 終了コード
    """
    script_path = PROJECT_ROOT / "scripts" / "run_site_identification.py"
    
    # 引数の構築
    cmd = [sys.executable, str(script_path)]
    
    # パスの追加
    if args_dict.get('rivers_path'):
        cmd.extend(['--rivers-path', args_dict['rivers_path']])
    
    if args_dict.get('gsw_path'):
        cmd.extend(['--gsw-path', args_dict['gsw_path']])
    
    if args_dict.get('output_path'):
        cmd.extend(['--output-path', args_dict['output_path']])
    
    if args_dict.get('pbf_path'):
        cmd.extend(['--pbf-path', args_dict['pbf_path']])
    
    # BBOXの追加
    if 'bbox' in args_dict and args_dict['bbox']:
        cmd.extend(['--bbox'] + [str(x) for x in args_dict['bbox']])
    
    # その他のオプション
    if args_dict.get('visualize'):
        cmd.append('--visualize')
    
    if args_dict.get('skip_water_freq'):
        cmd.append('--skip-water-freq')
    
    if args_dict.get('distance_threshold'):
        cmd.extend(['--distance-threshold', str(args_dict['distance_threshold'])])
    
    if args_dict.get('water_freq_threshold'):
        cmd.extend(['--water-freq-threshold', str(args_dict['water_freq_threshold'])])
    
    logger.info(f"サイト特定スクリプトを実行中: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("サイト特定スクリプトが正常に完了しました")
        return 0
    except subprocess.CalledProcessError as e:
        logger.error(f"サイト特定スクリプトの実行に失敗しました: {e}")
        logger.error(f"標準出力: {e.stdout}")
        logger.error(f"標準エラー: {e.stderr}")
        return e.returncode


def run_legacy_mode(args_dict):
    """
    従来の一体実行モードを実行する
    
    Args:
        args_dict: 引数辞書
        
    Returns:
        int: 終了コード
    """
    logger.info("レガシーモードで実行中...")
    logger.warning("レガシーモードは現在サポートされていません。")
    logger.warning("--mode both を使用することを推奨します。")
    return 1


def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(
        description='統合ハーモナイザー実行スクリプト'
    )
    
    # 実行モードの設定
    parser.add_argument(
        '--mode',
        type=str,
        choices=['both', 'root-extraction', 'site-identification', 'legacy'],
        default='both',
        help='実行モード: both=両方実行, root-extraction=辞書管理のみ, site-identification=サイト特定のみ, legacy=従来版'
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
        default=str(PROJECT_ROOT / 'data/output/candidates/site_candidates.csv'),
        help='サイト特定結果の出力ファイルパス'
    )
    parser.add_argument(
        '--root_output_path', 
        type=str, 
        default=str(PROJECT_ROOT / 'data/interim/root_analysis_results.json'),
        help='語根分析結果の出力ファイルパス'
    )
    
    # PBFファイルオプション
    parser.add_argument(
        '--pbf-path',
        type=str,
        default=str(PROJECT_ROOT / 'data/raw/osm/norte-latest.osm.pbf'),
        help='PBFファイルのパス'
    )
    
    # 可視化オプション
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='処理結果を可視化する'
    )
    
    # LLMオプション
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='LLMハーモナイゼーションのサンプルサイズ（コスト削減用）'
    )
    
    # ドライランオプション
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='実際の辞書更新を行わずに結果のみ表示'
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
    
    # 水域頻度計算スキップオプション
    parser.add_argument(
        '--skip-water-freq',
        action='store_true',
        help='水域頻度計算をスキップする（距離のみで候補抽出）'
    )
    
    # 閾値オプション
    parser.add_argument(
        '--distance-threshold',
        type=float,
        default=3.0,
        help='河道からの距離閾値（km）'
    )
    
    parser.add_argument(
        '--water-freq-threshold',
        type=float,
        default=20.0,
        help='水域頻度閾値（%）'
    )
    
    # 水域タグ除外ルール無効化オプション
    parser.add_argument(
        '--include-water-features',
        action='store_true',
        help='水域タグを持つ地物も地名候補として含める（デフォルトは除外）'
    )

    return parser.parse_args()


def main():
    """メイン処理"""
    # コマンドライン引数のパース
    args = parse_args()
    
    logger.info("=== 統合ハーモナイザースクリプトを開始 ===")
    logger.info(f"実行モード: {args.mode}")
    
    # 引数を辞書に変換
    args_dict = {
        'bbox': args.bbox,
        'pbf_path': args.pbf_path,
        'sample_size': args.sample_size,
        'dry_run': args.dry_run,
        'root_output_path': args.root_output_path,
        'rivers_path': args.rivers_path,
        'gsw_path': args.gsw_path,
        'output_path': args.output_path,
        'visualize': args.visualize,
        'skip_water_freq': args.skip_water_freq,
        'distance_threshold': args.distance_threshold,
        'water_freq_threshold': args.water_freq_threshold,
        'include_water_features': args.include_water_features
    }
    
    success = True
    
    # モードに応じて実行
    if args.mode == 'both':
        logger.info("=== 辞書管理と地理的サイト特定の両方を実行 ===")
        
        # 1. 語根抽出・辞書管理
        logger.info("--- 1/2: 語根抽出・辞書管理を実行 ---")
        root_result = run_root_extraction_script(args_dict)
        if root_result != 0:
            logger.error("語根抽出・辞書管理に失敗しました")
            success = False
        
        # 2. サイト特定
        logger.info("--- 2/2: 地理的サイト特定を実行 ---")
        site_result = run_site_identification_script(args_dict)
        if site_result != 0:
            logger.error("地理的サイト特定に失敗しました")
            success = False
    
    elif args.mode == 'root-extraction':
        logger.info("=== 語根抽出・辞書管理のみを実行 ===")
        root_result = run_root_extraction_script(args_dict)
        if root_result != 0:
            logger.error("語根抽出・辞書管理に失敗しました")
            success = False
    
    elif args.mode == 'site-identification':
        logger.info("=== 地理的サイト特定のみを実行 ===")
        site_result = run_site_identification_script(args_dict)
        if site_result != 0:
            logger.error("地理的サイト特定に失敗しました")
            success = False
    
    elif args.mode == 'legacy':
        logger.info("=== レガシーモードを実行 ===")
        legacy_result = run_legacy_mode(args_dict)
        if legacy_result != 0:
            logger.error("レガシーモードの実行に失敗しました")
            success = False
    
    # 結果サマリー
    if success:
        logger.info("=== 処理が正常に完了しました ===")
        if args.mode in ['both', 'root-extraction']:
            logger.info(f"📝 語根分析結果: {args.root_output_path}")
        if args.mode in ['both', 'site-identification']:
            logger.info(f"🎯 サイト候補結果: {args.output_path}")
    else:
        logger.error("=== 処理中にエラーが発生しました ===")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)