# DEMO FILE: コマンドラインインターフェース

"""
CLI: コマンドラインインターフェース

このモジュールは、tamagawa-to-z パッケージのコマンドラインインターフェースを提供します。
"""

import os
import sys
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Union, Optional, Any

# パッケージのインポート
from tamagawa_to_z import harmonizer, hydro_da, morph_da, agents, utils


def setup_logging(log_level: str = 'INFO') -> None:
    """ロギングの設定を行う
    
    Parameters
    ----------
    log_level : str, optional
        ログレベル
    """
    # ログレベルの設定
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = log_levels.get(log_level.upper(), logging.INFO)
    
    # ロガーの設定
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する
    
    Returns
    -------
    argparse.Namespace
        解析された引数
    """
    # パーサーの作成
    parser = argparse.ArgumentParser(
        description='tamagawa-to-z: アマゾン古河道・集落探索フレームワーク',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # サブパーサーの作成
    subparsers = parser.add_subparsers(dest='command', help='サブコマンド')
    
    # harmonizer コマンド
    parser_harmonizer = subparsers.add_parser('harmonizer', help='多言語トポニム解析')
    parser_harmonizer.add_argument('--config', type=str, help='設定ファイルのパス')
    parser_harmonizer.add_argument('--input', type=str, required=True, help='入力ファイルのパス')
    parser_harmonizer.add_argument('--output', type=str, required=True, help='出力ファイルのパス')
    parser_harmonizer.add_argument('--log-level', type=str, default='INFO', help='ログレベル')
    
    # hydro_da コマンド
    parser_hydro_da = subparsers.add_parser('hydro_da', help='水理学データ同化')
    parser_hydro_da.add_argument('--config', type=str, help='設定ファイルのパス')
    parser_hydro_da.add_argument('--dem', type=str, required=True, help='DEMファイルのパス')
    parser_hydro_da.add_argument('--output', type=str, required=True, help='出力ディレクトリのパス')
    parser_hydro_da.add_argument('--ensemble-size', type=int, default=50, help='アンサンブルサイズ')
    parser_hydro_da.add_argument('--log-level', type=str, default='INFO', help='ログレベル')
    
    # morph_da コマンド
    parser_morph_da = subparsers.add_parser('morph_da', help='地形変化データ同化')
    parser_morph_da.add_argument('--config', type=str, help='設定ファイルのパス')
    parser_morph_da.add_argument('--dem', type=str, required=True, help='DEMファイルのパス')
    parser_morph_da.add_argument('--water-history', type=str, help='水域履歴ファイルのパス')
    parser_morph_da.add_argument('--toponym-prob', type=str, help='地名確率マップファイルのパス')
    parser_morph_da.add_argument('--output', type=str, required=True, help='出力ディレクトリのパス')
    parser_morph_da.add_argument('--ensemble-size', type=int, default=30, help='アンサンブルサイズ')
    parser_morph_da.add_argument('--log-level', type=str, default='INFO', help='ログレベル')
    
    # agents コマンド
    parser_agents = subparsers.add_parser('agents', help='マルチエージェント')
    parser_agents.add_argument('--config', type=str, help='設定ファイルのパス')
    parser_agents.add_argument('--output', type=str, required=True, help='出力ディレクトリのパス')
    parser_agents.add_argument('--workflow', type=str, help='ワークフロー図の出力パス')
    parser_agents.add_argument('--log-level', type=str, default='INFO', help='ログレベル')
    
    # 引数の解析
    args = parser.parse_args()
    
    # コマンドが指定されていない場合はヘルプを表示
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    return args


def run_harmonizer(args: argparse.Namespace) -> None:
    """harmonizer コマンドを実行する
    
    Parameters
    ----------
    args : argparse.Namespace
        コマンドライン引数
    """
    # ロギングの設定
    setup_logging(args.log_level)
    logger = logging.getLogger('harmonizer')
    
    logger.info('多言語トポニム解析を開始します')
    
    # 設定の読み込み
    config = {}
    if args.config:
        logger.info(f'設定ファイルを読み込みます: {args.config}')
        config = utils.io.load_config(args.config)
    
    # 入力ファイルの読み込み
    logger.info(f'入力ファイルを読み込みます: {args.input}')
    input_data = utils.io.load_vector(args.input)
    
    # Harmonizer の初期化
    water_seeds = config.get('water_seeds', None)
    embedding_model = config.get('embedding_model', 'sentence-transformers/distiluse-base-multilingual-v2')
    
    logger.info(f'Harmonizer を初期化します: {embedding_model}')
    harmonizer_obj = harmonizer.Harmonizer(
        water_seeds=water_seeds,
        embedding_model=embedding_model
    )
    
    # 地名の処理
    logger.info('地名を処理します')
    processed = harmonizer_obj.process(input_data)
    
    # 水関連確率マップの生成
    logger.info('水関連確率マップを生成します')
    shape = config.get('shape', (100, 100))
    bounds = config.get('bounds', None)
    buffer_radius = config.get('buffer_radius', 10)
    
    prob_map = harmonizer_obj.create_probability_map(
        processed,
        shape=shape,
        bounds=bounds,
        buffer_radius=buffer_radius
    )
    
    # 出力ディレクトリの作成
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # 出力ファイルの保存
    logger.info(f'出力ファイルを保存します: {args.output}')
    _, ext = os.path.splitext(args.output)
    
    if ext.lower() in ['.tif', '.tiff']:
        # ラスターとして保存
        harmonizer_obj.save_probability_map(prob_map, args.output)
    else:
        # ベクターとして保存
        utils.io.save_vector(processed, args.output)
    
    logger.info('多言語トポニム解析が完了しました')


def run_hydro_da(args: argparse.Namespace) -> None:
    """hydro_da コマンドを実行する
    
    Parameters
    ----------
    args : argparse.Namespace
        コマンドライン引数
    """
    # ロギングの設定
    setup_logging(args.log_level)
    logger = logging.getLogger('hydro_da')
    
    logger.info('水理学データ同化を開始します')
    
    # 設定の読み込み
    config = {}
    if args.config:
        logger.info(f'設定ファイルを読み込みます: {args.config}')
        config = utils.io.load_config(args.config)
    
    # DEMファイルの読み込み
    logger.info(f'DEMファイルを読み込みます: {args.dem}')
    dem_data, dem_meta = utils.io.load_raster(args.dem)
    
    # 出力ディレクトリの作成
    os.makedirs(args.output, exist_ok=True)
    
    # HydraulicsDA の初期化
    logger.info(f'HydraulicsDA を初期化します: ensemble_size={args.ensemble_size}')
    hydraulics_da_obj = hydro_da.HydraulicsDA(
        config_file=args.config,
        work_dir=args.output,
        ensemble_size=args.ensemble_size
    )
    
    # モデルのセットアップ
    logger.info('モデルをセットアップします')
    hydraulics_da_obj.setup_model(args.dem)
    
    # EnKFの実行
    logger.info('EnKFを実行します')
    parameters, results = hydraulics_da_obj.run_enkf()
    
    # 結果の保存
    logger.info('結果を保存します')
    hydraulics_da_obj.save_results(results)
    
    # 水深データの抽出
    logger.info('水深データを抽出します')
    water_depth = hydraulics_da_obj.extract_water_depth(results)
    
    # 水深データの保存
    water_depth_path = os.path.join(args.output, 'water_depth.tif')
    logger.info(f'水深データを保存します: {water_depth_path}')
    utils.io.save_raster(water_depth, dem_meta, water_depth_path)
    
    logger.info('水理学データ同化が完了しました')


def run_morph_da(args: argparse.Namespace) -> None:
    """morph_da コマンドを実行する
    
    Parameters
    ----------
    args : argparse.Namespace
        コマンドライン引数
    """
    # ロギングの設定
    setup_logging(args.log_level)
    logger = logging.getLogger('morph_da')
    
    logger.info('地形変化データ同化を開始します')
    
    # 設定の読み込み
    config = {}
    if args.config:
        logger.info(f'設定ファイルを読み込みます: {args.config}')
        config = utils.io.load_config(args.config)
    
    # DEMファイルの読み込み
    logger.info(f'DEMファイルを読み込みます: {args.dem}')
    dem_data, dem_meta = utils.io.load_raster(args.dem)
    
    # 水域履歴ファイルの読み込み
    water_history = None
    if args.water_history:
        logger.info(f'水域履歴ファイルを読み込みます: {args.water_history}')
        water_history, _ = utils.io.load_raster(args.water_history)
    
    # 地名確率マップファイルの読み込み
    toponym_prob = None
    if args.toponym_prob:
        logger.info(f'地名確率マップファイルを読み込みます: {args.toponym_prob}')
        toponym_prob, _ = utils.io.load_raster(args.toponym_prob)
    
    # 出力ディレクトリの作成
    os.makedirs(args.output, exist_ok=True)
    
    # MorphDA の初期化
    logger.info(f'MorphDA を初期化します: ensemble_size={args.ensemble_size}')
    morph_da_obj = morph_da.MorphDA(
        config_file=args.config,
        work_dir=args.output,
        ensemble_size=args.ensemble_size
    )
    
    # モデルのセットアップ
    logger.info('モデルをセットアップします')
    morph_da_obj.setup_model(
        args.dem,
        water_history_file=args.water_history,
        toponym_prob_file=args.toponym_prob
    )
    
    # Ensemble Smootherの実行
    logger.info('Ensemble Smootherを実行します')
    past_dem, paleo_channel_prob = morph_da_obj.run_smoother(
        dem_data,
        water_history=water_history,
        toponym_prob=toponym_prob
    )
    
    # 結果の保存
    logger.info('結果を保存します')
    morph_da_obj.save_results(past_dem, paleo_channel_prob)
    
    # 古河道マップの作成
    logger.info('古河道マップを作成します')
    paleo_channel_map = morph_da_obj.create_paleochannel_map(paleo_channel_prob)
    
    # 古河道マップの保存
    paleo_channel_map_path = os.path.join(args.output, 'paleo_channel_map.tif')
    logger.info(f'古河道マップを保存します: {paleo_channel_map_path}')
    utils.io.save_raster(paleo_channel_map, dem_meta, paleo_channel_map_path)
    
    logger.info('地形変化データ同化が完了しました')


def run_agents(args: argparse.Namespace) -> None:
    """agents コマンドを実行する
    
    Parameters
    ----------
    args : argparse.Namespace
        コマンドライン引数
    """
    # ロギングの設定
    setup_logging(args.log_level)
    logger = logging.getLogger('agents')
    
    logger.info('マルチエージェントを開始します')
    
    # 設定の読み込み
    config = {}
    if args.config:
        logger.info(f'設定ファイルを読み込みます: {args.config}')
        config = utils.io.load_config(args.config)
    
    # 出力ディレクトリの作成
    os.makedirs(args.output, exist_ok=True)
    
    # AgentManager の初期化
    logger.info('AgentManager を初期化します')
    agent_manager = agents.AgentManager(
        config_file=args.config,
        work_dir=args.output
    )
    
    # エージェントの作成
    logger.info('エージェントを作成します')
    agent_manager.create_agents()
    
    # タスクの作成
    logger.info('タスクを作成します')
    agent_manager.create_tasks()
    
    # Crewの作成
    logger.info('Crewを作成します')
    agent_manager.create_crew()
    
    # ワークフロー図の生成
    if args.workflow:
        logger.info(f'ワークフロー図を生成します: {args.workflow}')
        mermaid_code = agent_manager.visualize_workflow(args.workflow)
    
    # Crewの実行
    logger.info('Crewを実行します')
    results = agent_manager.run()
    
    # 結果の保存
    results_path = os.path.join(args.output, 'results.yaml')
    logger.info(f'結果を保存します: {results_path}')
    utils.io.save_config(results, results_path)
    
    logger.info('マルチエージェントが完了しました')


def main() -> None:
    """メイン関数
    """
    # 引数の解析
    args = parse_args()
    
    # コマンドの実行
    if args.command == 'harmonizer':
        run_harmonizer(args)
    elif args.command == 'hydro_da':
        run_hydro_da(args)
    elif args.command == 'morph_da':
        run_morph_da(args)
    elif args.command == 'agents':
        run_agents(args)


if __name__ == '__main__':
    main()
