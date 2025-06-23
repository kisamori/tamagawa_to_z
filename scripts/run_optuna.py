#!/usr/bin/env python3
"""Hybrid-BO CLI - OptunaとLLMを組み合わせたパラメータ最適化."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
from collections import Counter

import pandas as pd

# パッケージのインポートパス追加
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tamagawa_to_z.dataset.splitter import DataSplitter
from tamagawa_to_z.tuning.optuna_hybrid import HybridBO, create_toponym_stats_from_all_roots

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Hybrid-BO最適化を実行する."""
    parser = argparse.ArgumentParser(
        description='OptunaとLLMを組み合わせたパラメータ最適化',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--trials', '-t',
        type=int,
        default=50,
        help='最適化試行回数 (デフォルト: 50)'
    )
    parser.add_argument(
        '--dataset-config', '-d',
        default='configs/dataset_split.yaml',
        help='データセット分割設定ファイル (デフォルト: configs/dataset_split.yaml)'
    )
    parser.add_argument(
        '--optuna-config', '-o',
        default='configs/optuna_space.yaml',
        help='Optuna設定ファイル (デフォルト: configs/optuna_space.yaml)'
    )
    parser.add_argument(
        '--sites', '-s',
        default='data/known/known_acre.kmz',
        help='遺跡ファイルパス (.kmz/.csv/.gpkg) (デフォルト: data/known/known_acre.kmz)'
    )
    parser.add_argument(
        '--toponym-stats',
        help='地名統計CSVファイル（指定しない場合はサンプルデータ使用）'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='詳細ログを表示'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='既存のstudyを再開する'
    )
    
    args = parser.parse_args()
    
    # パスオブジェクトに変換
    dataset_config = Path(args.dataset_config)
    optuna_config = Path(args.optuna_config)
    sites_file = Path(args.sites)
    toponym_stats = Path(args.toponym_stats) if args.toponym_stats else None
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=== Hybrid-BO最適化開始 ===")
    logger.info(f"試行回数: {args.trials}")
    logger.info(f"データセット設定: {dataset_config}")
    logger.info(f"Optuna設定: {optuna_config}")
    logger.info(f"遺跡ファイル: {sites_file}")
    
    try:
        # 入力ファイルの存在確認
        required_files = [dataset_config, optuna_config, sites_file]
        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"ファイルが見つかりません: {file_path}")
                sys.exit(1)
        
        # データ分割器初期化
        logger.info("データ分割器を初期化中...")
        splitter = DataSplitter(dataset_config, sites_file)
        
        # 地名統計読み込み
        if toponym_stats and toponym_stats.exists():
            logger.info(f"地名統計を読み込み中: {toponym_stats}")
            stats_df = pd.read_csv(toponym_stats)
            if 'root' in stats_df.columns and 'count' in stats_df.columns:
                stats = dict(zip(stats_df['root'], stats_df['count']))
            else:
                logger.warning("地名統計CSVの形式が不正です。all_roots.csvから統計を生成します。")
                stats = create_toponym_stats_from_all_roots()
        else:
            logger.info("地名統計ファイルが指定されていません。all_roots.csvから統計を生成します。")
            stats = create_toponym_stats_from_all_roots()
        
        logger.info(f"地名統計: {len(stats)} roots, 合計出現数: {sum(stats.values())}")
        
        # Hybrid-BO初期化
        logger.info("Hybrid-BOを初期化中...")
        hybrid_bo = HybridBO(
            data_splitter=splitter,
            toponym_stats=stats,
            config_path=optuna_config,
            n_trials=args.trials,
            resume=args.resume
        )
        
        # 既存study情報表示
        if args.resume:
            study_info = hybrid_bo.get_study_info()
            if study_info["n_trials"] > 0:
                logger.info(f"既存study発見: {study_info['n_trials']} trials, best={study_info['best_value']:.4f}")
            else:
                logger.info("既存studyが見つかりません。新しいstudyを開始します")
        else:
            logger.info("新しいstudyを開始します")
        
        # 最適化実行
        logger.info("最適化を実行中...")
        result = hybrid_bo.run()
        
        # 結果表示
        logger.info("=== 最適化結果 ===")
        logger.info(f"🏆 最良スコア: {result['score']:.4f}")
        logger.info(f"📏 距離しきい値: {result['distance_km']:.2f} km")
        logger.info(f"💧 水域出現率: {result['occ_pct']:.2f} %")
        logger.info(f"🔤 語根ウェイト数: {len(result['root_weights'])}")
        logger.info(f"🔢 最良試行番号: {result['trial_number']}")
        
        # 語根ウェイト上位表示
        sorted_weights = sorted(
            result['root_weights'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        logger.info("📊 語根ウェイト (上位10位):")
        for i, (root, weight) in enumerate(sorted_weights[:10]):
            logger.info(f"  {i+1:2d}. {root}: {weight:.3f}")
        
        # Study統計
        study_info = hybrid_bo.get_study_info()
        logger.info(f"📈 総試行数: {study_info['n_trials']}")
        
        # 最適化履歴（上位5位）
        history = result.get('optimization_history', [])
        if len(history) > 1:
            logger.info("🏅 上位5試行:")
            for i, trial in enumerate(history[:5]):
                logger.info(f"  {i+1}. Trial {trial['trial_number']}: {trial['score']:.4f}")
        
        logger.info("=== 最適化完了 ===")
        logger.info(f"結果ディレクトリ: {hybrid_bo.run_dir}")
        logger.info(f"結果ファイル: {hybrid_bo.run_dir}/best_params.json")
        logger.info(f"履歴ファイル: {hybrid_bo.run_dir}/optimization_history.json")
        
    except KeyboardInterrupt:
        logger.info("最適化が中断されました")
        sys.exit(130)
    except Exception as e:
        logger.error(f"最適化中にエラーが発生しました: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)




if __name__ == "__main__":
    main()