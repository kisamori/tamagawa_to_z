#!/usr/bin/env python3
"""データ分割CLI - 遺跡データをTrain/Val/Test-time/Test-regionに分割する."""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd

# パッケージのインポートパス追加
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tamagawa_to_z.dataset.splitter import DataSplitter, create_sample_master_csv

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """遺跡データを分割してGPKGファイルを生成する."""
    parser = argparse.ArgumentParser(
        description='遺跡データをTrain/Val/Test-time/Test-regionに分割する',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        default='configs/dataset_split.yaml',
        help='分割設定ファイルパス (デフォルト: configs/dataset_split.yaml)'
    )
    parser.add_argument(
        '--sites', '-s',
        default='data/known/known_acre.kmz',
        help='遺跡ファイルパス (.kmz/.csv/.gpkg) (デフォルト: data/known/known_acre.kmz)'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/known/split',
        help='出力ディレクトリパス (デフォルト: data/known/split)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='詳細ログを表示'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='実際にファイルを作成せずに分割結果のみ表示'
    )
    
    args = parser.parse_args()
    
    # パスオブジェクトに変換
    config = Path(args.config)
    sites_file = Path(args.sites)
    output_dir = Path(args.output)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=== データ分割開始 ===")
    logger.info(f"設定ファイル: {config}")
    logger.info(f"遺跡ファイル: {sites_file}")
    logger.info(f"出力ディレクトリ: {output_dir}")
    
    try:
        # 入力ファイルの存在確認
        if not config.exists():
            logger.error(f"設定ファイルが見つかりません: {config}")
            sys.exit(1)
            
        if not sites_file.exists():
            logger.error(f"遺跡ファイルが見つかりません: {sites_file}")
            if sites_file.suffix.lower() == '.csv':
                response = input("サンプルのknown_sites_master.csvを作成しますか？ [y/N]: ")
                if response.lower() in ['y', 'yes']:
                    sites_file.parent.mkdir(parents=True, exist_ok=True)
                    create_sample_master_csv(sites_file)
                    logger.info(f"サンプルファイルを作成しました: {sites_file}")
                else:
                    sys.exit(1)
            else:
                sys.exit(1)
        
        # データ分割器初期化
        logger.info("データ分割器を初期化中...")
        splitter = DataSplitter(config, sites_file)
        
        # 統計情報表示
        stats = splitter.get_stats()
        logger.info("=== データセット統計 ===")
        logger.info(f"総サイト数: {stats['total_sites']}")
        logger.info(f"文化圏分布: {stats['culture_distribution']}")
        logger.info(f"発見年範囲: {stats['discovery_year_range']['min']}-{stats['discovery_year_range']['max']}")
        logger.info(f"地理的範囲: {stats['geographic_bounds']}")
        
        # 分割実行
        logger.info("データ分割を実行中...")
        splits = splitter.split()
        
        # 分割結果表示
        logger.info("=== 分割結果 ===")
        for name, data in splits.items():
            if isinstance(data, dict):
                logger.info(f"{name}:")
                for sub_name, sub_data in data.items():
                    logger.info(f"  {sub_name}: {len(sub_data)} sites")
            else:
                logger.info(f"{name}: {len(data)} sites")
        
        if args.dry_run:
            logger.info("ドライランモードのため、ファイルは作成されませんでした")
            return
        
        # 出力ディレクトリ作成
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル出力
        logger.info("GPKGファイルを出力中...")
        files_created = []
        
        # train=0.0の場合、既存のtrain.gpkgファイルを削除
        train_file = output_dir / "train.gpkg"
        if "train" not in splits or len(splits.get("train", [])) == 0:
            if train_file.exists():
                train_file.unlink()
                logger.info(f"既存のトレーニングファイルを削除しました: {train_file}")
        
        for name, data in splits.items():
            if isinstance(data, dict):
                # test_region の場合
                for sub_name, sub_data in data.items():
                    if len(sub_data) > 0:
                        output_file = output_dir / f"{name}_{sub_name}.gpkg"
                        sub_data.to_file(output_file, driver="GPKG")
                        files_created.append(output_file)
                        logger.debug(f"作成: {output_file} ({len(sub_data)} sites)")
            else:
                # 通常の分割データ
                if len(data) > 0:
                    output_file = output_dir / f"{name}.gpkg"
                    data.to_file(output_file, driver="GPKG")
                    files_created.append(output_file)
                    logger.debug(f"作成: {output_file} ({len(data)} sites)")
        
        logger.info(f"✅ {len(files_created)} ファイルを作成しました:")
        for file_path in files_created:
            logger.info(f"  {file_path}")
        
        logger.info("=== データ分割完了 ===")
        
    except Exception as e:
        logger.error(f"データ分割中にエラーが発生しました: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()