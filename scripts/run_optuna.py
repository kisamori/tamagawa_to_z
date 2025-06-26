#!/usr/bin/env python3
"""
Real Optuna CLI - 実パイプラインを使ったパラメータ最適化

このスクリプトは、run_site_identification.py を実際に実行して
有効なパラメータ最適化を行います。

従来のrun_optuna.py の問題点（モックデータ使用）を解決し、
実際の地理空間解析結果に基づいた科学的に有効な最適化を実現します。
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import geopandas as gpd

# パッケージのインポートパス追加
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tamagawa_to_z.dataset.splitter import DataSplitter
from tamagawa_to_z.tuning.real_optuna_hybrid import RealHybridBO
from tamagawa_to_z.tuning.real_pipeline_runner import PipelineRunnerConfig
from tamagawa_to_z.config.region_config import RegionConfig, add_region_argument

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_validation_sites(sites_file: Path) -> gpd.GeoDataFrame:
    """検証用の既知サイトを読み込み"""
    try:
        if sites_file.suffix.lower() == '.kmz':
            # KMZファイルの処理（既存のDataSplitterを使用）
            # dataset_configにダミー設定を渡す
            dummy_config = Path(__file__).parent.parent / "configs/dataset_split.yaml"
            splitter = DataSplitter(dummy_config, sites_file)
            sites_gdf = splitter.sites
            
        elif sites_file.suffix.lower() == '.csv':
            # CSVファイルの処理
            df = pd.read_csv(sites_file)
            if 'lat' in df.columns and 'lon' in df.columns:
                from shapely.geometry import Point
                geometry = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
                sites_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            else:
                raise ValueError("CSV file must contain 'lat' and 'lon' columns")
                
        elif sites_file.suffix.lower() == '.gpkg':
            # GeoPackageファイルの処理
            sites_gdf = gpd.read_file(sites_file)
            
        else:
            raise ValueError(f"Unsupported file format: {sites_file.suffix}")
        
        logger.info(f"Loaded {len(sites_gdf)} validation sites from {sites_file}")
        return sites_gdf
        
    except Exception as e:
        logger.error(f"Failed to load validation sites: {e}")
        raise


def create_base_config(args) -> dict:
    """コマンドライン引数からベース設定を作成"""
    
    if args.test_config:
        # テスト用設定（高速）- 地域設定よりも優先
        config = PipelineRunnerConfig.create_test_config()
        logger.info("Using test configuration (fast mode)")
        
        # テストモードでは明示的に指定されたパラメータのみ上書きを許可
        if args.bbox:
            config["bbox"] = args.bbox
            logger.info(f"Test mode: Custom bbox override: {args.bbox}")
        
        if args.pbf_path:
            config["pbf-path"] = args.pbf_path
            logger.info(f"Test mode: Custom PBF path override: {args.pbf_path}")
        
        if args.rivers_path:
            config["rivers-path"] = args.rivers_path
            logger.info(f"Test mode: Custom rivers path override: {args.rivers_path}")
        
        if args.gsw_path:
            config["gsw-path"] = args.gsw_path
            logger.info(f"Test mode: Custom GSW path override: {args.gsw_path}")
        
        if args.skip_water_freq:
            config["skip-water-freq"] = True
            logger.info("Test mode: Water frequency calculation will be skipped")
    else:
        # 地域に基づいた設定を作成
        project_root = Path(__file__).parent.parent
        config = create_region_config(args.region, project_root)
        logger.info(f"Using {args.region} region configuration")
        
        # カスタム設定で上書き
        if args.bbox:
            config["bbox"] = args.bbox
            logger.info(f"Custom bbox: {args.bbox}")
        
        if args.pbf_path:
            config["pbf-path"] = args.pbf_path
        
        if args.rivers_path:
            config["rivers-path"] = args.rivers_path
        
        if args.gsw_path:
            config["gsw-path"] = args.gsw_path
        
        if args.skip_water_freq:
            config["skip-water-freq"] = True
            logger.info("Water frequency calculation will be skipped")
    
    return config


def create_region_config(region: str, project_root: Path) -> dict:
    """地域設定に基づいてパイプライン設定を作成"""
    region_config = RegionConfig()
    data_root = project_root / 'data/raw'
    
    config = {
        "bbox": region_config.get_bbox(region),
        "pbf-path": str(region_config.get_osm_pbf_path(region, data_root)),
        "rivers-path": str(region_config.get_hydrorivers_path(region, data_root)),
        "gsw-path": str(region_config.get_gsw_occurrence_path(region, data_root)),
        "output-path": str(project_root / 'data/output/candidates/paleochannel_candidates.csv'),
        "quiet": True,
        "no-visualize": True,
        "precompute-only": False
    }
    
    logger.info(f"地域 {region} の設定を作成: BBOX={config['bbox']}")
    return config


def main():
    """Real Optuna 最適化を実行"""
    parser = argparse.ArgumentParser(
        description='実パイプラインを使ったパラメータ最適化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本実行（Acre州）
  python run_real_optuna.py --region acre --trials 20 --sites data/known/known_acre.kmz
  
  # マラジオ島で実行
  python run_real_optuna.py --region marajo --trials 20 --sites data/known/known_marajo.kmz
  
  # テスト実行（高速、少数試行）
  python run_real_optuna.py --region acre --trials 3 --test-config --timeout 300
        """
    )
    
    # 地域引数を追加
    parser = add_region_argument(parser)
    
    # 基本パラメータ
    parser.add_argument(
        '--trials', '-t',
        type=int,
        default=20,
        help='最適化試行回数 (デフォルト: 20)'
    )
    
    parser.add_argument(
        '--sites', '-s',
        default=None,  # 地域設定から自動取得
        help='既知遺跡ファイル (.kmz/.csv/.gpkg) (地域設定を上書き)'
    )
    
    # 設定ファイル
    parser.add_argument(
        '--optuna-config', '-o',
        default='configs/optuna_space.yaml',
        help='Optuna設定ファイル (デフォルト: configs/optuna_space.yaml)'
    )
    
    # 実行制御
    parser.add_argument(
        '--resume',
        action='store_true',
        help='既存のstudyを再開する'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=1800,
        help='1試行あたりのタイムアウト時間（秒、デフォルト: 1800=30分）'
    )
    
    # データ設定
    parser.add_argument(
        '--bbox',
        type=float,
        nargs=4,
        metavar=('LON_MIN', 'LAT_MIN', 'LON_MAX', 'LAT_MAX'),
        help='対象領域のBBOX'
    )
    
    parser.add_argument(
        '--pbf-path',
        type=str,
        help='OSM PBFファイルパス'
    )
    
    parser.add_argument(
        '--rivers-path',
        type=str,
        help='HydroRIVERSシェープファイルパス'
    )
    
    parser.add_argument(
        '--gsw-path',
        type=str,
        help='GSW occurrenceファイルパス'
    )
    
    parser.add_argument(
        '--skip-water-freq',
        action='store_true',
        help='水域頻度計算をスキップ（高速化）'
    )
    
    # テスト・デバッグ
    parser.add_argument(
        '--test-config',
        action='store_true',
        help='テスト用設定を使用（高速モード）'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='詳細ログを表示'
    )
    
    args = parser.parse_args()
    
    # 地域設定を適用
    region_config = RegionConfig()
    data_root = Path(__file__).resolve().parents[1] / 'data'
    
    if args.sites is None:
        try:
            args.sites = str(region_config.get_known_sites_path(args.region, data_root / 'raw'))
            logger.info(f"🌍 地域 {args.region} の既知遺跡ファイル: {args.sites}")
        except Exception as e:
            logger.error(f"地域設定からの既知遺跡ファイル取得に失敗: {e}")
            logger.error("--sitesオプションで明示的に指定してください")
            sys.exit(1)
    
    # ロギングレベル設定
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # パス設定
    sites_file = Path(args.sites)
    optuna_config = Path(args.optuna_config)
    script_path = Path(__file__).parent / "run_site_identification.py"
    
    logger.info("=== Real Optuna 最適化開始 ===")
    logger.info(f"🌍 対象地域: {args.region}")
    logger.info(f"試行回数: {args.trials}")
    logger.info(f"既知サイト: {sites_file}")
    logger.info(f"Optuna設定: {optuna_config}")
    logger.info(f"タイムアウト: {args.timeout} 秒/試行")
    
    try:
        # 入力ファイル確認
        if not sites_file.exists():
            logger.error(f"既知サイトファイルが見つかりません: {sites_file}")
            logger.error("ファイルを作成するか、--sitesオプションで正しいパスを指定してください")
            sys.exit(1)
        
        if not optuna_config.exists():
            logger.error(f"Optuna設定ファイルが見つかりません: {optuna_config}")
            sys.exit(1)
        
        if not script_path.exists():
            logger.error(f"パイプラインスクリプトが見つかりません: {script_path}")
            sys.exit(1)
        
        # 検証サイト読み込み
        logger.info("既知サイトを読み込み中...")
        validation_sites = load_validation_sites(sites_file)
        
        # ベース設定作成
        logger.info("実行設定を構築中...")
        base_config = create_base_config(args)
        
        # 設定確認ログ
        logger.info("=== 実行設定 ===")
        for key, value in base_config.items():
            logger.info(f"  {key}: {value}")
        
        # 最適化器初期化
        logger.info("最適化器を初期化中...")
        optimizer = RealHybridBO(
            script_path=str(script_path),
            validation_sites=validation_sites,
            config_path=optuna_config,
            n_trials=args.trials,
            base_config=base_config,
            resume=args.resume,
            timeout_per_trial=args.timeout
        )
        
        # 既存study情報表示
        if args.resume:
            study_info = optimizer.get_study_info()
            if study_info["n_trials"] > 0:
                logger.info(f"既存study再開: {study_info['n_trials']} trials")
            else:
                logger.info("新しいstudyを開始します")
        else:
            logger.info("新しいstudyを開始します")
        
        # 最適化実行
        logger.info("🚀 実パイプライン最適化を実行中...")
        logger.info("注意: 各試行は実際の地理空間解析を実行するため時間がかかります")
        
        result = optimizer.run()
        
        # 結果表示
        logger.info("=== 最適化結果 ===")
        logger.info(f"🏆 最良スコア: {result['score']:.4f}")
        logger.info(f"📏 距離しきい値: {result['distance_km']:.2f} km")
        logger.info(f"💧 水域出現率: {result['occ_pct']:.2f} %")
        logger.info(f"🔤 語根ウェイト数: {len(result.get('root_weights', {}))}")
        logger.info(f"🔢 最良試行番号: {result['trial_number']}")
        
        # 語根ウェイト表示
        root_weights = result.get('root_weights', {})
        if root_weights:
            sorted_weights = sorted(root_weights.items(), key=lambda x: x[1], reverse=True)
            logger.info("📊 語根ウェイト (上位10位):")
            for i, (root, weight) in enumerate(sorted_weights[:10]):
                logger.info(f"  {i+1:2d}. {root}: {weight:.3f}")
        
        # 実行統計
        study_info = optimizer.get_study_info()
        logger.info(f"📈 総試行数: {study_info['n_trials']}")
        
        # 最適化履歴（上位5位）
        history = result.get('optimization_history', [])
        if len(history) > 1:
            history_sorted = sorted(history, key=lambda x: x.get('score', -1), reverse=True)
            logger.info("🏅 上位5試行:")
            for i, trial in enumerate(history_sorted[:5]):
                if trial.get('score') is not None:
                    logger.info(f"  {i+1}. Trial {trial['trial_number']}: {trial['score']:.4f}")
        
        logger.info("=== 最適化完了 ===")
        logger.info(f"結果ディレクトリ: {optimizer.run_dir}")
        logger.info(f"ベストパラメータ: {optimizer.run_dir}/best_params.json")
        logger.info(f"実行履歴: {optimizer.run_dir}/optimization_history.json")
        
        # 次のステップの案内
        logger.info("")
        logger.info("=== 次のステップ ===")
        logger.info("最適化されたパラメータを実パイプラインで実行:")
        logger.info(f"python scripts/run_site_identification.py \\")
        logger.info(f"  --dist-threshold {result['distance_km']:.2f} \\")
        logger.info(f"  --occ-threshold {result['occ_pct']:.2f} \\")
        if root_weights:
            weights_json = json.dumps(root_weights)
            logger.info(f"  --root-weights-json '{weights_json}' \\")
        logger.info(f"  --output-path optimized_candidates.csv")
        
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