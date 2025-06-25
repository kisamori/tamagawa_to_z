#!/usr/bin/env python3
"""
テスト用スクリプト：最適化されたOptuna実装のテスト

このスクリプトは、事前計算を活用した高速OptunaSことを確認します。
"""

import sys
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tamagawa_to_z.tuning.real_optuna_hybrid import RealHybridBO
from tamagawa_to_z.tuning.real_pipeline_runner import PipelineRunnerConfig

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_validation_sites():
    """テスト用の検証サイトを作成"""
    import geopandas as gpd
    from shapely.geometry import Point
    
    # アクレ州内のダミーサイト
    sites_data = [
        {"name": "Test Site A", "lat": -9.5, "lon": -68.0},
        {"name": "Test Site B", "lat": -10.0, "lon": -67.5},
    ]
    
    geometry = [Point(s["lon"], s["lat"]) for s in sites_data]
    gdf = gpd.GeoDataFrame(sites_data, geometry=geometry, crs="EPSG:4326")
    
    return gdf


def main():
    """テスト実行"""
    logger.info("=== 最適化されたOptuna実装のテスト開始 ===")
    
    # パス設定
    project_root = Path(__file__).parent.parent
    script_path = str(project_root / "scripts/run_site_identification.py")
    config_path = project_root / "configs/optuna_space.yaml"
    
    # 検証サイト
    validation_sites = create_test_validation_sites()
    
    # 高速テスト設定（小さなBBOX）
    test_config = {
        "bbox": [-68.0, -10.0, -67.5, -9.5],  # 小範囲
        "skip-water-freq": True  # 高速化のため水域頻度計算スキップ
    }
    
    try:
        logger.info("最適化器を初期化中...")
        optimizer = RealHybridBO(
            script_path=script_path,
            validation_sites=validation_sites,
            config_path=config_path,
            n_trials=2,  # テスト用に極少数
            base_config=test_config,
            timeout_per_trial=300  # 5分
        )
        
        logger.info("最適化を実行中...")
        result = optimizer.run()
        
        logger.info("=== テスト結果 ===")
        logger.info(f"🏆 最良スコア: {result['score']:.4f}")
        logger.info(f"📏 距離しきい値: {result['distance_km']:.2f} km") 
        logger.info(f"💧 水域出現率: {result['occ_pct']:.2f} %")
        logger.info(f"🔢 最良試行番号: {result['trial_number']}")
        logger.info(f"📁 出力ディレクトリ: {optimizer.run_dir}")
        
        logger.info("✅ 最適化されたOptuna実装のテストが成功しました！")
        
    except Exception as e:
        logger.error(f"❌ テスト中にエラーが発生: {e}")
        raise


if __name__ == "__main__":
    main()