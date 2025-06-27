#!/usr/bin/env python3
"""
事前計算の修正をテストするスクリプト
"""

import logging
from pathlib import Path
import sys

# パッケージのインポートパス追加
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_precompute_fix():
    """事前計算の修正をテスト"""
    logger.info("=== 事前計算修正テスト開始 ===")
    
    # run_site_identification.py を事前計算モードで実行
    script_path = project_root / "scripts/run_site_identification.py"
    test_output = "/tmp/test_precompute_candidates.csv"
    test_metrics = "/tmp/test_precompute_metrics.json"
    
    if not script_path.exists():
        logger.error(f"スクリプトが見つかりません: {script_path}")
        return False
    
    import subprocess
    
    # テスト用の最小コマンド
    cmd = [
        "python3", str(script_path),
        "--precompute-only",
        "--region", "test",  # テスト地域（小さな範囲）
        "--output-path", test_output,
        "--output-metrics-json", test_metrics,
        "--quiet"
    ]
    
    try:
        logger.info(f"実行コマンド: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,  # 1分でタイムアウト
            cwd=project_root
        )
        
        logger.info(f"終了コード: {result.returncode}")
        
        if result.returncode != 0:
            logger.error(f"実行失敗: {result.stderr}")
            return False
        
        # 出力ファイルの確認
        from pathlib import Path
        output_path = Path(test_output)
        if output_path.exists():
            logger.info(f"✅ 出力ファイル作成成功: {output_path}")
            logger.info(f"   ファイルサイズ: {output_path.stat().st_size} bytes")
            
            # CSVの内容確認
            import pandas as pd
            try:
                df = pd.read_csv(output_path)
                logger.info(f"   行数: {len(df)}")
                logger.info(f"   列: {list(df.columns)}")
                
                # 期待する列の確認
                expected_cols = ['dist_km', 'occ_pct']
                for col in expected_cols:
                    if col in df.columns:
                        logger.info(f"   ✅ {col} 列が存在")
                    else:
                        logger.warning(f"   ⚠️ {col} 列が不足")
                
                return True
                
            except Exception as e:
                logger.error(f"CSVファイル読み込みエラー: {e}")
                return False
        else:
            logger.error(f"❌ 出力ファイルが作成されていません: {output_path}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("実行がタイムアウトしました")
        return False
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        return False

def test_real_hybrid_bo():
    """RealHybridBOクラスのテスト"""
    logger.info("=== RealHybridBO修正テスト開始 ===")
    
    try:
        from tamagawa_to_z.tuning.real_optuna_hybrid import RealHybridBO
        from tamagawa_to_z.tuning.real_pipeline_runner import PipelineRunnerConfig
        import geopandas as gpd
        from shapely.geometry import Point
        
        # テスト用検証サイト
        sites_data = [{"name": "Test Site", "lat": -9.5, "lon": -68.0}]
        geometry = [Point(s["lon"], s["lat"]) for s in sites_data]
        validation_sites = gpd.GeoDataFrame(sites_data, geometry=geometry, crs="EPSG:4326")
        
        # テスト設定
        script_path = str(project_root / "scripts/run_site_identification.py")
        config_path = project_root / "configs/optuna_space.yaml"
        test_config = PipelineRunnerConfig.create_test_config()
        
        # RealHybridBO初期化
        optimizer = RealHybridBO(
            script_path=script_path,
            validation_sites=validation_sites,
            config_path=config_path,
            n_trials=1,  # テスト用に1回のみ
            base_config=test_config,
            timeout_per_trial=60
        )
        
        logger.info("✅ RealHybridBO初期化成功")
        logger.info(f"   実行ディレクトリ: {optimizer.run_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"RealHybridBO初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("修正テストを開始します...")
    
    # テスト1: 事前計算の修正
    test1_result = test_precompute_fix()
    
    # テスト2: RealHybridBOクラス
    test2_result = test_real_hybrid_bo()
    
    # 結果表示
    logger.info("=== テスト結果 ===")
    logger.info(f"事前計算テスト: {'✅ 成功' if test1_result else '❌ 失敗'}")
    logger.info(f"RealHybridBOテスト: {'✅ 成功' if test2_result else '❌ 失敗'}")
    
    if test1_result and test2_result:
        logger.info("🎉 全テスト成功！修正が完了しました。")
        sys.exit(0)
    else:
        logger.error("❌ 一部のテストが失敗しました。")
        sys.exit(1)