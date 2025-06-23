#!/usr/bin/env python3
"""最良パラメータ再実行CLI - 最適化で得られた最良パラメータで再実行."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

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
        
        # 評価結果格納
        evaluation_results = {}
        
        # Validation セットで評価
        logger.info("=== Validation セット評価 ===")
        val_score, val_fp = run_pipeline_with_params(
            distance_km=best_params['distance_km'],
            occ_pct=best_params['occ_pct'],
            root_weights=best_params['root_weights'],
            validation_set=splits['val'],
            return_fp=True,
            experiment_id=f"{experiment_id}_val"
        )
        
        evaluation_results['validation'] = {
            'score': val_score,
            'n_sites': len(splits['val']),
            'false_positives': val_fp
        }
        
        logger.info(f"✅ Validation スコア: {val_score:.4f} ({len(splits['val'])} sites)")
        
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




if __name__ == "__main__":
    main()