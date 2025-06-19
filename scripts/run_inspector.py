#!/usr/bin/env python3
"""Inspector-Validator Agent 実行スクリプト

このスクリプトは、多言語トポニム解析の結果を評価し、
改善提案を行うInspector-Validator Agentを実行します。

Usage:
    python scripts/run_inspector.py --candidates data/interim/acre_candidates.csv \\
                                   --known data/raw/known_sites.gpkg \\
                                   --output data/output/inspector_reports

Example:
    # 基本実行
    python scripts/run_inspector.py \\
        --candidates data/interim/acre_candidates.csv \\
        --known data/raw/known_sites.gpkg

    # メタ情報と辞書を含む実行
    python scripts/run_inspector.py \\
        --candidates data/interim/acre_candidates.csv \\
        --known data/raw/known_sites.gpkg \\
        --meta config/run_meta.yaml \\
        --dict data/dict/toponym_dict.csv \\
        --output reports/
"""

import argparse
import os
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from tamagawa_to_z.inspector_agent import run


def parse_arguments():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(
        description="Inspector-Validator Agent を実行して候補データを分析し改善提案を行います",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 必須引数
    parser.add_argument(
        "--candidates",
        required=True,
        help="候補データのCSVファイルパス（geometry列がWKT形式）"
    )
    
    parser.add_argument(
        "--known",
        required=True,
        help="既知遺跡データのファイルパス（.gpkg, .shp等のGISファイル）"
    )
    
    # オプション引数
    parser.add_argument(
        "--output",
        default="data/output/inspector_reports",
        help="出力ディレクトリ（デフォルト: data/output/inspector_reports）"
    )
    
    parser.add_argument(
        "--meta",
        help="メタ情報のYAMLファイルパス（run_id, region等の情報）"
    )
    
    parser.add_argument(
        "--dict",
        help="トポニム辞書のCSVファイルパス（将来の拡張用）"
    )
    
    parser.add_argument(
        "--api-key",
        help="OpenAI APIキー（環境変数OPENAI_API_KEYからも読み込み可能）"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細な実行ログを表示"
    )
    
    return parser.parse_args()


def validate_inputs(args):
    """入力ファイルの存在確認とバリデーション"""
    errors = []
    
    # 候補データファイルの確認
    if not os.path.exists(args.candidates):
        errors.append(f"候補データファイルが見つかりません: {args.candidates}")
    elif not args.candidates.endswith('.csv'):
        errors.append(f"候補データファイルはCSV形式である必要があります: {args.candidates}")
    
    # 既知遺跡データファイルの確認
    if not os.path.exists(args.known):
        errors.append(f"既知遺跡データファイルが見つかりません: {args.known}")
    elif not any(args.known.endswith(ext) for ext in ['.gpkg', '.shp', '.geojson']):
        errors.append(f"既知遺跡データファイルはGIS形式（.gpkg, .shp, .geojson）である必要があります: {args.known}")
    
    # メタファイルの確認（オプション）
    if args.meta and not os.path.exists(args.meta):
        errors.append(f"メタ情報ファイルが見つかりません: {args.meta}")
    
    # 辞書ファイルの確認（オプション）
    if args.dict and not os.path.exists(args.dict):
        errors.append(f"辞書ファイルが見つかりません: {args.dict}")
    
    # OpenAI APIキーの確認
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        errors.append("OpenAI APIキーが設定されていません。--api-key引数か環境変数OPENAI_API_KEYを設定してください。")
    
    if errors:
        print("❌ 入力エラー:")
        for error in errors:
            print(f"  • {error}")
        sys.exit(1)
    
    return api_key


def main():
    """メイン実行関数"""
    args = parse_arguments()
    
    # 入力バリデーション
    api_key = validate_inputs(args)
    
    # 実行パラメータの表示
    print("🔍 Inspector-Validator Agent を開始します")
    print("=" * 50)
    print(f"候補データ: {args.candidates}")
    print(f"既知遺跡データ: {args.known}")
    print(f"出力ディレクトリ: {args.output}")
    if args.meta:
        print(f"メタ情報: {args.meta}")
    if args.dict:
        print(f"辞書データ: {args.dict}")
    print("=" * 50)
    
    try:
        # 環境変数の設定（必要に応じて）
        if api_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Inspector-Validator Agent の実行
        print("📊 データ分析を開始しています...")
        
        results = run(
            candidates_path=args.candidates,
            known_sites_path=args.known,
            output_dir=args.output,
            meta_path=args.meta,
            dict_path=args.dict
        )
        
        # 結果の表示
        print("\\n✅ 分析が完了しました")
        print("=" * 50)
        print(f"実行ID: {results['run_id']}")
        print(f"実行時刻: {results['timestamp']}")
        
        # メトリクスのサマリー表示
        if 'metrics' in results:
            metrics = results['metrics']
            print("\\n📊 主要メトリクス:")
            
            # Recall@100
            if 'recall@100' in metrics:
                recall = metrics['recall@100']
                status = "🟢 良好" if recall >= 0.5 else "🟡 改善余地" if recall >= 0.3 else "🔴 低リコール"
                print(f"  • Recall@100: {recall:.3f} ({status})")
            
            # mAP
            if 'map' in metrics:
                map_score = metrics['map']
                status = "🟢 良好" if map_score >= 0.4 else "🟡 改善余地" if map_score >= 0.2 else "🔴 低精度"
                print(f"  • mAP: {map_score:.3f} ({status})")
            
            # Workload
            if 'workload' in metrics:
                workload = metrics['workload']
                status = "🟢 適正" if workload < 500 else "🟡 中負荷" if workload < 1000 else "🔴 高負荷"
                print(f"  • 候補数: {workload} ({status})")
        
        # 改善提案のサマリー表示
        if results.get('proposal'):
            proposal = results['proposal']
            print(f"\\n💡 改善提案: {proposal['action']}")
            print(f"  理由: {proposal['rationale']}")
            if proposal.get('priority'):
                priority_icon = "🔴" if proposal['priority'] == 'high' else "🟡" if proposal['priority'] == 'medium' else "🟢"
                print(f"  優先度: {priority_icon} {proposal['priority']}")
        
        # タイムスタンプディレクトリの計算
        timestamp = results['timestamp']
        timestamp_str = timestamp.replace(":", "-").replace(".", "-").split("T")[0] + "_" + timestamp.split("T")[1].replace(":", "-").split(".")[0]
        
        print("\\n📁 出力ファイル:")
        print(f"  • 改善計画: {args.output}/{timestamp_str}/plan_{results['run_id']}.yaml")
        print(f"  • 分析レポート: {args.output}/{timestamp_str}/report_{results['run_id']}.md")
        print(f"  • 詳細結果: {args.output}/{timestamp_str}/results_{results['run_id']}.json")
        
        print("\\n🎯 次のステップ:")
        print("  1. 生成されたMarkdownレポートを確認")
        print("  2. 改善計画YAMLを編集（必要に応じて）")
        print("  3. パラメータ設定を更新してHarmonizerを再実行")
        
    except KeyboardInterrupt:
        print("\\n⚠️  実行がユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ エラーが発生しました: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()