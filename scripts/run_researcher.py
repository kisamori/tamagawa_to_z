#!/usr/bin/env python3
"""Researcher Agent 実行スクリプト

このスクリプトは、Inspector Agent の出力を分析し、
改善提案を行うResearcher Agentを実行します。

Usage:
    python scripts/run_researcher.py --artefacts data/output/inspector_reports/latest \\
                                    --output data/output/research_reports

Example:
    # 基本実行
    python scripts/run_researcher.py \\
        --artefacts data/output/inspector_reports/2025-06-19_14-27-56 \\
        --output reports/research

    # 設定ファイル指定
    python scripts/run_researcher.py \\
        --artefacts data/output/inspector_reports/latest \\
        --output reports/research \\
        --config configs/researcher.yml
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# .envファイルを読み込み
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    # python-dotenvがインストールされていない場合はスキップ
    pass

from tamagawa_to_z.researcher_agent import run


def parse_arguments():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(
        description="Researcher Agent を実行してInspector Agent出力を分析し改善提案を行います",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 必須引数
    parser.add_argument(
        "--artefacts",
        required=True,
        help="IA出力とデータファイルのディレクトリパス"
    )
    
    parser.add_argument(
        "--output",
        default="data/output/research_reports",
        help="研究レポートと計画の出力ディレクトリ（デフォルト: data/output/research_reports）"
    )
    
    # オプション引数
    parser.add_argument(
        "--config",
        help="研究エージェント設定ファイルのパス（YAML形式）"
    )
    
    parser.add_argument(
        "--api-key",
        help="OpenAI APIキー（環境変数OPENAI_API_KEY_TIRE5からも読み込み可能）"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細な実行ログを表示"
    )
    
    return parser.parse_args()


def validate_inputs(args):
    """入力ディレクトリとファイルの存在確認"""
    errors = []
    
    # アーティファクトディレクトリの確認
    if not os.path.exists(args.artefacts):
        errors.append(f"アーティファクトディレクトリが見つかりません: {args.artefacts}")
    elif not os.path.isdir(args.artefacts):
        errors.append(f"アーティファクトパスはディレクトリである必要があります: {args.artefacts}")
    
    # 設定ファイルの確認（オプション）
    if args.config and not os.path.exists(args.config):
        errors.append(f"設定ファイルが見つかりません: {args.config}")
    
    # OpenAI APIキーの確認
    api_key = args.api_key or os.getenv("OPENAI_API_KEY_TIRE5")
    if not api_key:
        errors.append("OpenAI APIキーが設定されていません。--api-key引数か環境変数OPENAI_API_KEY_TIRE5を設定してください。")
    
    if errors:
        print("❌ 入力エラー:")
        for error in errors:
            print(f"  • {error}")
        sys.exit(1)
    
    return api_key


def find_latest_ia_output(base_dir):
    """最新のIA出力ディレクトリを探す"""
    base_path = Path(base_dir)
    
    # 'latest'シンボリックリンクがあるかチェック
    latest_link = base_path / "latest"
    if latest_link.exists() and latest_link.is_dir():
        return str(latest_link)
    
    # タイムスタンプ付きディレクトリを探す
    timestamp_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and any(char.isdigit() for char in item.name):
            timestamp_dirs.append(item)
    
    if timestamp_dirs:
        # 最新の（最後に変更された）ディレクトリを返す
        latest_dir = max(timestamp_dirs, key=lambda p: p.stat().st_mtime)
        return str(latest_dir)
    
    return str(base_path)  # フォールバック


def main():
    """メイン実行関数"""
    args = parse_arguments()
    
    # 入力バリデーション
    api_key = validate_inputs(args)
    
    # 最新のIA出力ディレクトリを解決
    artefacts_dir = find_latest_ia_output(args.artefacts)
    
    # タイムスタンプ付きの出力ディレクトリを作成
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamped_output_dir = os.path.join(args.output, timestamp)
    os.makedirs(timestamped_output_dir, exist_ok=True)
    
    # 実行パラメータの表示
    print("🔬 Researcher Agent を開始します")
    print("=" * 50)
    print(f"アーティファクト: {artefacts_dir}")
    print(f"出力ディレクトリ: {timestamped_output_dir}")
    if args.config:
        print(f"設定ファイル: {args.config}")
    print("=" * 50)
    
    try:
        # 環境変数の設定（必要に応じて）
        if api_key and not os.getenv("OPENAI_API_KEY_TIRE5"):
            os.environ["OPENAI_API_KEY_TIRE5"] = api_key
        
        # Researcher Agent の実行
        print("🧠 分析を開始しています...")
        
        report_path, plan_path = run(
            artefact_dir=artefacts_dir,
            output_dir=timestamped_output_dir,
            config_path=args.config,
            api_key=api_key
        )
        
        # 結果の表示
        print("\\n✅ 分析が完了しました")
        print("=" * 50)
        
        print("\\n📁 生成されたファイル:")
        print(f"  • 研究レポート: {report_path}")
        print(f"  • 改善計画: {plan_path}")
        
        print("\\n🎯 次のステップ:")
        print("  1. 研究レポート（Markdown）を確認してください")
        print("  2. 改善計画（YAML）から実装したい提案を選択してください")
        print("  3. 選択した提案のパラメータでシステムを再実行してください")
        print("\\n📊 提案の評価基準:")
        print("  • 改善ポテンシャル（60%）")
        print("  • アプローチの多様性（20%）")
        print("  • 実装コスト（20%）")
        
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