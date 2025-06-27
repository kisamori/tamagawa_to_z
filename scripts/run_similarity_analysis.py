#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_similarity_analysis.py: 考古学遺跡類似度分析・候補地ランキング専用スクリプト

既存の地名分析CSVファイルを使用して、機械学習による類似度分析を実行し、
候補地との比較・ランキングを行います。

使用方法:
    python scripts/run_similarity_analysis.py --region REGION [OPTIONS]

主要オプション:
    --region REGION       地域名（利用可能な地域はconfig/region_data_paths.yamlで設定）（必須）
    --mode MODE          分析モード（similarity_only: 既知遺跡のみ, candidate_ranking: 候補地比較）（デフォルト: candidate_ranking）
    --output-dir PATH     出力ディレクトリ（デフォルト: data/output/candidate_ranking/）
    --config PATH        地域設定ファイルのパス（デフォルト: config/region_data_paths.yaml）

AI分析機能:
    OpenAI o3-miniモデルを使用して、各候補地の類似性理由を分析します。
    特徴量の差分を基に、考古学的観点から類似性の理由を生成します。
    
    環境設定:
    export OPENAI_API_KEY="your_openai_api_key"
    
    ※API キーが設定されていない場合は、フォールバック分析を実行します。

地域データパス設定:
    config/region_data_paths.yamlファイルで地域別のデータパスを管理します。
    新しい地域を追加する場合は、このファイルにエントリを追加してください。
    
    現在利用可能な地域:
    - acre: アクレ州の考古学遺跡データ
    - marajo: マラジョ島の考古学遺跡データ

出力:
    data/output/candidate_ranking/candidate_ranking_{timestamp}/
    ├── similarity_scores_{region}.csv (既知遺跡の類似度)
    ├── candidate_ranking_{region}.csv (候補地ランキング + AI分析理由)
    ├── candidate_locations_{region}.csv (候補地位置情報 + AI分析理由)
    ├── candidate_locations_{region}.kmz (候補地位置KMZ + AI分析理由)
    ├── comparison_analysis_{region}.md (統合分析レポート)
    ├── analysis_config.yaml (設定・統計情報)
    └── candidate_ranking.log (処理ログ)

例:
    # 既知遺跡のみの類似度分析
    python scripts/run_similarity_analysis.py --region acre --mode similarity_only
    
    # 候補地を含む比較・ランキング（デフォルト）
    python scripts/run_similarity_analysis.py --region acre
    python scripts/run_similarity_analysis.py --region marajo --mode candidate_ranking
    
    # カスタム設定ファイルを使用
    python scripts/run_similarity_analysis.py --region new_region --config custom_regions.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
import zipfile
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import openai
import os
from typing import Dict, List

# プロジェクトのルートディレクトリをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# YAMLファイルから地域データパス設定を読み込み
def load_region_config(config_path="config/region_data_paths.yaml"):
    """地域設定ファイルを読み込み"""
    full_config_path = PROJECT_ROOT / config_path
    
    if not full_config_path.exists():
        raise FileNotFoundError(f"地域設定ファイルが見つかりません: {full_config_path}")
    
    with open(full_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 従来形式に変換
    region_data_paths = {}
    for region_id, region_config in config['regions'].items():
        region_data_paths[region_id] = {
            'known': region_config['known_sites']['csv'],
            'candidate': region_config['candidate_sites']['csv'],
            'name': region_config['name'],
            'description': region_config['description']
        }
    
    return region_data_paths

# 地域別データパス設定を動的に読み込み
try:
    REGION_DATA_PATHS = load_region_config()
except Exception as e:
    print(f"警告: 地域設定ファイルの読み込みに失敗しました: {e}")
    print("デフォルト設定を使用します。")
    # フォールバック用のデフォルト設定
    REGION_DATA_PATHS = {
        'acre': {
            'known': 'data/output/site_analysis/site_analysis_20250627_005446/site_toponym_analysis_acre.csv',
            'candidate': 'data/output/site_analysis/site_analysis_20250627_113813/site_toponym_analysis_acre.csv',
            'name': 'Acre',
            'description': 'アクレ州の考古学遺跡データ'
        },
        'marajo': {
            'known': 'data/output/site_analysis/site_analysis_20250627_005823/site_toponym_analysis_marajo.csv',
            'candidate': 'data/output/site_analysis/site_analysis_20250627_113813/site_toponym_analysis_acre.csv',
            'name': 'Marajó', 
            'description': 'マラジョ島の考古学遺跡データ'
        }
    }

try:
    from tamagawa_to_z.site_analysis import ArchaeologicalSimilarityAnalyzer
except ImportError as e:
    print(f"Error: 必要なモジュールのインポートに失敗しました: {e}")
    print("プロジェクトが正しくインストールされているか確認してください。")
    sys.exit(1)


def setup_logging(log_file: Path = None, level: str = "INFO") -> logging.Logger:
    """ログ設定"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ルートロガーの設定
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # ファイルハンドラー（オプション）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def create_output_directory(base_dir="data/output/candidate_ranking"):
    """タイムスタンプ付き出力ディレクトリを作成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"candidate_ranking_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, timestamp

def validate_region_data_paths(region):
    """地域データパスの存在確認（デフォルト設定用）"""
    return validate_region_data_paths_custom(region, REGION_DATA_PATHS)

def validate_region_data_paths_custom(region, region_config):
    """地域データパスの存在確認（カスタム設定対応）"""
    if region not in region_config:
        available_regions = list(region_config.keys())
        region_list = '\n'.join([f"  - {r}: {region_config[r].get('name', r)}" for r in available_regions])
        raise ValueError(f"サポートされていない地域です: {region}\n利用可能な地域:\n{region_list}")
    
    region_data = region_config[region]
    validated_paths = {}
    
    # known と candidate パスの検証
    for data_type in ['known', 'candidate']:
        if data_type not in region_data:
            raise KeyError(f"地域 '{region}' の設定に '{data_type}' パスが見つかりません")
        
        path = region_data[data_type]
        full_path = PROJECT_ROOT / path
        if not full_path.exists():
            raise FileNotFoundError(f"{data_type}データファイルが見つかりません: {full_path}")
        
        validated_paths[data_type] = path
    
    return validated_paths

def setup_openai_client():
    """OpenAI APIクライアントの設定"""
    api_key = os.getenv("OPENAI_API_KEY_TIRE5")
    if not api_key:
        raise ValueError("OPENAI_API_KEY_TIRE5環境変数が設定されていません")
    
    client = openai.OpenAI(api_key=api_key)
    return client

def analyze_similarity_reasons_with_llm(candidate_features, known_features_mean, analyzer, candidate_name, region):
    """LLMを使用して類似性の理由を分析"""
    try:
        client = setup_openai_client()
        
        # 特徴量の差分を計算
        feature_diff = {}
        feature_names = analyzer.scaled_features.columns.tolist()
        
        for i, feature_name in enumerate(feature_names):
            if i < len(candidate_features):
                candidate_val = candidate_features[i]
                known_val = known_features_mean[i] if i < len(known_features_mean) else 0
                feature_diff[feature_name] = {
                    'candidate': candidate_val,
                    'known_avg': known_val,
                    'difference': candidate_val - known_val
                }
        
        # 最も重要な特徴量（差分の大きい順）を選択
        important_features = sorted(
            feature_diff.items(), 
            key=lambda x: abs(x[1]['difference']), 
            reverse=True
        )[:10]
        
        # プロンプト作成
        prompt = f"""
考古学遺跡候補地「{candidate_name}」（{region}地域）が既知遺跡と類似している理由を分析してください。

以下は主要な特徴量の比較データです：

主要特徴量の差分（候補地値 vs 既知遺跡平均値）：
"""
        
        for feature_name, data in important_features:
            prompt += f"- {feature_name}: {data['candidate']:.3f} vs {data['known_avg']:.3f} (差分: {data['difference']:.3f})\n"
        
        prompt += f"""
特徴量の説明：
- distance系: 地名までの距離統計（近いほど密集地域）
- density系: 単位面積あたりの地名密度
- count_within系: 特定半径内の地名数
- ratio系: 各カテゴリー地名の比率
- angle系: 地名分布の方位特性
- river系: 河川との位置関係

この候補地が既知の考古学遺跡と類似している理由を、考古学的観点から200字以内で簡潔に説明してください。
地形的特徴、集落パターン、立地条件などの観点から分析してください。
"""
        
        response = client.chat.completions.create(
            model="o3",
            messages=[
                {"role": "user", "content": prompt}
            ],
            reasoning_effort="medium"
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"分析エラー: {str(e)}"

def calculate_site_centers(csv_path):
    """各サイトの中心座標を計算"""
    df = pd.read_csv(csv_path)
    site_centers = []
    
    for site_name in df['site_name'].unique():
        site_data = df[df['site_name'] == site_name]
        center_lat = site_data['toponym_lat'].mean()
        center_lon = site_data['toponym_lon'].mean()
        
        site_centers.append({
            'site_name': site_name,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'toponym_count': len(site_data)
        })
    
    return pd.DataFrame(site_centers)

def generate_kml_content(ranked_sites_df, title="Candidate Sites Ranking"):
    """KMLコンテンツを生成"""
    # KMLルート要素
    kml = Element('kml', xmlns="http://www.opengis.net/kml/2.2")
    document = SubElement(kml, 'Document')
    
    # ドキュメント名
    name_elem = SubElement(document, 'name')
    name_elem.text = title
    
    # スタイル定義（ランキング順に色分け）
    colors = [
        'ff0000ff',  # 1位: 赤
        'ff00ffff',  # 2位: 黄
        'ff00ff00',  # 3位: 緑
        'ffffff00',  # 4位: シアン
        'ffff0000',  # 5位: 青
        'ffff00ff',  # 6位: マゼンタ
        'ff808080',  # 7位以降: グレー
        'ff404040'   # 8位以降: 濃いグレー
    ]
    
    for i in range(min(8, len(ranked_sites_df))):
        style = SubElement(document, 'Style', id=f'rank{i+1}')
        icon_style = SubElement(style, 'IconStyle')
        color_elem = SubElement(icon_style, 'color')
        color_elem.text = colors[i] if i < len(colors) else colors[-1]
        scale_elem = SubElement(icon_style, 'scale')
        scale_elem.text = str(1.2 - i * 0.1) if i < 5 else '0.7'
    
    # 各候補地のプレースマーク
    for idx, (_, row) in enumerate(ranked_sites_df.iterrows()):
        placemark = SubElement(document, 'Placemark')
        
        # 名前
        name_elem = SubElement(placemark, 'name')
        name_elem.text = f"#{row['rank']} {row['site_name']}"
        
        # 説明
        description = SubElement(placemark, 'description')
        similarity_reason = row.get('similarity_reason', '分析なし')
        desc_text = f"""
        <![CDATA[
        <b>候補地ランキング: #{row['rank']}</b><br/>
        <b>サイト名:</b> {row['site_name']}<br/>
        <b>複合スコア:</b> {row['composite_score']:.4f}<br/>
        <b>kNNスコア:</b> {row['knn_score']:.4f}<br/>
        <b>クラスタスコア:</b> {row['cluster_score']:.4f}<br/>
        <b>ガウシアンスコア:</b> {row['gaussian_score']:.4f}<br/>
        <b>異常度スコア:</b> {row['anomaly_score']:.4f}<br/>
        <b>地名数:</b> {row['toponym_count']}件<br/>
        <b>座標:</b> {row['center_lat']:.6f}, {row['center_lon']:.6f}<br/>
        <br/>
        <b>🤖 AI分析による類似性理由:</b><br/>
        {similarity_reason}
        ]]>
        """
        description.text = desc_text
        
        # スタイル
        style_url = SubElement(placemark, 'styleUrl')
        style_url.text = f'#rank{min(idx+1, 8)}'
        
        # 座標
        point = SubElement(placemark, 'Point')
        coordinates = SubElement(point, 'coordinates')
        coordinates.text = f"{row['center_lon']},{row['center_lat']},0"
    
    return kml

def create_kmz_file(kml_content, output_path):
    """KMLからKMZファイルを作成"""
    # KMLを美しくフォーマット
    rough_string = tostring(kml_content, 'unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_kml = reparsed.toprettyxml(indent="  ")
    
    # KMZファイル作成
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as kmz:
        kmz.writestr('doc.kml', pretty_kml)

def generate_candidate_location_outputs(candidate_ranking_df, candidate_csv_path, output_dir, region):
    """候補地の位置情報をKMZとCSVで出力"""
    # 候補地の中心座標を計算
    site_centers_df = calculate_site_centers(candidate_csv_path)
    
    # ランキングデータと位置データをマージ
    ranked_sites_with_location = pd.merge(
        candidate_ranking_df, 
        site_centers_df, 
        on='site_name', 
        how='left'
    )
    
    # CSV出力（座標付きランキング）
    location_csv_path = output_dir / f"candidate_locations_{region}.csv"
    ranked_sites_with_location.to_csv(location_csv_path, index=False, encoding='utf-8')
    
    # KMZ出力
    kml_content = generate_kml_content(
        ranked_sites_with_location, 
        f"Archaeological Candidate Sites Ranking - {region.upper()}"
    )
    kmz_path = output_dir / f"candidate_locations_{region}.kmz"
    create_kmz_file(kml_content, kmz_path)
    
    return location_csv_path, kmz_path

def generate_comprehensive_report(known_scores_df, candidate_ranking_df, region, mode, analyzer):
    """統合分析レポートを生成"""
    report = []
    report.append(f"# 考古学遺跡類似度分析・候補地ランキングレポート（{region.upper()}地域）")
    report.append("")
    report.append("## 分析概要")
    report.append(f"- **対象地域**: {region}")
    report.append(f"- **分析モード**: {mode}")
    report.append(f"- **既知遺跡数**: {len(known_scores_df)}件")
    if candidate_ranking_df is not None:
        report.append(f"- **候補地数**: {len(candidate_ranking_df)}件")
    report.append(f"- **特徴量数**: {len(analyzer.feature_names)}件")
    report.append("")
    
    # 既知遺跡類似度分析結果
    report.append("## 既知遺跡類似度分析結果")
    report.append("既知遺跡間の地名分布パターン類似度をLeave-One-Out法で評価しました。")
    report.append("")
    report.append("### 統計サマリー")
    report.append(f"- **平均複合スコア**: {known_scores_df['composite_score'].mean():.4f}")
    report.append(f"- **標準偏差**: {known_scores_df['composite_score'].std():.4f}")
    report.append(f"- **最高スコア**: {known_scores_df['composite_score'].max():.4f}")
    report.append(f"- **最低スコア**: {known_scores_df['composite_score'].min():.4f}")
    report.append("")
    
    report.append("### 既知遺跡類似度ランキング")
    report.append("| 順位 | 遺跡名 | 複合スコア | kNN | クラスタ | ガウシアン | 異常度 |")
    report.append("|------|--------|------------|-----|-----------|------------|--------|")
    
    sorted_known = known_scores_df.sort_values('composite_score', ascending=False)
    for i, (_, row) in enumerate(sorted_known.iterrows(), 1):
        report.append(f"| {i} | {row['site_name']} | {row['composite_score']:.4f} | {row['knn_score']:.4f} | {row['cluster_score']:.4f} | {row['gaussian_score']:.4f} | {row['anomaly_score']:.4f} |")
    report.append("")
    
    # 候補地ランキング結果
    if candidate_ranking_df is not None:
        report.append("## 候補地ランキング結果")
        report.append("既知遺跡パターンとの類似度に基づいて候補地をランキングしました。")
        report.append("")
        report.append("### 統計サマリー")
        report.append(f"- **平均複合スコア**: {candidate_ranking_df['composite_score'].mean():.4f}")
        report.append(f"- **標準偏差**: {candidate_ranking_df['composite_score'].std():.4f}")
        report.append(f"- **最高スコア**: {candidate_ranking_df['composite_score'].max():.4f}")
        report.append(f"- **最低スコア**: {candidate_ranking_df['composite_score'].min():.4f}")
        report.append("")
        
        report.append("### 候補地ランキング（上位20位）")
        report.append("| 順位 | 候補地名 | 複合スコア | kNN | クラスタ | ガウシアン | 異常度 |")
        report.append("|------|----------|------------|-----|-----------|------------|--------|")
        
        top_candidates = candidate_ranking_df.head(20)
        for _, row in top_candidates.iterrows():
            report.append(f"| {row['rank']} | {row['site_name']} | {row['composite_score']:.4f} | {row['knn_score']:.4f} | {row['cluster_score']:.4f} | {row['gaussian_score']:.4f} | {row['anomaly_score']:.4f} |")
        report.append("")
        
        # 比較分析
        report.append("## 比較分析")
        known_mean = known_scores_df['composite_score'].mean()
        candidate_mean = candidate_ranking_df['composite_score'].mean()
        
        report.append(f"- **既知遺跡平均スコア**: {known_mean:.4f}")
        report.append(f"- **候補地平均スコア**: {candidate_mean:.4f}")
        report.append(f"- **スコア差**: {candidate_mean - known_mean:.4f}")
        
        high_scoring_candidates = len(candidate_ranking_df[candidate_ranking_df['composite_score'] > known_mean])
        report.append(f"- **既知遺跡平均を上回る候補地数**: {high_scoring_candidates}件 ({high_scoring_candidates/len(candidate_ranking_df)*100:.1f}%)")
        report.append("")
    
    # 機械学習手法の説明
    report.append("## 使用した機械学習手法")
    report.append("複合スコアは以下の4つの機械学習手法による類似度スコアの重み付き平均です：")
    report.append("")
    report.append("1. **k最近傍法（kNN）**: 特徴量空間での近傍距離による類似度（重み: 35%）")
    report.append("2. **クラスタリング**: K-meansクラスタ中心からの距離（重み: 25%）")
    report.append("3. **ガウシアン混合**: 確率分布による適合度（重み: 25%）")
    report.append("4. **異常検知**: Isolation Forestによる正常度評価（重み: 15%）")
    report.append("")
    report.append("**複合スコア計算式**: kNN×0.35 + クラスタ×0.25 + ガウシアン×0.25 + 異常度×0.15")
    report.append("")
    
    # 特徴量の説明
    report.append("## 特徴量の説明")
    report.append("分析では以下のカテゴリーの特徴量を使用しました：")
    report.append("")
    report.append("- **距離統計**: 地名までの距離の統計量（最小値、最大値、平均値、中央値、標準偏差、四分位数）")
    report.append("- **河川距離**: 河川までの距離統計")
    report.append("- **圏域密度**: 1km、2km、3km圏内の地名数と密度")
    report.append("- **地名カテゴリー分布**: waterway、natural、place、highway、landuse、man_made等の比率")
    report.append("- **方位分布**: 角度統計、象限分布、エントロピー")
    report.append("- **統計特徴量**: 歪度、尖度、四分位範囲等")
    report.append("")
    
    return "\n".join(report)

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='考古学遺跡類似度分析・候補地ランキング専用スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 必須引数
    available_regions = list(REGION_DATA_PATHS.keys())
    region_help = f"地域名（利用可能: {', '.join(available_regions)}）"
    
    parser.add_argument(
        '--region',
        required=True,
        help=region_help
    )
    
    # オプション引数
    parser.add_argument(
        '--mode',
        default='candidate_ranking',
        choices=['similarity_only', 'candidate_ranking'],
        help='分析モード（デフォルト: candidate_ranking）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default='data/output/candidate_ranking',
        help='出力ベースディレクトリ（デフォルト: data/output/candidate_ranking）'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default='config/region_data_paths.yaml',
        help='地域設定ファイルのパス（デフォルト: config/region_data_paths.yaml）'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='デバッグモードで実行'
    )
    
    args = parser.parse_args()
    
    try:
        # カスタム設定ファイルが指定されている場合は再読み込み
        region_config = REGION_DATA_PATHS
        if args.config != Path('config/region_data_paths.yaml'):
            region_config = load_region_config(str(args.config))
        
        # 地域データパス検証
        region_paths = validate_region_data_paths_custom(args.region, region_config)
        
        # タイムスタンプ付き出力ディレクトリ作成
        output_dir, timestamp = create_output_directory(str(args.output_dir))
        
        # ログ設定
        log_level = "DEBUG" if args.debug else "INFO"
        log_file = output_dir / "candidate_ranking.log"
        logger = setup_logging(log_file, log_level)
        
        logger.info("=== 考古学遺跡類似度分析・候補地ランキング開始 ===")
        logger.info(f"地域: {args.region}")
        logger.info(f"分析モード: {args.mode}")
        logger.info(f"出力ディレクトリ: {output_dir}")
        logger.info(f"既知遺跡データ: {region_paths['known']}")
        if args.mode == 'candidate_ranking':
            logger.info(f"候補地データ: {region_paths['candidate']}")
        
        print(f"📊 考古学遺跡{args.mode}分析を開始します")
        print(f"🌎 対象地域: {args.region}")
        print(f"📁 出力: {output_dir.name}")
        
        # 既知遺跡データで類似度分析器を初期化
        known_csv_path = PROJECT_ROOT / region_paths['known']
        logger.info("既知遺跡類似度分析器を初期化中...")
        analyzer = ArchaeologicalSimilarityAnalyzer(str(known_csv_path))
        
        # ステップ1: 既知遺跡データ読み込み・処理
        logger.info("既知遺跡データ読み込み中...")
        print("⏳ 既知遺跡データ読み込み中...")
        analyzer.load_data()
        
        # ステップ2: 特徴量エンジニアリング
        logger.info("特徴量エンジニアリング中...")
        print("⏳ 特徴量エンジニアリング中...")
        analyzer.engineer_features()
        
        # ステップ3: データ前処理
        logger.info("データ前処理中...")
        print("⏳ データ前処理中...")
        analyzer.preprocess_features()
        
        # ステップ4: 特徴量選択
        logger.info("高相関特徴量除去中...")
        print("⏳ 特徴量選択中...")
        analyzer.remove_high_correlation_features()
        
        # ステップ5: 類似度モデル構築
        logger.info("類似度モデル構築中...")
        print("⏳ 機械学習モデル構築中...")
        analyzer.build_similarity_models()
        
        # ステップ6: 既知遺跡類似度スコア計算
        logger.info("既知遺跡類似度スコア計算中...")
        print("⏳ 既知遺跡類似度スコア計算中...")
        scores_df = analyzer.calculate_similarity_scores()
        
        # 候補地ランキング処理
        candidate_ranking_df = None
        candidate_csv_path = None
        if args.mode == 'candidate_ranking':
            logger.info("候補地データ処理開始...")
            print("⏳ 候補地データ処理中...")
            
            # 候補地データ読み込み
            candidate_csv_path = PROJECT_ROOT / region_paths['candidate']
            candidate_analyzer = ArchaeologicalSimilarityAnalyzer(str(candidate_csv_path))
            candidate_analyzer.load_data()
            candidate_analyzer.engineer_features()
            
            # 候補地特徴量を既知遺跡と同じ前処理で変換
            logger.info("候補地特徴量前処理中...")
            print("⏳ 候補地特徴量前処理中...")
            
            # 候補地特徴量を全く同じパイプラインで処理
            # 1. 全特徴量を抽出（既知遺跡と同じ）
            candidate_features_full = candidate_analyzer.site_features[analyzer.feature_names].fillna(0)
            
            # 2. 既知遺跡と同じスケーリングを適用
            candidate_scaled_full = analyzer.scalers['standard'].transform(candidate_features_full)
            
            # 3. 既知遺跡と同じ高相関特徴量を除去
            # analyzer.scaled_featuresのカラム名を取得（フィルタリング後）
            remaining_features = analyzer.scaled_features.columns.tolist()
            
            # 全特徴量のDataFrameを作成してから必要な特徴量のみ選択
            candidate_scaled_df = pd.DataFrame(candidate_scaled_full, columns=analyzer.feature_names)
            candidate_scaled = candidate_scaled_df[remaining_features].values
            
            # 各候補地の類似度スコア計算
            logger.info("候補地類似度スコア計算中...")
            print("⏳ 候補地類似度スコア計算中...")
            
            candidate_scores = []
            for i, (_, candidate_site) in enumerate(candidate_analyzer.site_features.iterrows()):
                candidate_X = candidate_scaled[i:i+1]
                similarity_scores = analyzer._calculate_single_score(analyzer.scaled_features.values, candidate_X)
                similarity_scores['site_name'] = candidate_site['site_name']
                similarity_scores['region'] = args.region
                candidate_scores.append(similarity_scores)
            
            # 候補地ランキングDataFrame作成
            candidate_ranking_df = pd.DataFrame(candidate_scores)
            candidate_ranking_df = candidate_ranking_df.sort_values('composite_score', ascending=False)
            candidate_ranking_df['rank'] = range(1, len(candidate_ranking_df) + 1)
            
            # LLMによる類似性理由分析
            try:
                logger.info("LLMによる類似性理由分析中...")
                print("⏳ AI分析による類似性理由生成中...")
                
                # 既知遺跡の特徴量平均を計算
                known_features_mean = analyzer.scaled_features.mean().values
                
                similarity_reasons = []
                for i, (_, row) in enumerate(candidate_ranking_df.iterrows()):
                    site_name = row['site_name']
                    candidate_features = candidate_scaled[i]
                    
                    logger.info(f"  - {site_name}の類似性分析中...")
                    reason = analyze_similarity_reasons_with_llm(
                        candidate_features, 
                        known_features_mean, 
                        analyzer, 
                        site_name, 
                        args.region
                    )
                    similarity_reasons.append(reason)
                
                # 類似性理由をDataFrameに追加
                candidate_ranking_df['similarity_reason'] = similarity_reasons
                
            except Exception as e:
                logger.warning(f"LLM分析でエラーが発生しました: {e}")
                print(f"⚠️  AI分析をスキップしました: {e}")
                # フォールバック: 簡単な理由を生成
                candidate_ranking_df['similarity_reason'] = [
                    f"類似度スコア{row['composite_score']:.4f}により既知遺跡パターンとの類似性が検出されました。"
                    for _, row in candidate_ranking_df.iterrows()
                ]
            
            logger.info(f"候補地処理完了: {len(candidate_ranking_df)}件")
        
        # ステップ7: 結果出力
        logger.info("結果出力中...")
        print("⏳ 結果出力中...")
        
        # 既知遺跡類似度スコアCSV出力
        scores_output_path = output_dir / f"similarity_scores_{args.region}.csv"
        scores_df.to_csv(scores_output_path, index=False)
        logger.info(f"既知遺跡類似度スコアCSV出力: {scores_output_path}")
        
        # 候補地ランキングCSV出力
        candidate_output_path = None
        location_csv_path = None
        kmz_path = None
        if candidate_ranking_df is not None:
            candidate_output_path = output_dir / f"candidate_ranking_{args.region}.csv"
            candidate_ranking_df.to_csv(candidate_output_path, index=False)
            logger.info(f"候補地ランキングCSV出力: {candidate_output_path}")
            
            # 候補地位置情報出力（KMZ & CSV）
            logger.info("候補地位置情報ファイル生成中...")
            print("⏳ 候補地位置情報ファイル生成中...")
            location_csv_path, kmz_path = generate_candidate_location_outputs(
                candidate_ranking_df, candidate_csv_path, output_dir, args.region
            )
            logger.info(f"候補地位置CSV出力: {location_csv_path}")
            logger.info(f"候補地位置KMZ出力: {kmz_path}")
        
        # 統合分析レポート生成
        report_path = output_dir / f"comparison_analysis_{args.region}.md"
        report_content = generate_comprehensive_report(
            scores_df, candidate_ranking_df, args.region, args.mode, analyzer
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logger.info(f"統合分析レポート出力: {report_path}")
        
        # 設定ファイル出力
        config_path = output_dir / "analysis_config.yaml"
        config_data = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'region': args.region,
                'mode': args.mode,
                'known_sites_csv': str(known_csv_path),
                'candidate_sites_csv': str(candidate_csv_path) if args.mode == 'candidate_ranking' else None,
                'total_known_sites': len(scores_df),
                'total_candidate_sites': len(candidate_ranking_df) if candidate_ranking_df is not None else 0
            },
            'feature_engineering': {
                'total_features_generated': len(analyzer.feature_names),
                'final_features_count': analyzer.scaled_features.shape[1] if hasattr(analyzer, 'scaled_features') else len(analyzer.feature_names)
            },
            'model_configuration': {
                'knn_neighbors': 3,
                'clustering_components': 3,
                'composite_weights': {
                    'knn': 0.35,
                    'cluster': 0.25,
                    'gaussian': 0.25,
                    'anomaly': 0.15
                }
            },
            'results_summary': {
                'known_sites': {
                    'mean_composite_score': float(scores_df['composite_score'].mean()),
                    'std_composite_score': float(scores_df['composite_score'].std()),
                    'max_composite_score': float(scores_df['composite_score'].max()),
                    'min_composite_score': float(scores_df['composite_score'].min())
                }
            }
        }
        
        if candidate_ranking_df is not None:
            config_data['results_summary']['candidate_sites'] = {
                'mean_composite_score': float(candidate_ranking_df['composite_score'].mean()),
                'std_composite_score': float(candidate_ranking_df['composite_score'].std()),
                'max_composite_score': float(candidate_ranking_df['composite_score'].max()),
                'min_composite_score': float(candidate_ranking_df['composite_score'].min())
            }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"設定ファイル出力: {config_path}")
        
        # 完了メッセージ
        logger.info("=== 分析完了 ===")
        logger.info(f"処理した既知遺跡数: {len(scores_df)}件")
        if candidate_ranking_df is not None:
            logger.info(f"処理した候補地数: {len(candidate_ranking_df)}件")
        
        print(f"\\n✅ {args.mode}分析が完了しました!")
        print(f"📄 既知遺跡類似度: {scores_output_path.name}")
        if candidate_output_path:
            print(f"📄 候補地ランキング: {candidate_output_path.name}")
        if location_csv_path:
            print(f"📍 候補地位置CSV: {location_csv_path.name}")
        if kmz_path:
            print(f"🗺️  候補地位置KMZ: {kmz_path.name}")
        print(f"📋 統合分析レポート: {report_path.name}")
        print(f"⚙️  設定ファイル: {config_path.name}")
        print(f"📊 処理した既知遺跡数: {len(scores_df)}件")
        if candidate_ranking_df is not None:
            print(f"📊 処理した候補地数: {len(candidate_ranking_df)}件")
        
        # 上位結果の表示
        top_known = scores_df.sort_values('composite_score', ascending=False).head(3)
        print(f"\\n🏆 既知遺跡類似度スコア上位3:")
        for i, (_, row) in enumerate(top_known.iterrows(), 1):
            print(f"  {i}. {row['site_name']}: {row['composite_score']:.4f}")
        
        if candidate_ranking_df is not None:
            top_candidates = candidate_ranking_df.head(3)
            print(f"\\n🎯 候補地類似度スコア上位3:")
            for i, (_, row) in enumerate(top_candidates.iterrows(), 1):
                print(f"  {i}. {row['site_name']}: {row['composite_score']:.4f}")
        
    except Exception as e:
        print(f"\\n❌ エラーが発生しました: {e}")
        if args.debug:
            import traceback
            print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()