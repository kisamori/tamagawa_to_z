#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_root_extraction.py: 辞書作成と語根抽出専用スクリプト

このスクリプトは、run_harmonizer.pyのTask i)を独立実行するためのものです。
- 水関連語彙辞書(toponym_dict.csv)の作成
- 語根抽出(water_root.csv)
- LLMハーモナイゼーション
- 新語根発見分析

使用方法:
    python run_root_extraction.py [--bbox LON_MIN LAT_MIN LON_MAX LAT_MAX] [--sample-size N] [--pbf-path PATH]

オプション:
    --bbox COORDS         対象領域のBBOX (lon_min lat_min lon_max lat_max)
    --sample-size N       LLMハーモナイゼーションのサンプルサイズ（コスト削減用）
    --pbf-path PATH       PyrosmのPBFファイルパス
    --output-dir PATH     出力ディレクトリ
    --visualize           処理結果を可視化する
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# 地図背景用
try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# プロジェクトのルートディレクトリをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# 自作パッケージのインポート
from tamagawa_to_z.harmonizer.preprocess import (
    make_bbox_gdf, extract_toponyms_pyrosm, process_toponyms, DEFAULT_BBOX
)
from tamagawa_to_z.harmonizer.llm_layer.root_io import build_water_regex, build_root_regex
from tamagawa_to_z.config.region_config import RegionConfig, add_region_argument

# 環境変数読み込み
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path)
except ImportError:
    pass


def analyze_for_new_roots(toponyms):
    """
    地名リストから新語根候補をパターン分析で抽出
    
    Args:
        toponyms: 地名のリスト
        
    Returns:
        Dict[str, List[str]]: パターン -> 地名リストの辞書
    """
    candidates = defaultdict(list)
    
    # 既知の水関連語根（CSVから動的に読み込み）
    known_water_roots = set()
    try:
        from tamagawa_to_z.harmonizer.llm_layer.root_io import load_roots
        water_roots_df = load_roots()
        known_water_roots = set(water_roots_df['root'].str.lower().dropna())
        logger.debug(f"既知水語根をCSVから読み込み: {len(known_water_roots)}件")
    except Exception as e:
        logger.warning(f"CSVからの語根読み込みに失敗、デフォルトを使用: {e}")
        known_water_roots = {
            'rio', 'igarape', 'lagoa', 'porto', 'parana', 'igapo', 
            'baixio', 'furo', 'ressaca', 'camaa', 'charco'
        }
    
    # 単語ベースでのパターン分析
    word_patterns = defaultdict(list)
    
    logger.debug(f"   🔍 地名分析開始: {toponyms}")
    
    for toponym in toponyms:
        words = toponym.lower().split()
        logger.debug(f"   🔍 地名 '{toponym}' -> 単語: {words}")
        for word in words:
            # 3文字以上で、水関連の可能性がある単語
            if len(word) >= 3 and not word.isdigit():
                logger.debug(f"     - 単語 '{word}' を分析中...")
                
                # 既知の水関連語根かチェック
                is_water_related = word in known_water_roots
                
                # 語尾変化を考慮して語幹を抽出
                root_candidates = []
                
                # そのまま
                root_candidates.append(word)
                
                # 語尾のs, a, o, e除去を試行
                for suffix in ['s', 'a', 'o', 'e']:
                    if word.endswith(suffix) and len(word) > 3:
                        root_candidates.append(word[:-1])
                
                logger.debug(f"     - 語根候補: {root_candidates} (水関連: {is_water_related})")
                for root in root_candidates:
                    word_patterns[root].append(toponym)
                    logger.debug(f"       '{root}' <- '{toponym}'")
    
    # 1. 既知の水関連語根で2件以上の地名を持つもの
    for pattern, names in word_patterns.items():
        if len(names) >= 2 and pattern in known_water_roots:
            candidates[f"known_{pattern}"] = list(set(names))  # 重複除去
            logger.debug(f"   ✅ 既知水関連パターン '{pattern}' -> {len(candidates[f'known_{pattern}'])}件: {candidates[f'known_{pattern}']}")
    
    # 2. 未知の語根候補を探す（既知の水関連語根以外）
    for pattern, names in word_patterns.items():
        if (len(names) >= 2 and 
            pattern not in known_water_roots and 
            len(pattern) >= 3 and 
            not pattern.startswith(('de', 'da', 'do', 'das', 'dos', 'rua', 'avenida', 'travessa', 'estrada'))):
            # 既知語根ではない && 3文字以上 && 一般的な前置詞・道路名詞ではない
            candidates[f"unknown_{pattern}"] = list(set(names))  # 重複除去
            logger.debug(f"   🔍 未知語根候補 '{pattern}' -> {len(candidates[f'unknown_{pattern}'])}件: {candidates[f'unknown_{pattern}']}")
    
    # 3. 既知語根があるが新しい変形の可能性（参考情報として）
    if not candidates and len(toponyms) >= 2:
        # 地名タイプ別にグループ化（rio, porto等）
        water_type_groups = defaultdict(list)
        for toponym in toponyms:
            words = toponym.lower().split()
            for word in words:
                if word in known_water_roots:
                    water_type_groups[word].append(toponym)
        
        for water_type, names in water_type_groups.items():
            if len(names) >= 2:
                candidates[f"known_{water_type}"] = names
                logger.debug(f"   ℹ️ 既知水関連語根 '{water_type}' -> {len(names)}件: {names} (新語根ではないが分析対象)")
    
    logger.debug(f"   📊 最終結果: {len(candidates)}個のパターン")
    return dict(candidates)


def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(
        description='水関連語彙辞書作成と語根抽出専用スクリプト'
    )
    
    # 地域引数を追加
    parser = add_region_argument(parser)
    
    # BBOX オプション（地域設定で上書き可能）
    parser.add_argument(
        '--bbox',
        type=float,
        nargs=4,
        metavar=('LON_MIN', 'LAT_MIN', 'LON_MAX', 'LAT_MAX'),
        default=None,  # 地域設定から取得
        help='対象領域のBBOX (lon_min lat_min lon_max lat_max) (地域設定を上書き)'
    )
    
    # Pyrosmオプション（地域設定で上書き可能）
    parser.add_argument(
        '--pbf-path',
        type=str,
        default=None,  # 地域設定から取得
        help='PBFファイルのパス (地域設定を上書き)'
    )
    
    # LLMオプション
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='LLMハーモナイゼーションのサンプルサイズ（コスト削減用）'
    )
    
    # 出力オプション
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(PROJECT_ROOT / 'data/interim'),
        help='出力ディレクトリ'
    )
    
    # 可視化オプション
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='処理結果を可視化する'
    )
    parser.add_argument(
        '--viz-output-dir',
        type=str,
        default=str(PROJECT_ROOT / 'data/plots'),
        help='可視化画像の出力ディレクトリ'
    )
    
    # 水域タグ除外ルール無効化オプション
    parser.add_argument(
        '--include-water-features',
        action='store_true',
        help='水域タグを持つ地物も地名候補として含める（デフォルトは除外）'
    )
    
    # 新語根マージ無効化オプション
    parser.add_argument(
        '--no-merge-roots',
        action='store_true',
        help='新語根の自動マージを無効化する'
    )
    
    # OSMキー設定オプション
    parser.add_argument(
        '--osm-keys-config',
        type=str,
        default=str(PROJECT_ROOT / 'data/config/osm_keys.yaml'),
        help='OSMキー設定ファイルのパス'
    )
    parser.add_argument(
        '--osm-keys-mode',
        type=str,
        choices=['conservative', 'standard', 'extended', 'water_focused'],
        default='standard',
        help='OSMキー抽出モード (conservative/standard/extended/water_focused)'
    )
    
    # 語根カテゴリ関連オプション（新規）
    parser.add_argument(
        '--root-categories',
        type=str,
        nargs='+',
        default=['water'],
        help='抽出対象の語根カテゴリ (water terrain flora culture resource temporal)'
    )
    parser.add_argument(
        '--bbox-shift-km',
        type=float,
        default=0.0,
        help='BBOXを東西南北にシフトして再抽出する距離（クラスタリング用）'
    )
    
    # all_roots.csvマージ関連オプション
    parser.add_argument(
        '--create-all-roots',
        action='store_true',
        help='カテゴリ別CSVをall_roots.csvにマージして作成する'
    )
    parser.add_argument(
        '--all-roots-output',
        type=str,
        default=str(PROJECT_ROOT / 'data/dict/all_roots.csv'),
        help='all_roots.csvの出力パス'
    )
    
    return parser.parse_args()


def apply_region_config(args):
    """地域設定を引数に適用する"""
    region_config = RegionConfig()
    data_root = PROJECT_ROOT / 'data/raw'
    
    logger.info(f"🌍 地域設定を適用: {args.region}")
    
    # 地域情報を表示
    region_config.print_region_info(args.region)
    
    # BBOXの設定（コマンドライン引数で上書きされていない場合）
    if args.bbox is None:
        args.bbox = region_config.get_bbox(args.region)
        logger.info(f"📍 BBOX設定: {args.bbox}")
    
    # PBFファイルパスの設定（コマンドライン引数で上書きされていない場合）
    if args.pbf_path is None:
        args.pbf_path = str(region_config.get_osm_pbf_path(args.region, data_root))
        logger.info(f"🗺️ OSM PBF: {args.pbf_path}")
    
    return args


def load_osm_keys_config(config_path, mode='standard'):
    """
    OSMキー設定ファイルを読み込み、指定されたモードのキーリストを返す
    
    Args:
        config_path: 設定ファイルのパス
        mode: 抽出モード ('conservative', 'standard', 'extended', 'water_focused')
        
    Returns:
        List[str]: OSMキーのリスト
    """
    try:
        import yaml
        
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"OSMキー設定ファイルが見つかりません: {config_path}")
            logger.info("デフォルトのキーを使用します: ['place', 'landuse', 'man_made', 'highway']")
            return ['place', 'landuse', 'man_made', 'highway']
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 指定されたモードのキーを取得
        if 'extraction_modes' in config and mode in config['extraction_modes']:
            osm_keys = config['extraction_modes'][mode]
            logger.info(f"OSMキー設定モード '{mode}': {osm_keys}")
            return osm_keys
        else:
            logger.warning(f"指定されたモード '{mode}' が設定ファイルに見つかりません")
            # フォールバック: default_keysを使用
            if 'default_keys' in config:
                osm_keys = config['default_keys']
                logger.info(f"default_keysを使用: {osm_keys}")
                return osm_keys
            else:
                logger.info("ハードコードされたデフォルトを使用: ['place', 'landuse', 'man_made', 'highway']")
                return ['place', 'landuse', 'man_made', 'highway']
                
    except ImportError:
        logger.error("PyYAMLがインストールされていません。pip install pyyaml を実行してください。")
        logger.info("デフォルトのキーを使用します: ['place', 'landuse', 'man_made', 'highway']")
        return ['place', 'landuse', 'man_made', 'highway']
    except Exception as e:
        logger.error(f"OSMキー設定ファイルの読み込みエラー: {e}")
        logger.info("デフォルトのキーを使用します: ['place', 'landuse', 'man_made', 'highway']")
        return ['place', 'landuse', 'man_made', 'highway']


def collect_toponyms(bbox_coords, pbf_path, visualize=False, output_dir=None, include_water_features=False, osm_keys=None, root_categories=None):
    """地名収集（カテゴリ対応版）"""
    logger.info("=== 🌍 地名収集フェーズ開始 ===")
    
    if root_categories is None:
        root_categories = ['water']
    
    logger.info(f"対象カテゴリ: {root_categories}")
    
    # BBoxの作成
    bbox_gdf = make_bbox_gdf(*bbox_coords)
    bbox = bbox_gdf.geometry.iloc[0]
    logger.info(f"対象領域の境界: {bbox.bounds}")
    
    # カテゴリ別Regexパターンを構築
    try:
        logger.info("=== 🔧 Multi-Category Vocabulary Regex Construction ===")
        if len(root_categories) == 1 and root_categories[0] == 'water':
            # 後方互換性: waterのみの場合は旧関数を使用
            water_regex = build_water_regex()
            regex_dict = {'water': water_regex}
            logger.info("水系語根のみを使用（後方互換モード）")
        else:
            # 新形式: カテゴリ別処理
            regex_dict = build_root_regex(root_categories)
            logger.info(f"カテゴリ別語根を使用: {list(regex_dict.keys())}")
        logger.info("=== ✅ Regex Construction Completed ===")
    except Exception as e:
        logger.error(f"❌ 語根CSVからのRegex構築に失敗: {e}")
        logger.error("❌ 語彙フィルタリングを実行できません。処理を中止します。")
        raise RuntimeError(f"語根CSVが読み込めません: {e}")
    
    # Pyrosmを使用してローカルPBFファイルからカテゴリ別語彙地名を抽出
    logger.info(f"PyrosmでローカルPBFから{len(root_categories)}カテゴリの語彙地名を抽出しています...")
    if osm_keys:
        logger.info(f"OSMキー: {osm_keys}")
    try:
        if len(regex_dict) == 1 and 'water' in regex_dict:
            # 後方互換性: waterのみの場合
            names = extract_toponyms_pyrosm(bbox, pbf_path, regex=regex_dict['water'], include_water_features=include_water_features, osm_keys=osm_keys)
        else:
            # 新形式: カテゴリ別処理
            names = extract_toponyms_pyrosm(bbox, pbf_path, regex_dict=regex_dict, include_water_features=include_water_features, osm_keys=osm_keys)
        
        if names.empty:
            logger.warning("ローカルPBFからのデータ取得に失敗しました。")
        else:
            logger.info(f"ローカルPBFから{len(names)}件のトポニムを収集しました")
            # カテゴリ別統計を表示
            if 'root_category' in names.columns:
                category_counts = names['root_category'].value_counts()
                logger.info(f"カテゴリ別件数: {category_counts.to_dict()}")
    except Exception as e:
        logger.error(f"ローカルPBFデータ収集中にエラーが発生しました: {e}")
        logger.warning("空のデータセットで処理を続行します")
        names = gpd.GeoDataFrame([], columns=["name", "geometry", "source", "root_category"], crs="EPSG:4326")
    
    if names.empty:
        logger.warning("どのソースからもトポニムを収集できませんでした。処理を続行できない可能性があります。")
        return None
    
    logger.info(f"合計{len(names)}件のトポニムを収集しました")
    
    # 可視化
    if visualize and output_dir and not names.empty:
        os.makedirs(output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # 座標系をWeb Mercatorに変換（地図背景用）
        bbox_gdf_web = bbox_gdf.to_crs(epsg=3857)
        names_web = names.to_crs(epsg=3857)
        
        # まず対象領域の境界を描画（座標軸の範囲を設定）
        bbox_gdf_web.plot(ax=ax, color='none', edgecolor='red', linewidth=3, alpha=0.8)
        
        # トポニムを描画
        if 'source' in names_web.columns:
            names_web.plot(ax=ax, column='source', cmap='Set1', markersize=50, 
                          legend=True, alpha=0.8, edgecolors='white', linewidth=0.5)
        else:
            names_web.plot(ax=ax, color='blue', markersize=50, alpha=0.8, 
                          edgecolors='white', linewidth=0.5)
        
        # 地図の範囲を設定（bbox_gdf_webの範囲にマージンを追加）
        bounds = bbox_gdf_web.total_bounds
        margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
        ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
        ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
        
        # 地図背景を追加（座標軸の範囲設定後）
        if HAS_CONTEXTILY:
            try:
                ctx.add_basemap(ax, crs=bbox_gdf_web.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
                logger.info("🗺️ OpenStreetMap背景を追加しました")
            except Exception as e:
                logger.warning(f"地図背景の追加に失敗: {e}")
                logger.debug(f"エラー詳細: {str(e)}")
        
        ax.set_title('Distribution of Collected Toponyms (for Root Extraction)', fontsize=16, fontweight='bold')
        ax.set_xlabel('X Coordinate (Web Mercator)')
        ax.set_ylabel('Y Coordinate (Web Mercator)')
        
        # 凡例の調整
        if 'source' in names_web.columns:
            legend = ax.get_legend()
            if legend:
                legend.set_bbox_to_anchor((1.05, 1))
                legend.set_loc('upper left')
        
        plt.tight_layout()
        
        output_path = Path(output_dir) / 'root_extraction_toponyms_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 トポニム分布の可視化を保存: {output_path}")
    
    return names


def harmonize_toponyms(names, sample_size=None, visualize=False, output_dir=None):
    """LLMハーモナイゼーションと新語根発見"""
    logger.info("=== 🤖 LLMハーモナイゼーションフェーズ開始 ===")
    
    # トポニムの基本処理（正規化・タイプ推定）
    names = process_toponyms(names)
    logger.info(f"{len(names)}件のトポニムを処理しました")
    
    # 結果の確認
    if logger.level <= logging.INFO:
        sample_data = pd.DataFrame({
            'name': names['name'],
            'normalized_name': names['normalized_name'],
            'type': names['type']
        }).head(5)
        logger.info(f"処理結果サンプル:\n{sample_data}")
    
    # タイプ別の集計
    type_counts = names['type'].value_counts()
    logger.info(f"水系タイプ別の件数:\n{type_counts}")
    
    # LLMハーモナイゼーション
    try:
        # .envファイルを再読み込み（確実に環境変数を設定）
        try:
            from dotenv import load_dotenv
            env_path = PROJECT_ROOT / ".env"
            load_dotenv(env_path)
            logger.info(f"📁 .envファイルを再読み込みしました: {env_path}")
        except ImportError:
            logger.warning("python-dotenvが利用できません")
        
        # OpenAI APIキーの確認
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("⚠️  OpenAI APIキーが設定されていません")
            logger.warning("📝 .env.example を .env にコピーしてAPIキーを設定してください")
            logger.warning("🔄 LLMハーモナイゼーションをスキップして処理を続行します")
            return names, None
        else:
            logger.info(f"✅ OpenAI APIキーが設定されています (***{api_key[-4:]}) - LLMハーモナイゼーションを開始")
            
            # ToponymHarmonizerの初期化
            from tamagawa_to_z.harmonizer.llm_layer.harmonize import ToponymHarmonizer
            
            harmonizer = ToponymHarmonizer()
            harmonizer.prime_index()  # 既存辞書からEmbeddingインデックス構築
            
            # サンプルサイズの制限
            if sample_size and len(names) > sample_size:
                logger.info(f"🎯 LLMサンプルサイズ制限: {len(names)} → {sample_size}件")
                sample_names = names.sample(n=sample_size, random_state=42)
                logger.info(f"   サンプリングされた地名: {sample_names['name'].tolist()}")
            else:
                sample_names = names
                logger.info(f"🎯 全{len(names)}件でLLMハーモナイゼーションを実行")
            
            # LLMタグ付け実行（候補地名を類似度検索対象に含める）
            tagged_names = harmonizer.attach_llm_tags(
                sample_names, 
                name_column="name",
                include_candidates=True  # 候補地名を類似度検索対象に含める
            )
            
            # 新語根発見の試行
            logger.info("=== 🔍 新語根発見分析開始 ===")
            new_root_analysis = {}
            try:
                # 'different'判定された地名から新語根候補を抽出
                if 'relation' in tagged_names.columns:
                    different_mask = tagged_names['relation'] == 'different'
                    different_names = tagged_names[different_mask]['name'].dropna().tolist()
                    
                    if different_names:
                        logger.info(f"🔍 'different'判定された地名: {len(different_names)}件")
                        logger.info(f"   例: {', '.join(different_names[:3])}")
                        
                        # パターンが共通する地名をグループ化
                        logger.info(f"   📊 パターン分析を実行中...")
                        root_candidates = analyze_for_new_roots(different_names)
                        logger.info(f"   📊 検出されたパターン: {len(root_candidates)}個")
                        
                        if not root_candidates:
                            logger.info("   ⚠️ 共通パターンが見つかりませんでした")
                        else:
                            logger.info(f"   📋 検出されたパターン詳細: {list(root_candidates.keys())}")
                        
                        llm_proposal_attempted = False
                        for pattern, toponyms in root_candidates.items():
                            logger.info(f"📊 パターン '{pattern}': {len(toponyms)}件 - {', '.join(toponyms[:3])}")
                            
                            if len(toponyms) >= 2:  # 最低2件で候補とする
                                logger.info(f"🎯 新語根候補パターン '{pattern}': {len(toponyms)}件")
                                
                                # 未知語根（unknown_で始まる）の場合のみLLM提案を実行
                                if pattern.startswith('unknown_'):
                                    # 新語根提案を試行
                                    logger.info(f"   🤖 LLM新語根提案を実行中...")
                                    llm_proposal_attempted = True
                                    proposal = harmonizer.propose_new_root(
                                        candidate_toponyms=toponyms,
                                        min_frequency=2
                                    )
                                    
                                    if proposal:
                                        logger.info(f"✅ 新語根提案成功: {proposal.get('root')} ({proposal.get('lang')}) - {proposal.get('meaning_ja')}")
                                        new_root_analysis[pattern] = {
                                            'toponyms': toponyms,
                                            'proposal': proposal,
                                            'status': 'proposed'
                                        }
                                        logger.info("   📝 語根追加は手動で検討してください")
                                    else:
                                        logger.info(f"❌ パターン '{pattern}' は新語根として不適切と判定")
                                        new_root_analysis[pattern] = {
                                            'toponyms': toponyms,
                                            'proposal': None,
                                            'status': 'rejected'
                                        }
                                else:
                                    logger.info(f"   ℹ️ 既知語根パターンのため、LLM提案はスキップ")
                                    new_root_analysis[pattern] = {
                                        'toponyms': toponyms,
                                        'proposal': None,
                                        'status': 'known_root'
                                    }
                            else:
                                logger.info(f"   ⚠️ パターン '{pattern}' は最低頻度(2件)を満たしません")
                        
                        if not llm_proposal_attempted:
                            logger.info("🔍 新語根候補として適切なパターンが見つかりませんでした")
                            logger.info("   - 検出されたパターンはすべて既知語根または頻度不足です")
                    else:
                        logger.info("'different'判定された地名がありません")
                else:
                    logger.info("LLM結果に'relation'カラムがありません")
                    
            except Exception as e:
                logger.warning(f"新語根発見分析でエラー: {e}")
            
            logger.info("=== ✅ 新語根発見分析完了 ===")
            
            # サンプルの場合は元データにマージ
            if sample_size and len(names) > sample_size:
                logger.info("🔄 LLM結果を元データセットにマージ中...")
                llm_columns = [col for col in tagged_names.columns if col not in names.columns]
                if llm_columns:
                    names = names.merge(
                        tagged_names[['name'] + llm_columns], 
                        on='name', 
                        how='left'
                    )
                    logger.info(f"   追加されたLLMカラム: {llm_columns}")
            else:
                names = tagged_names
            
            logger.info("=== ✅ LLMハーモナイゼーション完了 ===")
            return names, new_root_analysis
    
    except Exception as e:
        logger.error(f"❌ LLMハーモナイゼーション中にエラーが発生: {e}")
        logger.warning("🔄 LLMハーモナイゼーションをスキップして処理を続行します")
        return names, None
    
    # 可視化
    if visualize and output_dir and len(type_counts) > 0:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        type_counts.plot(kind='bar')
        plt.title('Count by Water System Type (Root Extraction)')
        plt.xlabel('Water System Type')
        plt.ylabel('Count')
        plt.tight_layout()
        
        output_path = Path(output_dir) / 'root_extraction_type_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 水系タイプ分布の可視化を保存: {output_path}")
    
    return names, None


def create_backup(file_path):
    """ファイルのバックアップを作成"""
    if not file_path.exists():
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.parent / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
    
    import shutil
    shutil.copy2(file_path, backup_path)
    logger.info(f"📁 バックアップを作成: {backup_path}")
    return backup_path


def merge_new_roots_to_water_dict(new_root_analysis):
    """新語根をwater_roots.csvに自動マージ（水語彙のみ）"""
    water_roots_path = PROJECT_ROOT / 'data/dict/water_roots.csv'
    
    # バックアップ作成
    create_backup(water_roots_path)
    
    # 既存辞書読み込み
    existing_roots = pd.read_csv(water_roots_path)
    logger.info(f"📚 既存水語根数: {len(existing_roots)}件")
    
    # 新語根抽出（status == 'proposed'かつ水系のみ）
    new_roots_data = []
    for pattern, data in new_root_analysis.items():
        if data['status'] == 'proposed' and data['proposal']:
            proposal = data['proposal']
            
            # 水系語根かどうかを簡易判定（meaningに水関連キーワードが含まれるか）
            meaning_ja = proposal.get('meaning_ja', '').lower()
            meaning_en = proposal.get('meaning_en', '').lower()
            
            water_keywords_ja = ['水', '川', '湖', '沼', '池', '湘', '海', '浜', '港', '汰', '水路', '河', '河口', '湖沼', '湖水', '流', '渓', '港湾']
            water_keywords_en = ['water', 'river', 'stream', 'lake', 'pond', 'swamp', 'marsh', 'creek', 'channel', 'bay', 'harbor', 'lagoon', 'wetland', 'estuary', 'delta', 'mouth']
            
            is_water_related = any(keyword in meaning_ja for keyword in water_keywords_ja) or \
                              any(keyword in meaning_en for keyword in water_keywords_en)
            
            if is_water_related:
                new_root = {
                    'root': proposal.get('root'),
                    'lang': proposal.get('lang'),
                    'regex_token': proposal.get('root'),  # 簡単なパターンとしてrootをそのまま使用
                    'meaning_en': proposal.get('meaning_en'),
                    'meaning_ja': proposal.get('meaning_ja'),
                    'weight': proposal.get('confidence', 0.5)  # confidenceをweightとして使用、デフォルトは0.5
                }
                new_roots_data.append(new_root)
                logger.debug(f"💧 水系語根と判定: {proposal.get('root')} ({meaning_ja})")
            else:
                logger.debug(f"⚠️ 非水系語根と判定、スキップ: {proposal.get('root')} ({meaning_ja})")
    
    if not new_roots_data:
        logger.info("🔍 マージする新水系語根がありません")
        return
    
    new_roots_df = pd.DataFrame(new_roots_data)
    logger.info(f"🆕 新水系語根候補: {len(new_roots_df)}件")
    
    # 重複チェック（既存のrootと重複するものを除外）
    existing_roots_set = set(existing_roots['root'].str.lower())
    new_roots_filtered = new_roots_df[~new_roots_df['root'].str.lower().isin(existing_roots_set)]
    
    duplicates_count = len(new_roots_df) - len(new_roots_filtered)
    if duplicates_count > 0:
        logger.info(f"⚠️  重複により除外された新語根: {duplicates_count}件")
    
    if len(new_roots_filtered) == 0:
        logger.info("🔍 重複チェック後、マージする新水系語根がありません")
        return
    
    # マージして保存
    merged_roots = pd.concat([existing_roots, new_roots_filtered], ignore_index=True)
    merged_roots.to_csv(water_roots_path, index=False)
    
    logger.info(f"✅ 新水系語根を{len(new_roots_filtered)}件マージしました")
    logger.info(f"📚 更新後の水系語根数: {len(merged_roots)}件")
    
    # マージされた新語根を表示
    for _, row in new_roots_filtered.iterrows():
        logger.info(f"   + {row['root']} ({row['lang']}): {row['meaning_ja']}")
    
    # 非水系語根がある場合のアドバイス
    total_proposed = len([data for data in new_root_analysis.values() if data['status'] == 'proposed'])
    water_count = len(new_roots_data)
    if total_proposed > water_count:
        non_water_count = total_proposed - water_count
        logger.info(f"📝 注意: {non_water_count}件の非水系語根が発見されました")
        logger.info(f"       これらはカテゴリ別CSVに追加し、--create-all-roots でall_roots.csvを作成してください")
        logger.info(f"       例: poetry run python scripts/run_root_extraction.py --create-all-roots")


def create_all_roots_csv(categories=None, output_path=None):
    """カテゴリ別CSVをall_roots.csvにマージする"""
    if categories is None:
        categories = ['water', 'terrain', 'flora', 'culture', 'resource', 'temporal']
    
    if output_path is None:
        output_path = PROJECT_ROOT / 'data/dict/all_roots.csv'
    else:
        output_path = Path(output_path)
    
    logger.info("=== 🔄 All Roots CSV マージ開始 ===") 
    logger.info(f"対象カテゴリ: {categories}")
    logger.info(f"出力パス: {output_path}")
    
    all_roots_data = []
    category_counts = {}
    
    # カテゴリ設定ファイルの読み込み
    try:
        import yaml
        categories_config_path = PROJECT_ROOT / 'data/dict/root_categories.yaml'
        if categories_config_path.exists():
            with open(categories_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            categories_mapping = config.get('categories', {})
        else:
            categories_mapping = {}
        logger.info(f"カテゴリ設定を読み込み: {len(categories_mapping)}件")
    except Exception as e:
        logger.warning(f"カテゴリ設定の読み込みに失敗: {e}")
        categories_mapping = {}
    
    # 各カテゴリのCSVを読み込み、マージ
    for category in categories:
        try:
            # CSVファイルパスを決定
            if category in categories_mapping:
                csv_filename = categories_mapping[category]
            else:
                csv_filename = f"{category}_roots.csv"
            
            csv_path = PROJECT_ROOT / "data/dict" / csv_filename
            
            if not csv_path.exists():
                logger.warning(f"カテゴリ '{category}' のCSVファイルが見つかりません: {csv_path}")
                continue
            
            # CSVを読み込み
            df = pd.read_csv(csv_path, dtype=str)
            
            if df.empty:
                logger.warning(f"カテゴリ '{category}' のCSVファイルが空です: {csv_path}")
                continue
            
            # カテゴリカラムを追加（ない場合）
            if 'category' not in df.columns:
                df['category'] = category
            
            # データを追加
            all_roots_data.append(df)
            category_counts[category] = len(df)
            
            logger.info(f"✅ カテゴリ '{category}': {len(df)}件の語根を読み込み")
            
        except Exception as e:
            logger.error(f"❌ カテゴリ '{category}' の処理中にエラー: {e}")
            continue
    
    if not all_roots_data:
        raise ValueError("マージするデータがありません")
    
    # 全データを結合
    merged_df = pd.concat(all_roots_data, ignore_index=True)
    
    # 重複除去（root + category の組み合わせで）
    original_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['root', 'category'], keep='last')
    deduplicated_count = len(merged_df)
    
    if original_count > deduplicated_count:
        logger.info(f"🔄 重複除去: {original_count} → {deduplicated_count} ({original_count - deduplicated_count}件除去)")
    
    # バックアップ作成（既存ファイルがある場合）
    if output_path.exists():
        backup_path = create_backup(output_path)
        logger.info(f"💾 既存ファイルのバックアップ作成: {backup_path}")
    
    # ファイルに保存
    os.makedirs(output_path.parent, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    # 結果のサマリを表示
    logger.info(f"✅ all_roots.csv を作成完了: {output_path}")
    logger.info(f"📊 総語根数: {len(merged_df)}件")
    logger.info(f"📁 カテゴリ別内訳: {category_counts}")
    
    # 結果のサンプル表示
    if logger.level <= logging.INFO:
        sample_df = merged_df[['root', 'lang', 'category', 'meaning_ja']].head(10)
        logger.info(f"🔍 サンプルデータ:")
        for _, row in sample_df.iterrows():
            logger.info(f"   {row['root']:12} ({row['lang']:3}, {row['category']:8}) | {row['meaning_ja']}")
    
    return output_path


def save_results(names, new_root_analysis, output_dir, merge_roots=True):
    """結果の保存と新語根の自動マージ"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. ハーモナイゼーション済み地名辞書の保存
    names_output_path = Path(output_dir) / 'toponym_harmonization_results.csv'
    names.to_csv(names_output_path, index=False)
    logger.info(f"ハーモナイゼーション済み地名辞書を {names_output_path} に保存しました")
    
    # 2. 新語根分析結果の保存（data/dict/に保存）
    if new_root_analysis:
        dict_dir = PROJECT_ROOT / 'data/dict'
        os.makedirs(dict_dir, exist_ok=True)
        analysis_output_path = dict_dir / 'new_root_analysis.csv'
        
        analysis_rows = []
        for pattern, data in new_root_analysis.items():
            row = {
                'pattern': pattern,
                'toponym_count': len(data['toponyms']),
                'toponyms': '; '.join(data['toponyms']),
                'status': data['status'],
                'proposed_root': data['proposal'].get('root') if data['proposal'] else None,
                'proposed_lang': data['proposal'].get('lang') if data['proposal'] else None,
                'proposed_meaning_ja': data['proposal'].get('meaning_ja') if data['proposal'] else None,
                'proposed_meaning_en': data['proposal'].get('meaning_en') if data['proposal'] else None,
                'confidence': data['proposal'].get('confidence') if data['proposal'] else None,
            }
            analysis_rows.append(row)
        
        analysis_df = pd.DataFrame(analysis_rows)
        analysis_df.to_csv(analysis_output_path, index=False)
        logger.info(f"新語根分析結果を {analysis_output_path} に保存しました")
        
        # 提案された新語根の要約
        proposed_roots = analysis_df[analysis_df['status'] == 'proposed']
        if len(proposed_roots) > 0:
            logger.info(f"🎯 提案された新語根: {len(proposed_roots)}件")
            for _, row in proposed_roots.iterrows():
                logger.info(f"   - {row['proposed_root']} ({row['proposed_lang']}): {row['proposed_meaning_ja']}")
            
            # 3. 新語根の自動マージ（デフォルト有効）
            if merge_roots:
                logger.info("=== 🔄 新語根の自動マージ開始 ===")
                try:
                    merge_new_roots_to_water_dict(new_root_analysis)
                    logger.info("=== ✅ 新語根の自動マージ完了 ===")
                except Exception as e:
                    logger.error(f"❌ 新語根マージ中にエラー: {e}")
            else:
                logger.info("🔄 新語根マージは無効化されています")
        else:
            logger.info("🔍 今回は新語根の提案はありませんでした")


def main():
    """メイン処理"""
    args = parse_args()
    
    # 地域設定を適用
    try:
        args = apply_region_config(args)
    except Exception as e:
        logger.error(f"地域設定の適用に失敗: {e}")
        return
    
    logger.info("=== 🎯 辞書作成と語根抽出専用スクリプト開始 ===")
    
    # all_roots.csv作成モードの処理
    if args.create_all_roots:
        try:
            logger.info("=== 🔄 All Roots CSV マージモード ===")
            output_path = create_all_roots_csv(
                categories=args.root_categories,
                output_path=args.all_roots_output
            )
            logger.info(f"✅ all_roots.csv 作成完了: {output_path}")
            logger.info("=== ✅ All Roots CSV マージ完了 ===")
            return  # マージのみで終了
        except Exception as e:
            logger.error(f"❌ all_roots.csv 作成中にエラー: {e}")
            return
    
    # 可視化出力ディレクトリの設定
    viz_output_dir = None
    if args.visualize:
        # タイムスタンプ付きディレクトリを作成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_output_dir = Path(args.viz_output_dir) / f"root_extraction_{timestamp}"
        logger.info(f"📁 可視化出力ディレクトリ: {viz_output_dir}")
    
    # OSMキー設定の読み込み
    osm_keys = load_osm_keys_config(args.osm_keys_config, args.osm_keys_mode)
    
    # フェーズ1: 地名収集
    names = collect_toponyms(
        args.bbox, 
        args.pbf_path, 
        visualize=args.visualize, 
        output_dir=viz_output_dir, 
        include_water_features=args.include_water_features, 
        osm_keys=osm_keys,
        root_categories=args.root_categories
    )
    if names is None or len(names) == 0:
        logger.error("地名収集に失敗しました。処理を終了します。")
        return
    
    # フェーズ2: LLMハーモナイゼーション + 新語根発見
    names, new_root_analysis = harmonize_toponyms(
        names, 
        sample_size=args.sample_size, 
        visualize=args.visualize,
        output_dir=viz_output_dir
    )
    
    # フェーズ3: 結果保存
    save_results(names, new_root_analysis, args.output_dir, merge_roots=not args.no_merge_roots)
    
    logger.info("=== ✅ 辞書作成と語根抽出専用スクリプト完了 ===")
    logger.info(f"結果は {args.output_dir} に保存されました")


if __name__ == "__main__":
    main()