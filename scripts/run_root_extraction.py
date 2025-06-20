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
from tamagawa_to_z.harmonizer.llm_layer.root_io import build_water_regex

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
    
    # 既知の水関連語根
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
    
    # BBOX オプション
    parser.add_argument(
        '--bbox',
        type=float,
        nargs=4,
        metavar=('LON_MIN', 'LAT_MIN', 'LON_MAX', 'LAT_MAX'),
        default=list(DEFAULT_BBOX.bounds),
        help='対象領域のBBOX (lon_min lat_min lon_max lat_max)'
    )
    
    # Pyrosmオプション
    parser.add_argument(
        '--pbf-path',
        type=str,
        default=str(PROJECT_ROOT / 'data/raw/osm/norte-latest.osm.pbf'),
        help='PBFファイルのパス'
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
    
    return parser.parse_args()


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


def collect_toponyms(bbox_coords, pbf_path, visualize=False, output_dir=None, include_water_features=False, osm_keys=None):
    """地名収集"""
    logger.info("=== 🌍 地名収集フェーズ開始 ===")
    
    # BBoxの作成
    bbox_gdf = make_bbox_gdf(*bbox_coords)
    bbox = bbox_gdf.geometry.iloc[0]
    logger.info(f"対象領域の境界: {bbox.bounds}")
    
    # 最新のRegexパターンを構築
    try:
        logger.info("=== 🔧 Water Vocabulary Regex Construction ===")
        water_regex = build_water_regex()
        logger.info("=== ✅ Regex Construction Completed ===")
    except Exception as e:
        logger.error(f"❌ roots.csvからのRegex構築に失敗: {e}")
        logger.error("❌ 水語彙フィルタリングを実行できません。処理を中止します。")
        raise RuntimeError(f"water_roots.csvが読み込めません: {e}")
    
    # Pyrosmを使用してローカルPBFファイルから水語彙地名を抽出
    logger.info("PyrosmでローカルPBFから水語彙地名を抽出しています...")
    if osm_keys:
        logger.info(f"OSMキー: {osm_keys}")
    try:
        names = extract_toponyms_pyrosm(bbox, pbf_path, regex=water_regex, include_water_features=include_water_features, osm_keys=osm_keys)
        if names.empty:
            logger.warning("ローカルPBFからのデータ取得に失敗しました。")
        else:
            logger.info(f"ローカルPBFから{len(names)}件のトポニムを収集しました")
    except Exception as e:
        logger.error(f"ローカルPBFデータ収集中にエラーが発生しました: {e}")
        logger.warning("空のデータセットで処理を続行します")
        names = gpd.GeoDataFrame([], columns=["name", "geometry", "source"], crs="EPSG:4326")
    
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
            
            # LLMタグ付け実行
            tagged_names = harmonizer.attach_llm_tags(sample_names, name_column="name")
            
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
    """新語根をwater_roots.csvに自動マージ"""
    water_roots_path = PROJECT_ROOT / 'data/dict/water_roots.csv'
    
    # バックアップ作成
    create_backup(water_roots_path)
    
    # 既存辞書読み込み
    existing_roots = pd.read_csv(water_roots_path)
    logger.info(f"📚 既存語根数: {len(existing_roots)}件")
    
    # 新語根抽出（status == 'proposed'のみ）
    new_roots_data = []
    for pattern, data in new_root_analysis.items():
        if data['status'] == 'proposed' and data['proposal']:
            proposal = data['proposal']
            new_root = {
                'root': proposal.get('root'),
                'lang': proposal.get('lang'),
                'regex_token': proposal.get('root'),  # 簡単なパターンとしてrootをそのまま使用
                'meaning_en': proposal.get('meaning_en'),
                'meaning_ja': proposal.get('meaning_ja'),
                'weight': proposal.get('confidence', 0.5)  # confidenceをweightとして使用、デフォルトは0.5
            }
            new_roots_data.append(new_root)
    
    if not new_roots_data:
        logger.info("🔍 マージする新語根がありません")
        return
    
    new_roots_df = pd.DataFrame(new_roots_data)
    logger.info(f"🆕 新語根候補: {len(new_roots_df)}件")
    
    # 重複チェック（既存のrootと重複するものを除外）
    existing_roots_set = set(existing_roots['root'].str.lower())
    new_roots_filtered = new_roots_df[~new_roots_df['root'].str.lower().isin(existing_roots_set)]
    
    duplicates_count = len(new_roots_df) - len(new_roots_filtered)
    if duplicates_count > 0:
        logger.info(f"⚠️  重複により除外された新語根: {duplicates_count}件")
    
    if len(new_roots_filtered) == 0:
        logger.info("🔍 重複チェック後、マージする新語根がありません")
        return
    
    # マージして保存
    merged_roots = pd.concat([existing_roots, new_roots_filtered], ignore_index=True)
    merged_roots.to_csv(water_roots_path, index=False)
    
    logger.info(f"✅ 新語根を{len(new_roots_filtered)}件マージしました")
    logger.info(f"📚 更新後の語根数: {len(merged_roots)}件")
    
    # マージされた新語根を表示
    for _, row in new_roots_filtered.iterrows():
        logger.info(f"   + {row['root']} ({row['lang']}): {row['meaning_ja']}")


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
    
    logger.info("=== 🎯 辞書作成と語根抽出専用スクリプト開始 ===")
    
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
    names = collect_toponyms(args.bbox, args.pbf_path, visualize=args.visualize, output_dir=viz_output_dir, include_water_features=args.include_water_features, osm_keys=osm_keys)
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