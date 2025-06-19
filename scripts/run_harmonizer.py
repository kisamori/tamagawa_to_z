#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_harmonizer.py: アクレ州マデイラ川上流西部のS-1〜S-5パイプライン実行スクリプト

このスクリプトは、notebooks/01_harmonizer.ipynbの処理を通常のPythonスクリプトとして
実行できるようにしたものです。

使用方法:
    python run_harmonizer.py [--rivers_path PATH] [--gsw_path PATH] [--output_path PATH] [--visualize]

オプション:
    --rivers_path PATH    HydroRIVERSのシェープファイルパス
    --gsw_path PATH       GSW occurrenceのTIFFファイルパス
    --output_path PATH    出力ファイルパス
    --visualize           処理結果を可視化する（デフォルト: False）
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, box

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
from tamagawa_to_z.harmonizer import (
    make_bbox_gdf, process_toponyms,
    attach_distance, water_occurrence, filter_candidates, score_candidates
)
from tamagawa_to_z.harmonizer.preprocess import DEFAULT_BBOX
from tamagawa_to_z.harmonizer.llm_layer.root_io import build_water_regex

# 環境変数読み込み
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenvが利用できない場合はスキップ


def _analyze_for_new_roots(toponyms):
    """
    地名リストから新語根候補をパターン分析で抽出
    
    Args:
        toponyms: 地名のリスト
        
    Returns:
        Dict[str, List[str]]: パターン -> 地名リストの辞書
    """
    from collections import defaultdict
    import re
    
    candidates = defaultdict(list)
    
    # 既知の水関連語根
    known_water_roots = {
        'rio', 'igarape', 'lagoa', 'porto', 'parana', 'igapo', 
        'baixio', 'furo', 'ressaca', 'camaa', 'charco'
    }
    
    # 単語ベースでのパターン分析
    word_patterns = defaultdict(list)
    
    logger = logging.getLogger(__name__)
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
        description='アクレ州マデイラ川上流西部のS-1〜S-5パイプライン実行スクリプト'
    )
    
    # データパスの設定
    parser.add_argument(
        '--rivers_path', 
        type=str, 
        default=str(PROJECT_ROOT / 'data/raw/hydrorivers_sahydrorivers_sa/HydroRIVERS_v10_sa.shp'),
        help='HydroRIVERSのシェープファイルパス'
    )
    parser.add_argument(
        '--gsw_path', 
        type=str, 
        default=str(PROJECT_ROOT / 'data/raw/GSW_occurrence/occurrence_70W_10Sv1_4_2021.tif'),
        help='GSW occurrenceのTIFFファイルパス'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default=str(PROJECT_ROOT / 'data/interim/region_candidates.parquet'),
        help='出力ファイルパス'
    )
    
    # 可視化オプション
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='処理結果を可視化する'
    )
    
    # Pyrosmオプション
    parser.add_argument(
        '--use-pyrosm',
        action='store_true',
        help='PyrosmでローカルPBFファイルからデータを取得する'
    )
    parser.add_argument(
        '--pbf-path',
        type=str,
        default=str(PROJECT_ROOT / 'data/raw/osm/norte-latest.osm.pbf'),
        help='PBFファイルのパス'
    )
    
    # LLMオプション
    parser.add_argument(
        '--llm-sample-size',
        type=int,
        default=None,
        help='LLMハーモナイゼーションのサンプルサイズ（コスト削減用）'
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
    
    # 水域頻度計算スキップオプション
    parser.add_argument(
        '--skip-water-freq',
        action='store_true',
        help='水域頻度計算をスキップする（距離のみで候補抽出）'
    )

    return parser.parse_args()


def check_data_files(rivers_path, gsw_path):
    """入力データファイルの存在を確認する"""
    missing_files = []
    
    if not os.path.exists(rivers_path):
        missing_files.append(rivers_path)
        logger.warning(f"警告: {rivers_path} が見つかりません。")
        logger.warning("HydroRIVERS_SA.shp を data/raw/ ディレクトリに配置してください。")
        logger.warning("ダウンロード先: https://www.hydrosheds.org/products/hydrorivers")
    
    if not os.path.exists(gsw_path):
        missing_files.append(gsw_path)
        logger.warning(f"警告: {gsw_path} が見つかりません。")
        logger.warning("GSW_occurrence.tif を data/raw/ ディレクトリに配置してください。")
        logger.warning("ダウンロード先: https://global-surface-water.appspot.com/download")
    
    return missing_files


def ensure_data_dirs():
    """データディレクトリの存在を確認し、なければ作成する"""
    os.makedirs(PROJECT_ROOT / "data/raw", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "data/interim", exist_ok=True)
    logger.info("データディレクトリを確認しました")


def step1_define_bbox(bbox_coords, visualize=False):
    """S-1: 対象地域のBBox定義"""
    logger.info("S-1: 対象地域のBBox定義を実行中...")

    # BBoxの作成
    bbox_gdf = make_bbox_gdf(*bbox_coords)
    bbox = bbox_gdf.geometry.iloc[0]
    logger.info(f"対象領域の境界: {bbox.bounds}")
    
    # 可視化
    if visualize:
        fig, ax = plt.subplots(figsize=(10, 8))
        bbox_gdf.plot(ax=ax, color='none', edgecolor='red')
        ax.set_title('アクレ州マデイラ川上流西部の対象領域')
        plt.tight_layout()
        plt.show()
    
    return bbox_gdf


def step2_extract_toponyms(bbox_gdf, visualize=False, use_pyrosm=False, pbf_path=None):
    """S-2: 水場系トポニムの抽出"""
    logger.info("S-2: 水場系トポニムの抽出を実行中...")
    bbox = bbox_gdf.geometry.iloc[0]
    
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
    try:
        from tamagawa_to_z.harmonizer.preprocess import extract_toponyms_pyrosm
        names = extract_toponyms_pyrosm(bbox, pbf_path, regex=water_regex)
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
    else:
        logger.info(f"合計{len(names)}件のトポニムを収集しました")
    
    # 可視化
    if visualize:
        fig, ax = plt.subplots(figsize=(12, 10))
        bbox_gdf.plot(ax=ax, color='none', edgecolor='red')
        names.plot(ax=ax, column='source', cmap='Set1', markersize=20, legend=True)
        ax.set_title('収集したトポニムの分布')
        plt.tight_layout()
        plt.show()
    
    return names


def step3_process_toponyms(names, visualize=False, llm_sample_size=None):
    """S-3: クレンジング & タイプ付け & LLMハーモナイゼーション"""
    logger.info("S-3: クレンジング & タイプ付けを実行中...")
    
    # トポニムの処理
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
    
    # S-3b: LLMハーモナイゼーション（既存辞書との照合・新語根発見）
    logger.info("=== 🤖 S-3b: LLM Harmonization Starting ===")
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
        else:
            logger.info(f"✅ OpenAI APIキーが設定されています (***{api_key[-4:]}) - LLMハーモナイゼーションを開始")
            
            # ToponymHarmonizerの初期化
            from tamagawa_to_z.harmonizer.llm_layer.harmonize import ToponymHarmonizer
            
            harmonizer = ToponymHarmonizer()
            harmonizer.prime_index()  # 既存辞書からEmbeddingインデックス構築
            
            # サンプルサイズの制限
            if llm_sample_size and len(names) > llm_sample_size:
                logger.info(f"🎯 LLMサンプルサイズ制限: {len(names)} → {llm_sample_size}件")
                sample_names = names.sample(n=llm_sample_size, random_state=42)
                logger.info(f"   サンプリングされた地名: {sample_names['name'].tolist()}")
            else:
                sample_names = names
                logger.info(f"🎯 全{len(names)}件でLLMハーモナイゼーションを実行")
            
            # LLMタグ付け実行
            tagged_names = harmonizer.attach_llm_tags(sample_names, name_column="name")
            
            # S-3c: 新語根発見の試行
            logger.info("=== 🔍 S-3c: New Root Discovery Analysis ===")
            try:
                # 'different'判定された地名から新語根候補を抽出
                different_toponyms = []
                if 'relation' in tagged_names.columns:
                    different_mask = tagged_names['relation'] == 'different'
                    different_names = tagged_names[different_mask]['name'].dropna().tolist()
                    
                    if different_names:
                        logger.info(f"🔍 'different'判定された地名: {len(different_names)}件")
                        logger.info(f"   例: {', '.join(different_names[:3])}")
                        
                        # パターンが共通する地名をグループ化
                        logger.info(f"   📊 パターン分析を実行中...")
                        root_candidates = _analyze_for_new_roots(different_names)
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
                                        # 実際の語根追加は今回は手動とする（自動追加は慎重に）
                                        logger.info("   📝 語根追加は手動で検討してください")
                                    else:
                                        logger.info(f"❌ パターン '{pattern}' は新語根として不適切と判定")
                                else:
                                    logger.info(f"   ℹ️ 既知語根パターンのため、LLM提案はスキップ")
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
            
            logger.info("=== ✅ New Root Discovery Analysis Completed ===")
            
            # サンプルの場合は元データにマージ
            if llm_sample_size and len(names) > llm_sample_size:
                logger.info("🔄 LLM結果を元データセットにマージ中...")
                # LLM結果のカラムを元データに追加（該当する行のみ）
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
            
            logger.info("=== ✅ LLM Harmonization Completed ===")
    
    except Exception as e:
        logger.error(f"❌ LLMハーモナイゼーション中にエラーが発生: {e}")
        logger.warning("🔄 LLMハーモナイゼーションをスキップして処理を続行します")
    
    # 可視化
    if visualize:
        plt.figure(figsize=(10, 6))
        type_counts.plot(kind='bar')
        plt.title('水系タイプ別の件数')
        plt.xlabel('水系タイプ')
        plt.ylabel('件数')
        plt.tight_layout()
        plt.show()
    
    return names


def step4_calculate_distance(names, rivers_path, visualize=False):
    """S-4: 現河道との距離計算"""
    logger.info("S-4: 現河道との距離計算を実行中...")
    
    # HydroRIVERSファイルの存在確認
    if not os.path.exists(rivers_path):
        logger.error(f"エラー: {rivers_path} が見つかりません。")
        logger.error("このステップはスキップします。")
        return names
    
    # 距離計算
    logger.info("現河道との距離を計算しています...")
    names = attach_distance(names, rivers_path)
    logger.info(f"{len(names)}件のトポニムに距離情報を追加しました")
    
    # 結果の確認
    if logger.level <= logging.INFO:
        sample_data = names.sort_values('dist_km', ascending=False).head(5)[['name', 'type', 'dist_km']]
        logger.info(f"距離計算結果サンプル:\n{sample_data}")
    
    # 可視化
    if visualize and 'dist_km' in names.columns:
        # 距離の分布を可視化
        plt.figure(figsize=(10, 6))
        plt.hist(names['dist_km'], bins=20)
        plt.title('現河道からの距離の分布')
        plt.xlabel('距離 (km)')
        plt.ylabel('頻度')
        plt.axvline(x=3, color='red', linestyle='--', label='閾値 (3km)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return names


def step5_extract_candidates(names, gsw_path, visualize=False, skip_water_freq=False):
    """S-5: "川が無いのに川名が残る"ポイント抽出"""
    logger.info("S-5: 候補地点抽出を実行中...")
    
    if skip_water_freq:
        logger.info("水域頻度計算をスキップします（距離のみで候補抽出）")
        # 水域頻度を0に設定
        names = names.copy()
        names["occ_pct"] = 0
        logger.info(f"{len(names)}件のトポニムに仮の水域頻度情報（0%）を設定しました")
    else:
        # GSWファイルの存在確認
        if not os.path.exists(gsw_path):
            logger.error(f"エラー: {gsw_path} が見つかりません。")
            logger.error("このステップはスキップします。")
            return None
        
        # 水域頻度の計算
        logger.info("水域頻度を計算しています...")
        names = water_occurrence(names, gsw_path)
        logger.info(f"{len(names)}件のトポニムに水域頻度情報を追加しました")
    
    # 結果の確認
    if logger.level <= logging.INFO and 'occ_pct' in names.columns:
        sample_data = names.sort_values('occ_pct').head(5)[['name', 'type', 'dist_km', 'occ_pct']]
        logger.info(f"水域頻度計算結果サンプル:\n{sample_data}")
    
    # 候補地点の抽出
    if 'dist_km' in names.columns and 'occ_pct' in names.columns:
        # 閾値ベースでフィルタリング
        candidates = filter_candidates(names)
        logger.info(f"{len(candidates)}件の候補地点を抽出しました")
        
        # スコアリング
        candidates = score_candidates(candidates)
        logger.info(f"候補地点にスコアを付けました")
        
        # 結果の確認
        if logger.level <= logging.INFO:
            sample_data = candidates.sort_values('total_score', ascending=False).head(5)[
                ['name', 'type', 'dist_km', 'occ_pct', 'dist_score', 'occ_score', 'total_score']
            ]
            logger.info(f"候補地点スコアリング結果サンプル:\n{sample_data}")
        
        # 可視化
        if visualize:
            fig, ax = plt.subplots(figsize=(12, 10))
            names.plot(ax=ax, color='gray', alpha=0.5, markersize=10)
            candidates.plot(ax=ax, column='total_score', cmap='plasma', markersize=50, alpha=0.7, legend=True)
            ax.set_title('古河道候補地点')
            plt.tight_layout()
            plt.show()
        
        return candidates
    else:
        logger.warning("距離情報または水域頻度情報が不足しているため、候補地点を抽出できません。")
        return None


def save_results(candidates, output_path):
    """結果の保存"""
    if candidates is None or len(candidates) == 0:
        logger.warning("保存する候補地点がありません。")
        return
    
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # CSVとして保存（より安全）
    csv_path = output_path.replace('.parquet', '.csv')
    candidates.to_csv(csv_path, index=False)
    logger.info(f"{len(candidates)}件の候補地点を {csv_path} に保存しました")


def main():
    """メイン処理"""
    # コマンドライン引数のパース
    args = parse_args()
    
    # データディレクトリの確認
    ensure_data_dirs()
    
    # 入力データファイルの確認
    missing_files = check_data_files(args.rivers_path, args.gsw_path)
    if missing_files:
        logger.warning("一部のデータファイルが見つかりません。可能な処理のみ実行します。")
    
    # S-1: 対象地域のBBox定義
    bbox_gdf = step1_define_bbox(args.bbox, visualize=args.visualize)
    
    # S-2: 水場系トポニムの抽出
    names = step2_extract_toponyms(bbox_gdf, visualize=args.visualize, use_pyrosm=True, pbf_path=args.pbf_path)
    
    # S-3: クレンジング & タイプ付け
    names = step3_process_toponyms(names, visualize=args.visualize, llm_sample_size=args.llm_sample_size)
    
    # S-4: 現河道との距離計算
    names = step4_calculate_distance(names, args.rivers_path, visualize=args.visualize)
    
    # S-5: "川が無いのに川名が残る"ポイント抽出
    candidates = step5_extract_candidates(names, args.gsw_path, visualize=args.visualize, skip_water_freq=args.skip_water_freq)
    
    # 結果の保存
    save_results(candidates, args.output_path)
    
    logger.info("処理が完了しました。")


if __name__ == "__main__":
    main()
