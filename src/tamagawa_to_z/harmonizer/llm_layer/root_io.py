"""
Root Management for Water Toponyms

水語彙語根の管理とRegex生成機能
"""

import pandas as pd
import re
from pathlib import Path
from typing import List, Optional, Dict
import logging
import yaml

logger = logging.getLogger(__name__)

# プロジェクトルートからのパス
PROJECT_ROOT = Path(__file__).resolve().parents[4]
ROOT_PATH = PROJECT_ROOT / "data" / "dict" / "water_roots.csv"
ALL_ROOTS_PATH = PROJECT_ROOT / "data" / "dict" / "all_roots.csv"
CATEGORIES_PATH = PROJECT_ROOT / "data" / "dict" / "root_categories.yaml"


def load_roots(use_all_roots: bool = False) -> pd.DataFrame:
    """
    語根CSVを読み込む
    
    Args:
        use_all_roots: Trueの場合all_roots.csv、Falseの場合water_roots.csvを使用
    
    Returns:
        pd.DataFrame: 語根データ
        
    Raises:
        FileNotFoundError: CSVファイルが存在しない場合
    """
    if use_all_roots:
        csv_path = ALL_ROOTS_PATH
        csv_name = "all_roots.csv"
    else:
        csv_path = ROOT_PATH
        csv_name = "water_roots.csv"
    
    if not csv_path.exists():
        if use_all_roots:
            logger.warning(f"all_roots.csv not found at {csv_path}, falling back to water_roots.csv")
            return load_roots(use_all_roots=False)
        else:
            raise FileNotFoundError(f"{csv_name} not found at {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, dtype=str)
        logger.debug(f"Loaded {len(df)} root entries from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load {csv_name}: {e}")
        if use_all_roots:
            logger.warning(f"Falling back to water_roots.csv due to error loading all_roots.csv")
            return load_roots(use_all_roots=False)
        raise


def append_roots(df_new: pd.DataFrame, target_file: str = "water") -> None:
    """
    新しい語根をCSVに追記（重複排除）
    
    Args:
        df_new: 追加する語根データ
        target_file: 追加先ファイル ("water"|"all")
    """
    if df_new.empty:
        logger.debug("No new roots to append")
        return
    
    try:
        # 追加先ファイルを決定
        if target_file == "all":
            csv_path = ALL_ROOTS_PATH
            use_all_roots = True
            csv_name = "all_roots.csv"
        else:
            csv_path = ROOT_PATH
            use_all_roots = False
            csv_name = "water_roots.csv"
        
        # 既存データを読み込み
        df_old = load_roots(use_all_roots=use_all_roots)
        
        # 結合して重複排除
        if target_file == "all" and "category" in df_new.columns and "category" in df_old.columns:
            # all_roots.csvの場合はroot+categoryで重複排除
            df_all = pd.concat([df_old, df_new], ignore_index=True)
            df_all = df_all.drop_duplicates(subset=["root", "category"], keep="last")
        else:
            # water_roots.csvの場合はrootのみで重複排除
            df_all = pd.concat([df_old, df_new], ignore_index=True)
            df_all = df_all.drop_duplicates(subset=["root"], keep="last")
        
        # 保存
        df_all.to_csv(csv_path, index=False)
        
        new_count = len(df_all) - len(df_old)
        logger.info(f"Appended {new_count} new root entries to {csv_name}")
        
    except Exception as e:
        logger.error(f"Failed to append roots to {target_file}: {e}")
        raise


def build_root_regex(categories: List[str]) -> Dict[str, re.Pattern]:
    """
    カテゴリごとに語根CSVからRegexパターンを構築
    
    Args:
        categories: 抽出対象のカテゴリリスト (e.g., ["water", "terrain", ...])
        
    Returns:
        Dict[str, re.Pattern]: カテゴリ名をキーとするコンパイル済み正規表現の辞書
        
    Raises:
        FileNotFoundError: CSVファイルが存在しない場合
        ValueError: 有効な語根が見つからない場合
    """
    patterns = {}
    
    # カテゴリ設定ファイルの読み込み
    try:
        if CATEGORIES_PATH.exists():
            with open(CATEGORIES_PATH, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            categories_mapping = config.get('categories', {})
        else:
            # フォールバック: 水系のみ
            categories_mapping = {'water': 'water_roots.csv'}
            logger.warning(f"Categories config not found at {CATEGORIES_PATH}, using water-only fallback")
    except Exception as e:
        logger.warning(f"Failed to load categories config: {e}, using water-only fallback")
        categories_mapping = {'water': 'water_roots.csv'}
    
    for cat in categories:
        try:
            # CSVファイルパスを決定
            if cat in categories_mapping:
                csv_filename = categories_mapping[cat]
            else:
                csv_filename = f"{cat}_roots.csv"
                logger.warning(f"Category '{cat}' not in config, using default filename: {csv_filename}")
            
            csv_path = PROJECT_ROOT / "data" / "dict" / csv_filename
            
            if not csv_path.exists():
                logger.warning(f"CSV file not found for category '{cat}': {csv_path}")
                continue
            
            # CSVファイルを読み込み
            df_roots = pd.read_csv(csv_path, dtype=str)
            
            if df_roots.empty:
                logger.warning(f"No roots found in {csv_filename}")
                continue
            
            # 詳細ログ出力: CSVの内容
            logger.info(f"📁 Reading {cat} roots from: {csv_path}")
            logger.info(f"📊 Total entries in CSV: {len(df_roots)}")
            
            # 言語別統計
            if 'lang' in df_roots.columns:
                lang_stats = df_roots["lang"].value_counts().to_dict()
                lang_summary = ", ".join([f"{lang}:{count}" for lang, count in lang_stats.items()])
                logger.info(f"🌍 Language distribution for {cat}: {lang_summary}")
            
            # regex_tokenを収集（NaN値を除外）
            if 'regex_token' not in df_roots.columns:
                logger.warning(f"No 'regex_token' column in {csv_filename}")
                continue
            
            tokens = df_roots["regex_token"].dropna().unique()
            
            if len(tokens) == 0:
                logger.warning(f"No valid regex tokens found in {csv_filename}")
                continue
            
            # 詳細ログ出力: 使用される語根
            logger.info(f"🔧 Extracted {len(tokens)} regex tokens for {cat}:")
            for i, (_, row) in enumerate(df_roots.iterrows()):
                if pd.notna(row["regex_token"]):
                    meaning = row.get('meaning_en', row.get('meaning_ja', 'unknown'))
                    logger.info(f"   {i+1:2d}. {row['root']:10} ({row.get('lang', 'unknown'):3}) -> {row['regex_token']:15} | {meaning}")
            
            # Regexパターンを構築（大文字小文字無視、単語境界付き）
            joined = r'(?i)\b(' + '|'.join(tokens) + r')\b'
            pattern = re.compile(joined)
            
            patterns[cat] = pattern
            
            logger.info(f"✅ Successfully built {cat} regex pattern:")
            logger.info(f"   Pattern: {joined}")
            
        except Exception as e:
            logger.error(f"❌ Failed to build {cat} regex: {e}")
            continue
    
    if not patterns:
        raise ValueError(f"No valid patterns could be built for categories: {categories}")
    
    logger.info(f"✅ Built regex patterns for {len(patterns)} categories: {list(patterns.keys())}")
    return patterns


def build_all_roots_regex(categories: List[str] = None) -> re.Pattern:
    """
    all_roots.csvから指定カテゴリの語根Regexパターンを構築
    
    Args:
        categories: 対象カテゴリのリスト。Noneの場合は全カテゴリ
        
    Returns:
        re.Pattern: 統合されたコンパイル済み正規表現
        
    Raises:
        FileNotFoundError: CSVファイルが存在しない場合
        ValueError: 有効な語根が見つからない場合
    """
    try:
        df_roots = load_roots(use_all_roots=True)
        
        if df_roots.empty:
            raise ValueError("No roots found in all_roots.csv")
        
        # カテゴリフィルタリング
        if categories is not None:
            if 'category' in df_roots.columns:
                df_roots = df_roots[df_roots['category'].isin(categories)]
            else:
                logger.warning("Category column not found in all_roots.csv, using all roots")
        
        if df_roots.empty:
            raise ValueError(f"No roots found for categories: {categories}")
        
        logger.info(f"📁 Reading roots from all_roots.csv")
        logger.info(f"📊 Total entries: {len(df_roots)}")
        
        # カテゴリ別統計
        if 'category' in df_roots.columns:
            cat_stats = df_roots["category"].value_counts().to_dict()
            cat_summary = ", ".join([f"{cat}:{count}" for cat, count in cat_stats.items()])
            logger.info(f"🏷️ Category distribution: {cat_summary}")
        
        # regex_tokenを収集（NaN値を除外）
        tokens = df_roots["regex_token"].dropna().unique()
        
        if len(tokens) == 0:
            raise ValueError("No valid regex tokens found in all_roots.csv")
        
        # 詳細ログ出力: 使用される語根
        logger.info(f"🔧 Extracted {len(tokens)} regex tokens:")
        for i, (_, row) in enumerate(df_roots.iterrows()):
            if pd.notna(row["regex_token"]):
                category = row.get('category', 'unknown')
                meaning = row.get('meaning_en', row.get('meaning_ja', 'unknown'))
                logger.info(f"   {i+1:2d}. {row['root']:12} ({row.get('lang', 'unknown'):3}, {category:8}) -> {row['regex_token']:15} | {meaning}")
        
        # Regexパターンを構築（大文字小文字無視、単語境界付き）
        joined = r'(?i)\b(' + '|'.join(tokens) + r')\b'
        pattern = re.compile(joined)
        
        logger.info(f"✅ Successfully built all-roots regex pattern:")
        logger.info(f"   Categories: {categories if categories else 'all'}")
        logger.info(f"   Pattern length: {len(joined)} chars")
        logger.info(f"   Compiled regex ready for toponyms filtering")
        
        return pattern
        
    except Exception as e:
        logger.error(f"❌ Failed to build all-roots regex: {e}")
        raise


def build_water_regex() -> re.Pattern:
    """
    水語彙Regexパターンを構築（後方互換性のため）
    
    Returns:
        re.Pattern: コンパイル済み正規表現
        
    Raises:
        FileNotFoundError: CSVファイルが存在しない場合
        ValueError: 有効な語根が見つからない場合
    """
    # まずall_roots.csvからwaterカテゴリを試行
    try:
        return build_all_roots_regex(categories=["water"])
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Failed to load from all_roots.csv: {e}")
        logger.info("Falling back to water_roots.csv")
        
        # フォールバック: 従来のwater_roots.csvから構築
        patterns = build_root_regex(["water"])
        return patterns["water"]


def get_root_stats() -> dict:
    """
    語根辞書の統計情報を取得
    
    Returns:
        dict: 統計情報
    """
    try:
        df = load_roots()
        
        stats = {
            "total_roots": len(df),
            "languages": df["lang"].value_counts().to_dict(),
            "tokens_with_regex": df["regex_token"].notna().sum(),
            "roots_by_language": df.groupby("lang").size().to_dict()
        }
        
        return stats
        
    except Exception as e:
        logger.warning(f"Failed to get root stats: {e}")
        return {"error": str(e)}


def validate_root_entry(root_dict: dict) -> List[str]:
    """
    語根エントリーの妥当性を検証
    
    Args:
        root_dict: 検証する語根辞書
        
    Returns:
        List[str]: エラーメッセージのリスト（空の場合は妥当）
    """
    errors = []
    
    # 必須フィールドの確認
    required_fields = ["root", "regex_token"]
    for field in required_fields:
        if field not in root_dict or not root_dict[field]:
            errors.append(f"Required field '{field}' is missing or empty")
    
    # regex_tokenの妥当性確認
    if "regex_token" in root_dict and root_dict["regex_token"]:
        try:
            re.compile(root_dict["regex_token"])
        except re.error as e:
            errors.append(f"Invalid regex pattern in regex_token: {e}")
    
    # 言語コードの確認（存在する場合）
    if "lang" in root_dict and root_dict["lang"]:
        valid_langs = ["por", "tup", "araw", "mixed", "unknown"]
        if root_dict["lang"] not in valid_langs:
            errors.append(f"Invalid language code: {root_dict['lang']}. Must be one of {valid_langs}")
    
    return errors


def format_root_for_csv(root_dict: dict) -> dict:
    """
    語根辞書をCSV保存用に整形
    
    Args:
        root_dict: 整形する語根辞書
        
    Returns:
        dict: CSV保存用に整形された辞書
    """
    # 必要なフィールドのみ抽出し、デフォルト値を設定
    formatted = {
        "root": root_dict.get("root", "").strip(),
        "lang": root_dict.get("lang", "unknown").strip(),
        "regex_token": root_dict.get("regex_token", "").strip(),
        "meaning_en": root_dict.get("meaning_en", "").strip(),
        "meaning_ja": root_dict.get("meaning_ja", "").strip()
    }
    
    # 空文字列をNoneに変換（pandas処理用）
    for key, value in formatted.items():
        if value == "":
            formatted[key] = None
    
    return formatted