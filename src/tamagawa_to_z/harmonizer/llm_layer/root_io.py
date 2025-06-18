"""
Root Management for Water Toponyms

水語彙語根の管理とRegex生成機能
"""

import pandas as pd
import re
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# プロジェクトルートからのパス
ROOT_PATH = Path(__file__).resolve().parents[4] / "data" / "dict" / "water_roots.csv"


def load_roots() -> pd.DataFrame:
    """
    水語彙語根CSVを読み込む
    
    Returns:
        pd.DataFrame: 語根データ
        
    Raises:
        FileNotFoundError: CSVファイルが存在しない場合
    """
    if not ROOT_PATH.exists():
        raise FileNotFoundError(f"water_roots.csv not found at {ROOT_PATH}")
    
    try:
        df = pd.read_csv(ROOT_PATH, dtype=str)
        logger.debug(f"Loaded {len(df)} root entries from {ROOT_PATH}")
        return df
    except Exception as e:
        logger.error(f"Failed to load water_roots.csv: {e}")
        raise


def append_roots(df_new: pd.DataFrame) -> None:
    """
    新しい語根をCSVに追記（重複排除）
    
    Args:
        df_new: 追加する語根データ
    """
    if df_new.empty:
        logger.debug("No new roots to append")
        return
    
    try:
        # 既存データを読み込み
        df_old = load_roots()
        
        # 結合して重複排除（rootカラムベース）
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["root"], keep="last")
        
        # 保存
        df_all.to_csv(ROOT_PATH, index=False)
        
        new_count = len(df_all) - len(df_old)
        logger.info(f"Appended {new_count} new root entries to {ROOT_PATH}")
        
    except Exception as e:
        logger.error(f"Failed to append roots: {e}")
        raise


def build_water_regex() -> re.Pattern:
    """
    語根CSVから水語彙Regexパターンを構築
    
    Returns:
        re.Pattern: コンパイル済み正規表現
        
    Raises:
        FileNotFoundError: CSVファイルが存在しない場合
        ValueError: 有効な語根が見つからない場合
    """
    try:
        df_roots = load_roots()
        
        if df_roots.empty:
            raise ValueError("No roots found in water_roots.csv")
        
        # 詳細ログ出力: CSVの内容
        logger.info(f"📁 Reading water roots from: {ROOT_PATH}")
        logger.info(f"📊 Total entries in CSV: {len(df_roots)}")
        
        # 言語別統計
        lang_stats = df_roots["lang"].value_counts().to_dict()
        lang_summary = ", ".join([f"{lang}:{count}" for lang, count in lang_stats.items()])
        logger.info(f"🌍 Language distribution: {lang_summary}")
        
        # regex_tokenを収集（NaN値を除外）
        tokens = df_roots["regex_token"].dropna().unique()
        
        if len(tokens) == 0:
            raise ValueError("No valid regex tokens found in water_roots.csv")
        
        # 詳細ログ出力: 使用される語根
        logger.info(f"🔧 Extracted {len(tokens)} regex tokens:")
        for i, (_, row) in enumerate(df_roots.iterrows()):
            if pd.notna(row["regex_token"]):
                logger.info(f"   {i+1:2d}. {row['root']:10} ({row['lang']:3}) -> {row['regex_token']:15} | {row['meaning_en']}")
        
        # Regexパターンを構築（大文字小文字無視、単語境界付き）
        joined = r'(?i)\b(' + '|'.join(tokens) + r')\b'
        pattern = re.compile(joined)
        
        logger.info(f"✅ Successfully built water regex pattern:")
        logger.info(f"   Pattern: {joined}")
        logger.info(f"   Compiled regex ready for toponyms filtering")
        
        return pattern
        
    except Exception as e:
        logger.error(f"❌ Failed to build water regex: {e}")
        raise


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