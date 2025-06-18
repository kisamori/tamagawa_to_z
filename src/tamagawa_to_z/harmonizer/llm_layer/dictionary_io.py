"""
Toponym Dictionary CSV I/O Operations

多言語トポニム辞書のCSV読み書き機能を提供します。
重複除去、バックアップ、エラーハンドリングを含みます。
"""

import pandas as pd
import pathlib
import logging
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# プロジェクトルートからの相対パス
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.parent.parent
DICT_PATH = PROJECT_ROOT / "data" / "dict" / "toponym_dict.csv"

# 辞書CSVの列定義
COLUMNS = [
    "canonical_id",
    "canonical_name", 
    "variant_name",
    "root",
    "lang",
    "meaning_en",
    "meaning_ja",
    "confidence"
]

# 必須列
REQUIRED_COLUMNS = ["canonical_id", "canonical_name", "variant_name", "confidence"]


def ensure_dict_directory() -> None:
    """辞書ディレクトリが存在することを確認"""
    DICT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_dict(dict_path: Optional[pathlib.Path] = None) -> pd.DataFrame:
    """
    トポニム辞書CSVを読み込む
    
    Args:
        dict_path: 辞書ファイルパス（Noneの場合はデフォルトパス）
        
    Returns:
        辞書DataFrame
    """
    path = dict_path or DICT_PATH
    
    if path.exists():
        try:
            df = pd.read_csv(path, dtype=str)
            # 列の検証
            missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}")
            return df
        except Exception as e:
            logger.error(f"Failed to load dictionary from {path}: {e}")
            return pd.DataFrame(columns=COLUMNS)
    else:
        logger.info(f"Dictionary file not found at {path}. Creating empty DataFrame.")
        return pd.DataFrame(columns=COLUMNS)


def backup_dict(dict_path: Optional[pathlib.Path] = None) -> pathlib.Path:
    """
    現在の辞書ファイルをバックアップする
    
    Args:
        dict_path: 辞書ファイルパス
        
    Returns:
        バックアップファイルパス
    """
    path = dict_path or DICT_PATH
    
    if not path.exists():
        raise FileNotFoundError(f"Dictionary file not found: {path}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.parent / f"{path.stem}_backup_{timestamp}.csv"
    
    import shutil
    shutil.copy2(path, backup_path)
    logger.info(f"Dictionary backed up to: {backup_path}")
    
    return backup_path


def validate_entries(df_new: pd.DataFrame) -> pd.DataFrame:
    """
    新しいエントリーを検証する
    
    Args:
        df_new: 新しいエントリーのDataFrame
        
    Returns:
        検証済みDataFrame
    """
    # 必須列の確認
    missing_cols = set(REQUIRED_COLUMNS) - set(df_new.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # 空値の確認
    for col in REQUIRED_COLUMNS:
        if df_new[col].isna().any():
            logger.warning(f"Found NaN values in required column: {col}")
            df_new = df_new.dropna(subset=[col])
    
    # confidence値の範囲確認
    if "confidence" in df_new.columns:
        df_new["confidence"] = pd.to_numeric(df_new["confidence"], errors="coerce")
        invalid_conf = (df_new["confidence"] < 0) | (df_new["confidence"] > 1)
        if invalid_conf.any():
            logger.warning(f"Found {invalid_conf.sum()} entries with invalid confidence values")
            df_new = df_new[~invalid_conf]
    
    return df_new


def append_entries(
    df_new: pd.DataFrame, 
    dict_path: Optional[pathlib.Path] = None,
    create_backup: bool = True
) -> None:
    """
    新しいエントリーを辞書に追記する（重複除去付き）
    
    Args:
        df_new: 追加するエントリーのDataFrame
        dict_path: 辞書ファイルパス（Noneの場合はデフォルトパス）
        create_backup: バックアップを作成するかどうか
    """
    path = dict_path or DICT_PATH
    ensure_dict_directory()
    
    # 新しいエントリーの検証
    df_new = validate_entries(df_new.copy())
    if df_new.empty:
        logger.warning("No valid entries to append")
        return
    
    # 既存データの読み込み
    df_old = load_dict(path)
    
    # バックアップ作成
    if create_backup and path.exists() and not df_old.empty:
        backup_dict(path)
    
    # データの結合と重複除去
    df_all = pd.concat([df_old, df_new], ignore_index=True)
    
    # 重複除去（canonical_id + variant_name の組み合わせで）
    before_count = len(df_all)
    df_all = df_all.drop_duplicates(
        subset=["canonical_id", "variant_name"],
        keep="last"  # 最新のエントリーを保持
    )
    after_count = len(df_all)
    
    if before_count > after_count:
        logger.info(f"Removed {before_count - after_count} duplicate entries")
    
    # 保存
    try:
        df_all.to_csv(path, index=False)
        logger.info(f"Successfully appended {len(df_new)} entries to dictionary: {path}")
        logger.info(f"Total dictionary size: {len(df_all)} entries")
    except Exception as e:
        logger.error(f"Failed to save dictionary: {e}")
        raise


def get_dict_stats(dict_path: Optional[pathlib.Path] = None) -> dict:
    """
    辞書の統計情報を取得する
    
    Args:
        dict_path: 辞書ファイルパス
        
    Returns:
        統計情報辞書
    """
    df = load_dict(dict_path)
    
    if df.empty:
        return {"total_entries": 0}
    
    stats = {
        "total_entries": len(df),
        "unique_canonical_names": df["canonical_name"].nunique() if "canonical_name" in df.columns else 0,
        "unique_roots": df["root"].nunique() if "root" in df.columns else 0,
        "languages": df["lang"].value_counts().to_dict() if "lang" in df.columns else {},
        "avg_confidence": df["confidence"].astype(float).mean() if "confidence" in df.columns else 0
    }
    
    return stats


def search_variants(
    search_term: str, 
    dict_path: Optional[pathlib.Path] = None,
    fuzzy: bool = False
) -> pd.DataFrame:
    """
    辞書内でバリアントを検索する
    
    Args:
        search_term: 検索語
        dict_path: 辞書ファイルパス
        fuzzy: ファジー検索を行うかどうか
        
    Returns:
        マッチしたエントリーのDataFrame
    """
    df = load_dict(dict_path)
    
    if df.empty:
        return df
    
    search_term = search_term.lower()
    
    if fuzzy:
        # 部分一致検索
        mask = (
            df["variant_name"].str.lower().str.contains(search_term, na=False) |
            df["canonical_name"].str.lower().str.contains(search_term, na=False)
        )
    else:
        # 完全一致検索
        mask = (
            df["variant_name"].str.lower() == search_term
        )
    
    return df[mask]