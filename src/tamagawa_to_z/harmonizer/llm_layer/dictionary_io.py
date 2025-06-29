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
    "variant_id",
    "root",
    "lang",
    "meaning_en",
    "meaning_ja",
    "confidence"
]

# 必須列（variant_idは自動生成されるため除外）
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
    新しいエントリーを辞書に追記する（canonical_name統合とcanonical_id正規化付き）
    
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
    
    # データの結合
    df_all = pd.concat([df_old, df_new], ignore_index=True)
    
    # canonical_nameとvariant_nameの統合処理
    df_normalized = _normalize_canonical_structure(df_all)
    
    # 重複除去（variant_name単独で、variant_idは固有なので重複はないはず）
    before_count = len(df_normalized)
    df_final = df_normalized.drop_duplicates(
        subset=["variant_name"],
        keep="last"  # 最新のエントリーを保持
    )
    after_count = len(df_final)
    
    # 重複除去の詳細ログ
    removed_count = before_count - after_count
    if removed_count > 0:
        logger.info(f"Removed {removed_count} duplicate variant entries")
    
    # 保存
    try:
        df_final.to_csv(path, index=False)
        logger.info(f"Successfully appended {len(df_new)} entries to dictionary: {path}")
        logger.info(f"Total dictionary size: {len(df_final)} entries")
        
        # canonical統合の統計
        canonical_count = len(df_final['canonical_name'].unique())
        variant_count = len(df_final['variant_name'].unique())
        logger.info(f"Canonical names: {canonical_count} unique entries")
        logger.info(f"Variant names: {variant_count} unique entries")
        
    except Exception as e:
        logger.error(f"Failed to save dictionary: {e}")
        raise


def _normalize_canonical_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    canonical_nameに基づいてcanonical_idを正規化し、variant_nameに基づいてvariant_idを割り当てる
    
    Args:
        df: 統合前のDataFrame
        
    Returns:
        正規化されたDataFrame
    """
    if df.empty:
        return df
    
    logger.debug("Normalizing canonical structure...")
    
    # canonical_nameでグループ化
    canonical_groups = {}
    canonical_id_counter = 1
    
    # 既存のcanonical_nameとIDのマッピングを構築
    for _, row in df.iterrows():
        canonical_name = row['canonical_name']
        
        if canonical_name not in canonical_groups:
            # 新しいcanonical_nameに数値IDを割り当て
            canonical_groups[canonical_name] = str(canonical_id_counter)
            canonical_id_counter += 1
    
    # variant_id列が存在しない場合は追加
    if 'variant_id' not in df.columns:
        df['variant_id'] = ''
    
    # variant_nameでユニークなvariant_idを割り当て
    variant_groups = {}
    variant_id_counter = 1
    
    # 既存のvariant_nameとIDのマッピングを構築
    for _, row in df.iterrows():
        variant_name = row['variant_name']
        
        if variant_name not in variant_groups:
            # 新しいvariant_nameに数値IDを割り当て
            variant_groups[variant_name] = str(variant_id_counter)
            variant_id_counter += 1
    
    # 各行のcanonical_idとvariant_idを正規化
    df_normalized = df.copy()
    for idx, row in df_normalized.iterrows():
        canonical_name = row['canonical_name']
        variant_name = row['variant_name']
        
        correct_canonical_id = canonical_groups[canonical_name]
        correct_variant_id = variant_groups[variant_name]
        
        df_normalized.at[idx, 'canonical_id'] = correct_canonical_id
        df_normalized.at[idx, 'variant_id'] = correct_variant_id
    
    # グループ統計をログ出力
    for canonical_name, canonical_id in canonical_groups.items():
        variant_count = len(df_normalized[df_normalized['canonical_name'] == canonical_name])
        logger.debug(f"Canonical ID {canonical_id}: '{canonical_name}' ({variant_count} variants)")
    
    logger.debug(f"Assigned {len(variant_groups)} unique variant IDs")
    
    return df_normalized


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


def migrate_add_variant_id(dict_path: Optional[pathlib.Path] = None, create_backup: bool = True) -> bool:
    """
    既存の辞書にvariant_id列を追加するマイグレーション
    
    Args:
        dict_path: 辞書ファイルパス
        create_backup: バックアップを作成するかどうか
        
    Returns:
        bool: マイグレーション成功の可否
    """
    path = dict_path or DICT_PATH
    
    if not path.exists():
        logger.info(f"Dictionary file not found: {path}")
        return False
    
    # 現在の辞書を読み込み
    df = pd.read_csv(path, dtype=str)
    
    # variant_id列が既に存在する場合はスキップ
    if 'variant_id' in df.columns:
        logger.info("variant_id column already exists. Migration skipped.")
        return True
    
    logger.info(f"Starting migration: adding variant_id column to {path}")
    
    # バックアップ作成
    if create_backup:
        backup_dict(path)
    
    # variant_id列を追加
    df['variant_id'] = ''
    
    # variant_nameに基づいて固有IDを割り当て
    variant_groups = {}
    variant_id_counter = 1
    
    for idx, row in df.iterrows():
        variant_name = row['variant_name']
        
        if variant_name not in variant_groups:
            variant_groups[variant_name] = str(variant_id_counter)
            variant_id_counter += 1
        
        df.at[idx, 'variant_id'] = variant_groups[variant_name]
    
    # 列の順序を調整（variant_idをvariant_nameの後に配置）
    columns_order = []
    for col in COLUMNS:
        if col in df.columns:
            columns_order.append(col)
    
    # 残りの列を追加（relation, reasoning等）
    for col in df.columns:
        if col not in columns_order:
            columns_order.append(col)
    
    df = df[columns_order]
    
    # 保存
    try:
        df.to_csv(path, index=False)
        logger.info(f"Migration completed successfully. Added {len(variant_groups)} unique variant IDs.")
        return True
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False