"""
CLI: コマンドラインインターフェース

このモジュールは、tamagawa_to_z パッケージのコマンドラインインターフェースを提供します。
"""

import os
import sys
import click
import logging
import geopandas as gpd
from pathlib import Path

# パッケージのインポート
from tamagawa_to_z.harmonizer.preprocess import make_bbox_gdf, process_toponyms, extract_toponyms_pyrosm
from tamagawa_to_z.harmonizer.distance import attach_distance
from tamagawa_to_z.harmonizer.watermask import water_occurrence
from tamagawa_to_z.harmonizer.agent import filter_with_agent


def setup_logging(log_level: str = 'INFO') -> None:
    """ロギングの設定を行う
    
    Parameters
    ----------
    log_level : str, optional
        ログレベル
    """
    # ログレベルの設定
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = log_levels.get(log_level.upper(), logging.INFO)
    
    # ロガーの設定
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


@click.group()
def cli():
    """tamagawa_to_z: アマゾン古河道・集落探索フレームワーク"""
    pass


@cli.command()
@click.option("--out", default="data/interim/acre_candidates.parquet", help="出力ファイルのパス")
@click.option("--rivers", default="data/raw/HydroRIVERS_SA.shp", help="HydroRIVERSファイルのパス")
@click.option("--gsw", default="data/raw/GSW_occurrence.tif", help="GSW occurrenceファイルのパス")
@click.option("--pbf", default="data/raw/osm/norte-latest.osm.pbf", help="PBFファイルのパス")
@click.option("--log-level", default="INFO", help="ログレベル")
def acre_pipeline(out, rivers, gsw, pbf, log_level):
    """アクレ州マデイラ川上流西部のS-1〜S-5パイプラインを実行する"""
    # ロギングの設定
    setup_logging(log_level)
    logger = logging.getLogger("acre_pipeline")
    
    logger.info("アクレ州マデイラ川上流西部のS-1〜S-5パイプラインを開始します")
    
    # S-1: 対象地域のBBox定義
    logger.info("S-1: 対象地域のBBox定義")
    bbox = make_bbox_gdf().geometry.iloc[0]
    
    # S-2: 水場系トポニムの抽出
    logger.info("S-2: 水場系トポニムの抽出")
    logger.info("PyrosmでローカルPBFから水語彙地名を抽出しています...")
    names = extract_toponyms_pyrosm(bbox, pbf)
    logger.info(f"ローカルPBFから{len(names)}件のトポニムを収集しました")
    
    # S-3: クレンジング & タイプ付け
    logger.info("S-3: クレンジング & タイプ付け")
    names = process_toponyms(names)
    logger.info(f"{len(names)}件のトポニムを処理しました")
    
    # S-4: 現河道との距離計算
    logger.info("S-4: 現河道との距離計算")
    names = attach_distance(names, rivers)
    logger.info(f"{len(names)}件のトポニムに距離情報を追加しました")
    
    # S-5: "川が無いのに川名が残る"ポイント抽出
    logger.info("S-5: \"川が無いのに川名が残る\"ポイント抽出")
    
    # S-5前半: 水域頻度判定
    logger.info("S-5前半: 水域頻度判定")
    names = water_occurrence(names, gsw)
    logger.info(f"{len(names)}件のトポニムに水域頻度情報を追加しました")
    
    # S-5後半: LLM Agent判定
    logger.info("S-5後半: LLM Agent判定")
    candidates = filter_with_agent(names)
    logger.info(f"{len(candidates)}件の候補地点を抽出しました")
    
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(out), exist_ok=True)
    
    # 結果の保存
    logger.info(f"結果を保存しています: {out}")
    candidates.to_parquet(out, index=False)
    
    logger.info(f"アクレ州マデイラ川上流西部のS-1〜S-5パイプラインが完了しました")
    logger.info(f"保存された候補地点: {len(candidates)}件 → {out}")


@cli.command()
@click.option("--out", default="requirements.txt", help="出力ファイルのパス")
def export_requirements(out):
    """Poetry依存関係をrequirements.txtにエクスポートする"""
    os.system(f"poetry export --without-hashes --format=requirements.txt > {out}")
    print(f"依存関係をエクスポートしました: {out}")


def main():
    """メイン関数"""
    cli()


if __name__ == "__main__":
    main()
