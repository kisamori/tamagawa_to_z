#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_analyze_site.py: 既知遺跡周辺地名分析スクリプト

既知遺跡周辺の地名を極座標形式で分析し、川からの距離情報を含むCSVファイルを出力します。

使用方法:
    python scripts/run_analyze_site.py --region acre [OPTIONS]

主要オプション:
    --region REGION       地域指定（acre, marajo）
    --radius RADIUS       遺跡周辺検索半径（km、デフォルト5.0）
    --osm-keys-mode MODE  OSMキー抽出モード（デフォルト: water_focused）
    --visualize           結果可視化
    --output-dir PATH     出力ベースディレクトリ

出力:
    data/output/site_analysis/site_analysis_YYYYMMDD_HHMMSS/
    ├── site_toponym_analysis_{region}.csv
    ├── analysis_config.yaml
    ├── analysis_log.txt
    └── visualizations/ (--visualize使用時)
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime
import traceback

# PyYAMLのインポート
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Warning: PyYAMLがインストールされていません。設定ファイルの読み込みができません。")

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# 地図背景用
try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

# プロジェクトのルートディレクトリをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# 自作パッケージのインポート
try:
    from tamagawa_to_z.site_analysis import (
        ToponymExtractor, PolarConverter, RiverDistanceCalculator, CSVExporter,
        ArchaeologicalSimilarityAnalyzer
    )
except ImportError as e:
    print(f"Error: 必要なモジュールのインポートに失敗しました: {e}")
    print("プロジェクトが正しくインストールされているか確認してください。")
    sys.exit(1)


def setup_logging(log_file: Path, level: str = "INFO") -> logging.Logger:
    """ログ設定"""
    # ログレベルの設定
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # ログフォーマット
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ファイルハンドラー
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # ルートロガーの設定
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger


def load_config(config_path: Path) -> dict:
    """設定ファイルの読み込み"""
    if not HAS_YAML:
        raise ImportError("PyYAMLが必要です。pip install pyyamlでインストールしてください。")
    
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def load_region_config(region: str) -> dict:
    """地域設定の読み込み"""
    region_config_path = PROJECT_ROOT / "config" / "regions.yaml"
    
    if not region_config_path.exists():
        raise FileNotFoundError(f"地域設定ファイルが見つかりません: {region_config_path}")
    
    with open(region_config_path, 'r', encoding='utf-8') as f:
        regions_config = yaml.safe_load(f)
    
    if region not in regions_config.get('regions', {}):
        available_regions = list(regions_config.get('regions', {}).keys())
        raise ValueError(f"未定義の地域です: {region}. 利用可能: {available_regions}")
    
    return regions_config['regions'][region]


def load_osm_keys_config(osm_keys_mode: str) -> dict:
    """OSMキー設定の読み込み"""
    osm_keys_path = PROJECT_ROOT / "data" / "config" / "osm_keys.yaml"
    
    if not osm_keys_path.exists():
        raise FileNotFoundError(f"OSMキー設定ファイルが見つかりません: {osm_keys_path}")
    
    with open(osm_keys_path, 'r', encoding='utf-8') as f:
        osm_config = yaml.safe_load(f)
    
    if osm_keys_mode not in osm_config.get('extraction_modes', {}):
        available_modes = list(osm_config.get('extraction_modes', {}).keys())
        raise ValueError(f"未定義のOSMキーモード: {osm_keys_mode}. 利用可能: {available_modes}")
    
    return osm_config


def create_output_directory(base_dir: Path, timestamp_format: str) -> Path:
    """出力ディレクトリの作成"""
    timestamp = datetime.now().strftime(timestamp_format)
    output_dir = base_dir / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 可視化用ディレクトリも作成
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    return output_dir


def save_config_backup(config: dict, output_dir: Path, filename: str):
    """設定ファイルのバックアップ"""
    backup_path = output_dir / filename
    
    with open(backup_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def create_visualization(
    analysis_gdf: gpd.GeoDataFrame,
    sites_gdf: gpd.GeoDataFrame,
    output_dir: Path,
    region: str,
    config: dict
) -> None:
    """分析結果の可視化"""
    if not HAS_CONTEXTILY:
        logging.warning("contextilyがインストールされていないため、可視化をスキップします")
        return
    
    if analysis_gdf.empty:
        logging.warning("可視化対象のデータが空です")
        return
    
    logger = logging.getLogger(__name__)
    vis_dir = output_dir / "visualizations"
    
    try:
        # 図のサイズとDPI設定
        fig_config = config.get('visualization', {})
        figsize = fig_config.get('figure_size', [12, 8])
        dpi = fig_config.get('dpi', 300)
        
        # 1. 遺跡と地名の分布図
        logger.info("遺跡・地名分布図を作成中...")
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Web Mercatorに変換
        analysis_web = analysis_gdf.to_crs(epsg=3857)
        sites_web = sites_gdf.to_crs(epsg=3857)
        
        # 地名をプロット
        analysis_web.plot(ax=ax, color='blue', alpha=0.6, markersize=20, label='Toponyms')
        
        # 遺跡をプロット
        sites_web.plot(ax=ax, color='red', alpha=0.8, markersize=100, marker='^', label='Archaeological Sites')
        
        # 背景地図を追加
        try:
            ctx.add_basemap(ax, crs=analysis_web.crs, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            logger.warning(f"背景地図の追加に失敗: {e}")
        
        # 図の設定
        ax.set_title(f'Archaeological Sites and Toponyms - {region.title()}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # 軸ラベルを緯度経度に変換（概算）
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        
        plt.tight_layout()
        plt.savefig(vis_dir / f'site_toponym_distribution_{region}.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        # 2. 極座標分布図（ポーラープロット）
        if 'angle' in analysis_gdf.columns and 'radius' in analysis_gdf.columns:
            logger.info("極座標分布図を作成中...")
            fig, ax = plt.subplots(figsize=[8, 8], dpi=dpi, subplot_kw=dict(projection='polar'))
            
            # 角度をラジアンに変換（北を0、時計回り）
            angles_rad = np.radians(analysis_gdf['angle'])
            radii = analysis_gdf['radius']
            
            # ポーラープロットのスキャッター
            scatter = ax.scatter(angles_rad, radii, alpha=0.6, s=30, c=radii, cmap='viridis')
            
            # 設定
            ax.set_theta_zero_location('N')  # 北を0度に
            ax.set_theta_direction(-1)       # 時計回り
            ax.set_title(f'Polar Distribution of Toponyms - {region.title()}', fontsize=14, fontweight='bold', pad=20)
            ax.set_ylim(0, radii.max() * 1.1)
            
            # カラーバー
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Distance from Site (km)', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(vis_dir / f'polar_distribution_{region}.png', dpi=dpi, bbox_inches='tight')
            plt.close()
        
        # 3. 距離分布ヒストグラム
        if 'radius' in analysis_gdf.columns:
            logger.info("距離分布ヒストグラムを作成中...")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
            
            # 遺跡からの距離
            ax1.hist(analysis_gdf['radius'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Distance from Archaeological Site (km)', fontsize=10)
            ax1.set_ylabel('Frequency', fontsize=10)
            ax1.set_title('Distance Distribution', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # 川からの距離（データがある場合）
            if 'river_radius' in analysis_gdf.columns:
                river_data = analysis_gdf['river_radius'].dropna()
                if len(river_data) > 0:
                    ax2.hist(river_data, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
                    ax2.set_xlabel('Distance from Nearest River (km)', fontsize=10)
                    ax2.set_ylabel('Frequency', fontsize=10)
                    ax2.set_title('River Distance Distribution', fontsize=12, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'No river distance data', ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('River Distance Distribution (No Data)', fontsize=12)
            else:
                ax2.text(0.5, 0.5, 'No river distance data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('River Distance Distribution (No Data)', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(vis_dir / f'distance_distributions_{region}.png', dpi=dpi, bbox_inches='tight')
            plt.close()
        
        logger.info(f"可視化完了: {vis_dir}")
        
    except Exception as e:
        logger.error(f"可視化エラー: {e}")
        logger.error(traceback.format_exc())


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='既知遺跡周辺地名分析スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 必須引数
    parser.add_argument(
        '--region', 
        required=True,
        help='地域指定（例: acre, marajo）'
    )
    
    # オプション引数
    parser.add_argument(
        '--radius', 
        type=float,
        help='遺跡周辺検索半径（km、デフォルトは設定ファイルから読み込み）'
    )
    
    parser.add_argument(
        '--osm-keys-mode',
        default='water_focused',
        help='OSMキー抽出モード（デフォルト: water_focused）'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=PROJECT_ROOT / 'configs' / 'analyze_site.yaml',
        help='設定ファイルのパス'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='出力ベースディレクトリ（デフォルトは設定ファイルから読み込み）'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='結果の可視化を実行'
    )
    
    parser.add_argument(
        '--similarity-analysis',
        action='store_true',
        help='類似度分析を実行（機械学習による遺跡候補地スコアリング）'
    )
    
    parser.add_argument(
        '--cluster-sites',
        action='store_true',
        default=True,
        help='近接遺跡をクラスタリングして統合（デフォルト: True）'
    )
    
    parser.add_argument(
        '--no-cluster-sites',
        action='store_true',
        help='近接遺跡のクラスタリングを無効化'
    )
    
    parser.add_argument(
        '--cluster-distance',
        type=float,
        default=200.0,
        help='クラスタリング距離（メートル、デフォルト: 200m）'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='デバッグモードで実行'
    )
    
    parser.add_argument(
        '--max-sites',
        type=int,
        help='処理する遺跡数の上限（テスト用）'
    )
    
    args = parser.parse_args()
    
    try:
        # 設定ファイルの読み込み
        print(f"設定ファイルを読み込み中: {args.config}")
        config = load_config(args.config)
        
        # 地域設定の読み込み
        print(f"地域設定を読み込み中: {args.region}")
        region_config = load_region_config(args.region)
        
        # OSMキー設定の読み込み
        print(f"OSMキー設定を読み込み中: {args.osm_keys_mode}")
        osm_keys_config = load_osm_keys_config(args.osm_keys_mode)
        
        # 出力ディレクトリの作成
        base_output_dir = args.output_dir or Path(config['output']['base_dir'])
        output_dir = create_output_directory(
            base_output_dir, 
            config['output']['timestamp_dir_format']
        )
        
        print(f"出力ディレクトリ: {output_dir}")
        
        # ログ設定
        log_level = "DEBUG" if args.debug else config.get('logging', {}).get('level', 'INFO')
        log_file = output_dir / config['output']['log_filename']
        logger = setup_logging(log_file, log_level)
        
        logger.info("=== 既知遺跡周辺地名分析開始 ===")
        logger.info(f"地域: {args.region}")
        logger.info(f"出力ディレクトリ: {output_dir}")
        
        # 設定のバックアップ
        save_config_backup(config, output_dir, config['output']['config_backup'])
        
        # パラメータの決定
        radius_km = args.radius or config['analysis']['default_radius_km']
        logger.info(f"検索半径: {radius_km}km")
        logger.info(f"OSMキーモード: {args.osm_keys_mode}")
        
        # データファイルパスの構築
        data_dir = PROJECT_ROOT / "data"
        osm_pbf_path = data_dir / "raw" / "osm" / region_config['osm']['pbf_file']
        rivers_path = data_dir / "raw" / region_config['hydrorivers']['shapefile']
        
        logger.info(f"OSM PBFファイル: {osm_pbf_path}")
        logger.info(f"河川データ: {rivers_path}")
        
        # ファイル存在チェック
        if not osm_pbf_path.exists():
            raise FileNotFoundError(f"OSM PBFファイルが見つかりません: {osm_pbf_path}")
        if not rivers_path.exists():
            raise FileNotFoundError(f"河川データファイルが見つかりません: {rivers_path}")
        
        # 1. 遺跡データの読み込み
        logger.info("遺跡データを読み込み中...")
        extractor = ToponymExtractor(str(osm_pbf_path), osm_keys_config)
        sites_gdf = extractor.load_known_sites(region_config, args.region)
        
        # 遺跡数制限の適用
        max_sites = args.max_sites or config['processing']['max_sites_limit']
        if max_sites > 0 and len(sites_gdf) > max_sites:
            logger.info(f"遺跡数制限を適用: {len(sites_gdf)}件 → {max_sites}件")
            sites_gdf = sites_gdf.head(max_sites)
        
        logger.info(f"遺跡数: {len(sites_gdf)}件")
        
        # 2. 地名抽出
        logger.info("遺跡周辺地名を抽出中...")
        
        # クラスタリング設定
        cluster_sites = args.cluster_sites and not args.no_cluster_sites
        cluster_distance_m = args.cluster_distance
        
        toponyms_gdf = extractor.extract_toponyms_around_sites(
            sites_gdf, 
            radius_km, 
            args.osm_keys_mode,
            cluster_sites=cluster_sites,
            cluster_distance_m=cluster_distance_m
        )
        
        if toponyms_gdf.empty:
            logger.warning("地名が抽出されませんでした。処理を終了します。")
            return
        
        logger.info(f"抽出された地名数: {len(toponyms_gdf)}件")
        
        # 3. 極座標変換
        logger.info("極座標変換を実行中...")
        polar_config = config['analysis']['polar_coordinate_system']
        converter = PolarConverter(
            polar_config['angle_reference'],
            polar_config['angle_direction']
        )
        polar_gdf = converter.convert_to_polar(toponyms_gdf)
        polar_summary = converter.create_polar_summary(polar_gdf)
        
        # 4. 川距離計算
        logger.info("川距離を計算中...")
        max_river_search = config['processing']['max_river_search_distance']
        river_calc = RiverDistanceCalculator(str(rivers_path), max_river_search)
        
        # 地域のバウンディングボックスで河川データを読み込み
        bbox = region_config['bbox']
        river_calc.load_rivers(bbox)
        
        analysis_gdf = river_calc.calculate_river_distances(polar_gdf)
        river_summary = river_calc.create_river_distance_summary(analysis_gdf)
        
        # 地域・文化タグ情報を追加
        analysis_gdf['region'] = args.region
        if 'culture_tag' not in analysis_gdf.columns:
            # サイト名から文化タグを推定（簡易的）
            if args.region == 'acre':
                analysis_gdf['culture_tag'] = 'acre'
            elif args.region == 'marajo':
                analysis_gdf['culture_tag'] = 'marajo'
            else:
                analysis_gdf['culture_tag'] = 'unknown'
        
        # 5. CSV出力
        logger.info("CSV出力中...")
        exporter = CSVExporter(config)
        
        # メタデータの準備
        metadata = {
            'analysis_parameters': {
                'region': args.region,
                'radius_km': radius_km,
                'osm_keys_mode': args.osm_keys_mode,
                'max_river_search_distance': max_river_search
            },
            'data_sources': {
                'osm_pbf': str(osm_pbf_path),
                'rivers_shapefile': str(rivers_path),
                'region_config': region_config
            },
            'statistics': {
                'total_sites': len(sites_gdf),
                'total_toponyms': len(analysis_gdf),
                'polar_summary': polar_summary,
                'river_summary': river_summary
            }
        }
        
        csv_path = exporter.export_analysis_results(
            analysis_gdf, output_dir, args.region, metadata
        )
        
        # サマリレポートの作成
        summary_report = exporter.create_summary_report(
            analysis_gdf, polar_summary, river_summary, output_dir, args.region
        )
        
        # 6. 可視化（オプション）
        if args.visualize:
            logger.info("可視化を実行中...")
            # numpyのインポート（可視化で必要）
            import numpy as np
            create_visualization(analysis_gdf, sites_gdf, output_dir, args.region, config)
        
        # 完了メッセージ
        logger.info("=== 分析完了 ===")
        logger.info(f"出力ファイル: {csv_path}")
        logger.info(f"総地名数: {len(analysis_gdf)}件")
        logger.info(f"有効川距離データ: {river_summary.get('valid_river_distances', 0)}件")
        
        print(f"\n✅ 分析が完了しました!")
        print(f"📁 出力ディレクトリ: {output_dir}")
        print(f"📄 CSVファイル: {csv_path.name if csv_path else 'N/A'}")
        print(f"📊 地名数: {len(analysis_gdf)}件")
        
        if args.visualize:
            print(f"🖼️  可視化: {output_dir / 'visualizations'}")
        
        # 7. 類似度分析（オプション）
        if args.similarity_analysis:
            logger.info("類似度分析を実行中...")
            
            try:
                # 類似度分析器の初期化
                similarity_analyzer = ArchaeologicalSimilarityAnalyzer(str(csv_path))
                
                # 分析実行
                logger.info("データ読み込み中...")
                similarity_analyzer.load_data()
                
                logger.info("特徴量エンジニアリング中...")
                similarity_analyzer.engineer_features()
                
                logger.info("データ前処理中...")
                similarity_analyzer.preprocess_features()
                
                logger.info("高相関特徴量除去中...")
                similarity_analyzer.remove_high_correlation_features()
                
                logger.info("類似度モデル構築中...")
                similarity_analyzer.build_similarity_models()
                
                logger.info("類似度スコア計算中...")
                scores_df = similarity_analyzer.calculate_similarity_scores()
                
                # 類似度分析結果の出力
                similarity_output_dir = output_dir / "similarity_analysis"
                similarity_output_dir.mkdir(exist_ok=True)
                
                scores_output_path = similarity_output_dir / f"similarity_scores_{args.region}.csv"
                scores_df.to_csv(scores_output_path, index=False)
                
                # 分析レポート生成
                report_path = similarity_output_dir / f"similarity_report_{args.region}.md"
                report_content = similarity_analyzer.generate_analysis_report(scores_df)
                
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                logger.info(f"類似度分析完了: {similarity_output_dir}")
                print(f"📊 類似度分析結果: {similarity_output_dir}")
                
            except Exception as sim_error:
                logger.error(f"類似度分析エラー: {sim_error}")
                print(f"⚠️  類似度分析でエラーが発生しましたが、基本分析は完了しています: {sim_error}")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        if args.debug:
            print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()