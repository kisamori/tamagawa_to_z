"""
csv_exporter.py: CSV出力機能

遺跡周辺地名の分析結果をCSV形式で出力する機能を提供します。
"""

import logging
import csv
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

import pandas as pd
import geopandas as gpd
import yaml

logger = logging.getLogger(__name__)


class CSVExporter:
    """CSV出力クラス"""
    
    def __init__(self, config: Dict):
        """
        初期化
        
        Args:
            config: CSV出力設定辞書
        """
        self.config = config
        self.csv_format = config.get('csv_format', {})
        self.decimal_places = self.csv_format.get('decimal_places', {})
        
        logger.info("CSV出力器を初期化")
    
    def export_analysis_results(
        self, 
        analysis_gdf: gpd.GeoDataFrame, 
        output_dir: Path,
        region: str,
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        分析結果をCSVファイルとして出力
        
        Args:
            analysis_gdf: 分析結果のGeoDataFrame
            output_dir: 出力ディレクトリ
            region: 地域名
            metadata: メタデータ辞書
            
        Returns:
            出力ファイルのパス
        """
        if analysis_gdf.empty:
            logger.warning("出力対象のデータが空です")
            return None
        
        # ファイル名の生成
        filename = self.config['output']['csv_filename'].format(region=region)
        output_path = output_dir / filename
        
        logger.info(f"CSV出力開始: {output_path}")
        
        # データの準備
        export_df = self._prepare_export_data(analysis_gdf)
        
        # CSVファイルに出力
        try:
            export_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"CSV出力完了: {len(export_df)}件のデータ")
            
            # メタデータファイルも出力
            if metadata:
                self._export_metadata(output_dir, metadata, region)
            
            return output_path
            
        except Exception as e:
            logger.error(f"CSV出力エラー: {e}")
            raise
    
    def _prepare_export_data(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        出力用データの準備
        
        Args:
            gdf: 分析結果のGeoDataFrame
            
        Returns:
            出力用のDataFrame
        """
        # 必要な列のみを抽出
        required_columns = self.csv_format.get('columns', [])
        available_columns = [col for col in required_columns if col in gdf.columns]
        
        if not available_columns:
            logger.warning("出力対象の列が見つかりません。全ての列を出力します。")
            # 座標列を追加
            export_df = gdf.copy()
            export_df['toponym_lat'] = gdf.geometry.y
            export_df['toponym_lon'] = gdf.geometry.x
            # geometry列を除外
            export_df = export_df.drop('geometry', axis=1)
        else:
            export_df = gdf[available_columns].copy()
            
            # 座標情報を追加
            if 'toponym_lat' not in export_df.columns:
                export_df['toponym_lat'] = gdf.geometry.y
            if 'toponym_lon' not in export_df.columns:
                export_df['toponym_lon'] = gdf.geometry.x
        
        # 数値の丸め処理
        export_df = self._round_numeric_columns(export_df)
        
        # 列の並び順を調整
        export_df = self._reorder_columns(export_df)
        
        # 欠損値の処理
        export_df = self._handle_missing_values(export_df)
        
        return export_df
    
    def _round_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数値列の丸め処理
        
        Args:
            df: 対象のDataFrame
            
        Returns:
            丸め処理済みのDataFrame
        """
        result_df = df.copy()
        
        for column, decimal_places in self.decimal_places.items():
            if column in result_df.columns and pd.api.types.is_numeric_dtype(result_df[column]):
                result_df[column] = result_df[column].round(decimal_places)
        
        return result_df
    
    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        列の並び順を調整
        
        Args:
            df: 対象のDataFrame
            
        Returns:
            並び順調整済みのDataFrame
        """
        # 優先順位の高い列
        priority_columns = [
            'site_name', 'toponym_name', 'angle', 'radius', 
            'river_angle', 'river_radius', 'region', 'culture_tag',
            'toponym_lat', 'toponym_lon', 'site_lat', 'site_lon'
        ]
        
        # 存在する優先列
        existing_priority = [col for col in priority_columns if col in df.columns]
        
        # その他の列
        other_columns = [col for col in df.columns if col not in existing_priority]
        
        # 並び順を決定
        column_order = existing_priority + other_columns
        
        return df[column_order]
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        欠損値の処理
        
        Args:
            df: 対象のDataFrame
            
        Returns:
            欠損値処理済みのDataFrame
        """
        result_df = df.copy()
        
        # NaNを適切な値に置換
        # 文字列列はNaNを空文字に
        string_columns = result_df.select_dtypes(include=['object']).columns
        result_df[string_columns] = result_df[string_columns].fillna('')
        
        # 数値列はNaNをNullのまま（CSVでは空欄）
        # 特別な処理は行わない
        
        return result_df
    
    def _export_metadata(self, output_dir: Path, metadata: Dict, region: str):
        """
        メタデータファイルの出力
        
        Args:
            output_dir: 出力ディレクトリ
            metadata: メタデータ辞書
            region: 地域名
        """
        try:
            metadata_path = output_dir / f"analysis_metadata_{region}.yaml"
            
            # タイムスタンプを追加
            metadata['export_info'] = {
                'timestamp': datetime.now().isoformat(),
                'region': region,
                'exporter_version': '1.0.0'
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"メタデータファイル出力: {metadata_path}")
            
        except Exception as e:
            logger.error(f"メタデータ出力エラー: {e}")
    
    def create_summary_report(
        self, 
        analysis_gdf: gpd.GeoDataFrame,
        polar_summary: Dict,
        river_summary: Dict,
        output_dir: Path,
        region: str
    ) -> Path:
        """
        分析サマリレポートを作成
        
        Args:
            analysis_gdf: 分析結果のGeoDataFrame
            polar_summary: 極座標分析サマリ
            river_summary: 川距離分析サマリ
            output_dir: 出力ディレクトリ
            region: 地域名
            
        Returns:
            レポートファイルのパス
        """
        try:
            report_path = output_dir / f"analysis_summary_{region}.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# 遺跡周辺地名分析サマリレポート\n")
                f.write(f"地域: {region}\n")
                f.write(f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 基本統計
                f.write("## 基本統計\n")
                f.write(f"- 総地名数: {len(analysis_gdf)}件\n")
                
                if 'site_name' in analysis_gdf.columns:
                    site_count = analysis_gdf['site_name'].nunique()
                    f.write(f"- 対象遺跡数: {site_count}件\n")
                
                f.write("\n")
                
                # 極座標分析結果
                if polar_summary:
                    f.write("## 極座標分析\n")
                    angle_stats = polar_summary.get('angle_stats', {})
                    radius_stats = polar_summary.get('radius_stats', {})
                    
                    if angle_stats:
                        f.write("### 角度統計\n")
                        f.write(f"- 範囲: {angle_stats.get('min', 'N/A'):.1f}° - {angle_stats.get('max', 'N/A'):.1f}°\n")
                        f.write(f"- 平均: {angle_stats.get('mean', 'N/A'):.1f}°\n")
                        f.write(f"- 標準偏差: {angle_stats.get('std', 'N/A'):.1f}°\n")
                    
                    if radius_stats:
                        f.write("### 距離統計\n")
                        f.write(f"- 範囲: {radius_stats.get('min', 'N/A'):.3f}km - {radius_stats.get('max', 'N/A'):.3f}km\n")
                        f.write(f"- 平均: {radius_stats.get('mean', 'N/A'):.3f}km\n")
                        f.write(f"- 標準偏差: {radius_stats.get('std', 'N/A'):.3f}km\n")
                    
                    f.write("\n")
                
                # 川距離分析結果
                if river_summary:
                    f.write("## 川距離分析\n")
                    f.write(f"- 有効川距離データ: {river_summary.get('valid_river_distances', 0)}件\n")
                    f.write(f"- カバレッジ率: {river_summary.get('coverage_rate', 0):.1%}\n")
                    
                    river_radius_stats = river_summary.get('river_radius_stats', {})
                    if river_radius_stats:
                        f.write(f"- 川距離範囲: {river_radius_stats.get('min', 'N/A'):.3f}km - {river_radius_stats.get('max', 'N/A'):.3f}km\n")
                        f.write(f"- 川距離平均: {river_radius_stats.get('mean', 'N/A'):.3f}km\n")
                    
                    f.write("\n")
                
                # データ品質情報
                f.write("## データ品質\n")
                null_counts = analysis_gdf.isnull().sum()
                for column, null_count in null_counts.items():
                    if null_count > 0:
                        f.write(f"- {column}: {null_count}件の欠損値\n")
                
            logger.info(f"サマリレポート出力: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"サマリレポート出力エラー: {e}")
            return None