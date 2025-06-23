# DEMO FILE: Harmonizer メインクラス

"""
Harmonizer: 多言語トポニム解析のメインクラス

このクラスは、地名（トポニム）を解析して水に関連する地名を抽出・分類するための
機能を提供します。
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter
import unidecode
import re
from typing import Dict, List, Tuple, Union, Optional

# サブモジュールのインポート
from . import preprocess
from . import embed
from . import cluster


class HarmonizerPipeline:
    """パイプライン実行用のHarmonizerラッパークラス"""
    
    def __init__(self):
        self.config = {}
        self.harmonizer = Harmonizer()  # 実際のHarmonizerを使用
    
    def make_bbox_gdf(self):
        """バウンディングボックスGeoDataFrameを作成"""
        # 実装: アマゾン地域の基本的なバウンディングボックス
        from shapely.geometry import box
        
        # Acreリージョンを含む広めの範囲
        bbox = box(-74, -15, -60, -5)  # west, south, east, north
        
        gdf = gpd.GeoDataFrame(
            {'region': ['amazon_basin']},
            geometry=[bbox],
            crs="EPSG:4326"
        )
        return gdf
    
    def extract_toponyms_pyrosm(self, bbox_gdf):
        """pyrosmを使用して地名を抽出"""
        # 簡略化された実装：サンプルトポニムを生成
        import numpy as np
        np.random.seed(42)
        
        # 実際のアマゾン地名の例
        sample_toponyms = [
            'Rio Acre', 'Igarapé do Boi', 'Lago Grande', 'Rio Purus',
            'Igarapé Preto', 'Rio Branco', 'Seringal São Paulo',
            'Colocação Nazaré', 'Rio Juruá', 'Igarapé da Onça'
        ]
        
        n_toponyms = min(len(sample_toponyms), np.random.randint(5, 15))
        
        toponyms_data = []
        for i in range(n_toponyms):
            toponym = sample_toponyms[i % len(sample_toponyms)]
            
            # Acreリージョン内の座標
            lon = np.random.uniform(-70.5, -66.5)
            lat = np.random.uniform(-11.5, -8.5)
            
            toponyms_data.append({
                'name': toponym,
                'lat': lat,
                'lon': lon,
                'type': 'waterway',
                'osm_id': f'way_{1000000 + i}'
            })
        
        # 明示的にpd.DataFrameとして返す（GeoDataFrameではない）
        df = pd.DataFrame(toponyms_data)
        return df
    
    def process_toponyms(self, toponyms):
        """地名を処理"""
        if len(toponyms) == 0:
            return toponyms
            
        # 正規化された名前を追加
        toponyms = toponyms.copy()
        toponyms['normalized_name'] = toponyms['name'].str.lower().str.replace(' ', '_')
        
        # 語根を抽出（簡単な実装）
        def extract_root(name):
            name_lower = name.lower()
            if 'rio' in name_lower:
                return 'rio'
            elif 'igarape' in name_lower or 'igarapé' in name_lower:
                return 'igarape'
            elif 'lago' in name_lower:
                return 'lago'
            else:
                return 'agua'
        
        toponyms['root'] = toponyms['name'].apply(extract_root)
        # 明示的にpd.DataFrameとして返す
        return pd.DataFrame(toponyms)
    
    def attach_distance(self, processed_toponyms):
        """距離を計算して付与"""
        if len(processed_toponyms) == 0:
            return processed_toponyms
            
        # 水域との距離を計算（簡略化）
        import numpy as np
        processed_toponyms = processed_toponyms.copy()
        processed_toponyms['dist_km'] = np.random.exponential(
            scale=self.config.get('distance_threshold_km', 3.0),
            size=len(processed_toponyms)
        )
        # 明示的にpd.DataFrameとして返す
        return pd.DataFrame(processed_toponyms)
    
    def filter_candidates(self, with_distance):
        """候補地点をフィルタリング"""
        if len(with_distance) == 0:
            return with_distance
            
        import numpy as np
        
        # パラメータを取得
        distance_threshold = self.config.get('distance_threshold_km', 3.0)
        occ_threshold = self.config.get('occ_pct_threshold', 5.0)
        root_weights = self.config.get('root_weight_table', {})
        
        candidates = with_distance.copy()
        
        # 水域出現率を計算（簡略化）
        candidates['occ_pct'] = np.random.exponential(
            scale=occ_threshold,
            size=len(candidates)
        )
        
        # スコア計算
        def calculate_score(row):
            base_score = 0.5
            
            # 距離によるペナルティ
            if row['dist_km'] > distance_threshold:
                base_score *= 0.5
            
            # 水域出現率によるボーナス
            if row['occ_pct'] > occ_threshold:
                base_score *= 1.5
            
            # 語根ウェイトによる調整
            root = row.get('root', 'agua')
            weight = root_weights.get(root, 0.5)
            base_score *= weight
            
            return base_score
        
        candidates['total_score'] = candidates.apply(calculate_score, axis=1)
        candidates['is_candidate'] = candidates['total_score'] > 0.3
        
        # 候補のみを返す
        result = candidates[candidates['is_candidate']].reset_index(drop=True)
        # 明示的にpd.DataFrameとして返す
        return pd.DataFrame(result)


class Harmonizer:
    """多言語トポニム解析のメインクラス
    
    地名（トポニム）を解析して水に関連する地名を抽出・分類し、
    水関連確率マップ（Pwater(x)）を生成します。
    
    Attributes
    ----------
    water_seeds : List[str]
        水関連語彙のシードリスト
    embedding_model : str
        埋め込みモデルの名前
    embedding_dim : int
        埋め込みベクトルの次元数
    """
    
    def __init__(self, 
                 water_seeds: Optional[List[str]] = None,
                 embedding_model: str = "sentence-transformers/distiluse-base-multilingual-v2",
                 embedding_dim: int = 512):
        """Harmonizerの初期化
        
        Parameters
        ----------
        water_seeds : List[str], optional
            水関連語彙のシードリスト
        embedding_model : str, optional
            埋め込みモデルの名前
        embedding_dim : int, optional
            埋め込みベクトルの次元数
        """
        # デフォルトの水関連語彙（ポルトガル語）
        self.water_seeds = water_seeds or [
            "rio", "igarape", "lago", "parana", "cachoeira", 
            "corrego", "lagoa", "canal", "baia", "represa"
        ]
        
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        
        # 埋め込みモデルの初期化（実際の実装ではここでロード）
        # self.model = SentenceTransformer(embedding_model)
        
        print(f"Harmonizer initialized with {len(self.water_seeds)} water seeds")
        print(f"Using embedding model: {embedding_model} ({embedding_dim}d)")
    
    def process(self, 
                toponyms: Union[pd.DataFrame, gpd.GeoDataFrame],
                confidence_threshold: float = 0.85) -> gpd.GeoDataFrame:
        """地名データを処理する
        
        Parameters
        ----------
        toponyms : Union[pd.DataFrame, gpd.GeoDataFrame]
            地名データ
        confidence_threshold : float, optional
            信頼度閾値（この値以下の場合、人間の確認が必要）
            
        Returns
        -------
        gpd.GeoDataFrame
            処理済みの地名データ
        """
        # GeoDataFrameに変換
        if isinstance(toponyms, pd.DataFrame):
            if 'geometry' not in toponyms.columns:
                # 緯度経度から geometry を作成
                if all(col in toponyms.columns for col in ['lat', 'lon']):
                    geometry = [Point(lon, lat) for lon, lat in zip(toponyms['lon'], toponyms['lat'])]
                    toponyms = gpd.GeoDataFrame(toponyms, geometry=geometry, crs="EPSG:4326")
                else:
                    raise ValueError("DataFrame must have 'lat' and 'lon' columns")
        
        # 前処理
        processed = preprocess.normalize_toponyms(toponyms)
        
        # 埋め込み生成（実際の実装では self.model を使用）
        # embeddings = self.model.encode(processed['normalized_name'].tolist())
        # デモ用の簡易実装
        embeddings = embed.mock_embeddings(processed['normalized_name'].tolist(), self.embedding_dim)
        processed['embedding'] = list(embeddings)
        
        # 水関連度の計算
        water_scores = cluster.calculate_water_scores(
            processed['normalized_name'].tolist(),
            embeddings,
            self.water_seeds
        )
        processed['water_score'] = water_scores
        
        # 信頼度の計算
        processed['confidence'] = np.clip(water_scores * 1.2, 0, 1)  # 簡易的な計算
        
        # 人間の確認が必要なものをマーク
        processed['needs_review'] = processed['confidence'] < confidence_threshold
        
        return processed
    
    def create_probability_map(self, 
                              toponyms_gdf: gpd.GeoDataFrame,
                              shape: Tuple[int, int],
                              bounds: Optional[Tuple[float, float, float, float]] = None,
                              buffer_radius: int = 10) -> np.ndarray:
        """水関連確率マップを生成する
        
        Parameters
        ----------
        toponyms_gdf : gpd.GeoDataFrame
            処理済みの地名データ
        shape : Tuple[int, int]
            出力ラスターの形状 (height, width)
        bounds : Tuple[float, float, float, float], optional
            出力ラスターの範囲 (xmin, ymin, xmax, ymax)
        buffer_radius : int, optional
            バッファ半径（ピクセル単位）
            
        Returns
        -------
        np.ndarray
            水関連確率マップ
        """
        # 出力ラスターの初期化
        ny, nx = shape
        prob_map = np.zeros((ny, nx))
        
        # 範囲の設定
        if bounds is None:
            # GeoDataFrameの範囲を使用
            bounds = toponyms_gdf.total_bounds
        
        xmin, ymin, xmax, ymax = bounds
        
        # 地理座標系から画像座標系への変換関数
        def geo_to_pixel(x, y):
            px = int((x - xmin) / (xmax - xmin) * (nx - 1))
            py = int((ymax - y) / (ymax - ymin) * (ny - 1))  # 緯度は反転
            return px, py
        
        # 各地名ポイントの影響を追加
        for idx, row in toponyms_gdf.iterrows():
            # 水関連度スコアがない場合はスキップ
            if 'water_score' not in row:
                continue
                
            # ジオメトリから座標を取得
            x, y = row.geometry.x, row.geometry.y
            
            # 画像座標に変換
            px, py = geo_to_pixel(x, y)
            
            # 有効範囲内のみ処理
            if 0 <= px < nx and 0 <= py < ny:
                # バッファ領域の生成
                for dy in range(-buffer_radius, buffer_radius + 1):
                    for dx in range(-buffer_radius, buffer_radius + 1):
                        # バッファ内の距離を計算
                        distance = np.sqrt(dx**2 + dy**2)
                        if distance <= buffer_radius:
                            # 対象ピクセル
                            target_px, target_py = px + dx, py + dy
                            
                            # 有効範囲内のみ処理
                            if 0 <= target_px < nx and 0 <= target_py < ny:
                                # 距離に応じて減衰
                                weight = row['water_score'] * np.exp(-0.5 * (distance / (buffer_radius/2))**2)
                                
                                # 最大値を採用
                                prob_map[target_py, target_px] = max(prob_map[target_py, target_px], weight)
        
        # スムージング
        prob_map = gaussian_filter(prob_map, sigma=buffer_radius/3)
        
        # 0-1の範囲に正規化
        if prob_map.max() > 0:
            prob_map = prob_map / prob_map.max()
        
        return prob_map
    
    def save_probability_map(self, 
                            prob_map: np.ndarray,
                            output_path: str,
                            bounds: Optional[Tuple[float, float, float, float]] = None,
                            crs: str = "EPSG:4326"):
        """確率マップをGeoTIFFとして保存する
        
        Parameters
        ----------
        prob_map : np.ndarray
            確率マップ
        output_path : str
            出力ファイルパス
        bounds : Tuple[float, float, float, float], optional
            出力ラスターの範囲 (xmin, ymin, xmax, ymax)
        crs : str, optional
            座標参照系
        """
        # 範囲の設定
        ny, nx = prob_map.shape
        if bounds is None:
            # デフォルト範囲（0-1の正規化座標）
            bounds = (0, 0, 1, 1)
        
        xmin, ymin, xmax, ymax = bounds
        
        # 変換行列の計算
        transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, nx, ny)
        
        # GeoTIFFとして保存
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=ny,
            width=nx,
            count=1,
            dtype=prob_map.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(prob_map, 1)
        
        print(f"Probability map saved to {output_path}")
