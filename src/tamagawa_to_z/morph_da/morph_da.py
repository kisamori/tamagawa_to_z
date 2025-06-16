# DEMO FILE: MorphDA メインクラス

"""
MorphDA: 地形変化データ同化のメインクラス

このクラスは、Delft3D-FMとOpenDAを用いた地形変化データ同化のための
機能を提供します。
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import subprocess
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime, timedelta


class MorphDA:
    """地形変化データ同化のメインクラス
    
    Delft3D-FMとOpenDAを用いた地形変化データ同化を実行するためのクラスです。
    Ensemble Smootherを使用して、過去の地形（古河道）を推定します。
    
    Attributes
    ----------
    config : Dict[str, Any]
        設定情報
    work_dir : str
        作業ディレクトリ
    ensemble_size : int
        アンサンブルサイズ
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 work_dir: str = "./work",
                 ensemble_size: int = 30):
        """MorphDAの初期化
        
        Parameters
        ----------
        config_file : Optional[str], optional
            設定ファイルのパス
        work_dir : str, optional
            作業ディレクトリ
        ensemble_size : int, optional
            アンサンブルサイズ
        """
        self.work_dir = work_dir
        self.ensemble_size = ensemble_size
        
        # 設定ファイルの読み込み
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # デフォルト設定
            self.config = self._create_default_config()
        
        # 作業ディレクトリの作成
        os.makedirs(work_dir, exist_ok=True)
        
        print(f"MorphDA initialized with ensemble size: {ensemble_size}")
        print(f"Working directory: {work_dir}")
    
    def _create_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を作成する
        
        Returns
        -------
        Dict[str, Any]
            デフォルト設定
        """
        return {
            'model': {
                'name': 'Delft3D-FM',
                'config_file': 'model/flow2d3d_config.xml',
                'work_dir': 'model/work',
                'bin_dir': '/opt/delft3d/bin',
                'morphology': True  # 地形変化モジュールを有効化
            },
            'parameters': {
                'tau_cr': {  # 限界掃流力
                    'prior': 'lognormal(0.2,0.3)',
                    'min': 0.05,
                    'max': 1.0
                },
                'E': {  # 侵食係数
                    'prior': 'lognormal(1e-4,0.5)',
                    'min': 1e-5,
                    'max': 1e-3
                },
                'n_bed': {  # 河床の粗度係数
                    'prior': 'lognormal(0.03,0.2)',
                    'min': 0.01,
                    'max': 0.1
                }
            },
            'assimilation': {
                'scheme': 'EnSmoother',
                'ensemble_size': self.ensemble_size,
                'observations': [
                    {'dem': 'current_dem'},
                    {'watermask': 'JRC_GSW'}
                ],
                'cost_function': {
                    'dem_weight': 0.4,
                    'watermask_weight': 0.3,
                    'toponym_weight': 0.3
                }
            },
            'output': {
                'dir': 'output/smoother',
                'format': 'netcdf',
                'save_ensemble': True
            }
        }
    
    def save_config(self, output_path: str) -> None:
        """設定をYAMLファイルとして保存する
        
        Parameters
        ----------
        output_path : str
            出力ファイルパス
        """
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, sort_keys=False, default_flow_style=False)
        
        print(f"Configuration saved to {output_path}")
    
    def setup_model(self, 
                   current_dem_file: str,
                   water_history_file: Optional[str] = None,
                   toponym_prob_file: Optional[str] = None) -> None:
        """モデルをセットアップする
        
        Parameters
        ----------
        current_dem_file : str
            現在のDEMファイルのパス
        water_history_file : Optional[str], optional
            水域履歴ファイルのパス
        toponym_prob_file : Optional[str], optional
            地名確率マップファイルのパス
        """
        # 実際の実装では、Delft3D-FMのセットアップを行います
        # ここではデモ用の簡易実装
        
        print(f"Setting up model with current DEM: {current_dem_file}")
        if water_history_file:
            print(f"Water history: {water_history_file}")
        if toponym_prob_file:
            print(f"Toponym probability map: {toponym_prob_file}")
        
        # モデルディレクトリの作成
        model_dir = os.path.join(self.work_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        
        # 各アンサンブルメンバーのディレクトリ作成
        for i in range(self.ensemble_size):
            member_dir = os.path.join(model_dir, f"member_{i:03d}")
            os.makedirs(member_dir, exist_ok=True)
        
        print(f"Model setup completed. {self.ensemble_size} ensemble members created.")
    
    def generate_parameter_ensemble(self) -> Dict[str, np.ndarray]:
        """パラメータのアンサンブルを生成する
        
        Returns
        -------
        Dict[str, np.ndarray]
            パラメータのアンサンブル
        """
        # パラメータのアンサンブル生成
        np.random.seed(42)  # 再現性のため
        
        parameters = {}
        
        for param_name, param_config in self.config['parameters'].items():
            prior = param_config['prior']
            
            if 'lognormal' in prior:
                # 対数正規分布
                mu, sigma = map(float, prior.replace('lognormal(', '').replace(')', '').split(','))
                parameters[param_name] = np.random.lognormal(np.log(mu), sigma, self.ensemble_size)
            
            elif 'uniform' in prior:
                # 一様分布
                min_val, max_val = map(float, prior.replace('uniform(', '').replace(')', '').split(','))
                parameters[param_name] = np.random.uniform(min_val, max_val, self.ensemble_size)
            
            elif 'normal' in prior:
                # 正規分布
                mu, sigma = map(float, prior.replace('normal(', '').replace(')', '').split(','))
                parameters[param_name] = np.random.normal(mu, sigma, self.ensemble_size)
            
            else:
                # デフォルト（一様分布）
                min_val = param_config.get('min', 0.0)
                max_val = param_config.get('max', 1.0)
                parameters[param_name] = np.random.uniform(min_val, max_val, self.ensemble_size)
            
            # 範囲制限
            min_val = param_config.get('min', None)
            max_val = param_config.get('max', None)
            
            if min_val is not None:
                parameters[param_name] = np.maximum(parameters[param_name], min_val)
            
            if max_val is not None:
                parameters[param_name] = np.minimum(parameters[param_name], max_val)
        
        return parameters
    
    def generate_initial_dem_ensemble(self, 
                                     current_dem: np.ndarray,
                                     perturbation_scale: float = 0.2) -> np.ndarray:
        """初期DEMのアンサンブルを生成する
        
        Parameters
        ----------
        current_dem : np.ndarray
            現在のDEM
        perturbation_scale : float, optional
            摂動スケール
            
        Returns
        -------
        np.ndarray
            初期DEMのアンサンブル（shape: (ensemble_size, ny, nx)）
        """
        # 乱数シードを固定（再現性のため）
        np.random.seed(43)
        
        # DEMの形状
        ny, nx = current_dem.shape
        
        # 初期DEMのアンサンブル
        dem_ensemble = np.zeros((self.ensemble_size, ny, nx))
        
        # 各アンサンブルメンバーに対して初期DEMを生成
        for i in range(self.ensemble_size):
            # 現在のDEMをベースに摂動を加える
            perturbation = perturbation_scale * np.random.randn(ny, nx)
            
            # 古河道の痕跡を追加（ランダムな位置に）
            # 実際の実装では、より物理的に妥当な方法で生成
            
            # ランダムな位置
            center_y = np.random.randint(ny // 4, 3 * ny // 4)
            center_x = np.random.randint(nx // 4, 3 * nx // 4)
            
            # 河道の幅と深さ
            width = np.random.randint(5, 15)
            depth = np.random.uniform(0.5, 2.0)
            
            # 河道の形状（正弦波）
            y_indices, x_indices = np.indices((ny, nx))
            
            # 正弦波の河道
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(10, 30)
            frequency = np.random.uniform(0.01, 0.05)
            
            # 河道の中心線
            centerline_y = center_y + amplitude * np.sin(frequency * x_indices + phase)
            
            # 河道からの距離
            distance = np.abs(y_indices - centerline_y)
            
            # 河道の形状（ガウス関数）
            channel = depth * np.exp(-0.5 * (distance / width)**2)
            
            # DEMに河道を追加（標高を下げる）
            dem_with_channel = current_dem - channel
            
            # 摂動を加える
            dem_ensemble[i] = dem_with_channel + perturbation
        
        return dem_ensemble
    
    def run_smoother(self, 
                    current_dem: np.ndarray,
                    water_history: Optional[np.ndarray] = None,
                    toponym_prob: Optional[np.ndarray] = None,
                    parameters: Optional[Dict[str, np.ndarray]] = None,
                    dem_ensemble: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Ensemble Smootherを実行する
        
        Parameters
        ----------
        current_dem : np.ndarray
            現在のDEM
        water_history : Optional[np.ndarray], optional
            水域履歴データ
        toponym_prob : Optional[np.ndarray], optional
            地名確率マップ
        parameters : Optional[Dict[str, np.ndarray]], optional
            パラメータのアンサンブル
        dem_ensemble : Optional[np.ndarray], optional
            初期DEMのアンサンブル
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            過去のDEMと古河道の存在確率
        """
        # パラメータのアンサンブルが指定されていない場合は生成
        if parameters is None:
            parameters = self.generate_parameter_ensemble()
        
        # 初期DEMのアンサンブルが指定されていない場合は生成
        if dem_ensemble is None:
            dem_ensemble = self.generate_initial_dem_ensemble(current_dem)
        
        # 実際の実装では、OpenDAとDelft3D-FMを連携させてEnsemble Smootherを実行します
        # ここではデモ用の簡易実装
        
        print("Running Ensemble Smoother...")
        print(f"Parameters: {list(parameters.keys())}")
        
        # DEMの形状
        ny, nx = current_dem.shape
        
        # 水域履歴データがない場合はダミーデータを生成
        if water_history is None:
            water_history = np.zeros((ny, nx))
            
            # 現在の河川チャネルに基づいて水域履歴を生成
            # 低い標高ほど水域になりやすい
            normalized_dem = (current_dem.max() - current_dem) / (current_dem.max() - current_dem.min())
            water_history = normalized_dem**2  # 二乗して差を強調
            
            # ノイズ追加
            np.random.seed(44)  # 再現性のため
            water_history += 0.2 * (np.random.rand(ny, nx) - 0.5)
            water_history = np.clip(water_history, 0, 1)  # 0-1の範囲に収める
        
        # 地名確率マップがない場合はダミーデータを生成
        if toponym_prob is None:
            toponym_prob = np.zeros((ny, nx))
            
            # 現在の河川チャネルに基づいて地名確率マップを生成
            # 低い標高ほど水関連地名が多い
            normalized_dem = (current_dem.max() - current_dem) / (current_dem.max() - current_dem.min())
            toponym_prob = normalized_dem**1.5  # 1.5乗して差を強調
            
            # ノイズ追加
            np.random.seed(45)  # 再現性のため
            toponym_prob += 0.15 * (np.random.rand(ny, nx) - 0.5)
            toponym_prob = np.clip(toponym_prob, 0, 1)  # 0-1の範囲に収める
        
        # コスト関数の重み
        dem_weight = self.config['assimilation']['cost_function'].get('dem_weight', 0.4)
        watermask_weight = self.config['assimilation']['cost_function'].get('watermask_weight', 0.3)
        toponym_weight = self.config['assimilation']['cost_function'].get('toponym_weight', 0.3)
        
        # 各アンサンブルメンバーのコスト計算
        costs = np.zeros(self.ensemble_size)
        
        for i in range(self.ensemble_size):
            # 1. 現在のDEMとの差（RMSE）
            dem_cost = np.sqrt(np.mean((dem_ensemble[i] - current_dem)**2))
            
            # 2. 水域履歴との一致度（IoU）
            # 低い標高ほど水域になりやすい
            normalized_dem = (dem_ensemble[i].max() - dem_ensemble[i]) / (dem_ensemble[i].max() - dem_ensemble[i].min())
            predicted_water = normalized_dem**2  # 二乗して差を強調
            
            # IoUの計算
            intersection = np.sum(np.minimum(predicted_water, water_history))
            union = np.sum(np.maximum(predicted_water, water_history))
            watermask_cost = 1 - (intersection / (union + 1e-10))
            
            # 3. 地名確率マップとの一致度
            # 低い標高ほど水関連地名が多い
            predicted_toponym = normalized_dem**1.5  # 1.5乗して差を強調
            
            # 地名確率マップとの差
            toponym_cost = np.mean(np.abs(predicted_toponym - toponym_prob))
            
            # 総コスト
            costs[i] = dem_weight * dem_cost + watermask_weight * watermask_cost + toponym_weight * toponym_cost
        
        # コストが最小のアンサンブルメンバーを選択
        best_idx = np.argmin(costs)
        best_dem = dem_ensemble[best_idx].copy()
        
        # 古河道の存在確率の計算
        # 各アンサンブルメンバーの低標高領域を集計
        paleo_channel_prob = np.zeros((ny, nx))
        
        for i in range(self.ensemble_size):
            # 低標高領域の抽出
            normalized_dem = (dem_ensemble[i].max() - dem_ensemble[i]) / (dem_ensemble[i].max() - dem_ensemble[i].min())
            
            # 閾値以上の領域を河道とみなす
            threshold = 0.7
            channel_mask = normalized_dem > threshold
            
            # 確率マップに加算
            paleo_channel_prob[channel_mask] += 1
        
        # 確率に変換
        paleo_channel_prob /= self.ensemble_size
        
        print("Ensemble Smoother completed successfully.")
        
        return best_dem, paleo_channel_prob
    
    def save_results(self, 
                    past_dem: np.ndarray,
                    paleo_channel_prob: np.ndarray,
                    output_dir: Optional[str] = None) -> None:
        """結果を保存する
        
        Parameters
        ----------
        past_dem : np.ndarray
            過去のDEM
        paleo_channel_prob : np.ndarray
            古河道の存在確率
        output_dir : Optional[str], optional
            出力ディレクトリ
        """
        if output_dir is None:
            output_dir = os.path.join(self.work_dir, "output")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 結果をxarrayデータセットに変換
        ny, nx = past_dem.shape
        
        # 座標の設定
        coords = {
            'y': np.arange(ny),
            'x': np.arange(nx)
        }
        
        # データ変数の設定
        data_vars = {
            'past_dem': (['y', 'x'], past_dem),
            'paleo_channel_prob': (['y', 'x'], paleo_channel_prob)
        }
        
        # データセットの作成
        ds = xr.Dataset(data_vars, coords=coords)
        
        # NetCDFとして保存
        output_file = os.path.join(output_dir, "smoother_results.nc")
        ds.to_netcdf(output_file)
        
        print(f"Results saved to {output_file}")
    
    def create_paleochannel_map(self, 
                               paleo_channel_prob: np.ndarray,
                               threshold: float = 0.5) -> np.ndarray:
        """古河道マップを作成する
        
        Parameters
        ----------
        paleo_channel_prob : np.ndarray
            古河道の存在確率
        threshold : float, optional
            閾値
            
        Returns
        -------
        np.ndarray
            古河道マップ（0: 非河道, 1: 河道）
        """
        # 閾値以上の領域を河道とみなす
        paleo_channel_map = (paleo_channel_prob >= threshold).astype(np.int8)
        
        return paleo_channel_map
