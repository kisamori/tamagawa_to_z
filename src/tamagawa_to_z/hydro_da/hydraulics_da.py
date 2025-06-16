# DEMO FILE: HydraulicsDA メインクラス

"""
HydraulicsDA: 水理学データ同化のメインクラス

このクラスは、Delft3D-FMとOpenDAを用いた水理学データ同化のための
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


class HydraulicsDA:
    """水理学データ同化のメインクラス
    
    Delft3D-FMとOpenDAを用いた水理学データ同化を実行するためのクラスです。
    
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
                 ensemble_size: int = 50):
        """HydraulicsDAの初期化
        
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
        
        print(f"HydraulicsDA initialized with ensemble size: {ensemble_size}")
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
                'bin_dir': '/opt/delft3d/bin'
            },
            'parameters': {
                'Manning_n': {
                    'prior': 'lognormal(0.035,0.25)',
                    'min': 0.01,
                    'max': 0.1
                },
                'Ks': {
                    'prior': 'uniform(1e-6,1e-4)',
                    'min': 1e-7,
                    'max': 1e-3
                },
                'Q_factor': {
                    'prior': 'normal(1.0,0.2)',
                    'min': 0.5,
                    'max': 1.5
                }
            },
            'assimilation': {
                'scheme': 'EnKF',
                'ensemble_size': self.ensemble_size,
                'cadence': '6h',
                'observations': [
                    {'stage': 'ANA_gauge'},
                    {'watermask': 'Sentinel2'}
                ],
                'localization': {
                    'type': 'Gaspari-Cohn',
                    'horizontal_scale': 5000  # m
                }
            },
            'output': {
                'dir': 'output/enkf',
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
                   dem_file: str,
                   boundary_file: Optional[str] = None,
                   observation_file: Optional[str] = None) -> None:
        """モデルをセットアップする
        
        Parameters
        ----------
        dem_file : str
            DEMファイルのパス
        boundary_file : Optional[str], optional
            境界条件ファイルのパス
        observation_file : Optional[str], optional
            観測データファイルのパス
        """
        # 実際の実装では、Delft3D-FMのセットアップを行います
        # ここではデモ用の簡易実装
        
        print(f"Setting up model with DEM: {dem_file}")
        if boundary_file:
            print(f"Boundary conditions: {boundary_file}")
        if observation_file:
            print(f"Observation data: {observation_file}")
        
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
    
    def run_enkf(self, 
                parameters: Optional[Dict[str, np.ndarray]] = None,
                start_time: Optional[datetime] = None,
                end_time: Optional[datetime] = None,
                time_step: Optional[timedelta] = None) -> Tuple[Dict[str, np.ndarray], xr.Dataset]:
        """EnKFを実行する
        
        Parameters
        ----------
        parameters : Optional[Dict[str, np.ndarray]], optional
            パラメータのアンサンブル
        start_time : Optional[datetime], optional
            開始時刻
        end_time : Optional[datetime], optional
            終了時刻
        time_step : Optional[timedelta], optional
            時間ステップ
            
        Returns
        -------
        Tuple[Dict[str, np.ndarray], xr.Dataset]
            更新されたパラメータとシミュレーション結果
        """
        # パラメータのアンサンブルが指定されていない場合は生成
        if parameters is None:
            parameters = self.generate_parameter_ensemble()
        
        # 時間設定
        if start_time is None:
            start_time = datetime(2025, 1, 1)
        
        if end_time is None:
            end_time = start_time + timedelta(days=10)
        
        if time_step is None:
            time_step = timedelta(hours=1)
        
        # 時間配列の生成
        time_steps = []
        current_time = start_time
        while current_time <= end_time:
            time_steps.append(current_time)
            current_time += time_step
        
        # 実際の実装では、OpenDAとDelft3D-FMを連携させてEnKFを実行します
        # ここではデモ用の簡易実装
        
        print(f"Running EnKF from {start_time} to {end_time}")
        print(f"Time step: {time_step}")
        print(f"Parameters: {list(parameters.keys())}")
        
        # 簡易的なシミュレーション結果の生成
        # 実際の実装では、Delft3D-FMの実行結果を使用
        
        # 格子サイズ
        ny, nx = 100, 100
        
        # 水深の初期化
        water_depth = np.zeros((len(time_steps), self.ensemble_size, ny, nx))
        
        # 簡易的なシミュレーション
        for t, time in enumerate(time_steps):
            for i in range(self.ensemble_size):
                # 簡易的な水深計算
                # 実際はDelft3D-FMの実行結果
                
                # 基本的な水深パターン
                x = np.linspace(0, 1, nx)
                y = np.linspace(0, 1, ny)
                X, Y = np.meshgrid(x, y)
                
                # 時間変化
                time_factor = np.sin(2 * np.pi * t / len(time_steps))
                
                # 河川チャネル
                channel = 2 * np.exp(-100 * (Y - 0.5 + 0.1 * np.sin(10 * X))**2)
                
                # パラメータの影響
                param_effect = 1.0
                for param_name, param_values in parameters.items():
                    if param_name == 'Manning_n':
                        param_effect *= param_values[i] / 0.035
                    elif param_name == 'Ks':
                        param_effect *= param_values[i] / 5e-5
                    elif param_name == 'Q_factor':
                        param_effect *= param_values[i]
                
                # 水深計算
                water_depth[t, i] = channel * param_effect * (1 + 0.2 * time_factor) + 0.1 * np.random.randn(ny, nx)
                water_depth[t, i] = np.maximum(water_depth[t, i], 0)  # 負の水深を0に
        
        # 観測データの生成（実際のプロジェクトでは実測データを使用）
        # 真のパラメータに近いものを「観測」とする
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        channel = 2 * np.exp(-100 * (Y - 0.5 + 0.1 * np.sin(10 * X))**2)
        
        # 観測点の選択
        np.random.seed(43)  # 再現性のため
        obs_points = np.random.choice(ny * nx, 10, replace=False)  # 10観測点
        obs_y, obs_x = np.unravel_index(obs_points, (ny, nx))
        
        # 観測値の生成
        obs_values = np.zeros((len(time_steps), len(obs_points)))
        for t, time in enumerate(time_steps):
            time_factor = np.sin(2 * np.pi * t / len(time_steps))
            true_water_depth = channel * (1 + 0.2 * time_factor) + 0.05 * np.random.randn(ny, nx)
            true_water_depth = np.maximum(true_water_depth, 0)
            obs_values[t] = true_water_depth[obs_y, obs_x]
        
        # EnKFの実行（簡易版）
        # 実際はOpenDAが行う処理
        
        # 最終時刻のアンサンブル平均
        mean_water_depth = np.mean(water_depth[-1], axis=0)
        
        # 観測点での予測値
        predicted_values = np.array([water_depth[-1, i, obs_y, obs_x] for i in range(self.ensemble_size)])
        
        # カルマンゲインの計算（簡易版）
        innovation = obs_values[-1] - np.mean(predicted_values, axis=0)
        
        # アンサンブルの更新
        for i in range(self.ensemble_size):
            # 観測点での更新
            for j in range(len(obs_points)):
                y, x = obs_y[j], obs_x[j]
                water_depth[-1, i, y, x] += 0.7 * (obs_values[-1, j] - predicted_values[i, j])
        
        # パラメータの更新（簡易版）
        rmse_before = np.sqrt(np.mean((mean_water_depth[obs_y, obs_x] - obs_values[-1])**2))
        mean_water_depth_after = np.mean(water_depth[-1], axis=0)
        rmse_after = np.sqrt(np.mean((mean_water_depth_after[obs_y, obs_x] - obs_values[-1])**2))
        
        # パラメータの調整係数
        adjust_factor = rmse_before / (rmse_after + 1e-10)
        
        # パラメータの更新
        updated_parameters = {}
        for param_name, param_values in parameters.items():
            updated_parameters[param_name] = param_values * adjust_factor
        
        # 結果をxarrayデータセットに変換
        # 実際の実装では、Delft3D-FMの出力をxarrayに変換
        
        # 座標の設定
        coords = {
            'time': time_steps,
            'ensemble': np.arange(self.ensemble_size),
            'y': np.arange(ny),
            'x': np.arange(nx)
        }
        
        # データ変数の設定
        data_vars = {
            'water_depth': (['time', 'ensemble', 'y', 'x'], water_depth)
        }
        
        # データセットの作成
        ds = xr.Dataset(data_vars, coords=coords)
        
        print("EnKF completed successfully.")
        print(f"Updated parameters: {list(updated_parameters.keys())}")
        
        return updated_parameters, ds
    
    def save_results(self, 
                    results: xr.Dataset,
                    output_dir: Optional[str] = None) -> None:
        """シミュレーション結果を保存する
        
        Parameters
        ----------
        results : xr.Dataset
            シミュレーション結果
        output_dir : Optional[str], optional
            出力ディレクトリ
        """
        if output_dir is None:
            output_dir = os.path.join(self.work_dir, "output")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # NetCDFとして保存
        output_file = os.path.join(output_dir, "enkf_results.nc")
        results.to_netcdf(output_file)
        
        print(f"Results saved to {output_file}")
    
    def extract_water_depth(self, 
                           results: xr.Dataset,
                           time_index: int = -1,
                           ensemble_index: Optional[int] = None) -> np.ndarray:
        """水深データを抽出する
        
        Parameters
        ----------
        results : xr.Dataset
            シミュレーション結果
        time_index : int, optional
            時間インデックス
        ensemble_index : Optional[int], optional
            アンサンブルインデックス
            
        Returns
        -------
        np.ndarray
            水深データ
        """
        if ensemble_index is None:
            # アンサンブル平均
            return results['water_depth'].isel(time=time_index).mean(dim='ensemble').values
        else:
            # 特定のアンサンブルメンバー
            return results['water_depth'].isel(time=time_index, ensemble=ensemble_index).values
