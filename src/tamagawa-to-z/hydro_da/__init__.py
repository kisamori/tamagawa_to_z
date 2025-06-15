# DEMO FILE: hydro_da パッケージ初期化

"""
hydro_da: 水理学データ同化モジュール

このモジュールは、Delft3D-FMとOpenDAを用いた水理学データ同化のための
機能を提供します。
"""

# メインクラスのインポート
from .hydraulics_da import HydraulicsDA

__all__ = ['HydraulicsDA']
