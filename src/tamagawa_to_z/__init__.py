"""
tamagawa-to-z: アマゾン古河道・集落探索フレームワーク

このパッケージは、アマゾン流域における古河道や集落跡の探索を支援するためのフレームワークです。
以下の3つの主要コンポーネントを統合しています：

1. 多言語トポニム解析（System-A）
2. データ同化シミュレーション（System-B）
3. マルチエージェントシステム
"""

__version__ = "0.1.0"

# サブパッケージのインポート
from tamagawa_to_z import harmonizer
from tamagawa_to_z import utils

# CLI関数のインポート
from tamagawa_to_z.cli import main

__all__ = [
    'harmonizer',
    'utils',
    'main'
]
