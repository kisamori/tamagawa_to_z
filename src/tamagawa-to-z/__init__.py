# DEMO FILE: tamagawa-to-z パッケージ初期化

"""
tamagawa-to-z: アマゾン古河道・集落探索フレームワーク

このパッケージは、多言語トポニム解析、水理学データ同化、マルチエージェントを
組み合わせたフレームワークを提供します。
"""

__version__ = "0.1.0"

# サブパッケージのインポート
from . import harmonizer
from . import hydro_da
from . import morph_da
from . import agents
from . import utils

# 便利な関数をトップレベルに公開
from .harmonizer import Harmonizer
from .hydro_da import HydraulicsDA
from .morph_da import MorphDA
