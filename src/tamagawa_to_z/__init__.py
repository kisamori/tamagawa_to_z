# DEMO FILE: tamagawa_to_z パッケージ初期化

"""
tamagawa_to_z: アマゾン古河道・集落探索フレームワーク

このパッケージは、多言語トポニム解析、水理学データ同化、マルチエージェントを
組み合わせたフレームワークを提供します。
"""

__version__ = "0.1.0"

# サブパッケージのインポート
from . import harmonizer

# 依存関係が不足している可能性のあるモジュールは条件付きでインポート
try:
    from . import utils
except ImportError:
    pass

try:
    from . import hydro_da
    from .hydro_da import HydraulicsDA
except ImportError:
    pass

try:
    from . import morph_da
    from .morph_da import MorphDA
except ImportError:
    pass

try:
    from . import agents
except ImportError:
    pass

# 便利な関数をトップレベルに公開（条件付き）
try:
    from .harmonizer import Harmonizer
except ImportError:
    pass
