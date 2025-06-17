"""
harmonizer: 多言語トポニム解析モジュール

このパッケージは、地名データから水関連の地名を抽出・分類し、
水関連確率マップを生成するための機能を提供します。
"""

from tamagawa_to_z.harmonizer.preprocess import (
    normalize_name,
    infer_type,
    make_bbox_gdf,
    process_toponyms
)

from tamagawa_to_z.harmonizer.distance import (
    attach_distance,
    find_nearest_river,
    classify_by_distance
)

from tamagawa_to_z.harmonizer.watermask import (
    water_occurrence,
    buffer_occurrence,
    classify_by_occurrence,
    find_paleo_candidates
)

from tamagawa_to_z.harmonizer.agent import (
    filter_candidates,
    score_candidates
)

# メインクラスのインポート（条件付き）
try:
    from tamagawa_to_z.harmonizer.harmonizer import Harmonizer
    _has_harmonizer = True
except ImportError:
    _has_harmonizer = False

# __all__リストの作成
__all__ = [
    # preprocess
    'normalize_name',
    'infer_type',
    'make_bbox_gdf',
    'process_toponyms',
    
    # distance
    'attach_distance',
    'find_nearest_river',
    'classify_by_distance',
    
    # watermask
    'water_occurrence',
    'buffer_occurrence',
    'classify_by_occurrence',
    'find_paleo_candidates',
    
    # candidate
    'filter_candidates',
    'score_candidates'
]

# Harmonizerクラスが利用可能な場合は__all__に追加
if _has_harmonizer:
    __all__.append('Harmonizer')
