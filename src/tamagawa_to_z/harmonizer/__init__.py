"""
harmonizer: 多言語トポニム解析モジュール

このパッケージは、地名データから水関連の地名を抽出・分類し、
水関連確率マップを生成するための機能を提供します。
"""

from tamagawa_to_z.harmonizer.preprocess import (
    normalize_name,
    infer_type,
    make_bbox_gdf,
    collect_names,
    collect_osm_names,
    merge_toponyms,
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
    filter_with_agent,
    batch_filter_with_agent,
    score_candidates
)

__all__ = [
    # preprocess
    'normalize_name',
    'infer_type',
    'make_bbox_gdf',
    'collect_names',
    'collect_osm_names',
    'merge_toponyms',
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
    
    # agent
    'filter_with_agent',
    'batch_filter_with_agent',
    'score_candidates'
]
