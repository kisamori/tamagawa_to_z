"""
test_harmonizer: harmonizer モジュールのテスト

このモジュールは、harmonizer モジュールの機能をテストします。
"""

import os
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box

from tamagawa_to_z.harmonizer.preprocess import (
    normalize_name,
    infer_type,
    make_bbox_gdf,
    process_toponyms
)
from tamagawa_to_z.harmonizer.distance import (
    classify_by_distance
)
from tamagawa_to_z.harmonizer.watermask import (
    classify_by_occurrence,
    find_paleo_candidates
)


def test_normalize_name():
    """normalize_name 関数のテスト"""
    # 基本的な正規化
    assert normalize_name("Igarapé do Paxiúba") == "igarape do paxiuba"
    
    # アクセント除去
    assert normalize_name("Lagôa São João") == "lagoa sao joao"
    
    # 特殊文字の処理
    assert normalize_name("Rio-Açú (Grande)") == "rio acu grande"


def test_infer_type():
    """infer_type 関数のテスト"""
    # 水系タイプの推定
    assert infer_type("igarape do paxiuba") == "igarape"
    assert infer_type("lagoa grande") == "lagoa"
    assert infer_type("porto velho") == "porto"
    assert infer_type("igapo do rio") == "igapo"
    assert infer_type("baixio do sul") == "baixio"
    assert infer_type("furo do norte") == "furo"
    assert infer_type("parana do oeste") == "parana"
    
    # 水系タイプがない場合
    assert infer_type("rio amazonas") is None
    assert infer_type("cidade nova") is None


def test_make_bbox_gdf():
    """make_bbox_gdf 関数のテスト"""
    # BBoxの作成
    gdf = make_bbox_gdf()
    
    # 型の確認
    assert isinstance(gdf, gpd.GeoDataFrame)
    
    # 座標系の確認
    assert gdf.crs == "EPSG:4326"
    
    # ジオメトリの確認
    bbox = gdf.geometry.iloc[0]
    assert isinstance(bbox, box)
    
    # 境界の確認（デフォルトBBOX）
    bounds = bbox.bounds
    assert bounds == (-70.5, -11.5, -66.5, -8.5)


def test_process_toponyms():
    """process_toponyms 関数のテスト"""
    # テスト用データの作成
    data = {
        "name": ["Igarapé do Paxiúba", "Lagôa Grande", "Porto Velho"],
        "geometry": [
            Point(-69.5, -10.5),
            Point(-68.5, -9.5),
            Point(-67.5, -10.0)
        ],
        "source": ["bngb", "bngb", "osm"]
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    
    # 処理
    processed = process_toponyms(gdf)
    
    # 結果の確認
    assert "normalized_name" in processed.columns
    assert "type" in processed.columns
    
    # 正規化の確認
    assert processed["normalized_name"].iloc[0] == "igarape do paxiuba"
    
    # タイプ推定の確認
    assert processed["type"].iloc[0] == "igarape"
    assert processed["type"].iloc[1] == "lagoa"
    assert processed["type"].iloc[2] == "porto"



def test_classify_by_distance():
    """classify_by_distance 関数のテスト"""
    # テスト用データの作成
    data = {
        "name": ["Igarapé do Paxiúba", "Lagôa Grande", "Porto Velho", "Igarapé Pequeno"],
        "geometry": [
            Point(-69.5, -10.5),
            Point(-68.5, -9.5),
            Point(-67.5, -10.0),
            Point(-67.0, -9.0)
        ],
        "dist_km": [1.5, 2.8, 3.2, 5.7]
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    
    # 分類
    classified = classify_by_distance(gdf, threshold_km=3.0)
    
    # 結果の確認
    assert "distance_class" in classified.columns
    
    # 分類の確認
    assert classified["distance_class"].iloc[0] == "near"  # 1.5km
    assert classified["distance_class"].iloc[1] == "near"  # 2.8km
    assert classified["distance_class"].iloc[2] == "far"   # 3.2km
    assert classified["distance_class"].iloc[3] == "far"   # 5.7km


def test_classify_by_occurrence():
    """classify_by_occurrence 関数のテスト"""
    # テスト用データの作成
    data = {
        "name": ["Igarapé do Paxiúba", "Lagôa Grande", "Porto Velho", "Igarapé Pequeno"],
        "geometry": [
            Point(-69.5, -10.5),
            Point(-68.5, -9.5),
            Point(-67.5, -10.0),
            Point(-67.0, -9.0)
        ],
        "occ_pct": [2.3, 4.8, 7.2, 12.5]
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    
    # 分類
    classified = classify_by_occurrence(gdf, threshold_pct=5.0)
    
    # 結果の確認
    assert "water_class" in classified.columns
    
    # 分類の確認
    assert classified["water_class"].iloc[0] == "dry"  # 2.3%
    assert classified["water_class"].iloc[1] == "dry"  # 4.8%
    assert classified["water_class"].iloc[2] == "wet"  # 7.2%
    assert classified["water_class"].iloc[3] == "wet"  # 12.5%


def test_find_paleo_candidates():
    """find_paleo_candidates 関数のテスト"""
    # テスト用データの作成
    data = {
        "name": ["Igarapé do Paxiúba", "Lagôa Grande", "Porto Velho", "Igarapé Pequeno"],
        "geometry": [
            Point(-69.5, -10.5),
            Point(-68.5, -9.5),
            Point(-67.5, -10.0),
            Point(-67.0, -9.0)
        ],
        "dist_km": [1.5, 2.8, 3.2, 5.7],
        "occ_pct": [2.3, 4.8, 7.2, 12.5]
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    
    # 候補地点の抽出
    candidates = find_paleo_candidates(gdf, dist_threshold=3.0, occ_threshold=5.0)
    
    # 結果の確認
    assert len(candidates) == 1
    assert candidates["name"].iloc[0] == "Igarapé Pequeno"
    assert candidates["dist_km"].iloc[0] == 5.7
    assert candidates["occ_pct"].iloc[0] == 12.5
    assert candidates["is_candidate"].iloc[0] == True
