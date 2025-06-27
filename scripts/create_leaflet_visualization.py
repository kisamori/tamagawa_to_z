#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
create_leaflet_visualization.py: Leaflet + OpenStreetMapで既知の遺跡と候補地を表示

Google Maps APIキーが不要な版
"""

import os
import sys
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import tempfile
import zipfile
import json

# プロジェクトのルートディレクトリをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def extract_kmz_to_kml(kmz_path):
    """KMZファイルからKMLファイルを抽出"""
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(kmz_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            kml_path = os.path.join(temp_dir, 'doc.kml')
            if os.path.exists(kml_path):
                with open(kml_path, 'r', encoding='utf-8') as f:
                    return f.read()
    return None


def parse_kml_placemarks(kml_content):
    """KMLファイルからプレースマークを解析"""
    root = ET.fromstring(kml_content)
    
    # 名前空間を処理
    namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    placemarks = []
    for placemark in root.findall('.//kml:Placemark', namespace):
        name_elem = placemark.find('kml:name', namespace)
        desc_elem = placemark.find('kml:description', namespace)
        point_elem = placemark.find('.//kml:Point/kml:coordinates', namespace)
        
        if name_elem is not None and point_elem is not None:
            name = name_elem.text
            description = desc_elem.text if desc_elem is not None else ""
            coords = point_elem.text.strip()
            
            # 座標を解析 (lon,lat,alt形式)
            coord_parts = coords.split(',')
            if len(coord_parts) >= 2:
                try:
                    lon = float(coord_parts[0])
                    lat = float(coord_parts[1])
                    placemarks.append({
                        'name': name,
                        'description': description,
                        'lat': lat,
                        'lon': lon,
                        'type': 'existing_site'
                    })
                except ValueError:
                    continue
    
    return placemarks


def load_paleochannel_candidates(csv_path):
    """古河道候補地をCSVから読み込み"""
    candidates = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # geometry列からPOINT(lon lat)を解析
            geometry_str = row['geometry']
            match = re.search(r'POINT \(([^)]+)\)', geometry_str)
            if match:
                coords = match.group(1).split()
                if len(coords) >= 2:
                    try:
                        lon = float(coords[0])
                        lat = float(coords[1])
                        candidates.append({
                            'name': row['name'],
                            'lat': lat,
                            'lon': lon,
                            'type': 'candidate',
                            'score': float(row.get('total_score', 0)),
                            'distance_km': float(row.get('dist_km', 0)),
                            'water_freq': float(row.get('occ_pct', 0)),
                            'water_type': row.get('type', 'unknown')
                        })
                    except ValueError:
                        continue
    
    return candidates


def create_leaflet_html(existing_sites, candidates, output_path):
    """LeafletのHTMLファイルを作成"""
    
    # 中心点を計算（全ポイントの平均）
    all_lats = [site['lat'] for site in existing_sites + candidates]
    all_lons = [site['lon'] for site in existing_sites + candidates]
    center_lat = sum(all_lats) / len(all_lats) if all_lats else -10.0
    center_lon = sum(all_lons) / len(all_lons) if all_lons else -67.0
    
    # JavaScriptに渡すためのデータをJSON形式に変換
    existing_sites_json = json.dumps(existing_sites, ensure_ascii=False)
    candidates_json = json.dumps(candidates, ensure_ascii=False)
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>古河道候補地と既知の遺跡 - Leaflet版</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
        crossorigin=""/>
    
    <style>
        #map {{
            height: 100vh;
            width: 100%;
        }}
        .legend {{
            background: white;
            border: 1px solid #ccc;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            padding: 10px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            line-height: 18px;
        }}
        .legend h4 {{
            margin: 0 0 5px 0;
            font-size: 14px;
        }}
        .legend-item {{
            margin: 3px 0;
            display: flex;
            align-items: center;
        }}
        .legend-color {{
            width: 18px;
            height: 18px;
            border-radius: 50%;
            margin-right: 8px;
            border: 2px solid white;
            box-shadow: 0 0 2px rgba(0,0,0,0.5);
        }}
        .existing-site-icon {{
            background-color: #ff0000;
            width: 12px;
            height: 12px;
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <!-- Leaflet JavaScript -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
        crossorigin=""></script>
    
    <script>
        // 地図を初期化
        const map = L.map('map').setView([{center_lat}, {center_lon}], 10);

        // OpenStreetMapタイルレイヤーを追加
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }}).addTo(map);

        // 既知の遺跡データ
        const existingSites = {existing_sites_json};
        
        // 古河道候補地データ
        const candidates = {candidates_json};

        // アイコン定義
        const existingSiteIcon = L.icon({{
            iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34],
            shadowSize: [41, 41]
        }});

        // 既知の遺跡マーカーを追加
        existingSites.forEach(site => {{
            const marker = L.marker([site.lat, site.lon], {{
                icon: existingSiteIcon
            }}).addTo(map);

            marker.bindPopup(`
                <div>
                    <h3>${{site.name}}</h3>
                    <p><strong>タイプ:</strong> 既知の遺跡</p>
                    <p><strong>座標:</strong> ${{site.lat.toFixed(6)}}, ${{site.lon.toFixed(6)}}</p>
                    <p>${{site.description}}</p>
                </div>
            `);
        }});

        // 古河道候補地マーカーを追加
        candidates.forEach(candidate => {{
            // スコアに基づいて色を決定
            let color = '#FFD700'; // 低スコア（金色）
            let radius = 8;
            if (candidate.score > 0.6) {{
                color = '#FF4500'; // 高スコア（オレンジ赤）
                radius = 12;
            }} else if (candidate.score > 0.5) {{
                color = '#FF8C00'; // 中スコア（オレンジ）
                radius = 10;
            }}

            const marker = L.circleMarker([candidate.lat, candidate.lon], {{
                radius: radius,
                fillColor: color,
                color: '#ffffff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }}).addTo(map);

            marker.bindPopup(`
                <div>
                    <h3>${{candidate.name}}</h3>
                    <p><strong>タイプ:</strong> 古河道候補地</p>
                    <p><strong>スコア:</strong> ${{candidate.score.toFixed(3)}}</p>
                    <p><strong>水系タイプ:</strong> ${{candidate.water_type}}</p>
                    <p><strong>現河道からの距離:</strong> ${{candidate.distance_km.toFixed(1)}} km</p>
                    <p><strong>水域頻度:</strong> ${{candidate.water_freq.toFixed(1)}}%</p>
                    <p><strong>座標:</strong> ${{candidate.lat.toFixed(6)}}, ${{candidate.lon.toFixed(6)}}</p>
                </div>
            `);
        }});

        // 凡例を追加
        const legend = L.control({{position: 'bottomright'}});
        legend.onAdd = function (map) {{
            const div = L.DomUtil.create('div', 'legend');
            div.innerHTML = `
                <h4>凡例</h4>
                <div class="legend-item">
                    <div class="existing-site-icon"></div>
                    <span>既知の遺跡 (${{existingSites.length}}件)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #FF4500;"></div>
                    <span>高スコア候補地 (>0.6)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #FF8C00;"></div>
                    <span>中スコア候補地 (0.5-0.6)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #FFD700;"></div>
                    <span>低スコア候補地 (<0.5)</span>
                </div>
                <div style="margin-top: 5px; font-size: 10px; color: #666;">
                    候補地合計: ${{candidates.length}}件
                </div>
            `;
            return div;
        }};
        legend.addTo(map);

        // 地図にすべてのマーカーが表示されるように調整
        if (existingSites.length > 0 || candidates.length > 0) {{
            const group = new L.featureGroup();
            
            existingSites.forEach(site => {{
                group.addLayer(L.marker([site.lat, site.lon]));
            }});
            
            candidates.forEach(candidate => {{
                group.addLayer(L.circleMarker([candidate.lat, candidate.lon]));
            }});
            
            map.fitBounds(group.getBounds().pad(0.1));
        }}
    </script>
</body>
</html>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Leaflet可視化を作成')
    parser.add_argument('--output-dir', type=str, help='出力ディレクトリ（site_identification_xxxxxx形式）')
    parser.add_argument('--csv-path', type=str, help='候補地CSVファイルのパス')
    args = parser.parse_args()
    
    # ファイルパス
    kmz_path = PROJECT_ROOT / 'data/known/known_acre.kmz'
    
    # 出力ディレクトリが指定された場合
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_path = output_dir / 'leaflet_visualization.html'
        
        # CSVパスが指定されていない場合、デフォルトまたは同じディレクトリから探す
        if args.csv_path:
            csv_path = Path(args.csv_path)
        else:
            # 同じディレクトリまたはデフォルトから探す
            csv_candidates = [
                PROJECT_ROOT / 'data/output/candidates/paleochannel_candidates.csv',
                PROJECT_ROOT / 'data/interim/paleochannel_candidates.csv'
            ]
            csv_path = None
            for candidate in csv_candidates:
                if candidate.exists():
                    csv_path = candidate
                    break
            
            if csv_path is None:
                print(f"エラー: 候補地CSVファイルが見つかりません")
                return
    else:
        # 最新のsite_identificationディレクトリを自動検出
        plots_dir = PROJECT_ROOT / 'data/plots'
        site_dirs = [d for d in plots_dir.glob('site_identification_*') if d.is_dir()]
        
        if not site_dirs:
            print("エラー: site_identification_*ディレクトリが見つかりません")
            return
        
        # 最新のディレクトリを選択（名前順でソート）
        latest_dir = sorted(site_dirs)[-1]
        output_path = latest_dir / 'leaflet_visualization.html'
        
        # CSVファイルは従来の場所から読み込み
        csv_candidates = [
            PROJECT_ROOT / 'data/output/candidates/paleochannel_candidates.csv',
            PROJECT_ROOT / 'data/interim/paleochannel_candidates.csv'
        ]
        csv_path = None
        for candidate in csv_candidates:
            if candidate.exists():
                csv_path = candidate
                break
        
        if csv_path is None:
            print(f"エラー: 候補地CSVファイルが見つかりません")
            return
    
    print("=== Leaflet可視化作成開始 ===")
    print(f"出力先: {output_path}")
    print(f"CSVファイル: {csv_path}")
    
    # 既知の遺跡を読み込み
    print("KMZファイルから既知の遺跡を読み込み中...")
    kml_content = extract_kmz_to_kml(kmz_path)
    if kml_content is None:
        print(f"エラー: {kmz_path} からKMLファイルを抽出できませんでした")
        return
    
    existing_sites = parse_kml_placemarks(kml_content)
    print(f"既知の遺跡 {len(existing_sites)} 件を読み込みました")
    
    # 古河道候補地を読み込み
    print("CSVファイルから古河道候補地を読み込み中...")
    if not os.path.exists(csv_path):
        print(f"エラー: {csv_path} が見つかりません")
        return
    
    candidates = load_paleochannel_candidates(csv_path)
    print(f"古河道候補地 {len(candidates)} 件を読み込みました")
    
    # HTMLファイルを作成
    print("Leaflet HTMLファイルを作成中...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    create_leaflet_html(existing_sites, candidates, output_path)
    
    print(f"✅ Leaflet可視化を作成しました: {output_path}")
    print("\n使用方法:")
    print("ブラウザでHTMLファイルを開いてください（APIキー不要です）")
    print(f"ファイルの場所: {output_path}")


if __name__ == "__main__":
    main()