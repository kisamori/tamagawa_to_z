#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
create_google_maps_visualization.py: Google Mapsで既知の遺跡と候補地を表示

このスクリプトは以下を行います：
1. 既知の遺跡（KMZファイルから読み込み）
2. 古河道候補地（CSVファイルから読み込み）
3. 両方をGoogle Mapsで表示（異なるスタイルで）
"""

import os
import sys
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import tempfile
import zipfile

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


def create_google_maps_html(existing_sites, candidates, output_path):
    """Google MapsのHTMLファイルを作成"""
    
    # 中心点を計算（全ポイントの平均）
    all_lats = [site['lat'] for site in existing_sites + candidates]
    all_lons = [site['lon'] for site in existing_sites + candidates]
    center_lat = sum(all_lats) / len(all_lats) if all_lats else -10.0
    center_lon = sum(all_lons) / len(all_lons) if all_lons else -67.0
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>古河道候補地と既知の遺跡</title>
    <meta charset="utf-8">
    <style>
        #map {{
            height: 100vh;
            width: 100%;
        }}
        .legend {{
            background: white;
            border: 1px solid #ccc;
            border-radius: 3px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            margin: 10px;
            padding: 10px;
            font-family: Arial, sans-serif;
            font-size: 12px;
        }}
        .legend h4 {{
            margin: 0 0 10px 0;
            font-size: 14px;
        }}
        .legend-item {{
            margin: 5px 0;
            display: flex;
            align-items: center;
        }}
        .legend-color {{
            width: 15px;
            height: 15px;
            border-radius: 50%;
            margin-right: 5px;
            border: 2px solid white;
            box-shadow: 0 0 2px rgba(0,0,0,0.3);
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <script>
        function initMap() {{
            const map = new google.maps.Map(document.getElementById("map"), {{
                zoom: 10,
                center: {{ lat: {center_lat}, lng: {center_lon} }},
                mapTypeId: 'hybrid'
            }});

            // 既知の遺跡マーカー
            const existingSites = {existing_sites};
            existingSites.forEach(site => {{
                const marker = new google.maps.Marker({{
                    position: {{ lat: site.lat, lng: site.lon }},
                    map: map,
                    title: site.name,
                    icon: {{
                        url: 'https://maps.google.com/mapfiles/ms/icons/red-dot.png',
                        scaledSize: new google.maps.Size(32, 32)
                    }}
                }});

                const infoWindow = new google.maps.InfoWindow({{
                    content: `
                        <div>
                            <h3>${{site.name}}</h3>
                            <p><strong>タイプ:</strong> 既知の遺跡</p>
                            <p><strong>座標:</strong> ${{site.lat.toFixed(6)}}, ${{site.lon.toFixed(6)}}</p>
                            <p>${{site.description}}</p>
                        </div>
                    `
                }});

                marker.addListener('click', () => {{
                    infoWindow.open(map, marker);
                }});
            }});

            // 古河道候補地マーカー
            const candidates = {candidates};
            candidates.forEach(candidate => {{
                // スコアに基づいて色を決定
                let color = '#FFD700'; // 低スコア（金色）
                if (candidate.score > 0.6) {{
                    color = '#FF4500'; // 高スコア（オレンジ赤）
                }} else if (candidate.score > 0.5) {{
                    color = '#FF8C00'; // 中スコア（オレンジ）
                }}

                const marker = new google.maps.Marker({{
                    position: {{ lat: candidate.lat, lng: candidate.lon }},
                    map: map,
                    title: candidate.name,
                    icon: {{
                        path: google.maps.SymbolPath.CIRCLE,
                        scale: Math.max(8, candidate.score * 15),
                        fillColor: color,
                        fillOpacity: 0.8,
                        strokeColor: '#FFFFFF',
                        strokeWeight: 2
                    }}
                }});

                const infoWindow = new google.maps.InfoWindow({{
                    content: `
                        <div>
                            <h3>${{candidate.name}}</h3>
                            <p><strong>タイプ:</strong> 古河道候補地</p>
                            <p><strong>スコア:</strong> ${{candidate.score.toFixed(3)}}</p>
                            <p><strong>水系タイプ:</strong> ${{candidate.water_type}}</p>
                            <p><strong>現河道からの距離:</strong> ${{candidate.distance_km.toFixed(1)}} km</p>
                            <p><strong>水域頻度:</strong> ${{candidate.water_freq.toFixed(1)}}%</p>
                            <p><strong>座標:</strong> ${{candidate.lat.toFixed(6)}}, ${{candidate.lon.toFixed(6)}}</p>
                        </div>
                    `
                }});

                marker.addListener('click', () => {{
                    infoWindow.open(map, marker);
                }});
            }});

            // 凡例を追加
            const legend = document.createElement('div');
            legend.className = 'legend';
            legend.innerHTML = `
                <h4>凡例</h4>
                <div class="legend-item">
                    <img src="https://maps.google.com/mapfiles/ms/icons/red-dot.png" width="20" height="20">
                    <span>既知の遺跡</span>
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
            `;
            
            map.controls[google.maps.ControlPosition.RIGHT_BOTTOM].push(legend);
        }}
    </script>
    
    <script async defer
        src="https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&callback=initMap">
    </script>
</body>
</html>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Google Maps可視化を作成')
    parser.add_argument('--output-dir', type=str, help='出力ディレクトリ（site_identification_xxxxxx形式）')
    parser.add_argument('--csv-path', type=str, help='候補地CSVファイルのパス')
    args = parser.parse_args()
    
    # ファイルパス
    kmz_path = PROJECT_ROOT / 'data/known/known_acre.kmz'
    
    # 出力ディレクトリが指定された場合
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_path = output_dir / 'google_maps_visualization.html'
        
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
        output_path = latest_dir / 'google_maps_visualization.html'
        
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
    
    print("=== Google Maps可視化作成開始 ===")
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
    print("Google Maps HTMLファイルを作成中...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    create_google_maps_html(existing_sites, candidates, output_path)
    
    print(f"✅ Google Maps可視化を作成しました: {output_path}")
    print("\n使用方法:")
    print("1. HTMLファイル内の 'YOUR_API_KEY' をGoogle Maps APIキーに置換してください")
    print("2. ブラウザでHTMLファイルを開いてください")
    print("\nGoogle Maps API キーの取得方法:")
    print("https://developers.google.com/maps/documentation/javascript/get-api-key")


if __name__ == "__main__":
    main()