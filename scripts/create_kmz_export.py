#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
create_kmz_export.py: 古河道候補地と既知の遺跡をKMZ形式でエクスポート

このスクリプトは以下を行います：
1. 既知の遺跡（KMZファイルから読み込み）
2. 古河道候補地（CSVファイルから読み込み）
3. 両方をKMZ形式で出力（Google Maps/Google Earthで読み込み可能）
"""

import os
import sys
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import tempfile
import zipfile
from datetime import datetime
import argparse

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


def create_kml_content(existing_sites, candidates, experiment_id="export"):
    """KMLコンテンツを作成"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # KMLのルート要素
    kml = ET.Element('kml', xmlns='http://www.opengis.net/kml/2.2')
    document = ET.SubElement(kml, 'Document')
    
    # ドキュメント情報
    name_elem = ET.SubElement(document, 'name')
    name_elem.text = f'古河道候補地 - {experiment_id}'
    
    description_elem = ET.SubElement(document, 'description')
    description_elem.text = f'''
古河道候補地特定システムの結果
生成日時: {timestamp}
古河道候補地: {len(candidates)} 件
'''
    
    # スタイル定義
    # 古河道候補地用スタイル（スコア別）
    # 高スコア（>0.6）
    high_style = ET.SubElement(document, 'Style', id='candidate_high_style')
    icon_style = ET.SubElement(high_style, 'IconStyle')
    icon_scale = ET.SubElement(icon_style, 'scale')
    icon_scale.text = '1.0'
    icon_color = ET.SubElement(icon_style, 'color')
    icon_color.text = 'ff0000ff'  # 赤色
    icon_elem = ET.SubElement(icon_style, 'Icon')
    icon_href = ET.SubElement(icon_elem, 'href')
    icon_href.text = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    
    # 中スコア（0.5-0.6）
    mid_style = ET.SubElement(document, 'Style', id='candidate_mid_style')
    icon_style = ET.SubElement(mid_style, 'IconStyle')
    icon_scale = ET.SubElement(icon_style, 'scale')
    icon_scale.text = '0.8'
    icon_color = ET.SubElement(icon_style, 'color')
    icon_color.text = 'ff0080ff'  # オレンジ色
    icon_elem = ET.SubElement(icon_style, 'Icon')
    icon_href = ET.SubElement(icon_elem, 'href')
    icon_href.text = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    
    # 低スコア（<0.5）
    low_style = ET.SubElement(document, 'Style', id='candidate_low_style')
    icon_style = ET.SubElement(low_style, 'IconStyle')
    icon_scale = ET.SubElement(icon_style, 'scale')
    icon_scale.text = '0.6'
    icon_color = ET.SubElement(icon_style, 'color')
    icon_color.text = 'ff00ffff'  # 黄色
    icon_elem = ET.SubElement(icon_style, 'Icon')
    icon_href = ET.SubElement(icon_elem, 'href')
    icon_href.text = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    
    # フォルダ: 古河道候補地
    if candidates:
        candidates_folder = ET.SubElement(document, 'Folder')
        folder_name = ET.SubElement(candidates_folder, 'name')
        folder_name.text = f'候補地 ({len(candidates)} 件)'
        folder_desc = ET.SubElement(candidates_folder, 'description')
        folder_desc.text = 'AI分析により特定された古河道候補地点'
        
        # 全候補地を一つのフォルダに配置（スコア別分類なし）
        for candidate in candidates:
            placemark = ET.SubElement(candidates_folder, 'Placemark')
            
            name_elem = ET.SubElement(placemark, 'name')
            name_elem.text = f"候補地 (スコア: {candidate['score']:.3f})"
            
            description_elem = ET.SubElement(placemark, 'description')
            description_elem.text = f'''
<![CDATA[
<h3>候補地</h3>
<p><strong>元の名前:</strong> {candidate['name']}</p>
<p><strong>タイプ:</strong> 古河道候補地</p>
<p><strong>スコア:</strong> {candidate['score']:.3f}</p>
<p><strong>水系タイプ:</strong> {candidate['water_type']}</p>
<p><strong>現河道からの距離:</strong> {candidate['distance_km']:.1f} km</p>
<p><strong>水域頻度:</strong> {candidate['water_freq']:.1f}%</p>
<p><strong>座標:</strong> {candidate['lat']:.6f}, {candidate['lon']:.6f}</p>
]]>
'''
            
            # スコアに基づいてスタイルを決定
            if candidate['score'] > 0.6:
                style_id = 'candidate_high_style'
            elif candidate['score'] >= 0.5:
                style_id = 'candidate_mid_style'
            else:
                style_id = 'candidate_low_style'
            
            style_url = ET.SubElement(placemark, 'styleUrl')
            style_url.text = f'#{style_id}'
            
            point = ET.SubElement(placemark, 'Point')
            coordinates = ET.SubElement(point, 'coordinates')
            coordinates.text = f"{candidate['lon']},{candidate['lat']},0"
    
    # XMLを整形
    return ET.tostring(kml, encoding='unicode', method='xml')


def create_kmz_file(kml_content, output_path):
    """KMLコンテンツからKMZファイルを作成"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # KMLファイルを一時ディレクトリに作成
        kml_path = os.path.join(temp_dir, 'doc.kml')
        with open(kml_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(kml_content)
        
        # KMZファイルを作成（KMLファイルをZIP圧縮）
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as kmz_file:
            kmz_file.write(kml_path, 'doc.kml')


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='KMZ可視化を作成')
    parser.add_argument('--output-dir', type=str, help='出力ディレクトリ')
    parser.add_argument('--csv-path', type=str, help='候補地CSVファイルのパス')
    parser.add_argument('--experiment-id', type=str, default='export', help='実験ID')
    parser.add_argument('--known-sites', type=str, 
                       default=str(PROJECT_ROOT / 'data/known/known_acre.kmz'),
                       help='既知の遺跡KMZファイルのパス')
    args = parser.parse_args()
    
    print("=== KMZ可視化作成開始 ===")
    
    # 出力パス設定
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_path = output_dir / f'{args.experiment_id}_visualization.kmz'
    else:
        output_path = Path(f'{args.experiment_id}_visualization.kmz')
    
    # CSVパスが指定されていない場合、デフォルトから探す
    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
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
    
    print(f"出力先: {output_path}")
    print(f"CSVファイル: {csv_path}")
    
    # 既知の遺跡は含めない（候補地のみ出力）
    existing_sites = []
    
    # 古河道候補地を読み込み
    candidates = []
    if csv_path.exists():
        print("CSVファイルから古河道候補地を読み込み中...")
        candidates = load_paleochannel_candidates(csv_path)
        print(f"古河道候補地 {len(candidates)} 件を読み込みました")
    else:
        print(f"エラー: {csv_path} が見つかりません")
        return
    
    # KMLコンテンツを作成
    print("KMLコンテンツを作成中...")
    kml_content = create_kml_content(existing_sites, candidates, args.experiment_id)
    
    # KMZファイルを作成
    print("KMZファイルを作成中...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    create_kmz_file(kml_content, output_path)
    
    print(f"✅ KMZ可視化を作成しました: {output_path}")
    print("\n使用方法:")
    print("1. Google Maps (https://mymaps.google.com/) にアクセス")
    print("2. '新しい地図を作成' をクリック")
    print("3. 'インポート' をクリックしてKMZファイルをアップロード")
    print("4. または Google Earth でKMZファイルを直接開く")
    
    # 統計情報を表示
    if candidates:
        high_score = len([c for c in candidates if c['score'] > 0.6])
        mid_score = len([c for c in candidates if 0.5 <= c['score'] <= 0.6])
        low_score = len([c for c in candidates if c['score'] < 0.5])
        
        print(f"\n📊 候補地統計:")
        print(f"  高スコア (>0.6): {high_score} 件")
        print(f"  中スコア (0.5-0.6): {mid_score} 件")
        print(f"  低スコア (<0.5): {low_score} 件")
        print(f"  合計: {len(candidates)} 件")
    else:
        print("\n⚠️  候補地データが見つかりません")


if __name__ == "__main__":
    main()