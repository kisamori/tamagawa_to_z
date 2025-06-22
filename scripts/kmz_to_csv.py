#!/usr/bin/env python3
"""KMZファイルをCSVに変換するスクリプト

KMZファイルの内容を解析し、ファイル名、場所名、緯度、経度のCSVファイルとして出力します。

Usage:
    python scripts/kmz_to_csv.py data/known/known_acre.kmz data/known/known_acre.csv
"""

import argparse
import csv
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


def extract_kmz_to_csv(kmz_path: str, output_csv: str):
    """KMZファイルからCSVファイルを生成する"""
    
    kmz_file = Path(kmz_path)
    csv_file = Path(output_csv)
    
    if not kmz_file.exists():
        raise FileNotFoundError(f"KMZファイルが見つかりません: {kmz_path}")
    
    # CSVファイルの準備
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    
    # KMZファイルを開く（ZIPファイルとして）
    with zipfile.ZipFile(kmz_file, 'r') as kmz:
        # KMLファイルを探す
        kml_files = [name for name in kmz.namelist() if name.endswith('.kml')]
        
        if not kml_files:
            raise ValueError("KMZファイル内にKMLファイルが見つかりません")
        
        # 最初のKMLファイルを読み込む
        kml_content = kmz.read(kml_files[0]).decode('utf-8')
    
    # XMLを解析
    root = ET.fromstring(kml_content)
    
    # 名前空間の定義
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    # CSVファイルに書き込み
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # ヘッダー行を書き込み
        writer.writerow(['kmz_filename', 'place_name', 'latitude', 'longitude'])
        
        # Placemarkを探して処理
        placemarks = root.findall('.//kml:Placemark', ns)
        
        for placemark in placemarks:
            # 場所名を取得
            name_element = placemark.find('kml:name', ns)
            place_name = name_element.text if name_element is not None else ''
            
            # 座標を取得
            coordinates_element = placemark.find('.//kml:coordinates', ns)
            if coordinates_element is not None:
                coords_text = coordinates_element.text.strip()
                # 座標は "経度,緯度,高度" の形式
                coords_parts = coords_text.split(',')
                if len(coords_parts) >= 2:
                    try:
                        longitude = float(coords_parts[0])
                        latitude = float(coords_parts[1])
                        
                        # CSVに行を追加
                        writer.writerow([
                            kmz_file.name,
                            place_name,
                            latitude,
                            longitude
                        ])
                    except ValueError:
                        print(f"警告: 座標の変換に失敗しました - {place_name}: {coords_text}")
    
    print(f"✅ CSVファイルを生成しました: {csv_file}")
    print(f"   元ファイル: {kmz_file}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="KMZファイルをCSVに変換します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "kmz_file",
        help="変換するKMZファイルのパス"
    )
    
    parser.add_argument(
        "output_csv",
        help="出力するCSVファイルのパス"
    )
    
    args = parser.parse_args()
    
    try:
        extract_kmz_to_csv(args.kmz_file, args.output_csv)
        print("✅ 変換が完了しました")
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())