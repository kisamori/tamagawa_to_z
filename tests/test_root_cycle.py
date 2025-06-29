"""
Tests for the water vocabulary root cycle implementation

水語彙ルートサイクルのテスト
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import re
import sys
import os

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.tamagawa_to_z.harmonizer.llm_layer.root_io import (
    load_roots, append_roots, build_water_regex, 
    validate_root_entry, format_root_for_csv
)


class TestRootCycle(unittest.TestCase):
    """水語彙ルートサイクルのテストクラス"""
    
    def setUp(self):
        """テスト用の一時ディレクトリとファイルを設定"""
        self.test_dir = tempfile.mkdtemp()
        self.test_csv_path = Path(self.test_dir) / "test_water_roots.csv"
        
        # テスト用のCSVデータを作成
        test_data = pd.DataFrame([
            {
                "root": "igarape",
                "lang": "tup", 
                "regex_token": "igarap[eé]",
                "meaning_en": "small stream",
                "meaning_ja": "小川"
            },
            {
                "root": "lagoa",
                "lang": "por",
                "regex_token": "lagoa", 
                "meaning_en": "lagoon",
                "meaning_ja": "潟"
            }
        ])
        test_data.to_csv(self.test_csv_path, index=False)
        
        # root_io.pyのROOT_PATHを一時的に変更
        import src.tamagawa_to_z.harmonizer.llm_layer.root_io as root_io
        self.original_path = root_io.ROOT_PATH
        root_io.ROOT_PATH = self.test_csv_path
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        # 元のパスを復元
        import src.tamagawa_to_z.harmonizer.llm_layer.root_io as root_io
        root_io.ROOT_PATH = self.original_path
        
        # 一時ディレクトリを削除
        shutil.rmtree(self.test_dir)
    
    def test_load_roots(self):
        """roots CSVの読み込みテスト"""
        df = load_roots()
        
        self.assertEqual(len(df), 2)
        self.assertIn("igarape", df["root"].values)
        self.assertIn("lagoa", df["root"].values)
        self.assertEqual(df.loc[df["root"] == "igarape", "lang"].iloc[0], "tup")
    
    def test_append_roots(self):
        """新しい語根の追記テスト"""
        new_roots = pd.DataFrame([
            {
                "root": "rio",
                "lang": "por",
                "regex_token": "rio",
                "meaning_en": "river", 
                "meaning_ja": "川"
            }
        ])
        
        # 追記前の件数
        df_before = load_roots()
        count_before = len(df_before)
        
        # 新語根を追記
        append_roots(new_roots)
        
        # 追記後の確認
        df_after = load_roots()
        self.assertEqual(len(df_after), count_before + 1)
        self.assertIn("rio", df_after["root"].values)
    
    def test_append_roots_duplicates(self):
        """重複語根の追記テスト（重複排除）"""
        duplicate_roots = pd.DataFrame([
            {
                "root": "igarape",  # 既存の語根
                "lang": "tup",
                "regex_token": "igarap[eé]_new",  # 異なるregex
                "meaning_en": "updated meaning",
                "meaning_ja": "更新された意味"
            }
        ])
        
        # 追記前の件数
        count_before = len(load_roots())
        
        # 重複語根を追記
        append_roots(duplicate_roots)
        
        # 件数は増えないが、内容は更新される
        df_after = load_roots()
        self.assertEqual(len(df_after), count_before)
        
        # 更新された内容を確認
        igarape_row = df_after[df_after["root"] == "igarape"].iloc[0]
        self.assertEqual(igarape_row["regex_token"], "igarap[eé]_new")
        self.assertEqual(igarape_row["meaning_en"], "updated meaning")
    
    def test_build_water_regex(self):
        """水語彙Regexの構築テスト"""
        regex = build_water_regex()
        
        # 正規表現オブジェクトが返されることを確認
        self.assertIsInstance(regex, re.Pattern)
        
        # パターンマッチングの確認
        self.assertTrue(regex.search("Igarapé Pequeno"))  # 大文字小文字無視
        self.assertTrue(regex.search("lagoa azul"))       # 小文字
        self.assertFalse(regex.search("montanha"))        # 非水語彙
    
    def test_validate_root_entry(self):
        """語根エントリーの妥当性検証テスト"""
        # 有効なエントリー
        valid_entry = {
            "root": "porto",
            "regex_token": "porto",
            "lang": "por",
            "meaning_en": "port"
        }
        errors = validate_root_entry(valid_entry)
        self.assertEqual(len(errors), 0)
        
        # 無効なエントリー（必須フィールド不足）
        invalid_entry = {
            "root": "",  # 空の語根
            "lang": "por"
            # regex_tokenが不足
        }
        errors = validate_root_entry(invalid_entry)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("root" in error for error in errors))
        self.assertTrue(any("regex_token" in error for error in errors))
        
        # 無効な正規表現
        invalid_regex = {
            "root": "test",
            "regex_token": "[invalid",  # 閉じ括弧なし
        }
        errors = validate_root_entry(invalid_regex)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("regex" in error for error in errors))
    
    def test_format_root_for_csv(self):
        """CSV保存用フォーマットテスト"""
        raw_entry = {
            "root": "  yaru  ",  # 前後に空白
            "lang": "araw",
            "regex_token": "yaru",
            "meaning_en": "pool",
            "meaning_ja": "水溜まり",
            "extra_field": "ignored"  # 余分なフィールド
        }
        
        formatted = format_root_for_csv(raw_entry)
        
        # 必要なフィールドのみ含まれる
        expected_fields = {"root", "lang", "regex_token", "meaning_en", "meaning_ja"}
        self.assertEqual(set(formatted.keys()), expected_fields)
        
        # 空白がトリミングされる
        self.assertEqual(formatted["root"], "yaru")
        
        # 余分なフィールドは除外される
        self.assertNotIn("extra_field", formatted)
    
    def test_regex_pattern_integration(self):
        """Regexパターンの統合テスト（新語根追加→パターン更新）"""
        # 新しい語根を追加
        new_root = pd.DataFrame([{
            "root": "parana",
            "lang": "tup",
            "regex_token": "paran[aá]",
            "meaning_en": "large river",
            "meaning_ja": "大河"
        }])
        append_roots(new_root)
        
        # 更新されたRegexを構築
        regex = build_water_regex()
        
        # 新しい語根がパターンに含まれることを確認
        self.assertTrue(regex.search("Paraná"))
        self.assertTrue(regex.search("parana"))
        
        # 既存の語根も引き続き機能
        self.assertTrue(regex.search("igarapé"))
        self.assertTrue(regex.search("lagoa"))
    
    def test_empty_csv_handling(self):
        """空CSVファイルの処理テスト"""
        # 空のCSVを作成
        empty_csv = Path(self.test_dir) / "empty_roots.csv"
        pd.DataFrame(columns=["root", "lang", "regex_token", "meaning_en", "meaning_ja"]).to_csv(
            empty_csv, index=False
        )
        
        # パスを一時的に変更
        import src.tamagawa_to_z.harmonizer.llm_layer.root_io as root_io
        original_path = root_io.ROOT_PATH
        root_io.ROOT_PATH = empty_csv
        
        try:
            # 空のCSVからのRegex構築はエラーになるべき
            with self.assertRaises(ValueError):
                build_water_regex()
        finally:
            root_io.ROOT_PATH = original_path


class TestRootCycleIntegration(unittest.TestCase):
    """エンドツーエンドの統合テスト"""
    
    def test_full_cycle_simulation(self):
        """完全なサイクルのシミュレーション"""
        # 1. 初期状態：基本的な語根のみ
        initial_roots = ["igarape", "lagoa"]
        
        # 2. LLMが新しい語根を発見したと仮定
        llm_discovered_root = {
            "root": "ressaca",
            "lang": "por", 
            "regex_token": "ressaca",
            "meaning_en": "backwater",
            "meaning_ja": "後背水域"
        }
        
        # 3. 語根検証
        errors = validate_root_entry(llm_discovered_root)
        self.assertEqual(len(errors), 0, "新発見の語根は有効であるべき")
        
        # 4. CSV保存用フォーマット
        formatted = format_root_for_csv(llm_discovered_root)
        self.assertIn("root", formatted)
        self.assertIn("regex_token", formatted)
        
        # 5. Regexパターンテスト
        # この部分は実際のファイル操作なしでロジック確認のみ
        regex_pattern = r'(?i)\b(' + '|'.join([
            "igarap[eé]", "lagoa", formatted["regex_token"]
        ]) + r')\b'
        
        compiled_regex = re.compile(regex_pattern)
        self.assertTrue(compiled_regex.search("Vila Ressaca"))
        self.assertTrue(compiled_regex.search("Igarapé Novo"))


if __name__ == "__main__":
    unittest.main()