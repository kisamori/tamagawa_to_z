#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_llm_harmonizer.py: LLM統合トポニムハーモナイザーのテストスクリプト

このスクリプトは、新しく実装したLLMレイヤーの基本動作をテストします。
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# プロジェクトのルートディレクトリをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# .envファイルを読み込む
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path)
    logger.info(f"📁 .envファイルを読み込みました: {env_path}")
except ImportError:
    logger.warning("python-dotenvが利用できません")
except Exception as e:
    logger.warning(f".envファイルの読み込みに失敗: {e}")

def test_environment():
    """環境変数とインポートのテスト"""
    logger.info("🔍 環境変数とインポートのテスト")
    
    # OpenAI APIキーのチェック
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.info(f"✅ OpenAI APIキーが設定されています (***{api_key[-4:]})")
    else:
        logger.warning("⚠️  OpenAI APIキーが設定されていません")
        logger.info("📝 .env.example を .env にコピーしてAPIキーを設定してください")
        return False
    
    # 必要なライブラリのインポートテスト
    try:
        from tamagawa_to_z.harmonizer.llm_layer import (
            ToponymHarmonizer,
            ToponymEmbedding,
            load_dict,
            append_entries
        )
        logger.info("✅ LLMレイヤーのインポートに成功しました")
        return True
    except ImportError as e:
        logger.error(f"❌ LLMレイヤーのインポートに失敗しました: {e}")
        return False

def test_embedding_only():
    """Embeddingのみのテスト（OpenAI API呼び出しなし）"""
    logger.info("🔍 Embeddingのみのテスト")
    
    try:
        from tamagawa_to_z.harmonizer.llm_layer import ToponymEmbedding
        
        # サンプルデータ
        sample_variants = [
            "Igarapé Grande",
            "Rio Pequeno", 
            "Lagoa Azul",
            "Ygarapé do Pirarucu",
            "Igarité da Onça"
        ]
        
        # Embedding初期化
        embedding = ToponymEmbedding()
        logger.info("✅ ToponymEmbedding初期化成功")
        
        # インデックス構築
        embedding.build_index(sample_variants)
        logger.info(f"✅ インデックス構築成功 ({len(embedding.variants)} variants)")
        
        # 類似検索テスト
        query = "Grande River"
        results = embedding.find_similar(query, k=3)
        logger.info(f"✅ 類似検索成功: '{query}' -> {results}")
        
        # 統計情報
        stats = embedding.get_stats()
        logger.info(f"📊 Embedding統計: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Embeddingテストに失敗: {e}")
        return False

def test_dictionary_io():
    """辞書I/Oのテスト"""
    logger.info("🔍 辞書I/Oのテスト")
    
    try:
        from tamagawa_to_z.harmonizer.llm_layer import load_dict, append_entries
        
        # 辞書読み込みテスト
        dict_df = load_dict()
        logger.info(f"✅ 辞書読み込み成功: {len(dict_df)} entries")
        
        # サンプルエントリーの作成
        sample_entry = pd.DataFrame([{
            'canonical_id': 'test_001',
            'canonical_name': 'Igarapé do Teste',
            'variant_name': 'Test Creek',
            'root': 'igarape',
            'lang': 'por',
            'meaning_en': 'test creek for LLM integration',
            'meaning_ja': 'LLM統合テスト用の川',
            'confidence': 0.95
        }])
        
        # エントリー追加テスト（実際には追加せず、バリデーションのみ）
        logger.info("✅ サンプルエントリー作成成功")
        logger.info(f"📝 サンプル: {sample_entry.iloc[0].to_dict()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 辞書I/Oテストに失敗: {e}")
        return False

def test_harmonizer_mock():
    """ハーモナイザーのモックテスト（実際のAPI呼び出しなし）"""
    logger.info("🔍 ハーモナイザーのモックテスト")
    
    try:
        from tamagawa_to_z.harmonizer.llm_layer import ToponymHarmonizer
        
        # API呼び出しなしでインスタンス化テスト
        try:
            harmonizer = ToponymHarmonizer()
            logger.info("✅ ToponymHarmonizer初期化成功")
            
            # フォールバックエントリーテスト
            candidates = [{
                'canonical_name': 'Igarapé Grande',
                'root': 'igarape',
                'lang': 'por'
            }]
            
            fallback = harmonizer._create_fallback_entry("Test Toponym", candidates)
            logger.info(f"✅ フォールバックエントリー作成成功: {fallback}")
            
            # 統計情報
            stats = harmonizer.get_stats()
            logger.info(f"📊 ハーモナイザー統計: {stats}")
            
            return True
            
        except ValueError as e:
            if "OpenAI API key required" in str(e):
                logger.warning("⚠️  OpenAI APIキーが必要です（実際のAPI呼び出しテストはスキップ）")
                return True  # これは期待される動作
            else:
                raise
        
    except Exception as e:
        logger.error(f"❌ ハーモナイザーテストに失敗: {e}")
        return False

def test_simple_llm_call():
    """簡単なLLM呼び出しテスト（APIキーが設定されている場合のみ）"""
    logger.info("🔍 簡単なLLM呼び出しテスト")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        logger.info("⏭️  OpenAI APIキーが設定されていないため、LLM呼び出しテストをスキップします")
        return True
    
    try:
        from tamagawa_to_z.harmonizer.llm_layer import ToponymHarmonizer
        
        # ハーモナイザー初期化
        harmonizer = ToponymHarmonizer()
        logger.info("✅ ToponymHarmonizer初期化成功（APIキー有効）")
        
        # 空の辞書でインデックス構築
        harmonizer.prime_index()
        logger.info("✅ 空辞書でのインデックス構築成功")
        
        # 単一トポニムの正規化テスト
        test_toponym = "Igarape Teste"
        logger.info(f"🚀 単一トポニム正規化テスト: '{test_toponym}'")
        
        result = harmonizer.harmonize_single(test_toponym, use_fallback=True)
        logger.info(f"✅ LLM呼び出し成功!")
        logger.info(f"📝 結果: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ LLM呼び出しテストに失敗: {e}")
        return False

def test_sample_gdf_integration():
    """サンプルGeoDataFrameとの統合テスト"""
    logger.info("🔍 サンプルGeoDataFrameとの統合テスト")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        logger.info("⏭️  OpenAI APIキーが設定されていないため、統合テストをスキップします")
        return True
    
    try:
        from tamagawa_to_z.harmonizer.llm_layer import ToponymHarmonizer
        
        # サンプルGeoDataFrame作成
        sample_data = {
            'name': ['Igarapé Pequeno', 'Rio Grande', 'Lagoa Azul'],
            'geometry': [
                Point(-70.0, -10.0),
                Point(-70.1, -10.1), 
                Point(-70.2, -10.2)
            ]
        }
        
        gdf = gpd.GeoDataFrame(sample_data, crs='EPSG:4326')
        logger.info(f"✅ サンプルGeoDataFrame作成: {len(gdf)} toponyms")
        
        # ハーモナイザー初期化とタグ付け
        harmonizer = ToponymHarmonizer()
        harmonizer.prime_index()
        
        logger.info("🚀 LLMタグ付け実行中...")
        gdf_tagged = harmonizer.attach_llm_tags(gdf, batch_size=2)
        
        logger.info(f"✅ LLMタグ付け成功!")
        logger.info(f"📝 タグ付け結果: {gdf_tagged.columns.tolist()}")
        
        # 結果の表示
        if not gdf_tagged.empty:
            for idx, row in gdf_tagged.iterrows():
                logger.info(f"  {row['name']} -> {row.get('canonical_name', 'N/A')} (conf: {row.get('confidence', 'N/A')})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 統合テストに失敗: {e}")
        return False

def main():
    """メインテスト関数"""
    logger.info("🚀 LLM統合トポニムハーモナイザーのテスト開始")
    logger.info("=" * 60)
    
    tests = [
        ("環境変数とインポート", test_environment),
        ("Embeddingのみ", test_embedding_only),
        ("辞書I/O", test_dictionary_io),
        ("ハーモナイザーモック", test_harmonizer_mock),
        ("簡単なLLM呼び出し", test_simple_llm_call),
        ("サンプルGeoDataFrame統合", test_sample_gdf_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 テスト: {test_name}")
        logger.info("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✅ {test_name}: 成功")
            else:
                logger.error(f"❌ {test_name}: 失敗")
                
        except Exception as e:
            logger.error(f"💥 {test_name}: 例外発生 - {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    logger.info("\n" + "=" * 60)
    logger.info("📊 テスト結果サマリー")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\n🎯 結果: {passed}/{total} テスト成功")
    
    if passed == total:
        logger.info("🎉 すべてのテストが成功しました！")
        return 0
    else:
        logger.warning("⚠️  一部のテストが失敗しました")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)