"""
Tests for LLM Layer Multilingual Toponym Harmonization

多言語トポニムLLMレイヤーのテスト
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from tamagawa_to_z.harmonizer.llm_layer import (
    dictionary_io,
    agent_schema,
    ToponymEmbedding,
    ToponymHarmonizer
)


class TestDictionaryIO:
    """辞書I/Oテスト"""
    
    def test_load_empty_dict(self):
        """空の辞書読み込みテスト"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # ファイルが存在しない場合
            temp_path.unlink()
            df = dictionary_io.load_dict(temp_path)
            assert df.empty
            assert list(df.columns) == dictionary_io.COLUMNS
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_append_entries(self):
        """エントリー追加テスト"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # テストデータ
            df_new = pd.DataFrame([{
                'canonical_id': 'test_001',
                'canonical_name': 'Igarapé do Teste',
                'variant_name': 'Test Creek',
                'root': 'igarape',
                'lang': 'por',
                'meaning_en': 'test creek',
                'meaning_ja': 'テスト川',
                'confidence': 0.95
            }])
            
            # 追加
            dictionary_io.append_entries(df_new, temp_path, create_backup=False)
            
            # 確認
            df_loaded = dictionary_io.load_dict(temp_path)
            assert len(df_loaded) == 1
            assert df_loaded.iloc[0]['canonical_name'] == 'Igarapé do Teste'
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_duplicate_removal(self):
        """重複除去テスト"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # 重複データ
            df1 = pd.DataFrame([{
                'canonical_id': 'test_001',
                'canonical_name': 'Igarapé do Teste',
                'variant_name': 'Test Creek',
                'confidence': 0.90
            }])
            
            df2 = pd.DataFrame([{
                'canonical_id': 'test_001',
                'canonical_name': 'Igarapé do Teste Updated',
                'variant_name': 'Test Creek',
                'confidence': 0.95
            }])
            
            # 追加
            dictionary_io.append_entries(df1, temp_path, create_backup=False)
            dictionary_io.append_entries(df2, temp_path, create_backup=False)
            
            # 確認（重複除去され、最新が保持される）
            df_loaded = dictionary_io.load_dict(temp_path)
            assert len(df_loaded) == 1
            assert df_loaded.iloc[0]['canonical_name'] == 'Igarapé do Teste Updated'
            
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestAgentSchema:
    """エージェントスキーマテスト"""
    
    def test_validate_response_valid(self):
        """正常レスポンス検証テスト"""
        response = {
            'canonical_id': 'test_001',
            'canonical_name': 'Igarapé do Teste',
            'relation': 'same',
            'confidence': 0.95,
            'reasoning': 'Same toponym with minor spelling variation'
        }
        
        errors = agent_schema.validate_response(response)
        assert errors == {}
    
    def test_validate_response_invalid(self):
        """異常レスポンス検証テスト"""
        response = {
            'canonical_id': 'test_001',
            'relation': 'invalid_relation',
            'confidence': 1.5  # 範囲外
        }
        
        errors = agent_schema.validate_response(response)
        assert 'canonical_name' in errors  # 必須フィールド不足
        assert 'relation' in errors  # 無効な値
        assert 'confidence' in errors  # 範囲外
    
    def test_create_user_prompt(self):
        """ユーザープロンプト生成テスト"""
        candidates = [{
            'canonical_name': 'Igarapé Grande',
            'variant_name': 'Grande Creek',
            'root': 'igarape',
            'lang': 'por',
            'confidence': 0.90,
            'meaning_en': 'large creek'
        }]
        
        prompt = agent_schema.create_user_prompt('Grande Stream', candidates)
        assert 'Grande Stream' in prompt
        assert 'Igarapé Grande' in prompt
        assert 'igarape' in prompt


@pytest.mark.skipif(
    not os.getenv('OPENAI_API_KEY'),
    reason="OpenAI API key not available"
)
class TestToponymEmbedding:
    """トポニムEmbeddingテスト（依存関係が利用可能な場合のみ）"""
    
    def test_embedding_initialization(self):
        """Embedding初期化テスト"""
        try:
            embedding = ToponymEmbedding()
            assert embedding.model is not None
            assert embedding.normalize_text("Igarapé São João") == "igarape sao joao"
        except ImportError:
            pytest.skip("Required dependencies not available")
    
    def test_build_index(self):
        """インデックス構築テスト"""
        try:
            embedding = ToponymEmbedding()
            variants = ["Igarapé Grande", "Rio Pequeno", "Lagoa Azul"]
            
            embedding.build_index(variants)
            
            assert len(embedding.variants) == len(variants)
            assert embedding.embeddings is not None
            assert embedding.index is not None
            
        except ImportError:
            pytest.skip("Required dependencies not available")
    
    def test_find_similar(self):
        """類似検索テスト"""
        try:
            embedding = ToponymEmbedding()
            variants = ["Igarapé Grande", "Rio Grande", "Lagoa Grande"]
            embedding.build_index(variants)
            
            results = embedding.find_similar("Grande River", k=2)
            assert len(results) <= 2
            assert all(isinstance(score, float) for _, score in results)
            
        except ImportError:
            pytest.skip("Required dependencies not available")


class TestToponymHarmonizerMocked:
    """ToponymHarmonizerのモックテスト（API呼び出しなし）"""
    
    @patch('tamagawa_to_z.harmonizer.llm_layer.harmonize.OpenAI')
    @patch('tamagawa_to_z.harmonizer.llm_layer.harmonize.ToponymEmbedding')
    def test_harmonizer_initialization(self, mock_embedding, mock_openai):
        """ハーモナイザー初期化テスト"""
        # OpenAI APIキーをモック
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            harmonizer = ToponymHarmonizer()
            assert harmonizer.model == "gpt-4o-mini"
            assert harmonizer.max_retries == 3
    
    @patch('tamagawa_to_z.harmonizer.llm_layer.harmonize.OpenAI')
    @patch('tamagawa_to_z.harmonizer.llm_layer.harmonize.ToponymEmbedding')
    def test_create_fallback_entry(self, mock_embedding, mock_openai):
        """フォールバックエントリー作成テスト"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            harmonizer = ToponymHarmonizer()
            
            candidates = [{
                'root': 'igarape',
                'lang': 'por'
            }]
            
            result = harmonizer._create_fallback_entry("Test Toponym", candidates)
            
            assert result['canonical_name'] == "Test Toponym"
            assert result['relation'] == 'different'
            assert result['root'] == 'igarape'
            assert result['lang'] == 'por'
            assert result['confidence'] == 0.1


def test_schema_info():
    """スキーマ情報テスト"""
    info = agent_schema.SCHEMA_INFO
    
    assert 'function_schema' in info
    assert 'system_prompt' in info
    assert 'valid_relations' in info
    assert 'valid_languages' in info
    
    assert set(info['valid_relations']) == {'same', 'similar', 'different'}
    assert set(info['valid_languages']) == {'por', 'tup', 'mixed', 'unknown'}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])