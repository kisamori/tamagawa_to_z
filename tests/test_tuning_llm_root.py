"""Tests for LLM root weight inference functionality."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from tamagawa_to_z.tuning.llm_root import (
    get_root_weights, 
    clear_cache, 
    get_cache_info,
    create_mock_weights,
    _generate_cache_key,
    _parse_llm_response
)


class TestLLMRootWeights:
    """Test cases for LLM root weight functionality."""
    
    def test_generate_cache_key(self):
        """Test cache key generation is deterministic."""
        context1 = {
            "distance_km": 2.5,
            "occ_pct": 5.0,
            "toponym_stats": {"igarape": 10, "rio": 5},
            "false_pos": ["site1", "site2"]
        }
        
        context2 = context1.copy()
        
        # Same context should generate same key
        key1 = _generate_cache_key(context1, "gpt-4o-mini", 0.0)
        key2 = _generate_cache_key(context2, "gpt-4o-mini", 0.0)
        assert key1 == key2
        
        # Different context should generate different key
        context2["distance_km"] = 3.0
        key3 = _generate_cache_key(context2, "gpt-4o-mini", 0.0)
        assert key1 != key3
    
    def test_parse_llm_response_valid_json(self):
        """Test parsing valid LLM JSON response."""
        valid_response = """{
            "weights": {
                "igarape": 0.8,
                "rio": 0.6,
                "lago": 0.4
            }
        }"""
        
        weights = _parse_llm_response(valid_response)
        
        assert len(weights) == 3
        assert weights["igarape"] == 0.8
        assert weights["rio"] == 0.6
        assert weights["lago"] == 0.4
    
    def test_parse_llm_response_with_markdown(self):
        """Test parsing LLM response wrapped in markdown code blocks."""
        markdown_response = """```json
        {
            "weights": {
                "igarape": 0.9,
                "parana": 0.7
            }
        }
        ```"""
        
        weights = _parse_llm_response(markdown_response)
        
        assert len(weights) == 2
        assert weights["igarape"] == 0.9
        assert weights["parana"] == 0.7
    
    def test_parse_llm_response_weight_clamping(self):
        """Test that out-of-range weights are clamped."""
        response_with_bad_weights = """{
            "weights": {
                "root1": 1.5,
                "root2": 0.05,
                "root3": 0.5
            }
        }"""
        
        weights = _parse_llm_response(response_with_bad_weights)
        
        # Should clamp 1.5 to 1.0 and 0.05 to 0.1
        assert weights["root1"] == 1.0
        assert weights["root2"] == 0.1
        assert weights["root3"] == 0.5
    
    def test_parse_llm_response_invalid_json(self):
        """Test error handling for invalid JSON."""
        invalid_response = "This is not JSON at all"
        
        with pytest.raises(ValueError, match="Invalid JSON response"):
            _parse_llm_response(invalid_response)
    
    def test_parse_llm_response_missing_weights_key(self):
        """Test error handling for missing weights key."""
        missing_weights_response = """{
            "data": {
                "igarape": 0.8
            }
        }"""
        
        with pytest.raises(ValueError, match="Response missing 'weights' key"):
            _parse_llm_response(missing_weights_response)
    
    def test_create_mock_weights(self):
        """Test mock weight generation."""
        weights = create_mock_weights(5)
        
        assert len(weights) == 5
        assert all(0.1 <= w <= 1.0 for w in weights.values())
        assert all(isinstance(k, str) for k in weights.keys())
        assert all(isinstance(v, float) for v in weights.values())
    
    def test_cache_functionality(self):
        """Test cache operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = temp_dir
            
            # Clear cache initially
            cleared = clear_cache(cache_dir)
            
            # Get cache info
            info = get_cache_info(cache_dir)
            assert info["entry_count"] == 0
            assert info["cache_dir"] == cache_dir
    
    @patch('tamagawa_to_z.tuning.llm_root.OpenAI')
    def test_get_root_weights_with_mock_api(self, mock_openai_class):
        """Test get_root_weights with mocked OpenAI API."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """{
            "weights": {
                "igarape": 0.8,
                "rio": 0.6
            }
        }"""
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test context
        context = {
            "distance_km": 2.5,
            "occ_pct": 5.0,
            "toponym_stats": {"igarape": 10, "rio": 5},
            "false_pos": []
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            weights = get_root_weights(
                context=context,
                cache_dir=temp_dir
            )
            
            assert len(weights) == 2
            assert weights["igarape"] == 0.8
            assert weights["rio"] == 0.6
            
            # Verify API was called
            mock_client.chat.completions.create.assert_called_once()
            
            # Test caching - second call should not hit API
            mock_client.reset_mock()
            weights2 = get_root_weights(
                context=context,
                cache_dir=temp_dir
            )
            
            assert weights == weights2
            mock_client.chat.completions.create.assert_not_called()
    
    @patch('tamagawa_to_z.tuning.llm_root.OpenAI')
    def test_get_root_weights_api_error_fallback(self, mock_openai_class):
        """Test fallback behavior when OpenAI API fails."""
        # Setup mock to raise exception
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        context = {
            "distance_km": 2.5,
            "occ_pct": 5.0,
            "toponym_stats": {"igarape": 10},
            "false_pos": []
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should raise the exception (no fallback in this version)
            with pytest.raises(Exception, match="API Error"):
                get_root_weights(context=context, cache_dir=temp_dir)
    
    def test_context_validation(self):
        """Test that various context formats are handled gracefully."""
        base_context = {
            "distance_km": 2.5,
            "occ_pct": 5.0,
            "toponym_stats": {"igarape": 10},
            "false_pos": ["site1"]
        }
        
        # Test with empty false_pos
        context_empty_fp = base_context.copy()
        context_empty_fp["false_pos"] = []
        
        # Should not raise error
        key = _generate_cache_key(context_empty_fp, "gpt-4o-mini", 0.0)
        assert isinstance(key, str)
        
        # Test with large false_pos list
        context_large_fp = base_context.copy()
        context_large_fp["false_pos"] = [f"site_{i}" for i in range(100)]
        
        key2 = _generate_cache_key(context_large_fp, "gpt-4o-mini", 0.0)
        assert isinstance(key2, str)
        assert key != key2  # Different contexts should have different keys


class TestIntegration:
    """Integration tests for LLM root functionality."""
    
    def test_full_workflow_with_mock(self):
        """Test complete workflow with mocked API."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with mock weights (no API call)
            mock_weights = create_mock_weights(3)
            
            assert len(mock_weights) == 3
            assert all(0.1 <= w <= 1.0 for w in mock_weights.values())
            
            # Test cache operations
            initial_info = get_cache_info(temp_dir)
            assert initial_info["entry_count"] == 0
            
            # After clearing empty cache
            cleared = clear_cache(temp_dir)
            assert cleared == 0
            
            # Cache info should still show 0 entries
            final_info = get_cache_info(temp_dir)
            assert final_info["entry_count"] == 0