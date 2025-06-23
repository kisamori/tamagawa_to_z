"""Tests for pipeline runner functionality."""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
import geopandas as gpd
from unittest.mock import Mock, patch
import uuid

from tamagawa_to_z.tuning.pipeline_runner import (
    run_pipeline_with_params,
    _calculate_mock_metrics,
    _calculate_composite_score,
    _extract_false_positives
)


@pytest.fixture
def sample_validation_set():
    """Sample validation GeoDataFrame for testing."""
    return gpd.GeoDataFrame({
        'site_name': ['Site_A', 'Site_B', 'Site_C'],
        'culture_tag': ['acre', 'acre', 'casarabe'],
        'discovery_year': [2015, 2018, 2019]
    }, geometry=[
        gpd.points_from_xy([-68.0], [-9.0])[0],
        gpd.points_from_xy([-68.2], [-9.2])[0],
        gpd.points_from_xy([-63.0], [-17.8])[0]
    ], crs="EPSG:4326")


@pytest.fixture
def sample_candidates():
    """Sample candidates DataFrame for testing."""
    return pd.DataFrame([
        {
            'name': 'candidate_001',
            'lon': -68.1, 'lat': -9.1,
            'normalized_name': 'candidate_001',
            'type': 'igarape',
            'dist_km': 2.5,
            'occ_pct': 3.2,
            'total_score': 0.75,
            'is_candidate': True
        },
        {
            'name': 'candidate_002', 
            'lon': -68.3, 'lat': -9.3,
            'normalized_name': 'candidate_002',
            'type': 'rio',
            'dist_km': 1.8,
            'occ_pct': 4.1,
            'total_score': 0.68,
            'is_candidate': True
        },
        {
            'name': 'candidate_003',
            'lon': -63.1, 'lat': -17.9,
            'normalized_name': 'candidate_003', 
            'type': 'lago',
            'dist_km': 4.2,
            'occ_pct': 2.8,
            'total_score': 0.45,
            'is_candidate': False
        }
    ])


class TestPipelineRunner:
    """Test cases for pipeline runner functionality."""
    
    def test_calculate_mock_metrics(self, sample_candidates, sample_validation_set):
        """Test mock metrics calculation."""
        metrics = _calculate_mock_metrics(sample_candidates, sample_validation_set)
        
        # Check that all expected metrics are present
        assert 'recall_100' in metrics
        assert 'recall_50' in metrics
        assert 'map_score' in metrics
        assert 'workload' in metrics
        
        # Check value ranges
        assert 0 <= metrics['recall_100'] <= 1
        assert 0 <= metrics['recall_50'] <= 1
        assert 0 <= metrics['map_score'] <= 1
        assert metrics['workload'] == len(sample_candidates)
        
        # recall_100 should be >= recall_50
        assert metrics['recall_100'] >= metrics['recall_50']
    
    def test_calculate_composite_score(self):
        """Test composite score calculation."""
        # Test with good metrics
        good_metrics = {
            'recall_100': 0.8,
            'map_score': 0.6,
            'workload': 100
        }
        
        score = _calculate_composite_score(good_metrics)
        assert isinstance(score, float)
        
        # Test with high workload (should penalize)
        high_workload_metrics = {
            'recall_100': 0.8,
            'map_score': 0.6,
            'workload': 10000
        }
        
        score_penalized = _calculate_composite_score(high_workload_metrics)
        assert score_penalized < score  # Higher workload should result in lower score
        
        # Test with missing metrics (should handle gracefully)
        incomplete_metrics = {
            'recall_100': 0.5
        }
        
        score_incomplete = _calculate_composite_score(incomplete_metrics)
        assert isinstance(score_incomplete, float)
    
    def test_extract_false_positives(self, sample_candidates, sample_validation_set):
        """Test false positive extraction."""
        fp_examples = _extract_false_positives(
            sample_candidates, 
            sample_validation_set,
            max_examples=10
        )
        
        assert isinstance(fp_examples, list)
        assert len(fp_examples) <= 10
        assert all(isinstance(name, str) for name in fp_examples)
    
    @patch('tamagawa_to_z.tuning.pipeline_runner._run_existing_pipeline')
    def test_run_pipeline_with_params_score_only(self, mock_pipeline, sample_validation_set):
        """Test pipeline execution returning score only."""
        # Setup mock pipeline to return sample candidates
        mock_candidates = pd.DataFrame([
            {'name': 'test_candidate', 'lon': -68.0, 'lat': -9.0, 
             'total_score': 0.8, 'is_candidate': True}
        ])
        mock_pipeline.return_value = mock_candidates
        
        # Test parameters
        test_params = {
            'distance_km': 2.5,
            'occ_pct': 5.0,
            'root_weights': {'igarape': 0.8, 'rio': 0.6}
        }
        
        # Run pipeline
        score = run_pipeline_with_params(
            **test_params,
            validation_set=sample_validation_set,
            return_fp=False
        )
        
        assert isinstance(score, float)
        assert -5.0 <= score <= 5.0  # Reasonable score range
        
        # Verify mock was called
        mock_pipeline.assert_called_once()
    
    @patch('tamagawa_to_z.tuning.pipeline_runner._run_existing_pipeline')
    def test_run_pipeline_with_params_with_fp(self, mock_pipeline, sample_validation_set):
        """Test pipeline execution returning score and false positives."""
        # Setup mock
        mock_candidates = pd.DataFrame([
            {'name': 'fp_candidate_1', 'lon': -68.0, 'lat': -9.0, 
             'total_score': 0.8, 'is_candidate': True},
            {'name': 'fp_candidate_2', 'lon': -68.1, 'lat': -9.1,
             'total_score': 0.7, 'is_candidate': True}
        ])
        mock_pipeline.return_value = mock_candidates
        
        test_params = {
            'distance_km': 2.5,
            'occ_pct': 5.0,
            'root_weights': {'igarape': 0.8}
        }
        
        # Run with FP return
        score, fp_examples = run_pipeline_with_params(
            **test_params,
            validation_set=sample_validation_set,
            return_fp=True
        )
        
        assert isinstance(score, float)
        assert isinstance(fp_examples, list)
        assert len(fp_examples) <= 50  # Default max_examples
    
    @patch('tamagawa_to_z.tuning.pipeline_runner._run_existing_pipeline')
    def test_pipeline_with_experiment_id(self, mock_pipeline, sample_validation_set):
        """Test pipeline execution with custom experiment ID."""
        mock_pipeline.return_value = pd.DataFrame([
            {'name': 'test', 'total_score': 0.5, 'is_candidate': True}
        ])
        
        custom_id = "test_experiment_123"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch Path to use temp directory
            with patch('tamagawa_to_z.tuning.pipeline_runner.Path') as mock_path:
                mock_path.return_value = Path(temp_dir) / "dummy_params.json"
                mock_path.return_value.parent.mkdir = Mock()
                mock_path.return_value.parent.mkdir.return_value = None
                
                score = run_pipeline_with_params(
                    distance_km=2.0,
                    occ_pct=4.0,
                    root_weights={'test': 0.5},
                    validation_set=sample_validation_set,
                    experiment_id=custom_id
                )
        
        assert isinstance(score, float)
        # Verify the experiment_id was passed to the pipeline
        call_args = mock_pipeline.call_args
        assert call_args[1] == custom_id  # Second argument should be experiment_id
    
    def test_pipeline_error_handling(self, sample_validation_set):
        """Test error handling in pipeline execution."""
        # Test with invalid parameters that might cause errors
        with patch('tamagawa_to_z.tuning.pipeline_runner._run_existing_pipeline') as mock_pipeline:
            mock_pipeline.side_effect = Exception("Pipeline failed")
            
            with pytest.raises(RuntimeError, match="Pipeline failed"):
                run_pipeline_with_params(
                    distance_km=2.5,
                    occ_pct=5.0,
                    root_weights={'test': 0.5},
                    validation_set=sample_validation_set
                )
    
    def test_config_overrides(self, sample_validation_set):
        """Test configuration override functionality."""
        with patch('tamagawa_to_z.tuning.pipeline_runner._run_existing_pipeline') as mock_pipeline:
            mock_pipeline.return_value = pd.DataFrame([
                {'name': 'test', 'total_score': 0.5, 'is_candidate': True}
            ])
            
            overrides = {
                'custom_param': 'custom_value',
                'another_param': 42
            }
            
            score = run_pipeline_with_params(
                distance_km=2.0,
                occ_pct=4.0,
                root_weights={'test': 0.5},
                validation_set=sample_validation_set,
                config_overrides=overrides
            )
            
            # Check that overrides were included in the call
            call_args = mock_pipeline.call_args[0]
            param_patch = call_args[0]
            
            assert 'custom_param' in param_patch
            assert param_patch['custom_param'] == 'custom_value'
            assert param_patch['another_param'] == 42


class TestMockPipeline:
    """Test cases for mock pipeline functionality."""
    
    def test_mock_pipeline_generation(self):
        """Test mock pipeline generates reasonable data."""
        from tamagawa_to_z.tuning.pipeline_runner import _run_mock_pipeline
        
        test_params = {
            'distance_threshold_km': 3.0,
            'occ_pct_threshold': 5.0
        }
        
        candidates = _run_mock_pipeline(test_params, "test_experiment")
        
        # Check basic structure
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) > 0
        
        # Check required columns
        required_cols = ['name', 'lon', 'lat', 'total_score', 'is_candidate']
        assert all(col in candidates.columns for col in required_cols)
        
        # Check data validity
        assert candidates['lon'].between(-75, -60).all()  # Amazon region
        assert candidates['lat'].between(-15, -5).all()   # Amazon region
        assert candidates['total_score'].between(0, 1).all()
        assert candidates['is_candidate'].dtype == bool
    
    def test_mock_pipeline_parameter_influence(self):
        """Test that mock pipeline responds to parameter changes."""
        from tamagawa_to_z.tuning.pipeline_runner import _run_mock_pipeline
        
        # Run with different parameters
        params1 = {'distance_threshold_km': 1.0, 'occ_pct_threshold': 2.0}
        params2 = {'distance_threshold_km': 5.0, 'occ_pct_threshold': 10.0}
        
        candidates1 = _run_mock_pipeline(params1, "test1")
        candidates2 = _run_mock_pipeline(params2, "test2")
        
        # Both should generate data
        assert len(candidates1) > 0
        assert len(candidates2) > 0
        
        # Should have different characteristics (due to different random seeds based on experiment_id)
        assert not candidates1['total_score'].equals(candidates2['total_score'])