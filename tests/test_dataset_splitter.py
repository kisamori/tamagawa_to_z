"""Tests for dataset splitter functionality."""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
import geopandas as gpd
import yaml

from tamagawa_to_z.dataset.splitter import DataSplitter, create_sample_master_csv


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "use_train_split": True,
        "discovery_year_threshold": 2020,
        "val_ratio": 0.3,
        "culture_blocks": {
            "acre": {"sites": ["Site_A", "Site_B"]},
            "casarabe": {"sites": ["Site_C", "Site_D"]}
        },
        "buffer_radius_km": 0.5,
        "negative_ratio": 2,
        "target_crs": "EPSG:4326",
        "random_state": 42,
        "stratify_by_culture": True
    }


@pytest.fixture
def sample_sites_data():
    """Sample sites data for testing."""
    return pd.DataFrame([
        {"site_name": "Site_A", "lat": -9.0, "lon": -68.0, "culture_tag": "acre", "discovery_year": 2015},
        {"site_name": "Site_B", "lat": -9.2, "lon": -68.2, "culture_tag": "acre", "discovery_year": 2018},
        {"site_name": "Site_C", "lat": -17.8, "lon": -63.0, "culture_tag": "casarabe", "discovery_year": 2019},
        {"site_name": "Site_D", "lat": -17.6, "lon": -62.8, "culture_tag": "casarabe", "discovery_year": 2021},
        {"site_name": "Site_E", "lat": -2.5, "lon": -54.9, "culture_tag": "santarem", "discovery_year": 2022},
        {"site_name": "Site_F", "lat": -0.3, "lon": -49.6, "culture_tag": "marajo", "discovery_year": 2016}
    ])


class TestDataSplitter:
    """Test cases for DataSplitter class."""
    
    def test_initialization(self, sample_config, sample_sites_data):
        """Test DataSplitter initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create config file
            config_file = temp_path / "test_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(sample_config, f)
            
            # Create sites CSV
            sites_file = temp_path / "test_sites.csv"
            sample_sites_data.to_csv(sites_file, index=False)
            
            # Initialize splitter
            splitter = DataSplitter(config_file, sites_file)
            
            assert len(splitter.sites) == 6
            assert isinstance(splitter.sites, gpd.GeoDataFrame)
            assert splitter.sites.crs.to_string() == "EPSG:4326"
    
    def test_train_val_split(self, sample_config, sample_sites_data):
        """Test train/validation split functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            config_file = temp_path / "test_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(sample_config, f)
            
            sites_file = temp_path / "test_sites.csv"
            sample_sites_data.to_csv(sites_file, index=False)
            
            splitter = DataSplitter(config_file, sites_file)
            splits = splitter.split()
            
            # Check that train and val splits exist
            assert "train" in splits
            assert "val" in splits
            
            # Check total count
            train_count = len(splits["train"])
            val_count = len(splits["val"])
            
            # Sites before threshold (2020): 4 sites (2015, 2018, 2019, 2016)
            expected_pre_threshold = 4
            assert train_count + val_count == expected_pre_threshold
            
            # Check validation ratio approximately
            val_ratio = val_count / (train_count + val_count)
            assert abs(val_ratio - sample_config["val_ratio"]) < 0.2  # Allow some tolerance
    
    def test_val_only_split(self, sample_config, sample_sites_data):
        """Test validation-only split (no training set)."""
        config_no_train = sample_config.copy()
        config_no_train["use_train_split"] = False
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            config_file = temp_path / "test_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config_no_train, f)
            
            sites_file = temp_path / "test_sites.csv"
            sample_sites_data.to_csv(sites_file, index=False)
            
            splitter = DataSplitter(config_file, sites_file)
            splits = splitter.split()
            
            # Should have val but not train
            assert "val" in splits
            assert "train" not in splits
            
            # All pre-threshold sites should be in validation
            assert len(splits["val"]) == 4
    
    def test_test_time_split(self, sample_config, sample_sites_data):
        """Test test-time split based on discovery year."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            config_file = temp_path / "test_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(sample_config, f)
            
            sites_file = temp_path / "test_sites.csv"
            sample_sites_data.to_csv(sites_file, index=False)
            
            splitter = DataSplitter(config_file, sites_file)
            splits = splitter.split()
            
            # Check test-time split
            assert "test_time" in splits
            test_time_sites = splits["test_time"]
            
            # Sites with discovery_year >= 2020: Site_D (2021), Site_E (2022)
            assert len(test_time_sites) == 2
            
            # Check all test-time sites have discovery_year >= threshold
            for _, site in test_time_sites.iterrows():
                assert site["discovery_year"] >= sample_config["discovery_year_threshold"]
    
    def test_test_region_split(self, sample_config, sample_sites_data):
        """Test test-region split by culture."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            config_file = temp_path / "test_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(sample_config, f)
            
            sites_file = temp_path / "test_sites.csv"
            sample_sites_data.to_csv(sites_file, index=False)
            
            splitter = DataSplitter(config_file, sites_file)
            splits = splitter.split()
            
            # Check test-region split
            assert "test_region" in splits
            test_regions = splits["test_region"]
            
            # Should have acre and casarabe regions
            assert "acre" in test_regions
            assert "casarabe" in test_regions
            
            # Check acre region has correct sites
            acre_sites = test_regions["acre"]
            acre_names = set(acre_sites["site_name"])
            assert acre_names == {"Site_A", "Site_B"}
            
            # Check casarabe region
            casarabe_sites = test_regions["casarabe"]
            casarabe_names = set(casarabe_sites["site_name"])
            assert casarabe_names == {"Site_C", "Site_D"}
    
    def test_get_stats(self, sample_config, sample_sites_data):
        """Test dataset statistics generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            config_file = temp_path / "test_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(sample_config, f)
            
            sites_file = temp_path / "test_sites.csv"
            sample_sites_data.to_csv(sites_file, index=False)
            
            splitter = DataSplitter(config_file, sites_file)
            stats = splitter.get_stats()
            
            # Check basic stats
            assert stats["total_sites"] == 6
            assert "culture_distribution" in stats
            assert "discovery_year_range" in stats
            assert "geographic_bounds" in stats
            
            # Check culture distribution
            culture_dist = stats["culture_distribution"]
            assert culture_dist["acre"] == 2
            assert culture_dist["casarabe"] == 2
            
            # Check year range
            year_range = stats["discovery_year_range"]
            assert year_range["min"] == 2015
            assert year_range["max"] == 2022
    
    def test_missing_columns_error(self, sample_config):
        """Test error handling for missing required columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            config_file = temp_path / "test_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(sample_config, f)
            
            # Create sites CSV with missing columns
            incomplete_data = pd.DataFrame([
                {"site_name": "Site_A", "lat": -9.0, "lon": -68.0}  # Missing culture_tag, discovery_year
            ])
            
            sites_file = temp_path / "test_sites.csv"
            incomplete_data.to_csv(sites_file, index=False)
            
            # Should raise ValueError for missing columns
            with pytest.raises(ValueError, match="Missing required columns"):
                DataSplitter(config_file, sites_file)


def test_create_sample_master_csv():
    """Test sample CSV creation function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        sample_file = temp_path / "sample_sites.csv"
        
        create_sample_master_csv(sample_file)
        
        # Check file was created
        assert sample_file.exists()
        
        # Check content
        df = pd.read_csv(sample_file)
        assert len(df) > 0
        assert all(col in df.columns for col in ["site_name", "lat", "lon", "culture_tag", "discovery_year"])
        
        # Check data validity
        assert df["lat"].between(-20, 5).all()  # Amazon region latitudes
        assert df["lon"].between(-80, -40).all()  # Amazon region longitudes
        assert df["discovery_year"].between(2000, 2025).all()  # Reasonable years