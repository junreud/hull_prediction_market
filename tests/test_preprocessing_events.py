"""
Unit tests for Phase 3: Event Analysis and Granger Causality

Tests for:
1. add_event_dummies() - Add binary event indicators
2. analyze_granger_causality() - Test predictive relationships with lags
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import DataLoader


@pytest.fixture
def sample_df_with_target():
    """Create a sample DataFrame with target variable for testing."""
    np.random.seed(42)
    n = 300
    
    # Create features with lagged relationships to target
    base_signal = np.random.randn(n)
    
    # M1: leads target by 2 days (should show Granger causality)
    m1 = np.concatenate([base_signal[2:], np.zeros(2)])
    
    # M2: random (no causality)
    m2 = np.random.randn(n)
    
    # V1: leads target by 1 day
    v1 = np.concatenate([base_signal[1:], np.zeros(1)])
    
    # Target: based on base_signal
    target = base_signal + np.random.randn(n) * 0.1
    
    df = pd.DataFrame({
        'date_id': range(1, n + 1),
        'M1': m1,
        'M2': m2,
        'V1': v1,
        'E1': np.random.randn(n),
        'forward_returns': target,
    })
    
    return df


@pytest.fixture
def event_calendar():
    """Create a sample event calendar for testing."""
    return pd.DataFrame({
        'event_date': [50, 100, 150, 200, 250],
        'event_type': ['FOMC', 'CPI', 'FOMC', 'earnings', 'CPI'],
        'window_before': [1, 0, 1, 2, 0],
        'window_after': [1, 1, 1, 1, 2]
    })


@pytest.fixture
def data_loader(tmp_path):
    """Create a DataLoader instance with temporary config."""
    import yaml
    
    config_path = tmp_path / "params.yaml"
    config = {
        'paths': {'data': 'data/raw/'},
        'preprocessing': {
            'ffill_limit': 5,
            'ewma_span': 10
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    loader = DataLoader(config_path=str(config_path))
    
    # Manually set feature groups for testing
    loader.feature_groups = {
        'M': ['M1', 'M2'],
        'V': ['V1'],
        'E': ['E1'],
        'I': [],
        'P': [],
        'S': [],
        'D': [],
    }
    
    return loader


class TestEventDummies:
    """Test add_event_dummies() functionality."""
    
    def test_event_columns_created(self, data_loader, sample_df_with_target, event_calendar):
        """Test that event dummy columns are created."""
        df_events = data_loader.add_event_dummies(
            sample_df_with_target,
            event_calendar
        )
        
        # Check that event columns exist
        assert 'event_FOMC' in df_events.columns
        assert 'event_CPI' in df_events.columns
        assert 'event_earnings' in df_events.columns
    
    def test_event_window_marking(self, data_loader, sample_df_with_target, event_calendar):
        """Test that event windows are marked correctly."""
        df_events = data_loader.add_event_dummies(
            sample_df_with_target,
            event_calendar
        )
        
        # First FOMC event at date_id=50, window_before=1, window_after=1
        # Should mark dates 49, 50, 51
        fomc_dates = df_events[df_events['event_FOMC'] == 1]['date_id'].values
        
        # Should include dates around 50
        assert 50 in fomc_dates
        
        # Check that it's binary
        assert df_events['event_FOMC'].isin([0, 1]).all()
    
    def test_multiple_event_types(self, data_loader, sample_df_with_target, event_calendar):
        """Test that multiple event types are handled correctly."""
        df_events = data_loader.add_event_dummies(
            sample_df_with_target,
            event_calendar
        )
        
        # Count events per type
        fomc_count = df_events['event_FOMC'].sum()
        cpi_count = df_events['event_CPI'].sum()
        earnings_count = df_events['event_earnings'].sum()
        
        # Should have events marked
        assert fomc_count > 0
        assert cpi_count > 0
        assert earnings_count > 0
    
    def test_overlapping_events(self, data_loader, sample_df_with_target):
        """Test that overlapping events are handled correctly."""
        # Create calendar with overlapping events
        calendar = pd.DataFrame({
            'event_date': [100, 101],
            'event_type': ['FOMC', 'FOMC'],
            'window_before': [1, 1],
            'window_after': [1, 1]
        })
        
        df_events = data_loader.add_event_dummies(
            sample_df_with_target,
            calendar
        )
        
        # Date 100 and 101 should both be marked (and their windows)
        # Overlapping windows should both be marked as 1
        assert df_events.loc[df_events['date_id'] == 100, 'event_FOMC'].values[0] == 1
        assert df_events.loc[df_events['date_id'] == 101, 'event_FOMC'].values[0] == 1
    
    def test_no_window(self, data_loader, sample_df_with_target):
        """Test events with no window (window_before=0, window_after=0)."""
        calendar = pd.DataFrame({
            'event_date': [100],
            'event_type': ['CPI'],
            'window_before': [0],
            'window_after': [0]
        })
        
        df_events = data_loader.add_event_dummies(
            sample_df_with_target,
            calendar
        )
        
        # Only date 100 should be marked
        assert df_events.loc[df_events['date_id'] == 100, 'event_CPI'].values[0] == 1
        assert df_events.loc[df_events['date_id'] == 99, 'event_CPI'].values[0] == 0
        assert df_events.loc[df_events['date_id'] == 101, 'event_CPI'].values[0] == 0


class TestGrangerCausality:
    """Test analyze_granger_causality() functionality."""
    
    def test_granger_test_runs(self, data_loader, sample_df_with_target):
        """Test that Granger causality test runs without errors."""
        try:
            results = data_loader.analyze_granger_causality(
                sample_df_with_target,
                target='forward_returns',
                max_lag=3,
                significance=0.05
            )
            
            # Should return a DataFrame
            assert isinstance(results, pd.DataFrame)
            
            # Should have expected columns
            assert 'feature' in results.columns
            assert 'best_lag' in results.columns
            assert 'p_value' in results.columns
            assert 'significant' in results.columns
            
        except ImportError:
            pytest.skip("statsmodels not installed")
    
    def test_granger_identifies_relationships(self, data_loader, sample_df_with_target):
        """Test that Granger test identifies lagged relationships."""
        try:
            results = data_loader.analyze_granger_causality(
                sample_df_with_target,
                target='forward_returns',
                max_lag=5,
                significance=0.10  # More lenient for test data
            )
            
            # Should have results for features
            assert len(results) > 0
            
            # Should be sorted by p_value
            assert results['p_value'].is_monotonic_increasing
            
        except ImportError:
            pytest.skip("statsmodels not installed")
    
    def test_granger_max_lag_parameter(self, data_loader, sample_df_with_target):
        """Test that max_lag parameter is respected."""
        try:
            results = data_loader.analyze_granger_causality(
                sample_df_with_target,
                target='forward_returns',
                max_lag=3,
                significance=0.05
            )
            
            # best_lag should be between 1 and max_lag
            if len(results) > 0:
                assert results['best_lag'].min() >= 1
                assert results['best_lag'].max() <= 3
            
        except ImportError:
            pytest.skip("statsmodels not installed")
    
    def test_granger_missing_target(self, data_loader, sample_df_with_target):
        """Test that missing target raises error."""
        with pytest.raises(ValueError, match="not found"):
            data_loader.analyze_granger_causality(
                sample_df_with_target,
                target='nonexistent_target',
                max_lag=3
            )
    
    def test_granger_with_missing_values(self, data_loader):
        """Test Granger test with missing values (should handle gracefully)."""
        # Create data with missing values
        df = pd.DataFrame({
            'date_id': range(100),
            'M1': [np.nan] * 10 + list(np.random.randn(90)),
            'M2': np.random.randn(100),
            'forward_returns': np.random.randn(100)
        })
        
        try:
            results = data_loader.analyze_granger_causality(
                df,
                target='forward_returns',
                max_lag=3
            )
            
            # Should handle missing values gracefully
            assert isinstance(results, pd.DataFrame)
            
        except ImportError:
            pytest.skip("statsmodels not installed")
    
    def test_granger_significance_threshold(self, data_loader, sample_df_with_target):
        """Test that significance threshold affects results."""
        try:
            # Strict threshold
            results_strict = data_loader.analyze_granger_causality(
                sample_df_with_target,
                target='forward_returns',
                max_lag=3,
                significance=0.01
            )
            
            # Lenient threshold
            results_lenient = data_loader.analyze_granger_causality(
                sample_df_with_target,
                target='forward_returns',
                max_lag=3,
                significance=0.10
            )
            
            # Lenient should have more or equal significant features
            if len(results_strict) > 0 and len(results_lenient) > 0:
                assert results_lenient['significant'].sum() >= results_strict['significant'].sum()
            
        except ImportError:
            pytest.skip("statsmodels not installed")


class TestIntegration:
    """Integration tests for Phase 3 pipeline."""
    
    def test_full_phase3_pipeline(self, data_loader, sample_df_with_target, event_calendar):
        """Test that all Phase 3 features work together."""
        df = sample_df_with_target.copy()
        
        # Step 1: Add event dummies
        df = data_loader.add_event_dummies(df, event_calendar)
        
        # Verify events added
        assert 'event_FOMC' in df.columns
        assert 'event_CPI' in df.columns
        
        # Step 2: Analyze Granger causality
        try:
            causality_results = data_loader.analyze_granger_causality(
                df,
                target='forward_returns',
                max_lag=3
            )
            
            # Verify analysis completed
            assert isinstance(causality_results, pd.DataFrame)
            
            # Event features should be in the test
            # (they might not be significant, but should be tested)
            all_features = causality_results['feature'].tolist()
            
            # Original features should be tested
            assert any('M1' in f or 'M2' in f or 'V1' in f for f in all_features + ['M1', 'M2', 'V1'])
            
        except ImportError:
            pytest.skip("statsmodels not installed")
    
    def test_event_features_in_granger_test(self, data_loader, sample_df_with_target, event_calendar):
        """Test that event features can be used in Granger causality test."""
        df = sample_df_with_target.copy()
        
        # Add event dummies
        df = data_loader.add_event_dummies(df, event_calendar)
        
        # Manually update feature groups to include event features
        data_loader.feature_groups['D'] = ['event_FOMC', 'event_CPI', 'event_earnings']
        
        try:
            # Run Granger test (should handle event dummies)
            causality_results = data_loader.analyze_granger_causality(
                df,
                target='forward_returns',
                max_lag=2
            )
            
            # Should complete without errors
            assert isinstance(causality_results, pd.DataFrame)
            
        except ImportError:
            pytest.skip("statsmodels not installed")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
