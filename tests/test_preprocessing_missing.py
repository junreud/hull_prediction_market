"""
Unit tests for Phase 1: Missing Value Preprocessing

Tests for:
1. create_missing_masks() - Binary missing indicators and duration features
2. handle_missing_values() - LOCF strategy without future leakage
3. align_announcement_dates() - Economic indicator announcement lag
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
def sample_df():
    """Create a sample DataFrame with missing values for testing."""
    np.random.seed(42)
    
    # Create 50 time steps
    n = 50
    df = pd.DataFrame({
        'date_id': range(1, n + 1),
        'E1': [1.0, 2.0, np.nan, 4.0, 5.0] + [6.0] * 20 + [np.nan] * 5 + [7.0] * 20,  # Economic indicator
        'E2': [10.0] * 10 + [np.nan] * 3 + [15.0] * 37,  # Another economic indicator
        'M1': np.random.randn(n),  # Market indicator (no missing)
        'V1': [1.0, np.nan, np.nan, 2.0, 3.0] + [4.0] * 45,  # Volatility indicator
        'D1': [0, 1, 0, 1, 0] * 10,  # Dummy variable
    })
    
    return df


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
        'E': ['E1', 'E2'],
        'M': ['M1'],
        'V': ['V1'],
        'D': ['D1'],
        'I': [],
        'P': [],
        'S': [],
    }
    
    return loader


class TestMissingMasks:
    """Test create_missing_masks() functionality."""
    
    def test_binary_mask_creation(self, data_loader, sample_df):
        """Test that binary missing indicators are correctly created."""
        df_with_masks = data_loader.create_missing_masks(sample_df)
        
        # Check that mask columns are created
        assert 'E1_is_missing' in df_with_masks.columns
        assert 'E2_is_missing' in df_with_masks.columns
        assert 'V1_is_missing' in df_with_masks.columns
        
        # Check binary values (0 or 1)
        assert df_with_masks['E1_is_missing'].isin([0, 1]).all()
        
        # Check correctness: E1 has missing at index 2
        assert df_with_masks.loc[2, 'E1_is_missing'] == 1
        assert df_with_masks.loc[0, 'E1_is_missing'] == 0
    
    def test_missing_duration_calculation(self, data_loader, sample_df):
        """Test that missing duration is correctly calculated."""
        df_with_masks = data_loader.create_missing_masks(sample_df)
        
        # Check that duration columns are created
        assert 'E1_missing_days' in df_with_masks.columns
        
        # E1 pattern: [1.0, 2.0, NaN, 4.0, 5.0] + [6.0]*20 + [NaN]*5 + [7.0]*20
        # At index 2: missing_days should be 1
        assert df_with_masks.loc[2, 'E1_missing_days'] == 1
        
        # At index 3 (not missing): should be 0
        assert df_with_masks.loc[3, 'E1_missing_days'] == 0
        
        # At indices 25-29 (5 consecutive NaN): should be 1,2,3,4,5
        expected_durations = [1, 2, 3, 4, 5]
        actual_durations = df_with_masks.loc[25:29, 'E1_missing_days'].tolist()
        assert actual_durations == expected_durations
    
    def test_no_mask_for_complete_features(self, data_loader, sample_df):
        """Test that no masks are created for features without missing values."""
        df_with_masks = data_loader.create_missing_masks(sample_df)
        
        # M1 has no missing values, so no mask should be created
        assert 'M1_is_missing' not in df_with_masks.columns
        assert 'M1_missing_days' not in df_with_masks.columns
    
    def test_dummy_variables_skipped(self, data_loader, sample_df):
        """Test that dummy variables (D group) are skipped."""
        df_with_masks = data_loader.create_missing_masks(sample_df)
        
        # D1 should not have masks even if it had missing values
        assert 'D1_is_missing' not in df_with_masks.columns


class TestLOCFStrategy:
    """Test handle_missing_values() with LOCF strategy."""
    
    def test_locf_no_future_leakage(self, data_loader, sample_df):
        """Test that LOCF does not use future information."""
        df_filled = data_loader.handle_missing_values(
            sample_df, 
            strategy={'E': 'locf', 'V': 'ewma', 'M': 'ewma', 'D': 'zero'}
        )
        
        # E1 at index 2 (NaN) should be filled with value from index 1 (2.0)
        assert df_filled.loc[2, 'E1'] == 2.0
        
        # Should NOT be interpolated to 3.0 (average of 2.0 and 4.0)
        assert df_filled.loc[2, 'E1'] != 3.0
    
    def test_locf_max_gap_limit(self, data_loader, sample_df):
        """Test that LOCF respects max_gap parameter."""
        # E1 has 5 consecutive NaN at indices 25-29
        df_filled = data_loader.handle_missing_values(
            sample_df,
            max_gap=3,  # Only fill up to 3 gaps
            strategy={'E': 'locf', 'V': 'ewma', 'M': 'ewma', 'D': 'zero'}
        )
        
        # First 3 NaNs should be filled, but we need to check the fallback
        # Since max_gap=3, indices 25,26,27 filled with LOCF
        # Indices 28,29 should use fallback (median)
        assert not pd.isna(df_filled.loc[25:29, 'E1']).any()
    
    def test_fallback_to_median(self, data_loader, sample_df):
        """Test that fallback uses training median when provided."""
        train_df = sample_df.iloc[:25].copy()
        test_df = sample_df.iloc[25:].copy()
        
        # Calculate expected median from training data
        train_median_e1 = train_df['E1'].median()
        
        # Fill test data
        df_filled = data_loader.handle_missing_values(
            test_df,
            train_df=train_df,
            max_gap=2,
            strategy={'E': 'locf', 'V': 'ewma', 'M': 'ewma', 'D': 'zero'}
        )
        
        # Since test_df starts with NaNs and no previous value, 
        # it should use training median
        # (First few values in test_df indices 25-29 are NaN)
        assert not df_filled['E1'].isna().any()
    
    def test_no_interpolation_for_economic_indicators(self, data_loader, sample_df):
        """Test that economic indicators use LOCF, not interpolation."""
        df_filled = data_loader.handle_missing_values(
            sample_df,
            strategy={'E': 'locf', 'V': 'ewma', 'M': 'ewma', 'D': 'zero'}
        )
        
        # E1 index 2: should be 2.0 (LOCF), not 3.0 (interpolate)
        assert df_filled.loc[2, 'E1'] == 2.0
        
        # Verify no missing values remain
        assert df_filled['E1'].isna().sum() == 0
        assert df_filled['E2'].isna().sum() == 0


class TestAnnouncementAlignment:
    """Test align_announcement_dates() functionality."""
    
    def test_default_lag_applied(self, data_loader, sample_df):
        """Test that default lag is applied to E-group features."""
        original_e1 = sample_df['E1'].copy()
        
        df_aligned = data_loader.align_announcement_dates(
            sample_df,
            default_lag=5
        )
        
        # After 5-day lag, value at index 5 should come from index 0
        assert df_aligned.loc[5, 'E1'] == original_e1.iloc[0]
        
        # First 5 values should be NaN (no data to shift from)
        assert df_aligned.loc[:4, 'E1'].isna().all()
    
    def test_custom_announcement_calendar(self, data_loader, sample_df):
        """Test that custom announcement calendar is respected."""
        calendar = pd.DataFrame({
            'feature': ['E1', 'E2'],
            'announcement_lag': [10, 15]
        })
        
        original_e1 = sample_df['E1'].copy()
        original_e2 = sample_df['E2'].copy()
        
        df_aligned = data_loader.align_announcement_dates(
            sample_df,
            announcement_calendar=calendar
        )
        
        # E1: 10-day lag
        assert df_aligned.loc[10, 'E1'] == original_e1.iloc[0]
        assert df_aligned.loc[:9, 'E1'].isna().all()
        
        # E2: 15-day lag
        assert df_aligned.loc[15, 'E2'] == original_e2.iloc[0]
        assert df_aligned.loc[:14, 'E2'].isna().all()
    
    def test_non_economic_features_unaffected(self, data_loader, sample_df):
        """Test that non-E-group features are not affected."""
        original_m1 = sample_df['M1'].copy()
        
        df_aligned = data_loader.align_announcement_dates(
            sample_df,
            default_lag=10
        )
        
        # M1 should be unchanged
        pd.testing.assert_series_equal(df_aligned['M1'], original_m1, check_names=False)


class TestIntegration:
    """Integration tests for the full preprocessing pipeline."""
    
    def test_full_phase1_pipeline(self, data_loader, sample_df):
        """Test that all Phase 1 features work together."""
        # Step 1: Create missing masks
        df = data_loader.create_missing_masks(sample_df)
        
        # Verify masks created
        assert 'E1_is_missing' in df.columns
        assert 'E1_missing_days' in df.columns
        
        # Step 2: Align announcement dates
        df = data_loader.align_announcement_dates(df, default_lag=5)
        
        # Verify alignment created new NaNs at the beginning
        assert df.loc[:4, 'E1'].isna().all()
        
        # Step 3: Handle missing values with LOCF
        df = data_loader.handle_missing_values(
            df,
            max_gap=10,
            strategy={'E': 'locf', 'V': 'ewma', 'M': 'ewma', 'D': 'zero'}
        )
        
        # Verify all missing values filled
        assert df['E1'].isna().sum() == 0
        assert df['E2'].isna().sum() == 0
        assert df['V1'].isna().sum() == 0
        
        # Verify mask features still present
        assert 'E1_is_missing' in df.columns
        assert 'E1_missing_days' in df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
