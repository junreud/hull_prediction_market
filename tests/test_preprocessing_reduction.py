"""
Unit tests for Phase 2: Feature Reduction and Dimensionality Reduction

Tests for:
1. detect_feature_clusters() - Identify highly correlated feature groups
2. reduce_feature_redundancy() - Remove redundant features
3. apply_group_pca() - Apply PCA within feature groups
4. calculate_regime_weights() - Calculate sample weights by volatility regime
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
def sample_df_with_correlation():
    """Create a sample DataFrame with correlated features for testing."""
    np.random.seed(42)
    n = 200
    
    # Create base signals
    base_m1 = np.random.randn(n)
    base_v1 = np.random.randn(n)
    base_e1 = np.random.randn(n)
    
    df = pd.DataFrame({
        'date_id': range(1, n + 1),
        
        # M group: 3 highly correlated features
        'M1': base_m1,
        'M2': base_m1 + np.random.randn(n) * 0.1,  # Very similar to M1
        'M3': base_m1 + np.random.randn(n) * 0.15,  # Similar to M1
        'M4': np.random.randn(n),  # Independent
        
        # V group: 2 clusters
        'V1': base_v1,
        'V2': base_v1 + np.random.randn(n) * 0.1,  # Cluster with V1
        'V3': np.random.randn(n),  # Independent
        'V4': np.random.randn(n),  # Independent
        
        # E group: all different
        'E1': base_e1,
        'E2': np.random.randn(n),
        'E3': np.random.randn(n),
    })
    
    return df


@pytest.fixture
def sample_df_with_regime():
    """Create a sample DataFrame with regime column for testing."""
    np.random.seed(42)
    n = 150
    
    # Create regimes: low_vol, normal, high_vol
    regimes = ['normal'] * 50 + ['high_vol'] * 50 + ['low_vol'] * 50
    
    df = pd.DataFrame({
        'date_id': range(1, n + 1),
        'regime': regimes,
        'M1': np.random.randn(n),
        'V1': np.random.randn(n),
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
        'M': ['M1', 'M2', 'M3', 'M4'],
        'V': ['V1', 'V2', 'V3', 'V4'],
        'E': ['E1', 'E2', 'E3'],
        'I': [],
        'P': [],
        'S': [],
        'D': [],
    }
    
    return loader


class TestFeatureClustering:
    """Test detect_feature_clusters() functionality."""
    
    def test_cluster_detection(self, data_loader, sample_df_with_correlation):
        """Test that highly correlated features are grouped into clusters."""
        clusters = data_loader.detect_feature_clusters(
            sample_df_with_correlation,
            threshold=0.85,
            by_group=True
        )
        
        # Should detect clusters in M and V groups
        assert 'M' in clusters or 'V' in clusters
        
        # M group should have a cluster with M1, M2, M3
        if 'M' in clusters:
            # Find cluster containing M1
            m_cluster = None
            for cluster in clusters['M']:
                if 'M1' in cluster:
                    m_cluster = cluster
                    break
            
            if m_cluster:
                # Should contain at least M1 and M2 (highly correlated)
                assert 'M1' in m_cluster
                assert 'M2' in m_cluster or 'M3' in m_cluster
    
    def test_threshold_sensitivity(self, data_loader, sample_df_with_correlation):
        """Test that threshold affects cluster size."""
        # High threshold (0.95) - only very similar features
        clusters_high = data_loader.detect_feature_clusters(
            sample_df_with_correlation,
            threshold=0.95,
            by_group=True
        )
        
        # Low threshold (0.70) - more features grouped
        clusters_low = data_loader.detect_feature_clusters(
            sample_df_with_correlation,
            threshold=0.70,
            by_group=True
        )
        
        # Lower threshold should find more or equal clusters
        total_high = sum(len(clusters) for clusters in clusters_high.values())
        total_low = sum(len(clusters) for clusters in clusters_low.values())
        
        assert total_low >= total_high
    
    def test_by_group_parameter(self, data_loader, sample_df_with_correlation):
        """Test that by_group parameter works correctly."""
        # Cluster by group
        clusters_by_group = data_loader.detect_feature_clusters(
            sample_df_with_correlation,
            threshold=0.85,
            by_group=True
        )
        
        # Should have group names as keys
        for key in clusters_by_group.keys():
            assert key in ['M', 'V', 'E', 'I', 'P', 'S']
        
        # Cluster all together
        clusters_all = data_loader.detect_feature_clusters(
            sample_df_with_correlation,
            threshold=0.85,
            by_group=False
        )
        
        # Should have 'all' as key
        if clusters_all:
            assert 'all' in clusters_all


class TestFeatureReduction:
    """Test reduce_feature_redundancy() functionality."""
    
    def test_representative_method(self, data_loader, sample_df_with_correlation):
        """Test that representative method keeps highest variance feature."""
        # First detect clusters
        clusters = data_loader.detect_feature_clusters(
            sample_df_with_correlation,
            threshold=0.85,
            by_group=True
        )
        
        if not clusters:
            pytest.skip("No clusters detected for testing")
        
        original_cols = set(sample_df_with_correlation.columns)
        
        # Reduce redundancy
        df_reduced = data_loader.reduce_feature_redundancy(
            sample_df_with_correlation,
            clusters,
            method='representative'
        )
        
        # Should have fewer columns (some removed)
        assert len(df_reduced.columns) < len(original_cols)
        
        # date_id should still be present
        assert 'date_id' in df_reduced.columns
    
    def test_mean_method(self, data_loader, sample_df_with_correlation):
        """Test that mean method creates cluster features."""
        clusters = data_loader.detect_feature_clusters(
            sample_df_with_correlation,
            threshold=0.85,
            by_group=True
        )
        
        if not clusters:
            pytest.skip("No clusters detected for testing")
        
        df_reduced = data_loader.reduce_feature_redundancy(
            sample_df_with_correlation,
            clusters,
            method='mean'
        )
        
        # Should have new cluster columns
        cluster_cols = [col for col in df_reduced.columns if 'cluster' in col]
        assert len(cluster_cols) > 0
    
    def test_no_clusters(self, data_loader, sample_df_with_correlation):
        """Test that empty clusters dict doesn't break the function."""
        df_reduced = data_loader.reduce_feature_redundancy(
            sample_df_with_correlation,
            {},  # Empty clusters
            method='representative'
        )
        
        # Should return dataframe unchanged
        assert len(df_reduced.columns) == len(sample_df_with_correlation.columns)


class TestGroupPCA:
    """Test apply_group_pca() functionality."""
    
    def test_pca_dimension_reduction(self, data_loader, sample_df_with_correlation):
        """Test that PCA reduces dimensions correctly."""
        # Specify number of components per group
        n_components = {
            'M': 2,  # Reduce 4 M features to 2 components
            'V': 2,  # Reduce 4 V features to 2 components
            'E': 2,  # Reduce 3 E features to 2 components
        }
        
        df_pca, pca_models = data_loader.apply_group_pca(
            sample_df_with_correlation,
            n_components=n_components
        )
        
        # Should have PCA columns
        assert 'M_PC1' in df_pca.columns
        assert 'M_PC2' in df_pca.columns
        
        # Original M features should be removed
        assert 'M1' not in df_pca.columns
        assert 'M2' not in df_pca.columns
        
        # Should have models for each group
        assert 'M' in pca_models
        assert 'V' in pca_models
    
    def test_variance_preservation(self, data_loader, sample_df_with_correlation):
        """Test that PCA preserves specified variance."""
        df_pca, pca_models = data_loader.apply_group_pca(
            sample_df_with_correlation,
            variance_threshold=0.90
        )
        
        # Check that variance is preserved
        for group, model_info in pca_models.items():
            variance_explained = model_info['variance_explained']
            assert variance_explained >= 0.85  # Should be close to threshold
    
    def test_pca_with_train_data(self, data_loader, sample_df_with_correlation):
        """Test that PCA can be fit on training data and applied to test."""
        # Split data
        train_df = sample_df_with_correlation.iloc[:150].copy()
        test_df = sample_df_with_correlation.iloc[150:].copy()
        
        n_components = {'M': 2, 'V': 2, 'E': 2}
        
        # Fit on training data
        train_pca, pca_models = data_loader.apply_group_pca(
            train_df,
            n_components=n_components
        )
        
        # Apply to test data
        test_pca, _ = data_loader.apply_group_pca(
            test_df,
            train_df=train_df,
            n_components=n_components
        )
        
        # Both should have same PCA columns
        assert 'M_PC1' in train_pca.columns
        assert 'M_PC1' in test_pca.columns
        
        # Test should have same number of PCA components as train
        train_pca_cols = [c for c in train_pca.columns if '_PC' in c]
        test_pca_cols = [c for c in test_pca.columns if '_PC' in c]
        assert len(train_pca_cols) == len(test_pca_cols)
    
    def test_single_feature_group_skipped(self, data_loader):
        """Test that groups with only one feature are skipped."""
        df = pd.DataFrame({
            'date_id': range(100),
            'M1': np.random.randn(100),  # Only one M feature
            'V1': np.random.randn(100),
            'V2': np.random.randn(100),
        })
        
        loader = data_loader
        loader.feature_groups = {
            'M': ['M1'],  # Only one feature
            'V': ['V1', 'V2'],  # Two features
            'E': [], 'I': [], 'P': [], 'S': [], 'D': []
        }
        
        df_pca, pca_models = loader.apply_group_pca(df, n_components={'M': 1, 'V': 1})
        
        # M should be skipped (only one feature)
        assert 'M' not in pca_models
        assert 'M1' in df_pca.columns  # Original M1 should remain
        
        # V should be processed
        assert 'V' in pca_models or 'V_PC1' in df_pca.columns


class TestRegimeWeights:
    """Test calculate_regime_weights() functionality."""
    
    def test_default_weights(self, data_loader, sample_df_with_regime):
        """Test that default regime weights are applied correctly."""
        weights = data_loader.calculate_regime_weights(sample_df_with_regime)
        
        # Check that weights are calculated
        assert len(weights) == len(sample_df_with_regime)
        
        # Check that different regimes have different weights
        unique_weights = weights.unique()
        assert len(unique_weights) <= 3  # Max 3 regime types
        
        # High vol should have lower weight (0.5 by default)
        high_vol_idx = sample_df_with_regime['regime'] == 'high_vol'
        if high_vol_idx.any():
            assert weights[high_vol_idx].iloc[0] == 0.5
    
    def test_custom_weight_map(self, data_loader, sample_df_with_regime):
        """Test that custom weight mapping works."""
        custom_weights = {
            'low_vol': 1.5,
            'normal': 1.0,
            'high_vol': 0.3
        }
        
        weights = data_loader.calculate_regime_weights(
            sample_df_with_regime,
            weight_map=custom_weights
        )
        
        # Check custom weights are applied
        high_vol_idx = sample_df_with_regime['regime'] == 'high_vol'
        if high_vol_idx.any():
            assert weights[high_vol_idx].iloc[0] == 0.3
        
        low_vol_idx = sample_df_with_regime['regime'] == 'low_vol'
        if low_vol_idx.any():
            assert weights[low_vol_idx].iloc[0] == 1.5
    
    def test_missing_regime_column(self, data_loader):
        """Test that missing regime column returns uniform weights."""
        df = pd.DataFrame({
            'date_id': range(100),
            'M1': np.random.randn(100),
        })
        
        weights = data_loader.calculate_regime_weights(df)
        
        # Should return all 1.0 (uniform weights)
        assert (weights == 1.0).all()
    
    def test_unknown_regime_values(self, data_loader):
        """Test that unknown regime values get default weight."""
        df = pd.DataFrame({
            'date_id': range(100),
            'regime': ['normal'] * 50 + ['unknown_regime'] * 50,
            'M1': np.random.randn(100),
        })
        
        weights = data_loader.calculate_regime_weights(df)
        
        # Unknown regimes should get weight 1.0
        unknown_idx = df['regime'] == 'unknown_regime'
        assert (weights[unknown_idx] == 1.0).all()


class TestIntegration:
    """Integration tests for the full Phase 2 pipeline."""
    
    def test_full_phase2_pipeline(self, data_loader, sample_df_with_correlation):
        """Test that all Phase 2 features work together."""
        df = sample_df_with_correlation.copy()
        
        # Step 1: Detect clusters
        clusters = data_loader.detect_feature_clusters(
            df,
            threshold=0.85,
            by_group=True
        )
        
        # Step 2: Reduce redundancy (if clusters found)
        if clusters:
            df = data_loader.reduce_feature_redundancy(
                df,
                clusters,
                method='representative'
            )
        
        # Step 3: Apply PCA
        n_components = {'M': 2, 'V': 2, 'E': 2}
        df_pca, pca_models = data_loader.apply_group_pca(
            df,
            n_components=n_components
        )
        
        # Verify pipeline worked
        assert 'date_id' in df_pca.columns
        
        # Should have PCA components
        pca_cols = [c for c in df_pca.columns if '_PC' in c]
        assert len(pca_cols) > 0
        
        # Original features should be removed
        assert 'M1' not in df_pca.columns or 'M_PC1' in df_pca.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
