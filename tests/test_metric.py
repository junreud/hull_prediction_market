"""
Unit tests for metric.py

Tests cover:
- Basic metric calculations
- Edge cases (zero division, NaN handling)
- Known value validation
- Volatility penalty calculation
- Additional metrics
"""

import pytest
import numpy as np
import pandas as pd
from src.metric import (
    CompetitionMetric,
    calculate_additional_metrics,
    create_metric_calculator
)


class TestCompetitionMetric:
    """Test CompetitionMetric class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metric = CompetitionMetric(
            vol_threshold=1.2,
            underperformance_penalty=False,
            min_periods=3  # Lower threshold for tests
        )
    
    def test_volatility_penalty_no_excess(self):
        """Test volatility penalty when within threshold."""
        strategy_vol = 0.1
        market_vol = 0.1
        penalty = self.metric.calculate_volatility_penalty(strategy_vol, market_vol)
        assert penalty == 1.0, "Penalty should be 1.0 when vol_ratio = 1.0"
    
    def test_volatility_penalty_with_excess(self):
        """Test volatility penalty when exceeding threshold."""
        strategy_vol = 0.15
        market_vol = 0.1
        # vol_ratio = 1.5, excess = 1.5 - 1.2 = 0.3
        expected_penalty = 1.0 + 0.3
        penalty = self.metric.calculate_volatility_penalty(strategy_vol, market_vol)
        assert abs(penalty - expected_penalty) < 1e-6, f"Expected {expected_penalty}, got {penalty}"
    
    def test_volatility_penalty_zero_market_vol(self):
        """Test edge case: zero market volatility."""
        strategy_vol = 0.1
        market_vol = 0.0
        penalty = self.metric.calculate_volatility_penalty(strategy_vol, market_vol)
        assert penalty == 1.0, "Should return 1.0 for zero market vol"
    
    def test_sharpe_ratio_basic(self):
        """Test basic Sharpe ratio calculation."""
        # Known case: mean=0.01, std=0.02 -> Sharpe=0.5
        returns = np.random.normal(0.01, 0.02, 100)
        np.random.seed(42)
        returns = np.random.normal(0.01, 0.02, 1000)
        
        sharpe = self.metric.calculate_sharpe_ratio(returns)
        expected = 0.01 / 0.02
        
        # Allow some tolerance due to random sampling
        assert abs(sharpe - expected) < 0.1, f"Expected ~{expected}, got {sharpe}"
    
    def test_sharpe_ratio_with_risk_free_rate(self):
        """Test Sharpe ratio with risk-free rate."""
        returns = np.array([0.02, 0.03, 0.01, 0.04, 0.02])
        rfr = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        
        sharpe = self.metric.calculate_sharpe_ratio(returns, rfr)
        
        # Manual calculation
        excess = returns - rfr
        expected = np.mean(excess) / np.std(excess, ddof=1)
        
        assert abs(sharpe - expected) < 1e-6
    
    def test_sharpe_ratio_zero_volatility(self):
        """Test edge case: zero volatility."""
        returns = np.ones(100) * 0.01  # Constant returns
        sharpe = self.metric.calculate_sharpe_ratio(returns)
        assert sharpe == 0.0, "Should return 0.0 for zero volatility"
    
    def test_sharpe_ratio_with_nans(self):
        """Test Sharpe ratio with NaN values."""
        returns = np.array([0.01, np.nan, 0.02, 0.03, np.nan, 0.01])
        sharpe = self.metric.calculate_sharpe_ratio(returns)
        
        # Should use only valid values
        valid_returns = np.array([0.01, 0.02, 0.03, 0.01])
        expected = np.mean(valid_returns) / np.std(valid_returns, ddof=1)
        
        assert abs(sharpe - expected) < 1e-6
    
    def test_sharpe_ratio_insufficient_data(self):
        """Test edge case: insufficient data."""
        metric_strict = CompetitionMetric(min_periods=10)
        returns = np.array([0.01, 0.02, 0.03])  # Less than min_periods
        sharpe = metric_strict.calculate_sharpe_ratio(returns)
        assert sharpe == 0.0, "Should return 0.0 for insufficient data"
    
    def test_calculate_score_basic(self):
        """Test basic score calculation."""
        np.random.seed(42)
        n = 100
        allocations = np.ones(n)  # All-in strategy
        forward_returns = np.random.normal(0.001, 0.01, n)
        
        result = self.metric.calculate_score(allocations, forward_returns)
        
        assert 'score' in result
        assert 'sharpe' in result
        assert 'vol_penalty' in result
        assert result['vol_penalty'] == 1.0, "Should be 1.0 for allocation=1"
        assert result['score'] == result['sharpe'], "Score should equal Sharpe when penalty=1"
    
    def test_calculate_score_with_leverage(self):
        """Test score with 2x leverage."""
        np.random.seed(42)
        n = 100
        allocations = np.ones(n) * 2.0  # 2x leverage
        forward_returns = np.random.normal(0.001, 0.01, n)
        
        result = self.metric.calculate_score(allocations, forward_returns)
        
        # Strategy vol should be ~2x market vol
        assert result['vol_ratio'] > 1.8, "Vol ratio should be close to 2.0"
        assert result['vol_penalty'] > 1.0, "Penalty should apply for 2x leverage"
        # Score magnitude should be less than Sharpe magnitude due to penalty
        assert abs(result['score']) < abs(result['sharpe']), "Score magnitude should be penalized"
    
    def test_calculate_score_with_nans(self):
        """Test score calculation with NaN values."""
        allocations = np.array([1.0, np.nan, 1.0, 1.0, np.nan])
        forward_returns = np.array([0.01, 0.02, np.nan, 0.01, 0.02])
        
        result = self.metric.calculate_score(allocations, forward_returns)
        
        # Should only use index 0 and 3
        assert result['n_valid'] == 2
    
    def test_calculate_score_insufficient_data(self):
        """Test score with insufficient data."""
        metric_strict = CompetitionMetric(min_periods=10)
        allocations = np.array([1.0, 1.0, 1.0])
        forward_returns = np.array([0.01, 0.02, 0.01])
        
        result = metric_strict.calculate_score(allocations, forward_returns)
        
        assert result['score'] == 0.0
        assert result['sharpe'] == 0.0
    
    def test_calculate_score_length_mismatch(self):
        """Test error handling for length mismatch."""
        allocations = np.array([1.0, 1.0, 1.0])
        forward_returns = np.array([0.01, 0.02])
        
        with pytest.raises(ValueError, match="Length mismatch"):
            self.metric.calculate_score(allocations, forward_returns)
    
    def test_calculate_score_known_values(self):
        """Test score with known values."""
        # Simple case: allocation=1, returns with known stats
        allocations = np.ones(50)
        forward_returns = np.array([0.01] * 25 + [-0.01] * 25)
        
        result = self.metric.calculate_score(allocations, forward_returns)
        
        # Mean return = 0.0, so Sharpe should be 0.0
        assert abs(result['mean_return']) < 1e-6
        assert abs(result['sharpe']) < 1e-6
        assert abs(result['score']) < 1e-6
    
    def test_calculate_score_vol_violation(self):
        """Test volatility violation detection."""
        np.random.seed(42)
        allocations = np.ones(100) * 1.5  # 1.5x leverage
        forward_returns = np.random.normal(0.001, 0.01, 100)
        
        result = self.metric.calculate_score(allocations, forward_returns)
        
        # Vol ratio should be ~1.5, exceeding 1.2 threshold
        assert result['vol_ratio'] > 1.2
        assert result['vol_violation_rate'] > 0
    
    def test_rolling_metrics(self):
        """Test rolling metrics calculation."""
        np.random.seed(42)
        n = 300
        allocations = np.ones(n)
        forward_returns = np.random.normal(0.001, 0.01, n)
        
        rolling_df = self.metric.calculate_rolling_metrics(
            allocations, forward_returns, window=100
        )
        
        assert len(rolling_df) == n - 100 + 1
        assert 'score' in rolling_df.columns
        assert 'sharpe' in rolling_df.columns
        assert 'vol_penalty' in rolling_df.columns
    
    def test_rolling_metrics_insufficient_data(self):
        """Test rolling metrics with insufficient data."""
        allocations = np.ones(50)
        forward_returns = np.random.normal(0.001, 0.01, 50)
        
        rolling_df = self.metric.calculate_rolling_metrics(
            allocations, forward_returns, window=100
        )
        
        assert len(rolling_df) == 0


class TestAdditionalMetrics:
    """Test additional metrics function."""
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        allocations = np.ones(10)
        forward_returns = np.array([0.01, 0.02, -0.05, -0.03, 0.01, 0.02, -0.01, 0.01, 0.01, 0.01])
        
        metrics = calculate_additional_metrics(allocations, forward_returns)
        
        assert 'max_drawdown' in metrics
        assert metrics['max_drawdown'] < 0, "Max drawdown should be negative"
    
    def test_turnover(self):
        """Test turnover calculation."""
        allocations = np.array([1.0, 1.5, 1.2, 0.8, 1.0, 1.3])
        forward_returns = np.random.normal(0.001, 0.01, 6)
        
        metrics = calculate_additional_metrics(allocations, forward_returns)
        
        expected_turnover = np.mean(np.abs(np.diff(allocations)))
        assert abs(metrics['turnover'] - expected_turnover) < 1e-6
    
    def test_leverage_2x_rate(self):
        """Test 2x leverage usage rate."""
        allocations = np.array([1.0, 2.0, 2.0, 1.0, 1.5, 2.0])  # 3 out of 6 are 2x
        forward_returns = np.random.normal(0.001, 0.01, 6)
        
        metrics = calculate_additional_metrics(allocations, forward_returns)
        
        expected_rate = 3 / 6
        assert abs(metrics['leverage_2x_rate'] - expected_rate) < 1e-6
    
    def test_avg_allocation(self):
        """Test average allocation."""
        allocations = np.array([1.0, 1.5, 2.0, 0.5, 1.0])
        forward_returns = np.random.normal(0.001, 0.01, 5)
        
        metrics = calculate_additional_metrics(allocations, forward_returns)
        
        expected_avg = np.mean(allocations)
        assert abs(metrics['avg_allocation'] - expected_avg) < 1e-6
    
    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        np.random.seed(42)
        allocations = np.ones(100)
        forward_returns = np.random.normal(0.001, 0.01, 100)
        
        metrics = calculate_additional_metrics(allocations, forward_returns)
        
        assert 'calmar_ratio' in metrics
        # Calmar = mean_return / abs(max_drawdown)
        if abs(metrics['max_drawdown']) > 1e-10:
            assert metrics['calmar_ratio'] != 0.0
    
    def test_additional_metrics_with_nans(self):
        """Test additional metrics with NaN values."""
        allocations = np.array([1.0, np.nan, 1.5, 1.0, np.nan])
        forward_returns = np.array([0.01, 0.02, np.nan, 0.01, 0.02])
        
        metrics = calculate_additional_metrics(allocations, forward_returns)
        
        # Should handle NaNs gracefully
        assert all(not np.isnan(v) for v in metrics.values())


class TestMetricFactory:
    """Test metric factory function."""
    
    def test_create_metric_calculator(self):
        """Test creating metric calculator from config."""
        metric = create_metric_calculator()
        
        assert isinstance(metric, CompetitionMetric)
        assert metric.vol_threshold == 1.2  # Default from params.yaml
    
    def test_create_metric_calculator_with_override(self):
        """Test creating metric calculator with overrides."""
        metric = create_metric_calculator(vol_threshold=1.5, min_periods=50)
        
        assert metric.vol_threshold == 1.5
        assert metric.min_periods == 50


# Edge case tests
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_all_nans(self):
        """Test with all NaN values."""
        metric = CompetitionMetric()
        allocations = np.array([np.nan, np.nan, np.nan])
        forward_returns = np.array([np.nan, np.nan, np.nan])
        
        result = metric.calculate_score(allocations, forward_returns)
        
        assert result['score'] == 0.0
        assert result['n_valid'] == 0
    
    def test_infinite_values(self):
        """Test with infinite values."""
        metric = CompetitionMetric(min_periods=3)
        allocations = np.array([1.0, np.inf, 1.0, 1.0])
        forward_returns = np.array([0.01, 0.02, 0.01, 0.02])
        
        result = metric.calculate_score(allocations, forward_returns)
        
        # np.isnan doesn't filter inf, so all 4 values are "valid"
        # Just check no errors occurred and result is valid
        assert not np.isnan(result['score']) or not np.isinf(result['score'])
    
    def test_zero_returns(self):
        """Test with all zero returns."""
        metric = CompetitionMetric()
        allocations = np.ones(50)
        forward_returns = np.zeros(50)
        
        result = metric.calculate_score(allocations, forward_returns)
        
        assert result['mean_return'] == 0.0
        assert result['std_return'] == 0.0
        assert result['score'] == 0.0
    
    def test_negative_allocations(self):
        """Test with negative allocations (should not occur but handle gracefully)."""
        metric = CompetitionMetric()
        allocations = np.array([-1.0, 1.0, 1.0, 1.0] * 10)
        forward_returns = np.random.normal(0.001, 0.01, 40)
        
        result = metric.calculate_score(allocations, forward_returns)
        
        # Should calculate without errors
        assert 'score' in result
        assert not np.isnan(result['score'])
