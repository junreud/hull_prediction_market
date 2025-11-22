"""
Position Mapping and Risk Control Module

This module provides:
- Base position mapper interface
- Sharpe scaling strategy
- Quantile binning strategy
- Volatility targeting strategy
- Constraint validators
- Position optimization utilities
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize

from src.utils import get_logger, load_config, Timer
from src.metric import CompetitionMetric

logger = get_logger(log_file="logs/position.log", level="INFO")


class BasePositionMapper(ABC):
    """
    Base class for position mapping strategies.
    
    All position mapping strategies should inherit from this class
    and implement the map_positions method.
    """
    
    def __init__(self, config_path: str = "conf/params.yaml"):
        """
        Initialize position mapper.
        
        Parameters
        ----------
        config_path : str
            Path to configuration file
        """
        self.config = load_config(config_path)
        self.position_config = self.config.get('position', {})
        
        # Constraints
        constraints = self.position_config.get('constraints', {})
        self.min_allocation = constraints.get('min_allocation', 0.0)
        self.max_allocation = constraints.get('max_allocation', 2.0)
        self.max_vol_ratio = constraints.get('max_vol_ratio', 1.2)
        self.max_leverage_pct = constraints.get('max_leverage_pct', 0.10)
        
        logger.info(f"{self.__class__.__name__} initialized")
        logger.info(f"Allocation range: [{self.min_allocation}, {self.max_allocation}]")
        logger.info(f"Max vol ratio: {self.max_vol_ratio}")
        logger.info(f"Max leverage pct: {self.max_leverage_pct}")
    
    @abstractmethod
    def map_positions(
        self,
        r_hat: np.ndarray,
        sigma_hat: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Map predictions to positions.
        
        Parameters
        ----------
        r_hat : np.ndarray
            Predicted returns
        sigma_hat : np.ndarray
            Predicted volatility/risk
        **kwargs
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Position allocations (0 to 2)
        """
        pass
    
    def clip_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Clip positions to valid range.
        
        Parameters
        ----------
        positions : np.ndarray
            Raw position allocations
            
        Returns
        -------
        np.ndarray
            Clipped positions
        """
        return np.clip(positions, self.min_allocation, self.max_allocation)
    
    def validate_constraints(
        self,
        positions: np.ndarray,
        actual_returns: np.ndarray,
        market_returns: np.ndarray
    ) -> Dict[str, Union[bool, float]]:
        """
        Validate position constraints.
        
        Parameters
        ----------
        positions : np.ndarray
            Position allocations
        actual_returns : np.ndarray
            Actual market returns
        market_returns : np.ndarray
            Market returns (same as actual_returns)
            
        Returns
        -------
        dict
            Constraint validation results
        """
        results = {}
        
        # Calculate strategy returns
        strategy_returns = positions * actual_returns
        
        # Volatility ratio check
        strategy_vol = np.std(strategy_returns)
        market_vol = np.std(market_returns)
        vol_ratio = strategy_vol / (market_vol + 1e-10)
        vol_violations = np.sum(vol_ratio > self.max_vol_ratio)
        vol_violation_pct = vol_violations / len(positions)
        
        results['vol_ratio'] = vol_ratio
        results['vol_ratio_ok'] = vol_ratio <= self.max_vol_ratio
        results['vol_violation_pct'] = vol_violation_pct
        
        # Leverage check (2x positions)
        leverage_days = np.sum(positions >= 1.9)  # Close to 2.0
        leverage_pct = leverage_days / len(positions)
        
        results['leverage_days'] = leverage_days
        results['leverage_pct'] = leverage_pct
        results['leverage_ok'] = leverage_pct <= self.max_leverage_pct
        
        # Overall validation
        results['all_constraints_ok'] = (
            results['vol_ratio_ok'] and 
            results['leverage_ok']
        )
        
        logger.info("Constraint Validation:")
        logger.info(f"  Vol Ratio: {vol_ratio:.4f} (max: {self.max_vol_ratio})")
        logger.info(f"  Vol Violation %: {vol_violation_pct*100:.2f}%")
        logger.info(f"  Leverage Days: {leverage_days}/{len(positions)} ({leverage_pct*100:.2f}%)")
        logger.info(f"  All OK: {results['all_constraints_ok']}")
        
        return results


class SharpeScalingMapper(BasePositionMapper):
    """
    Sharpe-based position scaling strategy.
    
    Formula: a = clip(1 + k * tanh(b * r_hat / (sigma_hat + eps)), 0, 2)
    
    Parameters:
    - k: Scaling factor (controls overall position size)
    - b: Sensitivity parameter (controls response to signal strength)
    - eps: Small constant to avoid division by zero
    """
    
    def __init__(self, config_path: str = "conf/params.yaml"):
        """Initialize Sharpe scaling mapper."""
        super().__init__(config_path)

        logger.info("="*80)
        logger.info("Sharpe Scaling Mapper")
        logger.info("="*80)
        
        sharpe_config = self.position_config.get('sharpe_scaling', {})
        self.k = float(sharpe_config.get('k', 1.0))
        self.b = float(sharpe_config.get('b', 2.0))
        self.eps = float(sharpe_config.get('eps', 1e-6))
        
        logger.info(f"Sharpe Scaling Parameters: k={self.k}, b={self.b}, eps={self.eps}")
    
    def map_positions(
        self,
        r_hat: np.ndarray,
        sigma_hat: np.ndarray,
        k: Optional[float] = None,
        b: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Map predictions to positions using Sharpe scaling.
        
        Parameters
        ----------
        r_hat : np.ndarray
            Predicted returns
        sigma_hat : np.ndarray
            Predicted volatility
        k : float, optional
            Scaling factor (overrides config)
        b : float, optional
            Sensitivity parameter (overrides config)
            
        Returns
        -------
        np.ndarray
            Position allocations
        """
        # Use provided parameters or default from config
        k = k if k is not None else self.k
        b = b if b is not None else self.b
        
        # Calculate z-score (risk-adjusted signal)
        z = r_hat / (sigma_hat + self.eps)
        
        # Apply Sharpe scaling formula
        positions = 1.0 + k * np.tanh(b * z)
        
        # Clip to valid range
        positions = self.clip_positions(positions)
        
        return positions


class QuantileBinningMapper(BasePositionMapper):
    """
    Quantile-based binning strategy.
    
    Divides z-score (r_hat/sigma_hat) into quantile bins and assigns
    optimal allocation to each bin.
    
    IMPORTANT: Must call fit() on training data before using map_positions()!
    """
    
    def __init__(self, config_path: str = "conf/params.yaml"):
        """Initialize quantile binning mapper."""
        super().__init__(config_path)
        
        quantile_config = self.position_config.get('quantile', {})
        self.n_bins = int(quantile_config.get('n_bins', 7))
        self.allocations = np.array(quantile_config.get(
            'allocations',
            [0.0, 0.3, 0.6, 1.0, 1.4, 1.7, 2.0]
        ), dtype=float)
        
        # Ensure allocations match n_bins
        if len(self.allocations) != self.n_bins:
            logger.warning(f"Allocations length ({len(self.allocations)}) != n_bins ({self.n_bins})")
            logger.warning("Using default linear allocations")
            self.allocations = np.linspace(0.0, 2.0, self.n_bins)
        
        # Bin edges (fitted on training data)
        self.bin_edges = None
        self.is_fitted = False
        
        logger.info(f"Quantile Binning: {self.n_bins} bins")
        logger.info(f"Allocations: {self.allocations}")
    
    def fit(self, r_hat: np.ndarray, sigma_hat: np.ndarray) -> 'QuantileBinningMapper':
        """
        Fit bin edges on training data.
        
        Parameters
        ----------
        r_hat : np.ndarray
            Training predicted returns
        sigma_hat : np.ndarray
            Training predicted volatility
            
        Returns
        -------
        self : QuantileBinningMapper
            Fitted mapper
        """
        # Calculate z-score
        eps = 1e-6
        z = r_hat / (sigma_hat + eps)
        
        # Calculate quantile bins from training data
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        self.bin_edges = np.quantile(z, quantiles)
        
        self.is_fitted = True
        logger.info(f"Fitted bin edges: {self.bin_edges}")
        logger.info(f"Z-score range: [{z.min():.4f}, {z.max():.4f}]")
        
        return self
    
    def map_positions(
        self,
        r_hat: np.ndarray,
        sigma_hat: np.ndarray,
        allocations: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Map predictions to positions using quantile binning.
        
        Parameters
        ----------
        r_hat : np.ndarray
            Predicted returns
        sigma_hat : np.ndarray
            Predicted volatility
        allocations : np.ndarray, optional
            Bin allocations (overrides config)
            
        Returns
        -------
        np.ndarray
            Position allocations
        """
        # Check if fitted
        if not self.is_fitted:
            logger.warning("QuantileBinningMapper not fitted! Using simple mean-based binning.")
            # Fallback: fit on current data (not recommended for production)
            self.fit(r_hat, sigma_hat)
        
        # Use provided allocations or default from config
        allocations = allocations if allocations is not None else self.allocations
        
        # Calculate z-score
        eps = 1e-6
        z = r_hat / (sigma_hat + eps)
        
        # Assign positions based on pre-fitted bins
        positions = np.zeros_like(z)
        for i in range(self.n_bins):
            if i == 0:
                mask = z <= self.bin_edges[i + 1]
            elif i == self.n_bins - 1:
                mask = z > self.bin_edges[i]
            else:
                mask = (z > self.bin_edges[i]) & (z <= self.bin_edges[i + 1])
            
            positions[mask] = allocations[i]
        
        # Clip to valid range
        positions = self.clip_positions(positions)
        
        return positions


class VolatilityTargetingMapper(BasePositionMapper):
    """
    Volatility targeting strategy.
    
    Adjusts positions to target a specific volatility level,
    with constraints on daily position changes.
    """
    
    def __init__(self, config_path: str = "conf/params.yaml"):
        """Initialize volatility targeting mapper."""
        super().__init__(config_path)
        
        vol_config = self.position_config.get('vol_targeting', {})
        self.target_vol = float(vol_config.get('target_vol', 1.0))
        self.max_daily_change = float(vol_config.get('max_daily_change', 0.3))
        self.lookback_window = int(vol_config.get('lookback_window', 20))
        
        logger.info(f"Vol Targeting: target={self.target_vol}, max_change={self.max_daily_change}")
    
    def map_positions(
        self,
        r_hat: np.ndarray,
        sigma_hat: np.ndarray,
        market_vol: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Map predictions to positions using volatility targeting.
        
        Parameters
        ----------
        r_hat : np.ndarray
            Predicted returns
        sigma_hat : np.ndarray
            Predicted volatility
        market_vol : float, optional
            Market volatility (if not provided, calculated from sigma_hat)
            
        Returns
        -------
        np.ndarray
            Position allocations
        """
        # Base signal strength
        eps = 1e-6
        signal_strength = r_hat / (sigma_hat + eps)
        
        # Normalize to [0, 1] range using sigmoid
        normalized_signal = expit(signal_strength)
        
        # Calculate rolling market volatility if not provided
        if market_vol is None:
            market_vol = np.mean(sigma_hat)
        
        # Target position based on signal and volatility
        # When volatility is high, reduce position; when low, increase
        vol_scalar = (market_vol * self.target_vol) / (sigma_hat + eps)
        vol_scalar = np.clip(vol_scalar, 0.5, 2.0)  # Limit volatility adjustment
        
        # Base position from signal
        base_positions = normalized_signal * 2.0  # Scale to [0, 2]
        
        # Adjust by volatility
        positions = base_positions * vol_scalar
        
        # Apply daily change limit
        positions = self._apply_daily_change_limit(positions)
        
        # Clip to valid range
        positions = self.clip_positions(positions)
        
        return positions
    
    def _apply_daily_change_limit(self, positions: np.ndarray) -> np.ndarray:
        """
        Apply maximum daily position change constraint.
        
        Parameters
        ----------
        positions : np.ndarray
            Raw position allocations
            
        Returns
        -------
        np.ndarray
            Positions with change limit applied
        """
        limited_positions = np.zeros_like(positions)
        limited_positions[0] = positions[0]
        
        for i in range(1, len(positions)):
            change = positions[i] - limited_positions[i - 1]
            
            # Limit change
            if abs(change) > self.max_daily_change:
                change = np.sign(change) * self.max_daily_change
            
            limited_positions[i] = limited_positions[i - 1] + change
        
        return limited_positions


class PositionOptimizer:
    """
    Optimize position mapping parameters using custom metric.
    
    Uses Optuna or scipy.optimize to find optimal parameters
    that maximize the custom Sharpe-based metric.
    """
    
    def __init__(
        self,
        mapper: BasePositionMapper,
        config_path: str = "conf/params.yaml"
    ):
        """
        Initialize position optimizer.
        
        Parameters
        ----------
        mapper : BasePositionMapper
            Position mapping strategy to optimize
        config_path : str
            Path to configuration file
        """
        self.mapper = mapper
        self.config = load_config(config_path)
        
        logger.info(f"PositionOptimizer initialized for {mapper.__class__.__name__}")
    
    def optimize_sharpe_params(
        self,
        r_hat: np.ndarray,
        sigma_hat: np.ndarray,
        actual_returns: np.ndarray,
        k_range: Tuple[float, float] = (0.5, 2.0),
        b_range: Tuple[float, float] = (1.0, 5.0)
    ) -> Dict[str, float]:
        """
        Optimize Sharpe scaling parameters (k, b).
        
        Parameters
        ----------
        r_hat : np.ndarray
            Predicted returns
        sigma_hat : np.ndarray
            Predicted volatility
        actual_returns : np.ndarray
            Actual market returns
        k_range : tuple
            Range for k parameter
        b_range : tuple
            Range for b parameter
            
        Returns
        -------
        dict
            Optimal parameters
        """
        logger.info("="*80)
        logger.info("Optimizing Sharpe Scaling Parameters")
        logger.info("="*80)
        
        def objective(params):
            k, b = params
            
            # Map positions
            positions = self.mapper.map_positions(r_hat, sigma_hat, k=k, b=b)
            
            # Calculate custom metric (negative for minimization)
            metric_calc = CompetitionMetric()
            result = metric_calc.calculate_score(
                allocations=positions,
                forward_returns=actual_returns
            )
            score = result['score']
            
            return -score  # Minimize negative score
        
        # Initial guess
        x0 = [1.0, 2.0]
        
        # Bounds
        bounds = [k_range, b_range]
        
        # Optimize
        with Timer("Parameter optimization"):
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds
            )
        
        optimal_k, optimal_b = result.x
        optimal_score = -result.fun
        
        logger.info("\nOptimization Results:")
        logger.info(f"  Optimal k: {optimal_k:.4f}")
        logger.info(f"  Optimal b: {optimal_b:.4f}")
        logger.info(f"  Optimal score: {optimal_score:.6f}")
        
        return {
            'k': optimal_k,
            'b': optimal_b,
            'score': optimal_score
        }
    
    def optimize_quantile_allocations(
        self,
        r_hat: np.ndarray,
        sigma_hat: np.ndarray,
        actual_returns: np.ndarray,
        n_bins: int = 7
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Optimize quantile bin allocations.
        
        Parameters
        ----------
        r_hat : np.ndarray
            Predicted returns
        sigma_hat : np.ndarray
            Predicted volatility
        actual_returns : np.ndarray
            Actual market returns
        n_bins : int
            Number of bins
            
        Returns
        -------
        dict
            Optimal allocations and score
        """
        logger.info("="*80)
        logger.info("Optimizing Quantile Bin Allocations")
        logger.info("="*80)
        
        def objective(allocations):
            # Map positions
            positions = self.mapper.map_positions(r_hat, sigma_hat, allocations=allocations)
            
            # Calculate custom metric (negative for minimization)
            metric_calc = CompetitionMetric()
            result = metric_calc.calculate_score(
                allocations=positions,
                forward_returns=actual_returns
            )
            score = result['score']
            
            return -score
        
        # Initial guess (linear from 0 to 2)
        x0 = np.linspace(0.0, 2.0, n_bins)
        
        # Bounds (each allocation between 0 and 2)
        bounds = [(0.0, 2.0) for _ in range(n_bins)]
        
        # Optimize
        with Timer("Allocation optimization"):
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds
            )
        
        optimal_allocations = result.x
        optimal_score = -result.fun
        
        logger.info("\nOptimization Results:")
        logger.info(f"  Optimal allocations: {optimal_allocations}")
        logger.info(f"  Optimal score: {optimal_score:.6f}")
        
        return {
            'allocations': optimal_allocations,
            'score': optimal_score
        }


def create_position_mapper(
    strategy: str = 'sharpe_scaling',
    config_path: str = "conf/params.yaml"
) -> BasePositionMapper:
    """
    Factory function to create position mapper.
    
    Parameters
    ----------
    strategy : str
        Strategy name: 'sharpe_scaling', 'quantile', 'vol_targeting'
    config_path : str
        Path to configuration file
        
    Returns
    -------
    BasePositionMapper
        Position mapper instance
    """
    logger.info("="*80)
    logger.info("Create Position Allocations")
    logger.info("="*80)

    strategy_map = {
        'sharpe_scaling': SharpeScalingMapper,
        'quantile': QuantileBinningMapper,
        'vol_targeting': VolatilityTargetingMapper
    }
    
    if strategy not in strategy_map:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Available: {list(strategy_map.keys())}"
        )
    
    return strategy_map[strategy](config_path)
