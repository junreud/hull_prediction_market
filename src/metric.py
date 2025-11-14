"""
Custom evaluation metrics for Hull Tactical Market Prediction.

This module provides:
- Modified Sharpe ratio calculation
- Volatility penalty computation
- Underperformance penalty
- Strategy evaluation metrics
"""

from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
from src.utils import get_logger, load_config

logger = get_logger(log_file="logs/prediction_market.log", level="INFO")


class CompetitionMetric:
    """
    Competition metric calculator.
    
    Implements the modified Sharpe ratio with volatility penalty:
    score = mean(strategy_returns) / std(strategy_returns) / vol_penalty
    
    where:
    - strategy_returns = allocation_t × forward_returns_t
    - vol_penalty = 1 + max(0, (strategy_vol / market_vol) - 1.2)
    """
    
    def __init__(
        self,
        vol_threshold: float = 1.2,
        underperformance_penalty: bool = False,
        min_periods: int = 30,
        eps: float = 1e-10
    ):
        """
        Initialize metric calculator.
        
        Parameters
        ----------
        vol_threshold : float
            Maximum allowed strategy volatility ratio (default: 1.2)
        underperformance_penalty : bool
            Whether to apply penalty for underperforming market (default: False)
        min_periods : int
            Minimum number of periods required for valid calculation
        eps : float
            Small constant to prevent division by zero
        """
        self.vol_threshold = vol_threshold
        self.underperformance_penalty = underperformance_penalty
        self.min_periods = min_periods
        self.eps = eps
        
    def calculate_volatility_penalty(
        self,
        strategy_vol: float,
        market_vol: float
    ) -> float:
        """
        Calculate volatility penalty.
        
        vol_penalty = 1 + max(0, (strategy_vol / market_vol) - threshold)
        
        Parameters
        ----------
        strategy_vol : float
            Strategy volatility (std of strategy returns)
        market_vol : float
            Market volatility (std of market returns)
            
        Returns
        -------
        float
            Volatility penalty factor (≥ 1.0)
        """
        if market_vol < self.eps:
            logger.warning("Market volatility near zero, setting penalty to 1.0")
            return 1.0
        
        vol_ratio = strategy_vol / (market_vol + self.eps)
        penalty = 1.0 + max(0.0, vol_ratio - self.vol_threshold)
        
        return penalty
    
    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Parameters
        ----------
        returns : np.ndarray
            Array of returns
        risk_free_rate : np.ndarray, optional
            Risk-free rate for each period
            
        Returns
        -------
        float
            Sharpe ratio
        """
        # Remove NaN values
        valid_mask = ~np.isnan(returns)
        returns_clean = returns[valid_mask]
        
        if len(returns_clean) < self.min_periods:
            logger.warning(f"Insufficient data: {len(returns_clean)} < {self.min_periods}")
            return 0.0
        
        # Adjust for risk-free rate if provided
        if risk_free_rate is not None:
            rfr_clean = risk_free_rate[valid_mask]
            excess_returns = returns_clean - rfr_clean
        else:
            excess_returns = returns_clean
        
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)
        
        if std_return < self.eps:
            logger.warning("Return volatility near zero")
            return 0.0
        
        sharpe = mean_return / std_return
        
        return sharpe
    
    def calculate_score(
        self,
        allocations: np.ndarray,
        forward_returns: np.ndarray,
        market_returns: Optional[np.ndarray] = None,
        risk_free_rate: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate competition score and related metrics.
        
        Parameters
        ----------
        allocations : np.ndarray
            Strategy allocations (0 to 2)
        forward_returns : np.ndarray
            Forward returns (market returns)
        market_returns : np.ndarray, optional
            Market returns for comparison (defaults to forward_returns)
        risk_free_rate : np.ndarray, optional
            Risk-free rate for each period
            
        Returns
        -------
        dict
            Dictionary containing:
            - score: Final competition score
            - sharpe: Sharpe ratio before penalty
            - vol_penalty: Volatility penalty factor
            - strategy_vol: Strategy volatility
            - market_vol: Market volatility
            - vol_ratio: strategy_vol / market_vol
            - mean_return: Mean strategy return
            - std_return: Std strategy return
            - vol_violation_rate: Fraction of periods violating threshold
        """
        # Input validation
        if len(allocations) != len(forward_returns):
            raise ValueError(
                f"Length mismatch: allocations({len(allocations)}) != "
                f"forward_returns({len(forward_returns)})"
            )
        
        # Remove NaN values
        valid_mask = ~(np.isnan(allocations) | np.isnan(forward_returns))
        allocations_clean = allocations[valid_mask]
        forward_returns_clean = forward_returns[valid_mask]
        
        if len(allocations_clean) < self.min_periods:
            logger.warning(
                f"Insufficient valid data: {len(allocations_clean)} < {self.min_periods}"
            )
            return {
                'score': 0.0,
                'sharpe': 0.0,
                'vol_penalty': 1.0,
                'strategy_vol': 0.0,
                'market_vol': 0.0,
                'vol_ratio': 0.0,
                'mean_return': 0.0,
                'std_return': 0.0,
                'vol_violation_rate': 0.0,
                'n_valid': len(allocations_clean)
            }
        
        # Calculate strategy returns
        strategy_returns = allocations_clean * forward_returns_clean
        
        # Use forward_returns as market if not provided
        if market_returns is None:
            market_returns_clean = forward_returns_clean
        else:
            market_returns_clean = market_returns[valid_mask]
        
        # Calculate volatilities
        strategy_vol = np.std(strategy_returns, ddof=1)
        market_vol = np.std(market_returns_clean, ddof=1)
        
        # Calculate Sharpe ratio
        if risk_free_rate is not None:
            rfr_clean = risk_free_rate[valid_mask]
        else:
            rfr_clean = None
        
        sharpe = self.calculate_sharpe_ratio(strategy_returns, rfr_clean)
        
        # Calculate volatility penalty
        vol_penalty = self.calculate_volatility_penalty(strategy_vol, market_vol)
        
        # Calculate final score
        if abs(sharpe) < self.eps or vol_penalty < self.eps:
            score = 0.0
        else:
            score = sharpe / vol_penalty
        
        # Additional diagnostics
        vol_ratio = strategy_vol / (market_vol + self.eps)
        mean_return = np.mean(strategy_returns)
        std_return = strategy_vol
        
        # Calculate violation rate (rolling window check)
        # Approximate by checking if current vol_ratio exceeds threshold
        vol_violation_rate = float(vol_ratio > self.vol_threshold)
        
        # Apply underperformance penalty if enabled
        if self.underperformance_penalty:
            mean_market = np.mean(market_returns_clean)
            if mean_return < mean_market:
                underperf_penalty = 1.0 + (mean_market - mean_return)
                score = score / underperf_penalty
                logger.info(f"Underperformance penalty applied: {underperf_penalty:.4f}")
        
        return {
            'score': float(score),
            'sharpe': float(sharpe),
            'vol_penalty': float(vol_penalty),
            'strategy_vol': float(strategy_vol),
            'market_vol': float(market_vol),
            'vol_ratio': float(vol_ratio),
            'mean_return': float(mean_return),
            'std_return': float(std_return),
            'vol_violation_rate': float(vol_violation_rate),
            'n_valid': int(len(allocations_clean))
        }
    
    def calculate_rolling_metrics(
        self,
        allocations: np.ndarray,
        forward_returns: np.ndarray,
        window: int = 252,
        market_returns: Optional[np.ndarray] = None,
        risk_free_rate: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling window metrics.
        
        Parameters
        ----------
        allocations : np.ndarray
            Strategy allocations
        forward_returns : np.ndarray
            Forward returns
        window : int
            Rolling window size (default: 252 trading days ≈ 1 year)
        market_returns : np.ndarray, optional
            Market returns
        risk_free_rate : np.ndarray, optional
            Risk-free rate
            
        Returns
        -------
        pd.DataFrame
            DataFrame with rolling metrics
        """
        n = len(allocations)
        if n < window:
            logger.warning(f"Data length {n} < window {window}, returning empty DataFrame")
            return pd.DataFrame()
        
        strategy_returns = allocations * forward_returns
        if market_returns is None:
            market_returns = forward_returns
        
        results = []
        
        for i in range(window, n + 1):
            window_slice = slice(i - window, i)
            
            metrics = self.calculate_score(
                allocations[window_slice],
                forward_returns[window_slice],
                market_returns[window_slice],
                risk_free_rate[window_slice] if risk_free_rate is not None else None
            )
            metrics['period_end'] = i
            results.append(metrics)
        
        return pd.DataFrame(results)


def calculate_additional_metrics(
    allocations: np.ndarray,
    forward_returns: np.ndarray,
    market_returns: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate additional performance metrics.
    
    Parameters
    ----------
    allocations : np.ndarray
        Strategy allocations
    forward_returns : np.ndarray
        Forward returns
    market_returns : np.ndarray, optional
        Market returns for comparison
        
    Returns
    -------
    dict
        Additional metrics including:
        - max_drawdown: Maximum drawdown
        - calmar_ratio: Return / max drawdown
        - turnover: Average daily position change
        - leverage_2x_rate: Fraction of days with 2x leverage
        - avg_allocation: Average allocation
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(allocations) | np.isnan(forward_returns))
    allocations_clean = allocations[valid_mask]
    forward_returns_clean = forward_returns[valid_mask]
    
    if len(allocations_clean) == 0:
        return {
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'turnover': 0.0,
            'leverage_2x_rate': 0.0,
            'avg_allocation': 0.0
        }
    
    # Calculate cumulative returns
    strategy_returns = allocations_clean * forward_returns_clean
    cumulative = np.cumprod(1 + strategy_returns) - 1
    
    # Maximum drawdown
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / (1 + running_max)
    max_drawdown = np.min(drawdown)
    
    # Calmar ratio
    mean_return = np.mean(strategy_returns)
    calmar_ratio = mean_return / abs(max_drawdown) if abs(max_drawdown) > 1e-10 else 0.0
    
    # Turnover (average daily allocation change)
    turnover = np.mean(np.abs(np.diff(allocations_clean)))
    
    # 2x leverage usage rate
    leverage_2x_rate = np.mean(allocations_clean >= 1.9)  # Consider ≥1.9 as 2x
    
    # Average allocation
    avg_allocation = np.mean(allocations_clean)
    
    return {
        'max_drawdown': float(max_drawdown),
        'calmar_ratio': float(calmar_ratio),
        'turnover': float(turnover),
        'leverage_2x_rate': float(leverage_2x_rate),
        'avg_allocation': float(avg_allocation)
    }


def create_metric_calculator(
    config_path: str = "conf/params.yaml",
    **kwargs
) -> CompetitionMetric:
    """
    Factory function to create metric calculator from config.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    **kwargs
        Override configuration parameters
        
    Returns
    -------
    CompetitionMetric
        Configured metric calculator
    """
    config = load_config(config_path)
    metric_config = config.get('metric', {})
    
    # Merge config with kwargs
    params = {
        'vol_threshold': metric_config.get('vol_threshold', 1.2),
        'underperformance_penalty': metric_config.get('underperformance_penalty', False),
        'min_periods': metric_config.get('min_periods', 30),
    }
    params.update(kwargs)
    
    return CompetitionMetric(**params)
