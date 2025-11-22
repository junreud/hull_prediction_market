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
    
    Implements the competition metric (matches validation.py):
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    
    where:
    - sharpe = strategy_mean_excess_return / strategy_std * sqrt(252)
    - vol_penalty = 1 + max(0, (strategy_vol / market_vol) - 1.2)
    - return_penalty = 1 + (return_gap^2) / 100
    - return_gap = max(0, (market_return - strategy_return) * 100 * 252)
    """
    
    def __init__(
        self,
        vol_threshold: float = 1.2,
        use_return_penalty: bool = True,
        min_periods: int = 30,
        eps: float = 1e-10
    ):
        """
        Initialize metric calculator.
        
        Parameters
        ----------
        vol_threshold : float
            Maximum allowed strategy volatility ratio (default: 1.2)
        use_return_penalty : bool
            Whether to apply return penalty for underperforming market (default: True)
        min_periods : int
            Minimum number of periods required for valid calculation
        eps : float
            Small constant to prevent division by zero
        """
        self.vol_threshold = vol_threshold
        self.use_return_penalty = use_return_penalty
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
    
    def calculate_return_penalty(
        self,
        strategy_mean_excess_return: float,
        market_mean_excess_return: float,
        trading_days_per_yr: int = 252
    ) -> float:
        """
        Calculate return penalty (as per validation.py).
        
        return_penalty = 1 + (return_gap^2) / 100
        where return_gap = max(0, (market_return - strategy_return) * 252 * 100)
        
        Parameters
        ----------
        strategy_mean_excess_return : float
            Strategy mean excess return (daily)
        market_mean_excess_return : float
            Market mean excess return (daily)
        trading_days_per_yr : int
            Number of trading days per year (default: 252)
            
        Returns
        -------
        float
            Return penalty factor (≥ 1.0)
        """
        # Calculate return gap (matching Kaggle's validation.py - note: order is intentional)
        # Kaggle uses: * 100 * 252 (not * 252 * 100)
        return_gap = max(
            0.0,
            (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr
        )
        
        # Apply quadratic penalty
        penalty = 1.0 + (return_gap ** 2) / 100
        
        return penalty
    
    def calculate_score(
        self,
        allocations: np.ndarray,
        forward_returns: np.ndarray,
        market_returns: Optional[np.ndarray] = None,
        risk_free_rate: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate competition score and related metrics (matches validation.py).
        
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
            - score: Final competition score (adjusted_sharpe)
            - sharpe: Sharpe ratio before penalty
            - vol_penalty: Volatility penalty factor
            - return_penalty: Return penalty factor
            - strategy_vol: Strategy volatility (annualized %)
            - market_vol: Market volatility (annualized %)
            - vol_ratio: strategy_vol / market_vol
            - mean_return: Mean strategy return
            - std_return: Std strategy return
            - strategy_mean_excess_return: Strategy mean excess return
            - market_mean_excess_return: Market mean excess return
            - return_gap: Return gap (annualized %)
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
                'return_penalty': 1.0,
                'strategy_vol': 0.0,
                'market_vol': 0.0,
                'vol_ratio': 0.0,
                'mean_return': 0.0,
                'std_return': 0.0,
                'strategy_mean_excess_return': 0.0,
                'market_mean_excess_return': 0.0,
                'return_gap': 0.0,
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
        
        # Handle risk-free rate
        if risk_free_rate is not None:
            rfr_clean = risk_free_rate[valid_mask]
        else:
            rfr_clean = np.zeros_like(strategy_returns)
        
        # Calculate excess returns (matching validation.py)
        strategy_excess_returns = strategy_returns - rfr_clean
        market_excess_returns = market_returns_clean - rfr_clean
        
        # Calculate cumulative returns for mean calculation
        strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
        market_excess_cumulative = (1 + market_excess_returns).prod()
        
        # Calculate mean excess returns (geometric mean)
        n_periods = len(strategy_excess_returns)
        strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / n_periods) - 1
        market_mean_excess_return = (market_excess_cumulative) ** (1 / n_periods) - 1
        
        # Calculate volatilities (std of returns, not excess returns)
        strategy_std = np.std(strategy_returns, ddof=1)
        market_std = np.std(market_returns_clean, ddof=1)
        
        # Annualize volatilities (as percentage)
        trading_days_per_yr = 252
        strategy_vol = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)
        market_vol = float(market_std * np.sqrt(trading_days_per_yr) * 100)
        
        # Calculate Sharpe ratio (matching validation.py)
        if strategy_std < self.eps:
            logger.warning("Strategy volatility near zero")
            sharpe = 0.0
        else:
            sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
        
        # Calculate volatility penalty
        vol_penalty = self.calculate_volatility_penalty(strategy_vol, market_vol)
        
        # Calculate return penalty (if enabled)
        if self.use_return_penalty:
            return_penalty = self.calculate_return_penalty(
                strategy_mean_excess_return,
                market_mean_excess_return,
                trading_days_per_yr
            )
        else:
            return_penalty = 1.0
        
        # Calculate return gap for reporting (same order as validation.py)
        return_gap = max(
            0.0,
            (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr
        )
        
        # Calculate final score (adjusted Sharpe)
        if abs(sharpe) < self.eps or vol_penalty < self.eps or return_penalty < self.eps:
            score = 0.0
        else:
            score = sharpe / (vol_penalty * return_penalty)
        
        # Cap score at 1,000,000 (matching validation.py)
        score = min(float(score), 1_000_000)
        
        # Additional diagnostics
        vol_ratio = strategy_vol / (market_vol + self.eps)
        mean_return = np.mean(strategy_returns)
        std_return = strategy_std
        
        # Calculate violation rate (rolling window check)
        # Approximate by checking if current vol_ratio exceeds threshold
        vol_violation_rate = float(vol_ratio > self.vol_threshold)
        
        return {
            'score': float(score),
            'sharpe': float(sharpe),
            'vol_penalty': float(vol_penalty),
            'return_penalty': float(return_penalty),
            'strategy_vol': float(strategy_vol),
            'market_vol': float(market_vol),
            'vol_ratio': float(vol_ratio),
            'mean_return': float(mean_return),
            'std_return': float(std_return),
            'strategy_mean_excess_return': float(strategy_mean_excess_return),
            'market_mean_excess_return': float(market_mean_excess_return),
            'return_gap': float(return_gap),
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


def evaluate_return_model(
    r_hat: np.ndarray,
    actual_returns: np.ndarray,
    return_all_metrics: bool = True
) -> Dict[str, float]:
    """
    Evaluate return prediction model performance.
    
    This evaluates the quality of return predictions (r_hat) independently
    from position sizing or risk management.
    
    Parameters
    ----------
    r_hat : np.ndarray
        Predicted returns
    actual_returns : np.ndarray
        Actual forward returns
    return_all_metrics : bool
        Whether to return all metrics or just main score
        
    Returns
    -------
    dict
        Return model evaluation metrics including:
        - directional_accuracy: Percentage of correct direction predictions
        - correlation: Pearson correlation between prediction and actual
        - rank_correlation: Spearman rank correlation (robust to outliers)
        - information_coefficient: IC, correlation of ranked predictions/returns
        - mse: Mean squared error
        - mae: Mean absolute error
        - rmse: Root mean squared error
        - top_quintile_return: Mean return of top 20% predictions
        - bottom_quintile_return: Mean return of bottom 20% predictions
        - long_short_spread: Difference between top and bottom quintiles
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(r_hat) | np.isnan(actual_returns))
    r_hat_clean = r_hat[valid_mask]
    actual_clean = actual_returns[valid_mask]
    
    if len(r_hat_clean) < 30:
        logger.warning(f"Insufficient data for return model evaluation: {len(r_hat_clean)}")
        return {
            'directional_accuracy': 0.0,
            'correlation': 0.0,
            'rank_correlation': 0.0,
            'information_coefficient': 0.0,
            'mse': np.inf,
            'mae': np.inf,
            'rmse': np.inf,
            'top_quintile_return': 0.0,
            'bottom_quintile_return': 0.0,
            'long_short_spread': 0.0
        }
    
    # 1. Directional Accuracy (방향 예측 정확도)
    directional_accuracy = np.mean(np.sign(r_hat_clean) == np.sign(actual_clean))
    
    # 2. Pearson Correlation (선형 상관관계)
    correlation = np.corrcoef(r_hat_clean, actual_clean)[0, 1]
    
    # 3. Spearman Rank Correlation (순위 기반, 이상치에 강건)
    from scipy.stats import spearmanr
    rank_correlation = spearmanr(r_hat_clean, actual_clean)[0]
    
    # 4. Information Coefficient (IC) - 순위 상관관계
    r_hat_ranks = pd.Series(r_hat_clean).rank()
    actual_ranks = pd.Series(actual_clean).rank()
    information_coefficient = np.corrcoef(r_hat_ranks, actual_ranks)[0, 1]
    
    # 5. Error Metrics
    mse = np.mean((r_hat_clean - actual_clean) ** 2)
    mae = np.mean(np.abs(r_hat_clean - actual_clean))
    rmse = np.sqrt(mse)
    
    # 6. Quintile Analysis (상위/하위 20% 예측의 실제 수익률)
    top_20_threshold = np.percentile(r_hat_clean, 80)
    bottom_20_threshold = np.percentile(r_hat_clean, 20)
    
    top_quintile_mask = r_hat_clean >= top_20_threshold
    bottom_quintile_mask = r_hat_clean <= bottom_20_threshold
    
    top_quintile_return = np.mean(actual_clean[top_quintile_mask])
    bottom_quintile_return = np.mean(actual_clean[bottom_quintile_mask])
    long_short_spread = top_quintile_return - bottom_quintile_return
    
    results = {
        'directional_accuracy': float(directional_accuracy),
        'correlation': float(correlation),
        'rank_correlation': float(rank_correlation),
        'information_coefficient': float(information_coefficient),
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'top_quintile_return': float(top_quintile_return),
        'bottom_quintile_return': float(bottom_quintile_return),
        'long_short_spread': float(long_short_spread),
        'n_valid': int(len(r_hat_clean))
    }
    
    if return_all_metrics:
        return results
    else:
        # Return single score (IC is most important for return models)
        return information_coefficient


def evaluate_risk_model(
    sigma_hat: np.ndarray,
    actual_returns: np.ndarray,
    window: int = 20,
    return_all_metrics: bool = True
) -> Dict[str, float]:
    """
    Evaluate risk prediction model performance.
    
    This evaluates the quality of volatility/risk predictions independently
    from return predictions or position sizing.
    
    Parameters
    ----------
    sigma_hat : np.ndarray
        Predicted volatility/risk
    actual_returns : np.ndarray
        Actual forward returns (used to calculate realized volatility)
    window : int
        Window size for calculating realized volatility
    return_all_metrics : bool
        Whether to return all metrics or just main score
        
    Returns
    -------
    dict
        Risk model evaluation metrics including:
        - correlation: Correlation between predicted and realized volatility
        - rmse: Root mean squared error of volatility predictions
        - mae: Mean absolute error of volatility predictions
        - bias: Mean prediction error (positive = overestimation)
        - hit_rate: Accuracy of high volatility period detection
        - calibration_q20/q50/q80: Realized vol at different predicted quantiles
    """
    # Calculate realized volatility
    realized_vol = pd.Series(actual_returns).rolling(window=window, min_periods=10).std().values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(sigma_hat) | np.isnan(realized_vol))
    sigma_hat_clean = sigma_hat[valid_mask]
    realized_vol_clean = realized_vol[valid_mask]
    
    if len(sigma_hat_clean) < 30:
        logger.warning(f"Insufficient data for risk model evaluation: {len(sigma_hat_clean)}")
        return {
            'correlation': 0.0,
            'rmse': np.inf,
            'mae': np.inf,
            'bias': 0.0,
            'hit_rate': 0.0,
            'calibration_q20': 0.0,
            'calibration_q50': 0.0,
            'calibration_q80': 0.0
        }
    
    # 1. Correlation
    correlation = np.corrcoef(sigma_hat_clean, realized_vol_clean)[0, 1]
    
    # 2. RMSE
    rmse = np.sqrt(np.mean((sigma_hat_clean - realized_vol_clean) ** 2))
    
    # 3. MAE
    mae = np.mean(np.abs(sigma_hat_clean - realized_vol_clean))
    
    # 4. Bias (양수면 과대추정, 음수면 과소추정)
    bias = np.mean(sigma_hat_clean - realized_vol_clean)
    
    # 5. Hit Rate (고변동성 구간 탐지 정확도)
    high_risk_pred = sigma_hat_clean > np.percentile(sigma_hat_clean, 80)
    high_risk_actual = realized_vol_clean > np.percentile(realized_vol_clean, 80)
    hit_rate = np.mean(high_risk_pred == high_risk_actual)
    
    # 6. Calibration (예측 분위수별 실제 변동성)
    quantiles = [0.2, 0.5, 0.8]
    calibration = {}
    for q in quantiles:
        threshold = np.percentile(sigma_hat_clean, q * 100)
        mask = sigma_hat_clean >= threshold
        if mask.sum() > 0:
            calibration[f'calibration_q{int(q*100)}'] = float(np.mean(realized_vol_clean[mask]))
        else:
            calibration[f'calibration_q{int(q*100)}'] = 0.0
    
    results = {
        'correlation': float(correlation),
        'rmse': float(rmse),
        'mae': float(mae),
        'bias': float(bias),
        'hit_rate': float(hit_rate),
        **calibration,
        'n_valid': int(len(sigma_hat_clean))
    }
    
    if return_all_metrics:
        return results
    else:
        # Return single score (correlation is most important for risk models)
        return correlation


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
        'use_return_penalty': metric_config.get('use_return_penalty', True),
        'min_periods': metric_config.get('min_periods', 30),
    }
    params.update(kwargs)
    
    return CompetitionMetric(**params)