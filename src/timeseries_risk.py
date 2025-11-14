"""
Time-series based risk forecasting models.

This module provides:
1. EWMA (Exponentially Weighted Moving Average)
2. GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)
3. Hybrid ensemble combining multiple risk models
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pickle

from .utils import get_logger, load_config, Timer

logger = get_logger(__name__)


class EWMARiskForecaster:
    """
    EWMA (Exponentially Weighted Moving Average) risk forecaster.
    
    Features:
    - Fast and simple volatility prediction
    - Recent data gets higher weight
    - Single parameter (lambda) optimization
    - RiskMetrics standard: lambda = 0.94
    
    Formula:
        σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}
    """
    
    def __init__(
        self,
        lambda_: float = 0.94,
        min_periods: int = 20,
        config_path: str = "conf/params.yaml"
    ):
        """
        Initialize EWMA risk forecaster.
        
        Args:
            lambda_: Decay factor (0.9-0.99, higher = more smoothing)
            min_periods: Minimum periods for initial variance calculation
            config_path: Path to configuration file
        """
        config = load_config(config_path)
        ewma_config = config.get('risk', {}).get('ewma', {})
        
        self.lambda_ = lambda_ or ewma_config.get('lambda', 0.94)
        self.min_periods = min_periods or ewma_config.get('min_periods', 20)
        
        # Trained state
        self.variance_history = None
        self.is_fitted = False
        
        logger.info("EWMARiskForecaster initialized")
        logger.info(f"Lambda: {self.lambda_}, Min periods: {self.min_periods}")
    
    def fit(
        self,
        returns: Union[pd.Series, np.ndarray],
        date_ids: Optional[np.ndarray] = None
    ) -> 'EWMARiskForecaster':
        """
        Fit EWMA model on historical returns.
        
        Args:
            returns: Historical returns
            date_ids: Optional date identifiers (for logging)
            
        Returns:
            self
        """
        logger.info("="*80)
        logger.info("Fitting EWMA Risk Model")
        logger.info("="*80)
        
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        with Timer("EWMA fitting"):
            # Initialize variance array
            variance = np.zeros(len(returns))
            
            # Initial variance: simple variance of first min_periods
            if len(returns) >= self.min_periods:
                variance[0] = np.var(returns[:self.min_periods])
            else:
                variance[0] = np.var(returns)
            
            # EWMA recursion
            for t in range(1, len(returns)):
                variance[t] = (
                    self.lambda_ * variance[t-1] + 
                    (1 - self.lambda_) * returns[t-1]**2
                )
            
            self.variance_history = variance
            self.volatility_history = np.sqrt(variance)
            
            logger.info(f"\n✓ EWMA fitting complete")
            logger.info(f"  Samples: {len(returns)}")
            logger.info(f"  Mean volatility: {self.volatility_history.mean():.6f}")
            logger.info(f"  Volatility range: [{self.volatility_history.min():.6f}, {self.volatility_history.max():.6f}]")
        
        self.is_fitted = True
        return self
    
    def predict(
        self,
        returns: Optional[Union[pd.Series, np.ndarray]] = None,
        horizon: int = 1
    ) -> np.ndarray:
        """
        Predict future volatility.
        
        Args:
            returns: Recent returns (if None, use last fitted value)
            horizon: Forecast horizon (days ahead)
            
        Returns:
            Array of volatility predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if returns is None:
            # Use last variance from fitted history
            last_variance = self.variance_history[-1]
        else:
            if isinstance(returns, pd.Series):
                returns = returns.values
            
            # Update variance with new returns
            variance = self.variance_history[-1]
            for r in returns:
                variance = self.lambda_ * variance + (1 - self.lambda_) * r**2
            last_variance = variance
        
        # For EWMA, volatility converges to long-term mean
        # Multi-step forecast: variance stays constant (random walk assumption)
        if horizon == 1:
            return np.sqrt(last_variance)
        else:
            return np.full(horizon, np.sqrt(last_variance))
    
    def get_oof_predictions(
        self,
        returns: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """
        Get out-of-fold style predictions (one-step-ahead).
        
        For each time point, predict volatility using only past data.
        
        Args:
            returns: Historical returns
            
        Returns:
            Array of one-step-ahead volatility predictions
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        predictions = np.zeros(len(returns))
        variance = None
        
        for t in range(len(returns)):
            if t < self.min_periods:
                # Not enough data, use NaN
                predictions[t] = np.nan
            elif variance is None:
                # Initialize with variance of first min_periods
                variance = np.var(returns[:self.min_periods])
                predictions[t] = np.sqrt(variance)
            else:
                # Predict then update
                predictions[t] = np.sqrt(variance)
                variance = self.lambda_ * variance + (1 - self.lambda_) * returns[t]**2
        
        return predictions
    
    def save_model(self, output_path: str = "artifacts/ewma_risk_model.pkl"):
        """Save EWMA model state."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'lambda_': self.lambda_,
            'min_periods': self.min_periods,
            'variance_history': self.variance_history,
            'volatility_history': self.volatility_history,
            'is_fitted': self.is_fitted
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"✓ EWMA model saved to {output_path}")
    
    def load_model(self, model_path: str = "artifacts/ewma_risk_model.pkl"):
        """Load EWMA model state."""
        with open(model_path, 'rb') as f:
            state = pickle.load(f)
        
        self.lambda_ = state['lambda_']
        self.min_periods = state['min_periods']
        self.variance_history = state['variance_history']
        self.volatility_history = state['volatility_history']
        self.is_fitted = state['is_fitted']
        
        logger.info(f"✓ EWMA model loaded from {model_path}")


class GARCHRiskForecaster:
    """
    GARCH(1,1) risk forecaster.
    
    Features:
    - Statistically rigorous volatility modeling
    - Mean reversion to long-term volatility
    - Maximum Likelihood Estimation (MLE)
    - Handles volatility clustering
    
    Formula:
        σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}
    
    Requires: arch library (pip install arch)
    """
    
    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        mean: str = 'Zero',
        config_path: str = "conf/params.yaml"
    ):
        """
        Initialize GARCH risk forecaster.
        
        Args:
            p: GARCH order (lag of variance)
            q: ARCH order (lag of squared returns)
            mean: Mean model ('Zero', 'Constant', 'AR')
            config_path: Path to configuration file
        """
        config = load_config(config_path)
        garch_config = config.get('risk', {}).get('garch', {})
        
        self.p = p or garch_config.get('p', 1)
        self.q = q or garch_config.get('q', 1)
        self.mean = mean or garch_config.get('mean', 'Zero')
        
        # Fitted model
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        
        logger.info("GARCHRiskForecaster initialized")
        logger.info(f"GARCH({self.p},{self.q}), Mean: {self.mean}")
    
    def fit(
        self,
        returns: Union[pd.Series, np.ndarray],
        date_ids: Optional[np.ndarray] = None
    ) -> 'GARCHRiskForecaster':
        """
        Fit GARCH model using MLE.
        
        Args:
            returns: Historical returns (scaled by 100 for numerical stability)
            date_ids: Optional date identifiers
            
        Returns:
            self
        """
        logger.info("="*80)
        logger.info("Fitting GARCH Risk Model")
        logger.info("="*80)
        
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError(
                "arch library required for GARCH. "
                "Install with: pip install arch"
            )
        
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        with Timer("GARCH fitting"):
            # Scale returns by 100 for numerical stability (common practice)
            returns_scaled = returns * 100
            
            # Create GARCH model
            self.model = arch_model(
                returns_scaled,
                vol='Garch',
                p=self.p,
                q=self.q,
                mean=self.mean,
                rescale=False
            )
            
            # Fit using MLE
            self.fitted_model = self.model.fit(disp='off', show_warning=False)
            
            # Extract parameters
            params = self.fitted_model.params
            
            logger.info(f"\n✓ GARCH fitting complete")
            logger.info(f"  Samples: {len(returns)}")
            logger.info(f"\nEstimated parameters:")
            for name, value in params.items():
                logger.info(f"  {name}: {value:.6f}")
            
            # Model diagnostics
            logger.info(f"\nModel fit:")
            logger.info(f"  Log-Likelihood: {self.fitted_model.loglikelihood:.2f}")
            logger.info(f"  AIC: {self.fitted_model.aic:.2f}")
            logger.info(f"  BIC: {self.fitted_model.bic:.2f}")
        
        self.is_fitted = True
        return self
    
    def predict(
        self,
        returns: Optional[Union[pd.Series, np.ndarray]] = None,
        horizon: int = 1
    ) -> np.ndarray:
        """
        Forecast future volatility.
        
        Args:
            returns: Recent returns (if None, use fitted model state)
            horizon: Forecast horizon (days ahead)
            
        Returns:
            Array of volatility predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Generate forecast
        forecast = self.fitted_model.forecast(horizon=horizon, reindex=False)
        
        # Extract variance forecast and convert to volatility
        # Scale back by 100 (we scaled returns by 100 during fitting)
        variance_forecast = forecast.variance.values[-1, :]
        volatility_forecast = np.sqrt(variance_forecast) / 100
        
        if horizon == 1:
            return volatility_forecast[0]
        else:
            return volatility_forecast
    
    def get_oof_predictions(
        self,
        returns: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """
        Get conditional volatility from fitted model.
        
        Args:
            returns: Historical returns
            
        Returns:
            Array of conditional volatility (in-sample fitted values)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting predictions")
        
        # Conditional volatility from fitted model
        # Scale back by 100
        cond_vol = self.fitted_model.conditional_volatility / 100
        
        return cond_vol.values
    
    def save_model(self, output_path: str = "artifacts/garch_risk_model.pkl"):
        """Save GARCH fitted model."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'p': self.p,
            'q': self.q,
            'mean': self.mean,
            'fitted_model': self.fitted_model,
            'is_fitted': self.is_fitted
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"✓ GARCH model saved to {output_path}")
    
    def load_model(self, model_path: str = "artifacts/garch_risk_model.pkl"):
        """Load GARCH fitted model."""
        with open(model_path, 'rb') as f:
            state = pickle.load(f)
        
        self.p = state['p']
        self.q = state['q']
        self.mean = state['mean']
        self.fitted_model = state['fitted_model']
        self.is_fitted = state['is_fitted']
        
        logger.info(f"✓ GARCH model loaded from {model_path}")


class HybridRiskEnsemble:
    """
    Ensemble of multiple risk forecasting models.
    
    Combines:
    - LightGBM (feature-based ML)
    - GARCH (statistical time series)
    - EWMA (exponential smoothing)
    
    Strategies:
    - 'max': Most conservative (highest volatility)
    - 'weighted_avg': Weighted average with optimized weights
    - 'percentile': 75th percentile across models
    """
    
    def __init__(
        self,
        models: Dict[str, object],
        strategy: str = 'max',
        weights: Optional[List[float]] = None
    ):
        """
        Initialize hybrid risk ensemble.
        
        Args:
            models: Dictionary of model_name -> model_object
            strategy: Ensemble strategy ('max', 'weighted_avg', 'percentile')
            weights: Weights for weighted_avg (must sum to 1)
        """
        self.models = models
        self.strategy = strategy
        self.weights = weights
        
        valid_strategies = ['max', 'weighted_avg', 'percentile']
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        
        if strategy == 'weighted_avg' and weights is not None:
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.0")
        
        logger.info("HybridRiskEnsemble initialized")
        logger.info(f"Models: {list(models.keys())}")
        logger.info(f"Strategy: {strategy}")
    
    def predict(
        self,
        X: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate ensemble risk predictions.
        
        Args:
            X: Features for ML models (LightGBM)
            returns: Returns for time-series models (GARCH, EWMA)
            
        Returns:
            Combined risk predictions
        """
        predictions = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                # ML models (LightGBM) use features
                if X is not None and name.startswith('lgbm'):
                    pred = model.predict(X)
                # Time-series models use returns
                elif returns is not None and name in ['garch', 'ewma']:
                    if hasattr(model, 'get_oof_predictions'):
                        pred = model.get_oof_predictions(returns)
                    else:
                        pred = model.predict(returns)
                else:
                    continue
                
                predictions[name] = pred
        
        if len(predictions) == 0:
            raise ValueError("No valid predictions generated")
        
        # Combine using ensemble.py function
        from .ensemble import combine_risk_predictions
        
        combined = combine_risk_predictions(
            predictions,
            strategy=self.strategy,
            weights=self.weights
        )
        
        return combined
    
    def evaluate(
        self,
        y_true: np.ndarray,
        X: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance.
        
        Args:
            y_true: True risk labels
            X: Features for ML models
            returns: Returns for time-series models
            
        Returns:
            Performance metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Get ensemble prediction
        ensemble_pred = self.predict(X, returns)
        
        # Calculate metrics
        valid_idx = ~(np.isnan(y_true) | np.isnan(ensemble_pred))
        
        rmse = np.sqrt(mean_squared_error(y_true[valid_idx], ensemble_pred[valid_idx]))
        mae = mean_absolute_error(y_true[valid_idx], ensemble_pred[valid_idx])
        corr = np.corrcoef(y_true[valid_idx], ensemble_pred[valid_idx])[0, 1]
        
        metrics = {
            'ensemble_rmse': rmse,
            'ensemble_mae': mae,
            'ensemble_correlation': corr
        }
        
        # Individual model performance
        for name, model in self.models.items():
            if name.startswith('lgbm') and X is not None:
                pred = model.predict(X)
            elif name in ['garch', 'ewma'] and returns is not None:
                pred = model.get_oof_predictions(returns)
            else:
                continue
            
            valid_idx_model = ~(np.isnan(y_true) | np.isnan(pred))
            model_rmse = np.sqrt(mean_squared_error(
                y_true[valid_idx_model], pred[valid_idx_model]
            ))
            metrics[f'{name}_rmse'] = model_rmse
        
        return metrics
