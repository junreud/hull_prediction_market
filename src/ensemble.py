"""
Ensemble module for combining multiple model predictions.

Supports various ensemble strategies:
- Weighted averaging (simple, optimized, time-weighted)
- Stacking (meta-learner on top of base models)
- Blending (holdout set for meta-model)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import logging
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class ModelEnsemble:
    """
    Ensemble class for combining multiple model predictions.
    
    Strategies:
    - 'simple_average': Equal weights for all models
    - 'weighted_average': Custom weights (sum to 1)
    - 'optimized_weights': Find optimal weights using OOF predictions
    - 'time_weighted': Recent predictions get higher weights
    - 'stacking': Meta-model learns to combine predictions
    """
    
    def __init__(
        self,
        strategy: str = 'simple_average',
        weights: Optional[List[float]] = None,
        decay_factor: float = 0.95,
        meta_model = None
    ):
        """
        Initialize ensemble.
        
        Parameters
        ----------
        strategy : str
            Ensemble strategy to use
        weights : List[float], optional
            Weights for weighted_average strategy (must sum to 1)
        decay_factor : float
            Decay factor for time_weighted strategy (0-1)
            Higher = more weight to recent data
        meta_model : sklearn model, optional
            Meta-model for stacking (default: Ridge regression)
        """
        self.strategy = strategy
        self.weights = weights
        self.decay_factor = decay_factor
        self.meta_model = meta_model if meta_model is not None else Ridge(alpha=1.0)
        self.is_fitted = False
        
        # Validate strategy
        valid_strategies = [
            'simple_average', 'weighted_average', 'optimized_weights',
            'time_weighted', 'stacking'
        ]
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        
        # Validate weights if provided
        if strategy == 'weighted_average' and weights is not None:
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.0")
            if any(w < 0 for w in weights):
                raise ValueError("Weights must be non-negative")
    
    def fit(
        self,
        oof_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        date_ids: Optional[np.ndarray] = None
    ):
        """
        Fit ensemble on OOF predictions.
        
        Parameters
        ----------
        oof_predictions : Dict[str, np.ndarray]
            Dictionary of model_name -> OOF predictions
        y_true : np.ndarray
            True target values
        date_ids : np.ndarray, optional
            Date identifiers for time weighting
        """
        # Convert to matrix: (n_samples, n_models)
        model_names = list(oof_predictions.keys())
        X_oof = np.column_stack([oof_predictions[name] for name in model_names])
        
        if self.strategy == 'simple_average':
            n_models = len(model_names)
            self.weights = [1.0 / n_models] * n_models
            logger.info(f"Simple average: {n_models} models with equal weights")
        
        elif self.strategy == 'weighted_average':
            if self.weights is None:
                raise ValueError("weights must be provided for weighted_average")
            logger.info(f"Using provided weights: {self.weights}")
        
        elif self.strategy == 'optimized_weights':
            self.weights = self._optimize_weights(X_oof, y_true)
            logger.info(f"Optimized weights: {self.weights}")
        
        elif self.strategy == 'time_weighted':
            if date_ids is None:
                raise ValueError("date_ids required for time_weighted strategy")
            self.weights = self._compute_time_weights(
                X_oof, y_true, date_ids
            )
            logger.info(f"Time-weighted: decay={self.decay_factor}")
        
        elif self.strategy == 'stacking':
            logger.info("Fitting stacking meta-model...")
            self.meta_model.fit(X_oof, y_true)
            cv_scores = cross_val_score(
                self.meta_model, X_oof, y_true,
                cv=5, scoring='neg_root_mean_squared_error'
            )
            logger.info(f"Meta-model CV RMSE: {-cv_scores.mean():.6f} (±{cv_scores.std():.6f})")
        
        self.model_names = model_names
        self.is_fitted = True
    
    def predict(
        self,
        predictions: Dict[str, np.ndarray],
        date_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Parameters
        ----------
        predictions : Dict[str, np.ndarray]
            Dictionary of model_name -> predictions
        date_ids : np.ndarray, optional
            Date identifiers (for time_weighted)
        
        Returns
        -------
        np.ndarray
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Ensure model order matches training
        X_pred = np.column_stack([predictions[name] for name in self.model_names])
        
        if self.strategy == 'stacking':
            return self.meta_model.predict(X_pred)
        
        elif self.strategy == 'time_weighted' and date_ids is not None:
            # Apply exponential decay based on recency
            unique_dates = np.unique(date_ids)
            weights_by_date = {}
            for i, date in enumerate(unique_dates):
                # More recent dates get higher weight
                recency = len(unique_dates) - i - 1
                weights_by_date[date] = self.decay_factor ** recency
            
            # Apply time weights to model predictions
            sample_weights = np.array([weights_by_date[d] for d in date_ids])
            sample_weights = sample_weights / sample_weights.sum()
            
            # Weighted combination
            ensemble_pred = np.average(X_pred, axis=1, weights=self.weights)
            return ensemble_pred
        
        else:
            # Simple weighted average
            return np.average(X_pred, axis=1, weights=self.weights)
    
    def _optimize_weights(
        self,
        X_oof: np.ndarray,
        y_true: np.ndarray,
        method: str = 'constrained'
    ) -> List[float]:
        """
        Find optimal weights to minimize RMSE.
        
        Parameters
        ----------
        X_oof : np.ndarray
            OOF predictions (n_samples, n_models)
        y_true : np.ndarray
            True targets
        method : str
            'constrained' = sum to 1, non-negative
            'ridge' = Ridge regression (can be negative)
        
        Returns
        -------
        List[float]
            Optimal weights
        """
        from scipy.optimize import minimize
        
        n_models = X_oof.shape[1]
        
        if method == 'constrained':
            # Objective: minimize RMSE
            def objective(w):
                pred = X_oof @ w
                return np.sqrt(np.mean((y_true - pred) ** 2))
            
            # Constraints: sum to 1, non-negative
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
            bounds = [(0.0, 1.0)] * n_models
            
            # Initial guess: equal weights
            w0 = np.ones(n_models) / n_models
            
            result = minimize(
                objective, w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                logger.warning(f"Optimization failed: {result.message}")
                logger.warning("Falling back to equal weights")
                return list(w0)
            
            weights = result.x
            logger.info(f"Optimized RMSE: {result.fun:.6f}")
            
        else:  # ridge
            ridge = Ridge(alpha=1.0, fit_intercept=False, positive=True)
            ridge.fit(X_oof, y_true)
            weights = ridge.coef_
            weights = weights / weights.sum()  # Normalize
        
        return list(weights)
    
    def _compute_time_weights(
        self,
        X_oof: np.ndarray,
        y_true: np.ndarray,
        date_ids: np.ndarray
    ) -> List[float]:
        """
        Compute weights giving more importance to recent performance.
        
        Strategy: For each model, compute rolling RMSE with exponential decay,
        then assign weights inversely proportional to recent error.
        """
        n_models = X_oof.shape[1]
        unique_dates = np.unique(date_ids)
        
        # Compute performance by time window for each model
        model_scores = []
        for i in range(n_models):
            preds = X_oof[:, i]
            
            # Compute weighted RMSE (recent errors weighted higher)
            weights_by_date = {}
            for j, date in enumerate(unique_dates):
                recency = len(unique_dates) - j - 1
                weights_by_date[date] = self.decay_factor ** recency
            
            sample_weights = np.array([weights_by_date[d] for d in date_ids])
            sample_weights = sample_weights / sample_weights.sum()
            
            weighted_mse = np.average((y_true - preds) ** 2, weights=sample_weights)
            model_scores.append(np.sqrt(weighted_mse))
        
        model_scores = np.array(model_scores)
        logger.info(f"Model time-weighted RMSEs: {model_scores}")
        
        # Inverse weighting: better models get higher weight
        # Add small epsilon to avoid division by zero
        inverse_scores = 1.0 / (model_scores + 1e-8)
        weights = inverse_scores / inverse_scores.sum()
        
        return list(weights)
    
    def evaluate(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        date_ids: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance.
        
        Returns
        -------
        Dict[str, float]
            Performance metrics
        """
        ensemble_pred = self.predict(predictions, date_ids)
        
        rmse = np.sqrt(np.mean((y_true - ensemble_pred) ** 2))
        mae = np.mean(np.abs(y_true - ensemble_pred))
        
        # Correlation
        corr = np.corrcoef(y_true, ensemble_pred)[0, 1]
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'correlation': corr
        }
        
        # Individual model performance for comparison
        for name, pred in predictions.items():
            model_rmse = np.sqrt(np.mean((y_true - pred) ** 2))
            metrics[f'{name}_rmse'] = model_rmse
        
        return metrics
    
    def get_weights_df(self) -> pd.DataFrame:
        """Return weights as DataFrame for easy viewing."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted first")
        
        if self.strategy == 'stacking':
            # For stacking, show meta-model coefficients
            if hasattr(self.meta_model, 'coef_'):
                coef = self.meta_model.coef_
                return pd.DataFrame({
                    'model': self.model_names,
                    'coefficient': coef,
                    'normalized_weight': coef / coef.sum()
                }).sort_values('coefficient', ascending=False)
            else:
                return pd.DataFrame({'model': self.model_names})
        
        return pd.DataFrame({
            'model': self.model_names,
            'weight': self.weights
        }).sort_values('weight', ascending=False)


def combine_risk_predictions(
    risk_predictions: Dict[str, np.ndarray],
    strategy: str = 'max',
    weights: Optional[List[float]] = None
) -> np.ndarray:
    """
    Combine risk (sigma) predictions from multiple models.
    
    For risk, we typically want conservative estimates to avoid
    underestimating volatility.
    
    Parameters
    ----------
    risk_predictions : Dict[str, np.ndarray]
        Dictionary of model_name -> risk predictions
    strategy : str
        'max': Take maximum across models (most conservative)
        'weighted_avg': Weighted average
        'percentile': Take 75th percentile
    weights : List[float], optional
        Weights for weighted_avg strategy
    
    Returns
    -------
    np.ndarray
        Combined risk predictions
    """
    X_risk = np.column_stack(list(risk_predictions.values()))
    
    if strategy == 'max':
        return np.max(X_risk, axis=1)
    
    elif strategy == 'weighted_avg':
        if weights is None:
            weights = [1.0 / len(risk_predictions)] * len(risk_predictions)
        return np.average(X_risk, axis=1, weights=weights)
    
    elif strategy == 'percentile':
        return np.percentile(X_risk, 75, axis=1)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
