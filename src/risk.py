"""
Risk prediction module for volatility forecasting.

This module provides tools for:
1. Risk label creation (future volatility)
2. Risk forecasting models
3. Calibration assessment
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pickle

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .utils import get_logger, load_config, Timer
from .cv import PurgedWalkForwardCV
from .models import ReturnPredictor

logger = get_logger(__name__)


class RiskLabeler:
    """
    Create risk labels by calculating future volatility.
    
    Risk is defined as rolling standard deviation of forward returns
    over a future window (forward-looking volatility).
    """
    
    def __init__(
        self,
        window: int = 20,
        min_periods: Optional[int] = None,
        clip_threshold: float = 4.0,
        config_path: str = "conf/params.yaml"
    ):
        """
        Initialize risk labeler.
        
        Args:
            window: Rolling window size for volatility calculation
            min_periods: Minimum periods required for rolling calculation
            clip_threshold: MAD threshold for clipping extreme values
            config_path: Path to configuration file
        """
        # Load config
        config = load_config(config_path)
        risk_config = config.get('risk', {}).get('label', {})
        
        # Use config values or defaults
        self.window = window or risk_config.get('window', 20)
        self.min_periods = min_periods or risk_config.get('min_periods', 10)
        self.clip_threshold = clip_threshold or risk_config.get('clip_threshold', 4.0)
        
        logger.info("RiskLabeler initialized")
        logger.info(f"Window: {self.window}, Min periods: {self.min_periods}")
        logger.info(f"Clip threshold: {self.clip_threshold} MAD")
    
    def create_labels(self, df: pd.DataFrame, target_col: str = 'forward_returns') -> pd.Series:
        """
        Create risk labels (future volatility).
        
        Args:
            df: DataFrame with target column
            target_col: Name of target column
            
        Returns:
            Series with risk labels (future volatility)
        """
        logger.info("="*80)
        logger.info("Creating Risk Labels")
        logger.info("="*80)
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        with Timer("Risk label creation"):
            # 핵심 계산: 미래 rolling volatility
            # 역순으로 계산하여 "미래" 값을 얻음
            risk_labels = (
                df[target_col][::-1]                                       # forward_returns를 역순으로
                .rolling(window=self.window, min_periods=self.min_periods) # 미래 20일 window
                .std()                                                     # 표준편차 (변동성)
                [::-1]                                                     # 다시 정순으로
            )
            
            # Outlier 제거 (MAD 방식)
            median = risk_labels.median()
            mad = (risk_labels - median).abs().median()
            
            if mad > 0:
                lower_bound = median - self.clip_threshold * mad  # clip_threshold=4.0 MAD
                upper_bound = median + self.clip_threshold * mad
                
                n_clipped = ((risk_labels < lower_bound) | (risk_labels > upper_bound)).sum()
                risk_labels = risk_labels.clip(lower=lower_bound, upper=upper_bound)
                
                logger.info(f"Clipped {n_clipped} extreme values ({n_clipped/len(risk_labels)*100:.2f}%)")
            
            # Log statistics
            logger.info(f"\nRisk Label Statistics:")
            logger.info(f"  Count: {risk_labels.notna().sum()}")
            logger.info(f"  Missing: {risk_labels.isna().sum()} ({risk_labels.isna().sum()/len(risk_labels)*100:.2f}%)")
            logger.info(f"  Mean: {risk_labels.mean():.6f}")
            logger.info(f"  Std: {risk_labels.std():.6f}")
            logger.info(f"  Min: {risk_labels.min():.6f}")
            logger.info(f"  Median: {risk_labels.median():.6f}")
            logger.info(f"  Max: {risk_labels.max():.6f}")
            
        logger.info("\n✓ Risk labels created")
        
        return risk_labels
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'forward_returns') -> pd.DataFrame:
        """
        Create risk labels and add to DataFrame.
        
        Args:
            df: DataFrame with target column
            target_col: Name of target column
            
        Returns:
            DataFrame with added 'risk_label' column
        """
        df_copy = df.copy()
        df_copy['risk_label'] = self.create_labels(df, target_col)
        return df_copy


class RiskForecaster:
    """
    Risk forecasting model for predicting future volatility.
    
    Uses ReturnPredictor infrastructure for LightGBM training.
    """
    
    def __init__(
        self,
        model_params: Optional[Dict] = None,
        config_path: str = "conf/params.yaml"
    ):
        """
        Initialize risk forecaster.
        
        Args:
            model_params: LightGBM parameters (optional override)
            config_path: Path to configuration file
        """
        # Load config
        config = load_config(config_path)
        risk_config = config.get('risk', {})
        
        # Initialize ReturnPredictor for LightGBM infrastructure
        self.predictor = ReturnPredictor(model_type='lightgbm', config_path=config_path)
        
        # Override params if provided
        if model_params is not None:
            self.predictor.params.update(model_params)
        
        logger.info("RiskForecaster initialized using ReturnPredictor")
        logger.info(f"Model parameters: {self.predictor.params}")
    
    def train(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        risk_col: str = 'risk_label',
        n_folds: int = 5
    ) -> Tuple[np.ndarray, List]:
        """
        Train risk models with cross-validation.
        
        Args:
            df: DataFrame with features and risk labels (must contain 'date_id' column)
            feature_cols: List of feature column names
            risk_col: Name of risk label column
            n_folds: Number of CV folds
            
        Returns:
            Tuple of (OOF predictions, trained models)
        """
        logger.info("="*80)
        logger.info("Training Risk Forecaster")
        logger.info("="*80)
        
        # Validate inputs
        if risk_col not in df.columns:
            raise ValueError(f"Risk column '{risk_col}' not found in DataFrame")
        
        if 'date_id' not in df.columns:
            raise ValueError("DataFrame must contain 'date_id' column for CV")
        
        missing_features = set(feature_cols) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        logger.info(f"\nDataset:")
        logger.info(f"  Features: {len(feature_cols)}")
        logger.info(f"  Samples: {len(df)}")
        logger.info(f"  Risk column: {risk_col}")
        
        # Prepare data (copy to avoid modifying original)
        df_train = df.copy()
        df_train['target'] = df_train[risk_col]  # Rename for compatibility
        
        # Use ReturnPredictor's train_cv method
        with Timer("Cross-validation training"):
            results = self.predictor.train_cv(
                df=df_train,
                target_col='target',
                date_col='date_id'
            )
        
        oof_predictions = results['oof_predictions']
        
        # Log results
        logger.info(f"\n{'='*80}")
        logger.info("Cross-Validation Results")
        logger.info(f"{'='*80}")
        
        # Filter out NaN from both predictions AND targets
        valid_idx = ~np.isnan(oof_predictions) & ~np.isnan(df_train['target'].values)
        y_true = df_train['target'].values[valid_idx]
        y_pred = oof_predictions[valid_idx]
        
        oof_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        oof_mae = mean_absolute_error(y_true, y_pred)
        
        logger.info(f"\nOOF RMSE: {oof_rmse:.6f}")
        logger.info(f"OOF MAE: {oof_mae:.6f}")
        logger.info(f"OOF Coverage: {valid_idx.sum()}/{len(oof_predictions)} ({valid_idx.sum()/len(oof_predictions)*100:.1f}%)")
        
        logger.info("\n✓ Risk forecaster training complete")
        
        return oof_predictions, self.predictor.models
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict risk using ensemble of trained models.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of risk predictions (mean of all models)
        """
        return self.predictor.predict(X)
    
    def save_models(self, output_dir: str = "artifacts/models"):
        """
        Save trained models.
        
        Args:
            output_dir: Directory to save models
        """
        self.predictor.save_models(output_dir)
        logger.info(f"\n✓ All risk models saved to {output_dir}")
    
    def load_models(self, model_dir: str = "artifacts/models", n_models: Optional[int] = None):
        """
        Load trained models.
        
        Args:
            model_dir: Directory containing saved models
            n_models: Number of models to load (None = all available)
        """
        self.predictor.load_models(model_dir, n_models)
        logger.info(f"\n✓ Loaded risk models from {model_dir}")
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance from trained models.
        
        Args:
            importance_type: Type of importance ('gain', 'split', 'weight')
            
        Returns:
            DataFrame with feature importance
        """
        return self.predictor.get_feature_importance(importance_type)


class RiskCalibrator:
    """
    Calibration assessment for risk predictions.
    
    Evaluates how well predicted risks match actual observed volatility.
    """
    
    def __init__(self, config_path: str = "conf/params.yaml"):
        """
        Initialize risk calibrator.
        
        Args:
            config_path: Path to configuration file
        """
        # Load config
        config = load_config(config_path)
        calib_config = config.get('risk', {}).get('calibration', {})
        
        self.n_bins = calib_config.get('n_bins', 10)
        self.strategy = calib_config.get('strategy', 'quantile')
        self.min_samples_per_bin = calib_config.get('min_samples_per_bin', 50)
        
        logger.info("RiskCalibrator initialized")
        logger.info(f"Bins: {self.n_bins}, Strategy: {self.strategy}")
    
    def calibration_curve(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate calibration curve.
        
        Args:
            y_true: True risk values
            y_pred: Predicted risk values
            
        Returns:
            Tuple of (bin_edges, mean_predicted, mean_observed)
        """
        # Remove NaN values
        valid_idx = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_valid = y_true[valid_idx]
        y_pred_valid = y_pred[valid_idx]
        
        if len(y_true_valid) == 0:
            raise ValueError("No valid samples for calibration")
        
        # Create bins
        if self.strategy == 'quantile':
            bin_edges = np.percentile(y_pred_valid, np.linspace(0, 100, self.n_bins + 1))
        else:  # uniform
            bin_edges = np.linspace(y_pred_valid.min(), y_pred_valid.max(), self.n_bins + 1)
        
        # Ensure unique bin edges
        bin_edges = np.unique(bin_edges)
        n_bins_actual = len(bin_edges) - 1
        
        # Calculate mean predicted and observed for each bin
        mean_predicted = np.zeros(n_bins_actual)
        mean_observed = np.zeros(n_bins_actual)
        bin_counts = np.zeros(n_bins_actual, dtype=int)
        
        for i in range(n_bins_actual):
            if i == n_bins_actual - 1:
                mask = (y_pred_valid >= bin_edges[i]) & (y_pred_valid <= bin_edges[i + 1])
            else:
                mask = (y_pred_valid >= bin_edges[i]) & (y_pred_valid < bin_edges[i + 1])
            
            if mask.sum() > 0:
                mean_predicted[i] = y_pred_valid[mask].mean()
                mean_observed[i] = y_true_valid[mask].mean()
                bin_counts[i] = mask.sum()
        
        return bin_edges, mean_predicted, mean_observed, bin_counts
    
    def assess_calibration(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Assess calibration quality.
        
        Args:
            y_true: True risk values
            y_pred: Predicted risk values
            
        Returns:
            Dictionary with calibration metrics
        """
        logger.info("="*80)
        logger.info("Risk Calibration Assessment")
        logger.info("="*80)
        
        # Calculate calibration curve
        bin_edges, mean_pred, mean_obs, bin_counts = self.calibration_curve(y_true, y_pred)
        
        # Calculate calibration error (mean absolute difference)
        valid_bins = bin_counts >= self.min_samples_per_bin
        
        if valid_bins.sum() == 0:
            logger.warning("No bins with sufficient samples for calibration")
            calibration_error = np.nan
        else:
            calibration_error = np.abs(mean_pred[valid_bins] - mean_obs[valid_bins]).mean()
        
        # Calculate correlation between predicted and observed
        valid_idx = ~(np.isnan(y_true) | np.isnan(y_pred))
        correlation = np.corrcoef(y_true[valid_idx], y_pred[valid_idx])[0, 1]
        
        # Log results
        logger.info(f"\nCalibration Metrics:")
        logger.info(f"  Calibration Error (MAE): {calibration_error:.6f}")
        logger.info(f"  Correlation: {correlation:.4f}")
        logger.info(f"  Valid bins: {valid_bins.sum()}/{len(valid_bins)}")
        
        logger.info(f"\nBin Statistics:")
        for i in range(len(bin_edges) - 1):
            if bin_counts[i] >= self.min_samples_per_bin:
                logger.info(
                    f"  Bin {i+1}: Predicted={mean_pred[i]:.6f}, "
                    f"Observed={mean_obs[i]:.6f}, "
                    f"Count={bin_counts[i]}"
                )
        
        results = {
            'calibration_error': calibration_error,
            'correlation': correlation,
            'bin_edges': bin_edges,
            'mean_predicted': mean_pred,
            'mean_observed': mean_obs,
            'bin_counts': bin_counts,
            'valid_bins': valid_bins
        }
        
        logger.info("\n✓ Calibration assessment complete")
        
        return results
    
    def save_calibration_report(
        self,
        results: Dict,
        output_path: str = "results/risk_calibration_report.csv"
    ):
        """
        Save calibration report to CSV.
        
        Args:
            results: Calibration results from assess_calibration()
            output_path: Path to save report
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame
        n_bins = len(results['bin_edges']) - 1
        
        report_data = {
            'bin': range(1, n_bins + 1),
            'bin_lower': results['bin_edges'][:-1],
            'bin_upper': results['bin_edges'][1:],
            'mean_predicted': results['mean_predicted'],
            'mean_observed': results['mean_observed'],
            'count': results['bin_counts'],
            'valid': results['valid_bins']
        }
        
        df_report = pd.DataFrame(report_data)
        df_report.to_csv(output_file, index=False)
        
        logger.info(f"✓ Calibration report saved to {output_file}")
