"""
Optimize risk model ensemble combining LightGBM, GARCH, and EWMA.

This script:
1. Trains LightGBM risk model (feature-based)
2. Trains GARCH model (statistical time-series)
3. Trains EWMA model (exponential smoothing)
4. Combines them using various ensemble strategies
5. Optimizes ensemble weights for best risk prediction
6. Saves best ensemble configuration

Usage:
    python scripts/optimize_risk_ensemble.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.data import DataLoader
from src.features import FeatureEngineering
from src.risk import RiskLabeler, RiskForecaster
from src.timeseries_risk import EWMARiskForecaster, GARCHRiskForecaster, HybridRiskEnsemble
from src.ensemble import combine_risk_predictions
from src.utils import get_logger, Timer, load_config

logger = get_logger(log_file="logs/risk_ensemble_optimization.log", level="INFO")


class RiskEnsembleOptimizer:
    """
    Optimize ensemble of risk forecasting models.
    """
    
    def __init__(self, config_path: str = "conf/params.yaml"):
        """Initialize optimizer."""
        self.config_path = config_path
        self.config = load_config(config_path)
        
        # Components
        self.data_loader = DataLoader(config_path)
        self.feature_engineer = FeatureEngineering(config_path)
        self.risk_labeler = RiskLabeler(config_path=config_path)
        
        # Results storage
        self.results = {}
        
        logger.info("="*80)
        logger.info("RISK ENSEMBLE OPTIMIZATION PIPELINE")
        logger.info("="*80)
    
    def step1_prepare_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Step 1: Load and prepare data.
        
        Returns:
            Tuple of (DataFrame with features and risk labels, returns array)
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("="*80)
        
        with Timer("Data preparation", logger):
            # Load data
            logger.info("\n1.1 Loading data...")
            train_df, _ = self.data_loader.load_data()
            logger.info(f"✓ Loaded {len(train_df)} samples")
            
            # Preprocess
            logger.info("\n1.2 Preprocessing...")
            train_df, _ = self.data_loader.preprocess_timeseries(
                train_df,
                train_df=None,
                add_missing_indicators=True,
                add_regime_indicators=True,
                handle_outliers=True,
                normalize=True,
                scale=True,
                window=60
            )
            logger.info(f"✓ Preprocessing complete: {train_df.shape}")
            
            # Feature engineering
            logger.info("\n1.3 Feature engineering...")
            train_df = self.feature_engineer.fit_transform(train_df)
            logger.info(f"✓ Features created: {train_df.shape[1]} columns")
            
            # Create risk labels
            logger.info("\n1.4 Creating risk labels...")
            train_df = self.risk_labeler.fit_transform(train_df)
            logger.info(f"✓ Risk labels created")
            
            # Extract returns for time-series models
            returns = train_df['forward_returns'].values
            
            logger.info(f"\n✓ Data preparation complete")
            logger.info(f"  Shape: {train_df.shape}")
            logger.info(f"  Risk labels: {train_df['risk_label'].notna().sum()} valid")
        
        return train_df, returns
    
    def step2_train_lgbm_model(
        self,
        df: pd.DataFrame
    ) -> Tuple[RiskForecaster, np.ndarray]:
        """
        Step 2: Train LightGBM risk model.
        
        Returns:
            Tuple of (trained model, OOF predictions)
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 2: TRAIN LIGHTGBM RISK MODEL")
        logger.info("="*80)
        
        with Timer("LightGBM training", logger):
            # Load features from saved selection
            feature_path = Path("results/feature_selection/selected_features_risk_optimized.csv")
            if feature_path.exists():
                feature_df = pd.read_csv(feature_path)
                feature_cols = feature_df['feature'].tolist()
                feature_cols = [f for f in feature_cols if f in df.columns]
                logger.info(f"Using {len(feature_cols)} features from selection file")
            else:
                # Fallback: use all features except special columns
                exclude_cols = ['date_id', 'forward_returns', 'risk_label', 
                               'risk_free_rate', 'market_forward_excess_returns']
                feature_cols = [c for c in df.columns if c not in exclude_cols]
                logger.info(f"Using all {len(feature_cols)} features")
            
            # Load best params from risk optimization
            params_path = Path("artifacts/lightgbm_best_params_risk_optimized.json")
            if params_path.exists():
                with open(params_path) as f:
                    best_params = json.load(f)
                logger.info("Loaded optimized LightGBM parameters")
            else:
                best_params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'verbosity': -1,
                    'random_state': 42
                }
                logger.info("Using default LightGBM parameters")
            
            # Train model
            lgbm_model = RiskForecaster(
                model_params=best_params,
                config_path=self.config_path
            )
            
            oof_preds, models = lgbm_model.train(
                df=df,
                feature_cols=feature_cols,
                risk_col='risk_label',
                n_folds=5
            )
            
            # Calculate OOF score
            valid_idx = ~np.isnan(oof_preds)
            y_true = df.loc[valid_idx, 'risk_label'].values
            oof_score = np.sqrt(mean_squared_error(y_true, oof_preds[valid_idx]))
            
            logger.info(f"\n✓ LightGBM training complete")
            logger.info(f"  OOF RMSE: {oof_score:.6f}")
            
            self.results['lgbm'] = {
                'oof_rmse': oof_score,
                'n_folds': len(models)
            }
        
        return lgbm_model, oof_preds
    
    def step3_train_ewma_model(
        self,
        returns: np.ndarray
    ) -> Tuple[EWMARiskForecaster, np.ndarray]:
        """
        Step 3: Train EWMA risk model.
        
        Returns:
            Tuple of (trained model, OOF predictions)
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 3: TRAIN EWMA RISK MODEL")
        logger.info("="*80)
        
        with Timer("EWMA training", logger):
            # Try different lambda values
            best_lambda = 0.94
            best_score = float('inf')
            
            for lambda_ in [0.90, 0.92, 0.94, 0.96, 0.97]:
                ewma_model = EWMARiskForecaster(lambda_=lambda_)
                ewma_model.fit(returns)
                
                # Get OOF predictions
                oof_preds = ewma_model.get_oof_predictions(returns)
                
                # We don't have true labels for EWMA validation here
                # Use realized volatility as proxy
                # (This is approximate - in practice, use rolling realized vol)
                logger.info(f"  Lambda={lambda_}: predictions generated")
                
                # For now, select RiskMetrics standard
                if lambda_ == 0.94:
                    best_lambda = lambda_
                    best_ewma_model = ewma_model
                    best_oof_preds = oof_preds
            
            logger.info(f"\n✓ EWMA training complete")
            logger.info(f"  Selected lambda: {best_lambda}")
            logger.info(f"  Mean volatility: {np.nanmean(best_oof_preds):.6f}")
            
            self.results['ewma'] = {
                'lambda': best_lambda,
                'mean_volatility': float(np.nanmean(best_oof_preds))
            }
        
        return best_ewma_model, best_oof_preds
    
    def step4_train_garch_model(
        self,
        returns: np.ndarray
    ) -> Tuple[GARCHRiskForecaster, np.ndarray]:
        """
        Step 4: Train GARCH risk model.
        
        Returns:
            Tuple of (trained model, OOF predictions)
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 4: TRAIN GARCH RISK MODEL")
        logger.info("="*80)
        
        try:
            with Timer("GARCH training", logger):
                # Train GARCH(1,1)
                garch_model = GARCHRiskForecaster(p=1, q=1, mean='Zero')
                garch_model.fit(returns)
                
                # Get conditional volatility (in-sample)
                oof_preds = garch_model.get_oof_predictions(returns)
                
                logger.info(f"\n✓ GARCH training complete")
                logger.info(f"  Mean volatility: {np.nanmean(oof_preds):.6f}")
                
                self.results['garch'] = {
                    'p': garch_model.p,
                    'q': garch_model.q,
                    'mean_volatility': float(np.nanmean(oof_preds))
                }
            
            return garch_model, oof_preds
            
        except ImportError:
            logger.warning("\n⚠️  GARCH model skipped: 'arch' library not installed")
            logger.warning("   Install with: pip install arch")
            self.results['garch'] = {'status': 'skipped'}
            return None, None
        except Exception as e:
            logger.error(f"\n❌ GARCH training failed: {str(e)}")
            self.results['garch'] = {'status': 'failed', 'error': str(e)}
            return None, None
    
    def step5_ensemble_comparison(
        self,
        df: pd.DataFrame,
        lgbm_preds: np.ndarray,
        ewma_preds: np.ndarray,
        garch_preds: Optional[np.ndarray]
    ) -> pd.DataFrame:
        """
        Step 5: Compare ensemble strategies.
        
        Returns:
            DataFrame with comparison results
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 5: ENSEMBLE STRATEGY COMPARISON")
        logger.info("="*80)
        
        with Timer("Ensemble comparison", logger):
            # Get true risk labels
            y_true = df['risk_label'].values
            
            # Prepare predictions dictionary
            predictions = {
                'lgbm': lgbm_preds,
                'ewma': ewma_preds
            }
            
            if garch_preds is not None:
                predictions['garch'] = garch_preds
            
            comparison_data = []
            
            # Strategy 1: Max (most conservative)
            logger.info("\n5.1 Evaluating 'max' strategy...")
            ensemble_max = combine_risk_predictions(predictions, strategy='max')
            
            valid_idx = ~(np.isnan(y_true) | np.isnan(ensemble_max))
            rmse_max = np.sqrt(mean_squared_error(y_true[valid_idx], ensemble_max[valid_idx]))
            mae_max = mean_absolute_error(y_true[valid_idx], ensemble_max[valid_idx])
            corr_max = np.corrcoef(y_true[valid_idx], ensemble_max[valid_idx])[0, 1]
            
            comparison_data.append({
                'strategy': 'max',
                'rmse': rmse_max,
                'mae': mae_max,
                'correlation': corr_max,
                'params': 'conservative (highest vol)'
            })
            logger.info(f"  RMSE: {rmse_max:.6f}, MAE: {mae_max:.6f}, Corr: {corr_max:.4f}")
            
            # Strategy 2: Percentile 75
            logger.info("\n5.2 Evaluating 'percentile' strategy...")
            ensemble_pct = combine_risk_predictions(predictions, strategy='percentile')
            
            valid_idx = ~(np.isnan(y_true) | np.isnan(ensemble_pct))
            rmse_pct = np.sqrt(mean_squared_error(y_true[valid_idx], ensemble_pct[valid_idx]))
            mae_pct = mean_absolute_error(y_true[valid_idx], ensemble_pct[valid_idx])
            corr_pct = np.corrcoef(y_true[valid_idx], ensemble_pct[valid_idx])[0, 1]
            
            comparison_data.append({
                'strategy': 'percentile_75',
                'rmse': rmse_pct,
                'mae': mae_pct,
                'correlation': corr_pct,
                'params': '75th percentile'
            })
            logger.info(f"  RMSE: {rmse_pct:.6f}, MAE: {mae_pct:.6f}, Corr: {corr_pct:.4f}")
            
            # Strategy 3: Weighted average (equal)
            logger.info("\n5.3 Evaluating 'weighted_avg' (equal) strategy...")
            n_models = len(predictions)
            weights_equal = [1.0 / n_models] * n_models
            ensemble_avg = combine_risk_predictions(
                predictions, strategy='weighted_avg', weights=weights_equal
            )
            
            valid_idx = ~(np.isnan(y_true) | np.isnan(ensemble_avg))
            rmse_avg = np.sqrt(mean_squared_error(y_true[valid_idx], ensemble_avg[valid_idx]))
            mae_avg = mean_absolute_error(y_true[valid_idx], ensemble_avg[valid_idx])
            corr_avg = np.corrcoef(y_true[valid_idx], ensemble_avg[valid_idx])[0, 1]
            
            comparison_data.append({
                'strategy': 'weighted_avg_equal',
                'rmse': rmse_avg,
                'mae': mae_avg,
                'correlation': corr_avg,
                'params': f'weights={weights_equal}'
            })
            logger.info(f"  RMSE: {rmse_avg:.6f}, MAE: {mae_avg:.6f}, Corr: {corr_avg:.4f}")
            
            # Strategy 4: Weighted average (optimized)
            logger.info("\n5.4 Optimizing weights...")
            best_weights = self._optimize_risk_weights(predictions, y_true)
            ensemble_opt = combine_risk_predictions(
                predictions, strategy='weighted_avg', weights=best_weights
            )
            
            valid_idx = ~(np.isnan(y_true) | np.isnan(ensemble_opt))
            rmse_opt = np.sqrt(mean_squared_error(y_true[valid_idx], ensemble_opt[valid_idx]))
            mae_opt = mean_absolute_error(y_true[valid_idx], ensemble_opt[valid_idx])
            corr_opt = np.corrcoef(y_true[valid_idx], ensemble_opt[valid_idx])[0, 1]
            
            comparison_data.append({
                'strategy': 'weighted_avg_optimized',
                'rmse': rmse_opt,
                'mae': mae_opt,
                'correlation': corr_opt,
                'params': f'weights={[f"{w:.3f}" for w in best_weights]}'
            })
            logger.info(f"  RMSE: {rmse_opt:.6f}, MAE: {mae_opt:.6f}, Corr: {corr_opt:.4f}")
            logger.info(f"  Optimized weights: {dict(zip(predictions.keys(), [f'{w:.3f}' for w in best_weights]))}")
            
            # Individual model performance for reference
            for name, pred in predictions.items():
                valid_idx = ~(np.isnan(y_true) | np.isnan(pred))
                if valid_idx.sum() > 0:
                    rmse_single = np.sqrt(mean_squared_error(y_true[valid_idx], pred[valid_idx]))
                    comparison_data.append({
                        'strategy': f'single_{name}',
                        'rmse': rmse_single,
                        'mae': mean_absolute_error(y_true[valid_idx], pred[valid_idx]),
                        'correlation': np.corrcoef(y_true[valid_idx], pred[valid_idx])[0, 1],
                        'params': 'baseline'
                    })
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('rmse', ascending=True)
            
            logger.info("\n" + "="*80)
            logger.info("ENSEMBLE COMPARISON RESULTS")
            logger.info("="*80)
            logger.info(f"\n{comparison_df.to_string()}")
            
            # Save comparison
            output_path = Path("results/risk_ensemble_comparison.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(output_path, index=False)
            logger.info(f"\n✓ Saved comparison to {output_path}")
            
            self.results['comparison'] = comparison_df.to_dict('records')
        
        return comparison_df
    
    def _optimize_risk_weights(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray
    ) -> List[float]:
        """Optimize ensemble weights to minimize RMSE."""
        from scipy.optimize import minimize
        
        model_names = list(predictions.keys())
        X = np.column_stack([predictions[name] for name in model_names])
        
        # Remove NaN rows
        valid_idx = ~(np.isnan(y_true) | np.any(np.isnan(X), axis=1))
        X_valid = X[valid_idx]
        y_valid = y_true[valid_idx]
        
        def objective(weights):
            ensemble = X_valid @ weights
            return np.sqrt(np.mean((y_valid - ensemble) ** 2))
        
        # Constraints: sum to 1, non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * len(model_names)
        w0 = np.ones(len(model_names)) / len(model_names)
        
        result = minimize(
            objective, w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return list(result.x)
    
    def step6_save_best_ensemble(
        self,
        comparison_df: pd.DataFrame,
        lgbm_model: RiskForecaster,
        ewma_model: EWMARiskForecaster,
        garch_model: Optional[GARCHRiskForecaster]
    ):
        """
        Step 6: Save best ensemble configuration.
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 6: SAVE BEST ENSEMBLE")
        logger.info("="*80)
        
        with Timer("Saving ensemble", logger):
            # Get best ensemble (excluding single models)
            ensemble_rows = comparison_df[~comparison_df['strategy'].str.startswith('single')]
            best_row = ensemble_rows.iloc[0]
            
            logger.info(f"\nBest ensemble: {best_row['strategy']}")
            logger.info(f"  RMSE: {best_row['rmse']:.6f}")
            logger.info(f"  MAE: {best_row['mae']:.6f}")
            logger.info(f"  Correlation: {best_row['correlation']:.4f}")
            
            # Save ensemble config
            ensemble_config = {
                'strategy': best_row['strategy'],
                'rmse': float(best_row['rmse']),
                'mae': float(best_row['mae']),
                'correlation': float(best_row['correlation']),
                'params': best_row['params'],
                'models_used': []
            }
            
            # Save individual models
            logger.info("\nSaving individual models...")
            lgbm_model.save_models(output_dir="artifacts/models_risk_ensemble/lgbm")
            ensemble_config['models_used'].append('lgbm')
            
            ewma_model.save_model("artifacts/models_risk_ensemble/ewma_model.pkl")
            ensemble_config['models_used'].append('ewma')
            
            if garch_model is not None:
                garch_model.save_model("artifacts/models_risk_ensemble/garch_model.pkl")
                ensemble_config['models_used'].append('garch')
            
            # Save ensemble configuration
            config_path = Path("artifacts/best_risk_ensemble.json")
            with open(config_path, 'w') as f:
                json.dump(ensemble_config, f, indent=2)
            
            logger.info(f"✓ Ensemble configuration saved to {config_path}")
            
            self.results['best_ensemble'] = ensemble_config
    
    def run_full_optimization(self) -> Dict:
        """
        Run complete risk ensemble optimization pipeline.
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING RISK ENSEMBLE OPTIMIZATION")
        logger.info("="*80)
        
        # Step 1: Prepare data
        df, returns = self.step1_prepare_data()
        
        # Step 2: Train LightGBM
        lgbm_model, lgbm_preds = self.step2_train_lgbm_model(df)
        
        # Step 3: Train EWMA
        ewma_model, ewma_preds = self.step3_train_ewma_model(returns)
        
        # Step 4: Train GARCH
        garch_model, garch_preds = self.step4_train_garch_model(returns)
        
        # Step 5: Compare ensembles
        comparison_df = self.step5_ensemble_comparison(
            df, lgbm_preds, ewma_preds, garch_preds
        )
        
        # Step 6: Save best ensemble
        self.step6_save_best_ensemble(
            comparison_df, lgbm_model, ewma_model, garch_model
        )
        
        logger.info("\n" + "="*80)
        logger.info("✓ RISK ENSEMBLE OPTIMIZATION COMPLETE!")
        logger.info("="*80)
        
        return self.results


def main():
    """Main execution function."""
    
    optimizer = RiskEnsembleOptimizer(config_path="conf/params.yaml")
    results = optimizer.run_full_optimization()
    
    logger.info("\n✅ Risk ensemble optimization complete!")
    logger.info(f"Best strategy: {results['best_ensemble']['strategy']}")
    logger.info(f"Best RMSE: {results['best_ensemble']['rmse']:.6f}")


if __name__ == "__main__":
    main()
