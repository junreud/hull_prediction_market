"""
Complete optimization pipeline for position mapping strategies.

This script:
1. Loads or generates return predictions (r_hat) and risk predictions (sigma_hat)
2. Tests multiple position mapping strategies
3. Optimizes strategy parameters using Optuna
4. Validates constraints (vol ratio, leverage limits)
5. Saves best strategy and parameters

Usage:
    python scripts/optimize_position_strategy.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, List, Tuple, Optional
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.data import DataLoader
from src.features import FeatureEngineering
from src.models import ReturnPredictor
from src.risk import RiskLabeler, RiskForecaster
from src.position import SharpeScalingMapper, QuantileBinningMapper
from src.metric import CompetitionMetric
from src.utils import get_logger, Timer, load_config

logger = get_logger(log_file="logs/position_optimization.log", level="INFO")


class PositionStrategyOptimizer:
    """
    End-to-end optimization pipeline for position mapping strategies.
    """
    
    def __init__(self, config_path: str = "conf/params.yaml"):
        """
        Initialize optimizer.
        
        Parameters
        ----------
        config_path : str
            Path to configuration file
        """
        self.config_path = config_path
        self.config = load_config(config_path)
        
        # Components
        self.data_loader = DataLoader(config_path)
        self.metric_calc = CompetitionMetric()
        
        # Results storage
        self.results = {}
        
        logger.info("="*80)
        logger.info("POSITION STRATEGY OPTIMIZATION PIPELINE")
        logger.info("="*80)
    
    def step1_load_predictions(
        self,
        return_model_dir: str = "artifacts/models_optimized",
        risk_model_dir: str = "artifacts/models_risk_optimized"
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Step 1: Load or generate return and risk predictions.
        
        Parameters
        ----------
        return_model_dir : str
            Directory containing return models
        risk_model_dir : str
            Directory containing risk models
            
        Returns
        -------
        Tuple[pd.DataFrame, np.ndarray, np.ndarray]
            (DataFrame with features and targets, return predictions, risk predictions)
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 1: LOAD/GENERATE PREDICTIONS")
        logger.info("="*80)
        
        with Timer("Prediction Loading", logger):
            # Load data
            logger.info("\n1.1 Loading training data...")
            train_df, _ = self.data_loader.load_data()
            logger.info(f"✓ Loaded {len(train_df)} samples with {train_df.shape[1]} columns")
            
            # Preprocess data (same as optimize_return_model.py)
            logger.info("\n1.2 Preprocessing data...")
            train_df, _ = self.data_loader.preprocess_timeseries(
                train_df,
                train_df=None,
                handle_outliers=True,
                winsorize_limits=(0.001, 0.001),
                winsorize_method='rolling',
                normalize=True,
                normalize_method='rank_gauss',
                scale=True,
                scale_method='robust',
                window=60
            )
            logger.info(f"✓ Preprocessing complete: {train_df.shape}")
            
            # Apply feature engineering
            logger.info("\n1.3 Applying feature engineering...")
            feature_engineer = FeatureEngineering(config_path=self.config_path)
            train_df = feature_engineer.fit_transform(train_df)
            logger.info(f"✓ Features created: {train_df.shape[1]} columns")
            
            # Try to load existing predictions
            return_pred_path = Path("artifacts/oof_return_predictions.npy")
            risk_pred_path = Path("artifacts/oof_risk_predictions.npy")
            
            if return_pred_path.exists() and risk_pred_path.exists():
                logger.info("\n1.3 Loading existing OOF predictions...")
                r_hat = np.load(return_pred_path)
                sigma_hat = np.load(risk_pred_path)
                logger.info(f"✓ Loaded predictions: r_hat={r_hat.shape}, sigma_hat={sigma_hat.shape}")
            else:
                logger.info("\n1.3 Generating OOF predictions from models...")
                r_hat, sigma_hat = self._generate_oof_predictions(
                    train_df, return_model_dir, risk_model_dir
                )
                
                # Save predictions
                np.save(return_pred_path, r_hat)
                np.save(risk_pred_path, sigma_hat)
                logger.info(f"✓ Saved predictions to artifacts/")
            
            # Validate predictions
            valid_mask = ~(np.isnan(r_hat) | np.isnan(sigma_hat))
            logger.info(f"\n1.3 Prediction statistics:")
            logger.info(f"  Valid samples: {valid_mask.sum()}/{len(r_hat)} ({valid_mask.sum()/len(r_hat)*100:.1f}%)")
            logger.info(f"  r_hat range: [{np.nanmin(r_hat):.6f}, {np.nanmax(r_hat):.6f}]")
            logger.info(f"  sigma_hat range: [{np.nanmin(sigma_hat):.6f}, {np.nanmax(sigma_hat):.6f}]")
            
            self.results['predictions'] = {
                'n_valid': valid_mask.sum(),
                'n_total': len(r_hat),
                'r_hat_mean': float(np.nanmean(r_hat)),
                'sigma_hat_mean': float(np.nanmean(sigma_hat))
            }
            
            return train_df, r_hat, sigma_hat
    
    def _generate_oof_predictions(
        self,
        train_df: pd.DataFrame,
        return_model_dir: str,
        risk_model_dir: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate OOF predictions using saved models.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data
        return_model_dir : str
            Return model directory
        risk_model_dir : str
            Risk model directory
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (Return predictions, Risk predictions)
        """
        logger.info("Generating OOF predictions (this may take a while)...")
        
        # Generate return predictions
        logger.info("\nGenerating return predictions...")
        return_predictor = ReturnPredictor(config_path=self.config_path)
        
        # Load models manually
        return_model_path = Path(return_model_dir)
        model_files = sorted(return_model_path.glob("lightgbm_fold_*.pkl"))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {return_model_dir}")
        
        logger.info(f"Loading {len(model_files)} return models...")
        return_predictor.models = []
        for model_file in model_files:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
                return_predictor.models.append(model)
        
        # Load feature names from saved feature selection file
        feature_selection_path = Path("results/feature_selection/selected_features_optimized.csv")
        if feature_selection_path.exists():
            feature_df = pd.read_csv(feature_selection_path)
            feature_cols = feature_df['feature'].tolist()
            # Filter to only features that exist in train_df
            feature_cols = [f for f in feature_cols if f in train_df.columns]
            logger.info(f"Using {len(feature_cols)} features from feature selection file (filtered to available)")
        else:
            raise FileNotFoundError(f"Feature selection file not found: {feature_selection_path}")
        
        X_return = train_df[feature_cols].values
        r_hat = return_predictor.predict(X_return)
        
        # Generate risk predictions
        logger.info("\nGenerating risk predictions...")
        
        # Create risk labels using fit_transform (returns DataFrame)
        risk_labeler = RiskLabeler()
        train_df = risk_labeler.fit_transform(train_df, target_col='forward_returns')
        logger.info("✓ Risk labels created")
        
        risk_forecaster = RiskForecaster(config_path=self.config_path)
        
        # Load models manually
        risk_model_path = Path(risk_model_dir)
        risk_model_files = sorted(risk_model_path.glob("lightgbm_fold_*.pkl"))
        
        if not risk_model_files:
            raise FileNotFoundError(f"No model files found in {risk_model_dir}")
        
        logger.info(f"Loading {len(risk_model_files)} risk models...")
        risk_forecaster.models = []
        for model_file in risk_model_files:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
                risk_forecaster.models.append(model)
        
        # Load feature names from saved feature selection file
        risk_feature_selection_path = Path("results/feature_selection/selected_features_risk_optimized.csv")
        if risk_feature_selection_path.exists():
            risk_feature_df = pd.read_csv(risk_feature_selection_path)
            risk_feature_cols = risk_feature_df['feature'].tolist()
            # Filter to only features that exist in train_df
            risk_feature_cols = [f for f in risk_feature_cols if f in train_df.columns]
            logger.info(f"Using {len(risk_feature_cols)} features from risk feature selection file (filtered to available)")
        else:
            raise FileNotFoundError(f"Risk feature selection file not found: {risk_feature_selection_path}")
        
        X_risk = train_df[risk_feature_cols].values
        sigma_hat = risk_forecaster.predict(X_risk)
        
        logger.info(f"✓ Generated predictions: r_hat={r_hat.shape}, sigma_hat={sigma_hat.shape}")
        
        return r_hat, sigma_hat
    
    def step2_optimize_sharpe_scaling(
        self,
        train_df: pd.DataFrame,
        r_hat: np.ndarray,
        sigma_hat: np.ndarray,
        n_trials: int = 100
    ) -> Dict:
        """
        Step 2: Optimize Sharpe Scaling strategy parameters.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with actual returns
        r_hat : np.ndarray
            Return predictions
        sigma_hat : np.ndarray
            Risk predictions
        n_trials : int
            Number of optimization trials
            
        Returns
        -------
        dict
            Best parameters and score
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 2: OPTIMIZE SHARPE SCALING STRATEGY")
        logger.info("="*80)
        
        with Timer("Sharpe Scaling Optimization", logger):
            # Get actual returns
            actual_returns = train_df['forward_returns'].values
            
            # Create mapper
            mapper = SharpeScalingMapper(config_path=self.config_path)
            
            def objective(trial):
                """Optuna objective function."""
                # Sample parameters
                k = trial.suggest_float('k', 0.1, 2.0)
                b = trial.suggest_float('b', 0.5, 5.0)
                
                # Generate positions
                positions = mapper.map_positions(r_hat, sigma_hat, k=k, b=b)
                
                # Calculate metric (using allocations and forward_returns)
                result = self.metric_calc.calculate_score(
                    allocations=positions,
                    forward_returns=actual_returns
                )
                score = result['score']
                
                # Check constraints
                strategy_vol = result['strategy_vol']
                market_vol = result['market_vol']
                vol_ratio = strategy_vol / (market_vol + 1e-10)
                
                # Penalize constraint violations
                if vol_ratio > mapper.max_vol_ratio:
                    penalty = (vol_ratio - mapper.max_vol_ratio) * 10
                    score -= penalty
                
                # Check leverage usage
                leverage_pct = np.sum(positions >= 1.9) / len(positions)
                if leverage_pct > mapper.max_leverage_pct:
                    penalty = (leverage_pct - mapper.max_leverage_pct) * 20
                    score -= penalty
                
                return score
            
            # Run optimization
            logger.info(f"\n2.1 Running Optuna optimization ({n_trials} trials)...")
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
            )
            
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            # Get best results
            best_params = study.best_params
            best_score = study.best_value
            
            logger.info(f"\n✓ Sharpe Scaling optimization complete")
            logger.info(f"  Best score: {best_score:.6f}")
            logger.info(f"  Best k: {best_params['k']:.4f}")
            logger.info(f"  Best b: {best_params['b']:.4f}")
            
            # Generate positions with best params
            best_positions = mapper.map_positions(
                r_hat, sigma_hat, 
                k=best_params['k'], 
                b=best_params['b']
            )
            
            # Validate constraints
            validation = mapper.validate_constraints(
                best_positions, actual_returns, actual_returns
            )
            
            self.results['sharpe_scaling'] = {
                'best_params': best_params,
                'best_score': float(best_score),
                'n_trials': n_trials,
                'validation': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                              for k, v in validation.items()}
            }
            
            return self.results['sharpe_scaling']
    
    def step3_optimize_quantile_binning(
        self,
        train_df: pd.DataFrame,
        r_hat: np.ndarray,
        sigma_hat: np.ndarray,
        n_trials: int = 100
    ) -> Dict:
        """
        Step 3: Optimize Quantile Binning strategy.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with actual returns
        r_hat : np.ndarray
            Return predictions
        sigma_hat : np.ndarray
            Risk predictions
        n_trials : int
            Number of optimization trials
            
        Returns
        -------
        dict
            Best parameters and score
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 3: OPTIMIZE QUANTILE BINNING STRATEGY")
        logger.info("="*80)
        
        with Timer("Quantile Binning Optimization", logger):
            # Get actual returns
            actual_returns = train_df['forward_returns'].values
            
            # Create mapper
            mapper = QuantileBinningMapper(config_path=self.config_path)
            n_bins = mapper.n_bins
            
            def objective(trial):
                """Optuna objective function."""
                # Sample allocations for each bin
                # Bins are sorted from lowest to highest z-score
                allocations = []
                for i in range(n_bins):
                    # Lower bins (negative z) should have lower allocations
                    if i < n_bins // 2:
                        # Bearish bins: 0.0 to 1.0
                        alloc = trial.suggest_float(f'bin_{i}', 0.0, 1.0)
                    else:
                        # Bullish bins: 0.5 to 2.0
                        alloc = trial.suggest_float(f'bin_{i}', 0.5, 2.0)
                    allocations.append(alloc)
                
                # Generate positions
                positions = mapper.map_positions(
                    r_hat, sigma_hat, allocations=np.array(allocations)
                )
                
                # Calculate metric (using allocations and forward_returns)
                result = self.metric_calc.calculate_score(
                    allocations=positions,
                    forward_returns=actual_returns
                )
                score = result['score']
                
                # Check constraints
                strategy_vol = result['strategy_vol']
                market_vol = result['market_vol']
                vol_ratio = result['vol_ratio']
                
                # Penalize violations
                if vol_ratio > mapper.max_vol_ratio:
                    score -= (vol_ratio - mapper.max_vol_ratio) * 10
                
                leverage_pct = np.sum(positions >= 1.9) / len(positions)
                if leverage_pct > mapper.max_leverage_pct:
                    score -= (leverage_pct - mapper.max_leverage_pct) * 20
                
                return score
            
            # Run optimization
            logger.info(f"\n3.1 Running Optuna optimization ({n_trials} trials, {n_bins} bins)...")
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
            )
            
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            # Get best results
            best_params = study.best_params
            best_score = study.best_value
            
            # Extract allocations
            best_allocations = [best_params[f'bin_{i}'] for i in range(n_bins)]
            
            logger.info(f"\n✓ Quantile Binning optimization complete")
            logger.info(f"  Best score: {best_score:.6f}")
            logger.info(f"  Best allocations: {[f'{a:.2f}' for a in best_allocations]}")
            
            # Generate positions with best params
            best_positions = mapper.map_positions(
                r_hat, sigma_hat, allocations=np.array(best_allocations)
            )
            
            # Validate constraints
            validation = mapper.validate_constraints(
                best_positions, actual_returns, actual_returns
            )
            
            self.results['quantile_binning'] = {
                'best_allocations': best_allocations,
                'best_score': float(best_score),
                'n_bins': n_bins,
                'n_trials': n_trials,
                'validation': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                              for k, v in validation.items()}
            }
            
            return self.results['quantile_binning']
    
    def step4_compare_strategies(
        self,
        train_df: pd.DataFrame,
        r_hat: np.ndarray,
        sigma_hat: np.ndarray
    ) -> pd.DataFrame:
        """
        Step 4: Compare all optimized strategies.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data
        r_hat : np.ndarray
            Return predictions
        sigma_hat : np.ndarray
            Risk predictions
            
        Returns
        -------
        pd.DataFrame
            Strategy comparison results
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 4: STRATEGY COMPARISON")
        logger.info("="*80)
        
        with Timer("Strategy Comparison", logger):
            actual_returns = train_df['forward_returns'].values
            market_vol = np.nanstd(actual_returns)
            
            comparison_data = []
            
            # 1. Sharpe Scaling
            if 'sharpe_scaling' in self.results:
                logger.info("\n4.1 Evaluating Sharpe Scaling...")
                params = self.results['sharpe_scaling']['best_params']
                mapper = SharpeScalingMapper(config_path=self.config_path)
                positions = mapper.map_positions(r_hat, sigma_hat, k=params['k'], b=params['b'])
                
                result = self.metric_calc.calculate_score(
                    allocations=positions,
                    forward_returns=actual_returns
                )
                
                comparison_data.append({
                    'strategy': 'Sharpe Scaling',
                    'score': result['score'],
                    'mean_return': result['mean_return'],
                    'volatility': result['strategy_vol'],
                    'vol_ratio': result['vol_ratio'],
                    'sharpe_ratio': result['sharpe'],
                    'leverage_pct': np.sum(positions >= 1.9) / len(positions),
                    'params': f"k={params['k']:.3f}, b={params['b']:.3f}"
                })
            
            # 2. Quantile Binning
            if 'quantile_binning' in self.results:
                logger.info("\n4.2 Evaluating Quantile Binning...")
                allocations = self.results['quantile_binning']['best_allocations']
                mapper = QuantileBinningMapper(config_path=self.config_path)
                positions = mapper.map_positions(r_hat, sigma_hat, allocations=np.array(allocations))
                
                result = self.metric_calc.calculate_score(
                    allocations=positions,
                    forward_returns=actual_returns
                )
                
                comparison_data.append({
                    'strategy': 'Quantile Binning',
                    'score': result['score'],
                    'mean_return': result['mean_return'],
                    'volatility': result['strategy_vol'],
                    'vol_ratio': result['vol_ratio'],
                    'sharpe_ratio': result['sharpe'],
                    'leverage_pct': np.sum(positions >= 1.9) / len(positions),
                    'params': f"bins={len(allocations)}"
                })
            
            # 3. Baseline (always 1.0 allocation)
            logger.info("\n4.3 Evaluating Baseline (buy-and-hold)...")
            baseline_positions = np.ones_like(actual_returns)
            baseline_result = self.metric_calc.calculate_score(
                allocations=baseline_positions,
                forward_returns=actual_returns
            )
            
            comparison_data.append({
                'strategy': 'Baseline (1.0)',
                'score': baseline_result['score'],
                'mean_return': baseline_result['mean_return'],
                'volatility': baseline_result['strategy_vol'],
                'vol_ratio': baseline_result['vol_ratio'],
                'sharpe_ratio': baseline_result['sharpe'],
                'leverage_pct': 0.0,
                'params': 'allocation=1.0'
            })
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('score', ascending=False)
            
            logger.info("\n" + "="*80)
            logger.info("STRATEGY COMPARISON RESULTS")
            logger.info("="*80)
            logger.info(f"\n{comparison_df.to_string()}")
            
            # Save comparison
            comparison_path = Path("results/position_strategy_comparison.csv")
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(comparison_path, index=False)
            logger.info(f"\n✓ Saved comparison to {comparison_path}")
            
            self.results['comparison'] = comparison_df.to_dict('records')
            
            return comparison_df
    
    def step5_save_best_strategy(self, comparison_df: pd.DataFrame) -> Dict:
        """
        Step 5: Save the best performing strategy.
        
        Parameters
        ----------
        comparison_df : pd.DataFrame
            Strategy comparison results
            
        Returns
        -------
        dict
            Best strategy info
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 5: SAVE BEST STRATEGY")
        logger.info("="*80)
        
        with Timer("Saving Best Strategy", logger):
            best_row = comparison_df.iloc[0]
            best_strategy = best_row['strategy']
            
            logger.info(f"\n5.1 Best strategy: {best_strategy}")
            logger.info(f"  Score: {best_row['score']:.6f}")
            logger.info(f"  Vol Ratio: {best_row['vol_ratio']:.4f}")
            logger.info(f"  Leverage %: {best_row['leverage_pct']*100:.2f}%")
            
            # Save best strategy configuration
            best_config = {
                'strategy_name': best_strategy,
                'score': float(best_row['score']),
                'metrics': {
                    'mean_return': float(best_row['mean_return']),
                    'volatility': float(best_row['volatility']),
                    'vol_ratio': float(best_row['vol_ratio']),
                    'sharpe_ratio': float(best_row['sharpe_ratio']),
                    'leverage_pct': float(best_row['leverage_pct'])
                }
            }
            
            # Add strategy-specific parameters
            if best_strategy == 'Sharpe Scaling':
                best_config['parameters'] = self.results['sharpe_scaling']['best_params']
            elif best_strategy == 'Quantile Binning':
                best_config['parameters'] = {
                    'allocations': self.results['quantile_binning']['best_allocations'],
                    'n_bins': self.results['quantile_binning']['n_bins']
                }
            
            # Save to file
            config_path = Path("artifacts/best_position_strategy.json")
            with open(config_path, 'w') as f:
                json.dump(best_config, f, indent=2)
            
            logger.info(f"\n✓ Saved best strategy to {config_path}")
            
            self.results['best_strategy'] = best_config
            
            return best_config
    
    def step6_predict_test(
        self,
        return_model_dir: str,
        risk_model_dir: str,
        best_strategy_config: Dict
    ) -> pd.DataFrame:
        """
        Step 6: Predict on test data and generate submission.
        
        Parameters
        ----------
        return_model_dir : str
            Directory with saved return models
        risk_model_dir : str
            Directory with saved risk models
        best_strategy_config : Dict
            Best position strategy configuration
            
        Returns
        -------
        pd.DataFrame
            Submission dataframe with date_id and allocation
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 6: TEST PREDICTION & SUBMISSION GENERATION")
        logger.info("="*80)
        
        with Timer("Test Prediction", logger):
            # Load test data
            logger.info("\n6.1 Loading test data...")
            _, test_df = self.data_loader.load_data()
            logger.info(f"✓ Test data loaded: {test_df.shape}")
            
            # Preprocess test data (use fitted transformers)
            logger.info("\n6.2 Preprocessing test data...")
            test_processed, _ = self.data_loader.preprocess_timeseries(
                test_df,
                train_df=None,  # Use fitted transformers
                handle_outliers=True,
                winsorize_limits=(0.001, 0.001),
                winsorize_method='rolling',
                normalize=True,
                normalize_method='rank_gauss',
                scale=True,
                scale_method='robust',
                window=60
            )
            logger.info(f"✓ Preprocessing complete: {test_processed.shape}")
            
            # Feature engineering (use transform, not fit_transform)
            logger.info("\n6.3 Applying feature engineering...")
            feature_engineer = FeatureEngineering(config_path=self.config_path)
            test_features = feature_engineer.transform(test_processed)
            logger.info(f"✓ Features created: {test_features.shape[1]} columns")
            
            # Generate return predictions
            logger.info("\n6.4 Generating return predictions...")
            return_predictor = ReturnPredictor(config_path=self.config_path)
            
            # Load return models
            return_model_path = Path(return_model_dir)
            return_model_files = sorted(return_model_path.glob("lightgbm_fold_*.pkl"))
            
            if not return_model_files:
                raise FileNotFoundError(f"No return models found in {return_model_dir}")
            
            logger.info(f"Loading {len(return_model_files)} return models...")
            return_models = []
            for model_file in return_model_files:
                with open(model_file, 'rb') as f:
                    return_models.append(pickle.load(f))
            
            # Get return feature columns
            feature_path = Path("results/feature_selection/selected_features_optimized.csv")
            if not feature_path.exists():
                raise FileNotFoundError(f"Feature list not found: {feature_path}")
            
            return_features = pd.read_csv(feature_path)['feature'].tolist()
            
            # Filter to only features that exist in test data
            available_features = [f for f in return_features if f in test_features.columns]
            missing_features = set(return_features) - set(available_features)
            
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features in test data, adding with 0.0")
                for col in missing_features:
                    test_features[col] = 0.0
            
            # Ensure exact feature order and count
            X_test_return = test_features[return_features].values
            logger.info(f"Test return features shape: {X_test_return.shape}, Expected features: {len(return_features)}")
            
            # Ensemble prediction (mean of all folds, disable shape check for flexibility)
            r_hat_test = np.mean([
                model.predict(X_test_return, predict_disable_shape_check=True) 
                for model in return_models
            ], axis=0)
            logger.info(f"✓ Return predictions: range=[{r_hat_test.min():.6f}, {r_hat_test.max():.6f}]")
            
            # Generate risk predictions
            logger.info("\n6.5 Generating risk predictions...")
            risk_forecaster = RiskForecaster(config_path=self.config_path)
            
            # Load risk models
            risk_model_path = Path(risk_model_dir)
            risk_model_files = sorted(risk_model_path.glob("lightgbm_fold_*.pkl"))
            
            if not risk_model_files:
                raise FileNotFoundError(f"No risk models found in {risk_model_dir}")
            
            logger.info(f"Loading {len(risk_model_files)} risk models...")
            risk_models = []
            for model_file in risk_model_files:
                with open(model_file, 'rb') as f:
                    risk_models.append(pickle.load(f))
            
            # Get risk feature columns
            risk_feature_path = Path("results/feature_selection/selected_features_risk_optimized.csv")
            if not risk_feature_path.exists():
                raise FileNotFoundError(f"Risk feature list not found: {risk_feature_path}")
            
            risk_features = pd.read_csv(risk_feature_path)['feature'].tolist()
            
            # Filter to only features that exist in test data
            available_features = [f for f in risk_features if f in test_features.columns]
            missing_features = set(risk_features) - set(available_features)
            
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features in test data, adding with 0.0")
                for col in missing_features:
                    test_features[col] = 0.0
            
            # Ensure exact feature order and count
            X_test_risk = test_features[risk_features].values
            logger.info(f"Test risk features shape: {X_test_risk.shape}, Expected features: {len(risk_features)}")
            
            # Ensemble prediction (disable shape check for flexibility)
            sigma_hat_test = np.mean([
                model.predict(X_test_risk, predict_disable_shape_check=True) 
                for model in risk_models
            ], axis=0)
            logger.info(f"✓ Risk predictions: range=[{sigma_hat_test.min():.6f}, {sigma_hat_test.max():.6f}]")
            
            # Apply position mapping strategy
            logger.info(f"\n6.6 Applying position strategy: {best_strategy_config['strategy_name']}...")
            
            if best_strategy_config['strategy_name'] == 'Sharpe Scaling':
                mapper = SharpeScalingMapper(config_path=self.config_path)
                params = best_strategy_config['parameters']
                allocations = mapper.map_positions(
                    r_hat_test,
                    sigma_hat_test,
                    target_sharpe=params['target_sharpe'],
                    max_position=params['max_position'],
                    min_position=params['min_position']
                )
            
            elif best_strategy_config['strategy_name'] == 'Quantile Binning':
                mapper = QuantileBinningMapper(config_path=self.config_path)
                params = best_strategy_config['parameters']
                
                # Fit mapper on test predictions (using quantiles)
                mapper.fit(r_hat_test, sigma_hat_test)
                allocations = mapper.map_positions(r_hat_test, sigma_hat_test, allocations=np.array(params['allocations']))
            
            else:
                raise ValueError(f"Unknown strategy: {best_strategy_config['strategy_name']}")
            
            logger.info(f"✓ Allocations: range=[{allocations.min():.4f}, {allocations.max():.4f}]")
            logger.info(f"  Mean: {allocations.mean():.4f}, Std: {allocations.std():.4f}")
            
            # Create submission
            submission = pd.DataFrame({
                'date_id': test_features['date_id'].astype(int),
                'allocation': allocations.astype(float)
            })
            
            # Validate submission
            logger.info("\n6.7 Validating submission...")
            assert list(submission.columns) == ['date_id', 'allocation'], "Wrong column names!"
            assert submission['allocation'].isna().sum() == 0, "Contains NaN values!"
            assert (submission['allocation'] >= 0).all(), "Contains values < 0!"
            assert (submission['allocation'] <= 2).all(), "Contains values > 2!"
            logger.info("✓ Validation passed")
            
            # Save submission
            output_path = Path("submissions/submission.parquet")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            submission.to_parquet(output_path, index=False, engine='pyarrow')
            
            logger.info(f"\n✓ Submission saved to {output_path}")
            logger.info(f"  Rows: {len(submission)}")
            logger.info(f"  Allocation stats:")
            logger.info(f"    Min: {submission['allocation'].min():.4f}")
            logger.info(f"    Max: {submission['allocation'].max():.4f}")
            logger.info(f"    Mean: {submission['allocation'].mean():.4f}")
            logger.info(f"    Median: {submission['allocation'].median():.4f}")
            
            self.results['test_predictions'] = {
                'n_samples': len(submission),
                'allocation_mean': float(submission['allocation'].mean()),
                'allocation_std': float(submission['allocation'].std()),
                'allocation_min': float(submission['allocation'].min()),
                'allocation_max': float(submission['allocation'].max()),
                'output_path': str(output_path)
            }
            
            return submission
    
    def run_full_optimization(
        self,
        return_model_dir: str = "artifacts/models_optimized",
        risk_model_dir: str = "artifacts/models_risk_optimized",
        n_trials_sharpe: int = 100,
        n_trials_quantile: int = 100,
        predict_test: bool = False
    ) -> Dict:
        """
        Run the complete position strategy optimization pipeline.
        
        Parameters
        ----------
        return_model_dir : str
            Return model directory
        risk_model_dir : str
            Risk model directory
        n_trials_sharpe : int
            Number of trials for Sharpe scaling
        n_trials_quantile : int
            Number of trials for Quantile binning
            
        Returns
        -------
        dict
            Complete optimization results
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING POSITION STRATEGY OPTIMIZATION")
        logger.info("="*80)
        
        start_time = pd.Timestamp.now()
        
        try:
            # Step 1: Load/generate predictions
            train_df, r_hat, sigma_hat = self.step1_load_predictions(
                return_model_dir, risk_model_dir
            )
            
            # Step 2: Optimize Sharpe Scaling
            self.step2_optimize_sharpe_scaling(
                train_df, r_hat, sigma_hat, n_trials=n_trials_sharpe
            )
            
            # Step 3: Optimize Quantile Binning
            self.step3_optimize_quantile_binning(
                train_df, r_hat, sigma_hat, n_trials=n_trials_quantile
            )
            
            # Step 4: Compare strategies
            comparison_df = self.step4_compare_strategies(
                train_df, r_hat, sigma_hat
            )
            
            # Step 5: Save best strategy
            best_strategy = self.step5_save_best_strategy(comparison_df)
            
            # Step 6: Predict on test data (optional)
            if predict_test:
                submission = self.step6_predict_test(
                    return_model_dir,
                    risk_model_dir,
                    best_strategy
                )
            
            # Final summary
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("\n" + "="*80)
            logger.info("OPTIMIZATION COMPLETE")
            logger.info("="*80)
            logger.info(f"Total time: {duration:.1f}s")
            logger.info(f"Best strategy: {best_strategy['strategy_name']}")
            logger.info(f"Best score: {best_strategy['score']:.6f}")
            
            if predict_test:
                logger.info(f"✓ Submission generated: {self.results['test_predictions']['output_path']}")
            
            self.results['duration_seconds'] = duration
            self.results['status'] = 'success'
            
            # Save complete results
            results_path = Path("artifacts/position_optimization_results.json")
            with open(results_path, 'w') as f:
                # Convert numpy types to Python types
                def convert_types(obj):
                    if isinstance(obj, dict):
                        return {k: convert_types(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_types(v) for v in obj]
                    elif isinstance(obj, (bool, np.bool_)):
                        return bool(obj)
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    else:
                        return obj
                
                json.dump(convert_types(self.results), f, indent=2)
            
            logger.info(f"✓ Saved complete results to {results_path}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}", exc_info=True)
            self.results['status'] = 'failed'
            self.results['error'] = str(e)
            raise


def main():
    """Main execution function."""
    logger.info("="*80)
    logger.info("POSITION STRATEGY OPTIMIZATION")
    logger.info("="*80)
    
    # Create optimizer
    optimizer = PositionStrategyOptimizer(config_path="conf/params.yaml")
    
    # Run full optimization
    results = optimizer.run_full_optimization(
        return_model_dir="artifacts/models_optimized",
        risk_model_dir="artifacts/models_risk_optimized",
        n_trials_sharpe=100,      # Adjust based on time/resources
        n_trials_quantile=100,    # Adjust based on time/resources
        predict_test=True         # Generate submission.parquet
    )
    
    logger.info("\n✅ Position strategy optimization complete!")
    logger.info(f"Best strategy: {results['best_strategy']['strategy_name']}")
    logger.info(f"Best score: {results['best_strategy']['score']:.6f}")
    
    if 'test_predictions' in results:
        logger.info(f"✅ Submission file created: {results['test_predictions']['output_path']}")


if __name__ == "__main__":
    main()
