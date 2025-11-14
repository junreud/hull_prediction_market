"""
Optimize ensemble of return prediction models.

This script:
1. Trains multiple models with different feature sets
2. Combines them using various ensemble strategies
3. Optimizes ensemble weights for best Sharpe ratio
4. Saves best ensemble configuration
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from typing import Dict, List
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import DataLoader
from src.features import FeatureEngineering
from src.models import ReturnPredictor
from src.ensemble import ModelEnsemble, combine_risk_predictions
from src.position import QuantileBinningMapper
from src.cv import PurgedWalkForwardCV
from src.utils import setup_logging, Timer

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def train_diverse_models(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    config_path: str = "conf/params.yaml"
) -> Dict[str, Dict]:
    """
    Train multiple models with different configurations for diversity.
    
    Returns
    -------
    Dict[str, Dict]
        Dictionary with model_name -> {
            'oof_predictions': np.ndarray,
            'models': List[model],
            'feature_importance': pd.DataFrame
        }
    """
    logger.info("=" * 80)
    logger.info("TRAINING DIVERSE MODELS FOR ENSEMBLE")
    logger.info("=" * 80)
    
    results = {}
    
    # Model 1: Standard features (current best)
    logger.info("\n[Model 1] Standard features")
    with Timer("Model 1 training"):
        predictor1 = ReturnPredictor(
            model_type='lightgbm',
            config_path=config_path
        )
        
        # Load best params from previous optimization
        best_params_path = project_root / "artifacts" / "lightgbm_best_params_optimized.json"
        if best_params_path.exists():
            with open(best_params_path) as f:
                best_params = json.load(f)
            predictor1.params.update(best_params)
            logger.info("Loaded optimized parameters")
        
        result1 = predictor1.train_cv(
            df=df,
            target_col=target_col,
            date_col=date_col
        )
        
        results['standard'] = {
            'oof_predictions': result1['oof_predictions'],
            'models': predictor1.models,
            'feature_importance': result1.get('feature_importance'),
            'oof_score': result1['oof_score']
        }
        logger.info(f"Standard model OOF RMSE: {result1['oof_score']:.6f}")
    
    # Model 2: Higher tree complexity (more depth, fewer leaves)
    logger.info("\n[Model 2] Higher tree complexity")
    with Timer("Model 2 training"):
        predictor2 = ReturnPredictor(
            model_type='lightgbm',
            config_path=config_path
        )
        
        # Modify for more complex trees
        complex_params = {
            'max_depth': 8,  # Deeper trees
            'num_leaves': 31,  # Fewer leaves (less complex per tree)
            'min_child_samples': 10,  # Smaller min samples
            'learning_rate': 0.03,  # Slower learning
            'n_estimators': 300  # More trees
        }
        predictor2.params.update(complex_params)
        
        result2 = predictor2.train_cv(
            df=df,
            target_col=target_col,
            date_col=date_col
        )
        
        results['complex'] = {
            'oof_predictions': result2['oof_predictions'],
            'models': predictor2.models,
            'feature_importance': result2.get('feature_importance'),
            'oof_score': result2['oof_score']
        }
        logger.info(f"Complex model OOF RMSE: {result2['oof_score']:.6f}")
    
    # Model 3: Regularized (L1/L2)
    logger.info("\n[Model 3] Regularized model")
    with Timer("Model 3 training"):
        predictor3 = ReturnPredictor(
            model_type='lightgbm',
            config_path=config_path
        )
        
        # Strong regularization
        reg_params = {
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'min_gain_to_split': 0.01,
            'feature_fraction': 0.7,  # Feature sampling
            'bagging_fraction': 0.7,  # Row sampling
            'bagging_freq': 1
        }
        predictor3.params.update(reg_params)
        
        result3 = predictor3.train_cv(
            df=df,
            target_col=target_col,
            date_col=date_col
        )
        
        results['regularized'] = {
            'oof_predictions': result3['oof_predictions'],
            'models': predictor3.models,
            'feature_importance': result3.get('feature_importance'),
            'oof_score': result3['oof_score']
        }
        logger.info(f"Regularized model OOF RMSE: {result3['oof_score']:.6f}")
    
    return results


def optimize_ensemble_weights(
    model_results: Dict[str, Dict],
    y_true: np.ndarray,
    date_ids: np.ndarray,
    market_returns: np.ndarray
) -> Dict:
    """
    Find optimal ensemble weights that maximize Sharpe ratio.
    
    Uses Optuna to search for best weights.
    """
    import optuna
    
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZING ENSEMBLE WEIGHTS")
    logger.info("=" * 80)
    
    # Extract OOF predictions
    oof_preds = {
        name: result['oof_predictions']
        for name, result in model_results.items()
    }
    
    # Load position mapper from saved config
    position_config_path = project_root / "artifacts" / "best_position_strategy.json"
    
    # Always create mapper with default config
    position_mapper = QuantileBinningMapper(config_path="conf/params.yaml")
    
    # If we have optimized allocations, we'll use them in map_positions
    optimal_allocations = None
    if position_config_path.exists():
        with open(position_config_path) as f:
            position_config = json.load(f)
        
        strategy_name = position_config['strategy_name']
        if 'Quantile' in strategy_name and 'allocations' in position_config.get('parameters', {}):
            optimal_allocations = np.array(position_config['parameters']['allocations'])
            logger.info(f"Loaded optimized allocations: {optimal_allocations}")
        logger.info(f"Loaded position strategy: {strategy_name}")
    else:
        logger.info("Using default Quantile Binning strategy")
    
    def objective(trial):
        """Objective function: maximize Sharpe ratio."""
        n_models = len(oof_preds)
        
        # Suggest weights (will be normalized to sum to 1)
        raw_weights = [
            trial.suggest_float(f'weight_{i}', 0.0, 1.0)
            for i in range(n_models)
        ]
        
        # Normalize
        total = sum(raw_weights)
        if total == 0:
            return -999.0
        weights = [w / total for w in raw_weights]
        
        # Create weighted ensemble
        ensemble = ModelEnsemble(
            strategy='weighted_average',
            weights=weights
        )
        ensemble.model_names = list(oof_preds.keys())
        ensemble.is_fitted = True
        
        # Get ensemble predictions
        ensemble_pred = ensemble.predict(oof_preds)
        
        # Assume constant risk for optimization
        # (In practice, would use risk model predictions)
        sigma_hat = np.std(y_true) * np.ones_like(ensemble_pred)
        
        # Convert to positions
        if optimal_allocations is not None:
            allocations = position_mapper.map_positions(
                r_hat=ensemble_pred,
                sigma_hat=sigma_hat,
                allocations=optimal_allocations
            )
        else:
            allocations = position_mapper.calculate_positions(
                r_hat=ensemble_pred,
                sigma_hat=sigma_hat
            )
        
        # Calculate Sharpe
        strategy_returns = allocations * market_returns
        market_vol = np.std(market_returns)
        strategy_vol = np.std(strategy_returns)
        
        vol_ratio = strategy_vol / (market_vol + 1e-8)
        vol_penalty = 1.0 + max(0.0, vol_ratio - 1.2)
        
        mean_return = np.mean(strategy_returns)
        sharpe = (mean_return / (strategy_vol + 1e-8)) / vol_penalty
        
        return sharpe
    
    # Run optimization
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(
        objective,
        n_trials=100,
        show_progress_bar=True
    )
    
    # Extract best weights
    best_trial = study.best_trial
    n_models = len(oof_preds)
    raw_weights = [best_trial.params[f'weight_{i}'] for i in range(n_models)]
    total = sum(raw_weights)
    best_weights = [w / total for w in raw_weights]
    
    logger.info(f"\nBest Sharpe: {best_trial.value:.6f}")
    logger.info("Best weights:")
    for name, weight in zip(oof_preds.keys(), best_weights):
        logger.info(f"  {name}: {weight:.4f}")
    
    return {
        'weights': best_weights,
        'model_names': list(oof_preds.keys()),
        'best_sharpe': best_trial.value,
        'study': study
    }


def main():
    """Main optimization pipeline."""
    
    # Step 1: Load data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOAD DATA")
    logger.info("=" * 80)
    
    with Timer("Data loading"):
        loader = DataLoader(config_path="conf/params.yaml")
        train_df, _ = loader.load_data()  # Returns tuple (train, test)
        logger.info(f"Loaded {len(train_df)} samples")
    
    # Step 2: Feature engineering
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 80)
    
    with Timer("Feature engineering"):
        engineer = FeatureEngineering(config_path="conf/params.yaml")
        df_features = engineer.fit_transform(train_df)
        logger.info(f"Created {len(df_features.columns)} features")
    
    # Step 3: Feature selection
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: FEATURE SELECTION")
    logger.info("=" * 80)
    
    with Timer("Feature selection"):
        feature_cols = [
            col for col in df_features.columns
            if col not in ['date_id', 'forward_returns', 'symbol']
        ]
        
        # Simple variance-based selection
        variances = df_features[feature_cols].var()
        selected_features = variances[variances > 1e-6].index.tolist()
        
        logger.info(f"Selected {len(selected_features)} features (variance > 1e-6)")
        
        # Use selected features
        df_selected = df_features[['date_id', 'forward_returns'] + selected_features].copy()
    
    # Step 4: Train diverse models
    model_results = train_diverse_models(
        df=df_selected,
        target_col='forward_returns',
        date_col='date_id'
    )
    
    # Step 5: Test different ensemble strategies
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: COMPARE ENSEMBLE STRATEGIES")
    logger.info("=" * 80)
    
    # Extract OOF predictions and true values
    oof_preds = {
        name: result['oof_predictions']
        for name, result in model_results.items()
    }
    
    # Get true values from first model result
    first_result = list(model_results.values())[0]
    y_true = df_selected['forward_returns'].values
    date_ids = df_selected['date_id'].values
    
    # Align with OOF predictions (some samples may be excluded)
    oof_mask = ~np.isnan(first_result['oof_predictions'])
    y_true_oof = y_true[oof_mask]
    date_ids_oof = date_ids[oof_mask]
    
    # Filter OOF predictions
    oof_preds_filtered = {
        name: preds[oof_mask]
        for name, preds in oof_preds.items()
    }
    
    strategies_to_test = [
        'simple_average',
        'optimized_weights',
        'time_weighted',
        'stacking'
    ]
    
    results_comparison = {}
    
    for strategy in strategies_to_test:
        logger.info(f"\nTesting strategy: {strategy}")
        
        try:
            if strategy == 'time_weighted':
                ensemble = ModelEnsemble(strategy=strategy, decay_factor=0.95)
            else:
                ensemble = ModelEnsemble(strategy=strategy)
            
            # Fit ensemble
            ensemble.fit(
                oof_predictions=oof_preds_filtered,
                y_true=y_true_oof,
                date_ids=date_ids_oof if strategy == 'time_weighted' else None
            )
            
            # Evaluate
            metrics = ensemble.evaluate(
                predictions=oof_preds_filtered,
                y_true=y_true_oof,
                date_ids=date_ids_oof if strategy == 'time_weighted' else None
            )
            
            results_comparison[strategy] = metrics
            logger.info(f"  RMSE: {metrics['rmse']:.6f}")
            logger.info(f"  Correlation: {metrics['correlation']:.4f}")
            
            # Show weights
            weights_df = ensemble.get_weights_df()
            logger.info(f"\n{weights_df.to_string(index=False)}")
            
        except Exception as e:
            logger.error(f"Failed to test {strategy}: {e}")
            continue
    
    # Step 6: Optimize for Sharpe ratio
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: OPTIMIZE FOR SHARPE RATIO")
    logger.info("=" * 80)
    
    # Get market returns for Sharpe calculation
    market_returns = y_true_oof  # forward_returns is the market return
    
    # Optimize weights
    optimization_result = optimize_ensemble_weights(
        model_results=model_results,
        y_true=y_true_oof,
        date_ids=date_ids_oof,
        market_returns=market_returns
    )
    
    # Step 7: Save results
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: SAVE RESULTS")
    logger.info("=" * 80)
    
    output_dir = project_root / "artifacts"
    output_dir.mkdir(exist_ok=True)
    
    # Save best ensemble configuration
    ensemble_config = {
        'strategy': 'weighted_average',
        'weights': optimization_result['weights'],
        'model_names': optimization_result['model_names'],
        'best_sharpe': optimization_result['best_sharpe'],
        'individual_model_scores': {
            name: result['oof_score']
            for name, result in model_results.items()
        }
    }
    
    config_path = output_dir / "ensemble_config.json"
    with open(config_path, 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    logger.info(f"Saved ensemble config to {config_path}")
    
    # Save comparison results
    comparison_df = pd.DataFrame(results_comparison).T
    comparison_path = output_dir / "ensemble_comparison.csv"
    comparison_df.to_csv(comparison_path)
    logger.info(f"Saved comparison to {comparison_path}")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("ENSEMBLE OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nBest ensemble Sharpe: {optimization_result['best_sharpe']:.6f}")
    logger.info("\nModel weights:")
    for name, weight in zip(optimization_result['model_names'], optimization_result['weights']):
        logger.info(f"  {name}: {weight:.4f}")
    
    logger.info(f"\nIndividual model RMSEs:")
    for name, score in ensemble_config['individual_model_scores'].items():
        logger.info(f"  {name}: {score:.6f}")


if __name__ == "__main__":
    main()
