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
from src.metric import evaluate_return_model
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
    objective_type: str = 'combined'
) -> Dict:
    """
    Find optimal ensemble weights that maximize return model metrics.
    
    Uses Optuna to search for best weights based on IC and Spread.
    
    Parameters
    ----------
    model_results : Dict[str, Dict]
        Dictionary of model results with OOF predictions
    y_true : np.ndarray
        True target values
    objective_type : str
        Optimization objective:
        - 'ic': Maximize Information Coefficient
        - 'spread': Maximize Long-Short Spread
        - 'combined': Maximize IC + (Spread × 100) [RECOMMENDED]
    
    Returns
    -------
    Dict
        Optimization results with weights and metrics
    """
    import optuna
    
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZING ENSEMBLE WEIGHTS FOR RETURN MODEL")
    logger.info("=" * 80)
    logger.info(f"Objective: {objective_type}")
    
    # Extract OOF predictions
    oof_preds = {
        name: result['oof_predictions']
        for name, result in model_results.items()
    }
    
    def objective(trial):
        """Objective function: maximize IC + Spread."""
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
        
        # Evaluate return model metrics
        metrics = evaluate_return_model(ensemble_pred, y_true, return_all_metrics=True)
        
        ic = metrics['information_coefficient']
        spread = metrics['long_short_spread']
        
        # Store metrics for analysis
        trial.set_user_attr('ic', ic)
        trial.set_user_attr('spread', spread)
        trial.set_user_attr('correlation', metrics['correlation'])
        trial.set_user_attr('directional_accuracy', metrics['directional_accuracy'])
        
        # Calculate objective based on type
        if objective_type == 'ic':
            return ic
        elif objective_type == 'spread':
            return spread
        else:  # combined
            # IC weight: 1.0, Spread weight: 100.0 (scale 0.001 → 0.1)
            combined_score = ic + (spread * 100)
            trial.set_user_attr('combined_score', combined_score)
            return combined_score
    
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
    
    # Get best metrics
    best_ic = best_trial.user_attrs.get('ic', 0.0)
    best_spread = best_trial.user_attrs.get('spread', 0.0)
    best_corr = best_trial.user_attrs.get('correlation', 0.0)
    best_dir_acc = best_trial.user_attrs.get('directional_accuracy', 0.0)
    
    logger.info(f"\n{'='*80}")
    logger.info("OPTIMIZATION RESULTS")
    logger.info(f"{'='*80}")
    
    if objective_type == 'combined':
        logger.info(f"\n🎯 Best Combined Score: {best_trial.value:.6f}")
        logger.info(f"   → IC: {best_ic:.6f}")
        logger.info(f"   → Spread: {best_spread:.6f} ({best_spread*100:.4f}%)")
    elif objective_type == 'ic':
        logger.info(f"\n🎯 Best IC: {best_trial.value:.6f}")
    else:  # spread
        logger.info(f"\n🎯 Best Spread: {best_trial.value:.6f} ({best_trial.value*100:.4f}%)")
    
    logger.info(f"\n📊 Additional Metrics:")
    logger.info(f"   Correlation: {best_corr:.4f}")
    logger.info(f"   Directional Accuracy: {best_dir_acc:.2%}")
    
    logger.info(f"\n⚖️  Best Ensemble Weights:")
    for name, weight in zip(oof_preds.keys(), best_weights):
        logger.info(f"   {name:15s}: {weight:.4f}")
    
    return {
        'weights': best_weights,
        'model_names': list(oof_preds.keys()),
        'best_score': best_trial.value,
        'best_ic': best_ic,
        'best_spread': best_spread,
        'best_correlation': best_corr,
        'best_directional_accuracy': best_dir_acc,
        'objective_type': objective_type,
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
    
    # Step 6: Optimize for IC and Spread
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: OPTIMIZE ENSEMBLE WEIGHTS")
    logger.info("=" * 80)
    logger.info("\nOptimizing for Return Model Performance (IC + Spread)")
    
    # Optimize weights
    optimization_result = optimize_ensemble_weights(
        model_results=model_results,
        y_true=y_true_oof,
        objective_type='combined'  # 'ic', 'spread', or 'combined'
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
        'objective_type': optimization_result['objective_type'],
        'metrics': {
            'best_score': optimization_result['best_score'],
            'ic': optimization_result['best_ic'],
            'spread': optimization_result['best_spread'],
            'correlation': optimization_result['best_correlation'],
            'directional_accuracy': optimization_result['best_directional_accuracy']
        },
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
    
    logger.info(f"\n🎯 Optimization Objective: {optimization_result['objective_type']}")
    
    if optimization_result['objective_type'] == 'combined':
        logger.info(f"\n📊 Best Combined Score: {optimization_result['best_score']:.6f}")
        logger.info(f"   → IC: {optimization_result['best_ic']:.6f}")
        logger.info(f"   → Spread: {optimization_result['best_spread']:.6f} ({optimization_result['best_spread']*100:.4f}%)")
    elif optimization_result['objective_type'] == 'ic':
        logger.info(f"\n📊 Best IC: {optimization_result['best_score']:.6f}")
    else:  # spread
        logger.info(f"\n📊 Best Spread: {optimization_result['best_score']:.6f} ({optimization_result['best_score']*100:.4f}%)")
    
    logger.info(f"\n⚖️  Model Weights:")
    for name, weight in zip(optimization_result['model_names'], optimization_result['weights']):
        logger.info(f"   {name:15s}: {weight:.4f}")
    
    logger.info(f"\n🔍 Individual Model RMSEs:")
    for name, score in ensemble_config['individual_model_scores'].items():
        logger.info(f"   {name:15s}: {score:.6f}")


if __name__ == "__main__":
    main()
