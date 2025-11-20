"""
Risk Model Hyperparameter Tuning Script

This script tunes the risk prediction model (sigma_hat) using Optuna.
Optimizes LightGBM hyperparameters for volatility forecasting.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.data import DataLoader
from src.features_risk import FeatureRiskEngineering
from src.risk import RiskLabeler
from src.tuner import OptunaLightGBMTuner
from src.utils import get_logger, load_config, Timer
import warnings
warnings.filterwarnings('ignore')

# Logger
logger = get_logger(log_file="logs/risk_tuning.log", level="INFO")


def tune_risk_model(
    config_path: str = "conf/params.yaml",
    n_samples: int = None,
    n_trials: int = 20,
    timeout: int = 1800  # 30 minutes
):
    """
    Tune risk prediction model hyperparameters.
    
    Parameters
    ----------
    config_path : str
        Path to config file
    n_samples : int, optional
        Number of samples to use (for testing)
    n_trials : int
        Number of Optuna trials
    timeout : int
        Time limit in seconds
    """
    logger.info("="*80)
    logger.info("Risk Model Hyperparameter Tuning")
    logger.info("="*80)
    
    # Load config
    config = load_config(config_path)
    
    # ========================================================================
    # Step 1: Load Data
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 1: Loading Data")
    
    with Timer("Loading data"):
        loader = DataLoader(config_path)
        df_train, df_test = loader.load_data()
    
    if n_samples:
        logger.info(f"Using {n_samples} samples for testing")
        df_train = df_train.head(n_samples)
    
    logger.info(f"Data shape: {df_train.shape}")
    logger.info(f"Date range: {df_train['date_id'].min()} to {df_train['date_id'].max()}")
    
    # ========================================================================
    # Step 2: Feature Engineering
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 2: Feature Engineering")
    
    with Timer("Feature engineering"):
        engineer = FeatureRiskEngineering(config_path)
        df_engineered = engineer.fit_transform(df_train)
    
    logger.info(f"Features created: {df_engineered.shape[1]}")
    
    # ========================================================================
    # Step 3: Feature Selection
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 3: Feature Selection")
    
    with Timer("Feature selection"):
        df_selected, feature_cols = engineer.select_features_by_importance(
            df_engineered,
            target_col='forward_returns',  # Forward returns created by FeatureRiskEngineering
            method='correlation',
            top_n=100
        )
        
        # Remove correlated features
        df_selected, removed_features = engineer.remove_correlated_features(df_selected, threshold=0.95)
        
        # Update feature list
        feature_cols = [col for col in df_selected.columns 
                       if col not in ['date_id', 'forward_returns', 'risk_free_rate',
                                     'market_forward_excess_returns', 'risk_label']]
    
    logger.info(f"Selected features: {len(feature_cols)}")
    
    logger.info("\nDataset:")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Samples: {len(df_selected)}")
    
    # ========================================================================
    # Step 4: Create Risk Labels
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 4: Creating Risk Labels")
    
    with Timer("Risk label creation"):
        risk_config = config.get('risk', {}).get('label', {})
        labeler = RiskLabeler(
            window=risk_config.get('window', 20),
            min_periods=risk_config.get('min_periods', 10),
            clip_threshold=risk_config.get('clip_threshold', 4.0)
        )
        
        df_selected = labeler.fit_transform(
            df_selected,
            target_col='forward_returns'  # Forward returns column name
        )
    
    # Check risk labels
    valid_labels = df_selected['risk_label'].notna().sum()
    logger.info(f"Risk labels created: {valid_labels} valid samples")
    logger.info(f"\nRisk Label Statistics:")
    logger.info(f"  Count: {valid_labels}")
    logger.info(f"  Mean: {df_selected['risk_label'].mean():.6f}")
    logger.info(f"  Std: {df_selected['risk_label'].std():.6f}")
    logger.info(f"  Min: {df_selected['risk_label'].min():.6f}")
    logger.info(f"  Max: {df_selected['risk_label'].max():.6f}")
    
    # ========================================================================
    # Step 5: Hyperparameter Tuning
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 5: Hyperparameter Tuning with Optuna")
    
    with Timer("Hyperparameter tuning"):
        # Initialize tuner with 'risk' config section
        tuner = OptunaLightGBMTuner(
            config_path=config_path,
            config_section='risk',  # Use risk config section
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=1,  # Use 1 job for stability
            random_state=42
        )
        
        # Optimize
        best_params = tuner.optimize(
            df=df_selected,
            feature_cols=feature_cols,
            target_col='risk_label',  # Target is risk_label
            study_name='risk_lgbm_tuning'
        )
    
    # ========================================================================
    # Step 6: Save Results
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 6: Saving Results")
    
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    
    # Save best parameters
    import json
    params_file = output_dir / "lightgbm_best_params_risk.json"
    with open(params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"✓ Best parameters saved to {params_file}")
    
    # Save optimization history
    if tuner.study:
        history_df = tuner.study.trials_dataframe()
        history_file = output_dir / "risk_optimization_history.csv"
        history_df.to_csv(history_file, index=False)
        logger.info(f"✓ Optimization history saved to {history_file}")
        
        # Plot optimization history (if possible)
        try:
            import optuna.visualization as vis
            import plotly
            
            # Optimization history
            fig = vis.plot_optimization_history(tuner.study)
            fig.write_html(output_dir / "risk_optimization_history.html")
            logger.info(f"✓ Optimization plot saved to risk_optimization_history.html")
            
            # Parameter importances
            fig = vis.plot_param_importances(tuner.study)
            fig.write_html(output_dir / "risk_param_importances.html")
            logger.info(f"✓ Parameter importance plot saved")
        except Exception as e:
            logger.warning(f"Could not create plots: {e}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("Risk Model Tuning Summary")
    logger.info("="*80)
    
    logger.info("\n✅ Completed Tasks:")
    logger.info("  1. Data Loading and Preparation")
    logger.info("  2. Feature Engineering")
    logger.info("  3. Feature Selection")
    logger.info("  4. Risk Label Creation")
    logger.info("  5. Hyperparameter Tuning")
    logger.info("  6. Results Saved")
    
    logger.info("\n📁 Results saved to:")
    logger.info(f"  - {params_file}")
    logger.info(f"  - {history_file}")
    
    logger.info("\n📊 Best Results:")
    logger.info(f"  Best RMSE: {tuner.best_score:.6f}")
    logger.info(f"  Best Trial: {tuner.study.best_trial.number}")
    logger.info(f"  Total Trials: {len(tuner.study.trials)}")
    
    logger.info("\n🎯 Next Steps:")
    logger.info("  1. Review best parameters")
    logger.info("  2. Update params.yaml with best params")
    logger.info("  3. Retrain risk model with optimized params")
    logger.info("  4. Generate final OOF predictions")
    
    logger.info("\n" + "="*80)
    logger.info("Risk Model Tuning Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tune risk prediction model")
    parser.add_argument(
        "--config",
        type=str,
        default="conf/params.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples to use (for testing)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Time limit in seconds"
    )
    
    args = parser.parse_args()
    
    tune_risk_model(
        config_path=args.config,
        n_samples=args.n_samples,
        n_trials=args.n_trials,
        timeout=args.timeout
    )
