"""
Hyperparameter tuning module using Optuna.

This module provides:
- LightGBM hyperparameter optimization
- CatBoost hyperparameter optimization
- Time series cross-validation
- Early stopping and pruning
- Best parameters tracking
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, Optional, Callable, Any, List
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import lightgbm as lgb
import pickle
import json

from src.cv import CVStrategy, create_cv_strategy
from src.utils import get_logger, load_config, Timer

logger = get_logger(log_file="logs/tuner.log", level="INFO")


class OptunaLightGBMTuner:
    """
    Hyperparameter tuning for LightGBM using Optuna.
    
    Features:
    - Bayesian optimization with TPE sampler
    - Time series cross-validation
    - Early stopping and pruning
    - Parameter space customization
    """
    
    def __init__(
        self,
        config_path: str = "conf/params.yaml",
        config_section: str = "tuning",
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        random_state: int = 42
    ):
        """
        Initialize Optuna tuner.
        
        Parameters
        ----------
        config_path : str
            Path to configuration file
        config_section : str
            Config section to use ('tuning' for return model, 'risk' for risk model)
        n_trials : int
            Number of optimization trials
        timeout : int, optional
            Time limit in seconds
        n_jobs : int
            Number of parallel jobs
        random_state : int
            Random seed
        """
        self.config = load_config(config_path)
        self.config_section = config_section
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # CV strategy
        self.cv_strategy = create_cv_strategy(config_path)
        
        # Tuning configuration - support both 'tuning' and 'risk' sections
        if config_section == 'risk':
            tuning_config = self.config.get('risk', {}).get('lightgbm', {})
        else:
            tuning_config = self.config.get('tuning', {}).get('lightgbm', {})
        
        self.param_space = tuning_config.get('param_space', {})
        self.fixed_params = tuning_config.get('fixed_params', {})
        
        # Study
        self.study = None
        self.best_params = None
        self.best_score = None
        
        logger.info(f"OptunaLightGBMTuner initialized (config_section={config_section})")
        logger.info(f"Trials: {n_trials}, Timeout: {timeout}, Jobs: {n_jobs}")
    
    def _get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define hyperparameter search space.
        
        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object
            
        Returns
        -------
        dict
            Suggested parameters
        """
        params = {}
        
        # Get param space from config or use defaults
        space = self.param_space if self.param_space else {
            'num_leaves': {'type': 'int', 'low': 20, 'high': 100},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'feature_fraction': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'bagging_fraction': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'bagging_freq': {'type': 'int', 'low': 1, 'high': 10},
            'min_child_samples': {'type': 'int', 'low': 10, 'high': 100},
            'max_depth': {'type': 'int', 'low': 3, 'high': 12},
            'reg_alpha': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
            'reg_lambda': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'log': True},
        }
        
        # Suggest parameters based on type
        for param_name, param_config in space.items():
            param_type = param_config.get('type', 'float')
            
            if param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high']
                )
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
        
        # Add fixed parameters
        params.update(self.fixed_params)
        
        # Add default parameters
        default_params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'random_state': self.random_state,
            'n_estimators': 1000,
            'early_stopping_rounds': 50
        }
        
        for key, value in default_params.items():
            if key not in params:
                params[key] = value
        
        return params
    
    def _objective(
        self,
        trial: optuna.Trial,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str
    ) -> float:
        """
        Objective function for Optuna.
        
        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial
        df : pd.DataFrame
            Full dataframe with features and target
        feature_cols : list
            List of feature column names
        target_col : str
            Target column name (e.g., 'forward_returns' or 'risk_label')
            
        Returns
        -------
        float
            Validation score (lower is better)
        """
        # Get parameters
        params = self._get_param_space(trial)
        
        # CV scores
        fold_scores = []
        
        # Get CV folds
        cv_splits = list(self.cv_strategy.get_folds(df))
        
        # Train each fold
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_train = df.iloc[train_idx][feature_cols]
            y_train = df.iloc[train_idx][target_col]
            X_val = df.iloc[val_idx][feature_cols]
            y_val = df.iloc[val_idx][target_col]
            
            # Drop NaN values
            train_mask = ~(y_train.isna())
            val_mask = ~(y_val.isna())
            
            X_train = X_train[train_mask]
            y_train = y_train[train_mask]
            X_val = X_val[val_mask]
            y_val = y_val[val_mask]
            
            if len(X_train) == 0 or len(X_val) == 0:
                continue
            
            # Train model
            model = lgb.LGBMRegressor(**params)
            
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=params.get('early_stopping_rounds', 50)),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            # Validate
            y_pred = model.predict(X_val)
            score = np.sqrt(np.mean((y_val - y_pred) ** 2))
            fold_scores.append(score)
            
            # Report intermediate value for pruning
            trial.report(score, fold_idx)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Return mean CV score
        mean_score = np.mean(fold_scores)
        return mean_score
    
    def optimize(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        load_if_exists: bool = False
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Parameters
        ----------
        df : pd.DataFrame
            Full dataframe with features and target
        feature_cols : list
            List of feature column names
        target_col : str
            Target column name (e.g., 'forward_returns' or 'risk_label')
        study_name : str, optional
            Name for the study
        storage : str, optional
            Database URL for study storage
        load_if_exists : bool
            Load existing study if available
            
        Returns
        -------
        dict
            Best parameters found
        """
        logger.info("="*80)
        logger.info("Starting Hyperparameter Optimization")
        logger.info("="*80)
        logger.info(f"\nFeatures: {len(feature_cols)}")
        logger.info(f"Samples: {len(df)}")
        logger.info(f"Target: {target_col}")
        logger.info(f"\nTrials: {self.n_trials}")
        logger.info(f"Timeout: {self.timeout}")
        logger.info(f"Jobs: {self.n_jobs}")
        
        with Timer("Hyperparameter Optimization"):
            # Create study
            sampler = TPESampler(seed=self.random_state)
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
            
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                load_if_exists=load_if_exists,
                direction='minimize',
                sampler=sampler,
                pruner=pruner
            )
            
            # Optimize
            self.study.optimize(
                lambda trial: self._objective(trial, df, feature_cols, target_col),
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.n_jobs,
                show_progress_bar=True
            )
            
            # Store best results
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value
            
            logger.info("\n" + "="*80)
            logger.info("Optimization Results")
            logger.info("="*80)
            logger.info(f"\nBest Score (RMSE): {self.best_score:.6f}")
            logger.info(f"Best Trial: {self.study.best_trial.number}")
            logger.info(f"\nBest Parameters:")
            for param, value in self.best_params.items():
                logger.info(f"  {param}: {value}")
            
            return self.best_params
    
    def get_best_params(self, include_fixed: bool = True) -> Dict[str, Any]:
        """
        Get best parameters with fixed params.
        
        Parameters
        ----------
        include_fixed : bool
            Include fixed parameters
            
        Returns
        -------
        dict
            Best parameters
        """
        if self.best_params is None:
            raise ValueError("No optimization has been run yet")
        
        params = self.best_params.copy()
        
        if include_fixed:
            # Add fixed parameters
            params.update(self.fixed_params)
            
            # Add default parameters
            default_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'random_state': self.random_state,
                'n_estimators': 1000,
                'early_stopping_rounds': 50
            }
            
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
        
        return params
    
    def save_study(
        self,
        study_path: str = "artifacts/tuning/lightgbm_study.pkl",
        params_path: str = "artifacts/tuning/lightgbm_best_params.json"
    ):
        """
        Save study and best parameters.
        
        Parameters
        ----------
        study_path : str
            Path to save study
        params_path : str
            Path to save parameters
        """
        # Create directory
        Path(study_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save study
        with open(study_path, 'wb') as f:
            pickle.dump(self.study, f)
        logger.info(f"Study saved to {study_path}")
        
        # Save best parameters
        best_params = self.get_best_params(include_fixed=True)
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        logger.info(f"Best parameters saved to {params_path}")
        
        # Save optimization history
        history_path = str(Path(params_path).parent / "optimization_history.csv")
        trials_df = self.study.trials_dataframe()
        trials_df.to_csv(history_path, index=False)
        logger.info(f"Optimization history saved to {history_path}")
    
    def load_study(
        self,
        study_path: str = "artifacts/tuning/lightgbm_study.pkl"
    ):
        """
        Load saved study.
        
        Parameters
        ----------
        study_path : str
            Path to study file
        """
        with open(study_path, 'rb') as f:
            self.study = pickle.load(f)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"Study loaded from {study_path}")
        logger.info(f"Best Score: {self.best_score:.6f}")
    
    def plot_optimization_history(
        self,
        save_path: str = "results/tuning/optimization_history.png"
    ):
        """
        Plot optimization history.
        
        Parameters
        ----------
        save_path : str
            Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization import plot_optimization_history
            
            fig = plot_optimization_history(self.study)
            
            # Create directory
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save plot
            fig.write_image(save_path)
            logger.info(f"Optimization history plot saved to {save_path}")
        except ImportError:
            logger.warning("plotly or kaleido not installed. Skipping plot.")
    
    def plot_param_importances(
        self,
        save_path: str = "results/tuning/param_importances.png"
    ):
        """
        Plot parameter importances.
        
        Parameters
        ----------
        save_path : str
            Path to save plot
        """
        try:
            from optuna.visualization import plot_param_importances
            
            fig = plot_param_importances(self.study)
            
            # Create directory
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save plot
            fig.write_image(save_path)
            logger.info(f"Parameter importances plot saved to {save_path}")
        except ImportError:
            logger.warning("plotly or kaleido not installed. Skipping plot.")


if __name__ == "__main__":
    # Example usage
    from src.data import load_train_data
    from src.features import FeatureEngineering
    
    logger.info("Testing OptunaLightGBMTuner")
    
    # Load data
    df = load_train_data()
    df = df.head(3000)  # Use subset for testing
    
    # Create features
    fe = FeatureEngineering()
    df_features = fe.create_all_features(df)
    
    # Select features
    df_features = fe.remove_correlated_features(df_features, threshold=0.95)
    
    # Prepare data
    feature_cols = [
        col for col in df_features.columns 
        if col not in ['date_id', 'forward_returns', 'risk_free_rate', 
                      'market_forward_excess_returns']
    ]
    
    X = df_features[feature_cols]
    y = df_features['forward_returns']
    
    # Create tuner
    tuner = OptunaLightGBMTuner(
        n_trials=20,  # Small number for testing
        n_jobs=1,
        random_state=42
    )
    
    # Optimize
    best_params = tuner.optimize(X, y, study_name="lightgbm_test")
    
    # Save results
    tuner.save_study()
    
    logger.info("Tuning test complete")
