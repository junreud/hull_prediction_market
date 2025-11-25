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
from src.metric import evaluate_return_model, evaluate_risk_model

logger = get_logger(log_file="logs/tuner.log", level="INFO")

# Avoid circular import by lazy importing ReturnPredictor
def _get_return_predictor():
    """Lazy import to avoid circular dependency."""
    from src.models import ReturnPredictor
    return ReturnPredictor


class OptunaLightGBMTuner:
    """
    Hyperparameter tuning for LightGBM using Optuna.
    
    Features:
    - Bayesian optimization with TPE sampler
    - Time series cross-validation
    - Early stopping and pruning
    - Parameter space customization
    - Custom objective functions (RMSE, IC, Spread, or Combined)
    """
    
    def __init__(
        self,
        config_path: str = "conf/params.yaml",
        config_section: str = "tuning",
        n_trials: int = 30,
        timeout: Optional[int] = None,
        n_jobs: int = 5,
        random_state: int = 42,
        objective_type: str = 'return_ic',  # 'return_ic',
        model_type: str = 'lightgbm'  # 'lightgbm' or 'catboost'
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
        objective_type : str
            Type of objective to optimize:
            
            **Return Model Metrics** (from evaluate_return_model):
            - 'return_ic': Maximize Information Coefficient (RECOMMENDED)
            
            **Risk Model Metrics** (from evaluate_risk_model):
            - 'risk_correlation': Maximize correlation with realized volatility (RECOMMENDED)
        """
        self.config = load_config(config_path)
        self.config_section = config_section
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.objective_type = objective_type
        self.model_type = model_type
        
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
        logger.info(f"Model type: {model_type}")
        logger.info(f"Objective type: {objective_type}")
        logger.info(f"Trials: {n_trials}, Timeout: {timeout}, Jobs: {n_jobs}")
        
        # Store config path for ReturnPredictor
        self.config['_config_path'] = config_path
    
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
        Objective function for Optuna using ReturnPredictor.
        
        This ensures the tuning process uses the EXACT same model
        as the final training step.
        
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
            Validation score (lower is better for 'rmse', higher is better for others)
        """
        # Get parameters from trial
        params = self._get_param_space(trial)
        
        # Create ReturnPredictor with trial parameters
        ReturnPredictor = _get_return_predictor()
        predictor = ReturnPredictor(
            model_type=self.model_type,
            config_path=self.config.get('_config_path', 'conf/params.yaml')
        )
        
        # Update predictor params with trial params
        predictor.params.update(params)
        
        # Prepare dataframe with only needed columns
        df_subset = df[feature_cols + [target_col, 'date_id']].copy()
        
        # Collect all predictions for metric calculation
        all_y_val = []
        all_y_pred = []
        
        # Get CV folds
        cv_splits = list(self.cv_strategy.get_folds(df_subset))
        
        # Train each fold using ReturnPredictor.train_fold
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_train = df_subset.iloc[train_idx][feature_cols]
            y_train = df_subset.iloc[train_idx][target_col]
            X_val = df_subset.iloc[val_idx][feature_cols]
            y_val = df_subset.iloc[val_idx][target_col]
            
            # Drop NaN values
            train_mask = ~(y_train.isna())
            val_mask = ~(y_val.isna())
            
            X_train = X_train[train_mask]
            y_train = y_train[train_mask]
            X_val = X_val[val_mask]
            y_val = y_val[val_mask]
            
            if len(X_train) == 0 or len(X_val) == 0:
                continue
            
            # Train fold using ReturnPredictor (same as final training)
            model, val_score, y_pred = predictor.train_fold(
                X_train, y_train, X_val, y_val, fold_idx
            )
            
            # Collect predictions for metric calculation
            all_y_val.extend(y_val.values)
            all_y_pred.extend(y_pred)
            
            # Report intermediate value for pruning (use RMSE for pruning)
            rmse = np.sqrt(np.mean((y_val.values - y_pred) ** 2))
            trial.report(rmse, fold_idx)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Calculate final score based on objective type
        # Convert to numpy arrays
        all_y_val = np.array(all_y_val)
        all_y_pred = np.array(all_y_pred)
        
        # Calculate return model metrics
        if self.objective_type.startswith('return_'):
            metrics = evaluate_return_model(all_y_pred, all_y_val, return_all_metrics=True)
            
            if self.objective_type == 'return_ic':
                # Maximize IC (negate for minimization)
                ic = metrics['information_coefficient']
                trial.set_user_attr('ic', ic)
                return -ic  # Optuna minimizes, so negate
            else:
                raise ValueError(f"Unknown return objective_type: {self.objective_type}")
        
        # Calculate risk model metrics
        elif self.objective_type.startswith('risk_'):
            metrics = evaluate_risk_model(all_y_pred, all_y_val, return_all_metrics=True)
            
            if self.objective_type == 'risk_correlation':
                # Maximize correlation with realized volatility (negate for minimization)
                corr = metrics['correlation']
                trial.set_user_attr('correlation', corr)
                return -corr
            else:
                raise ValueError(f"Unknown risk objective_type: {self.objective_type}")
        
        else:
            raise ValueError(f"Unknown objective_type: {self.objective_type}")
    
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
                direction='minimize',  # Always minimize (IC/Spread are negated in _objective)
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
            
            # Display best score based on objective type
            metric_name = self.objective_type.replace('return_', '').replace('risk_', '').upper()
            logger.info(f"\nBest Score ({metric_name}): {-self.best_score:.6f}")  # Negate back
            
            # Show additional metrics if available
            if hasattr(self.study.best_trial, 'user_attrs') and self.study.best_trial.user_attrs:
                logger.info(f"\nBest Trial Metrics:")
                for attr_name, attr_value in self.study.best_trial.user_attrs.items():
                    logger.info(f"  {attr_name}: {attr_value:.6f}")
            
            logger.info(f"\nBest Trial: {self.study.best_trial.number}")
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
            
            # Add default parameters based on model type
            if self.model_type == 'lightgbm':
                default_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': 'rmse',
                    'verbosity': -1,
                    'random_state': self.random_state,
                    'n_estimators': 1000,
                    'early_stopping_rounds': 50
                }
            elif self.model_type == 'catboost':
                default_params = {
                    'loss_function': 'RMSE',
                    'verbose': False,
                    'random_seed': self.random_state,
                    'iterations': 1000,
                    'early_stopping_rounds': 50,
                    'allow_writing_files': False
                }
            else:
                default_params = {}
            
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
        
        return params