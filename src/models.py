"""
Model training module for Hull Tactical Market Prediction.

This module provides:
- LightGBM and CatBoost model training
- Cross-validation with time series splits
- OOF (Out-of-Fold) predictions
- Feature importance analysis
- Model persistence
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle

from src.cv import CVStrategy, create_cv_strategy
from src.metric import CompetitionMetric
from src.utils import get_logger, load_config, Timer

logger = get_logger(log_file="logs/models.log", level="INFO")


def convert_lgbm_to_catboost_params(lgbm_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert LightGBM parameters to CatBoost parameters.
    
    Parameters
    ----------
    lgbm_params : dict
        LightGBM parameters
        
    Returns
    -------
    dict
        CatBoost parameters
    """
    catboost_params = {}
    
    # Parameter mapping: LightGBM -> CatBoost
    param_mapping = {
        'num_leaves': 'max_leaves',
        'learning_rate': 'learning_rate',
        'n_estimators': 'iterations',
        'max_depth': 'depth',
        'min_child_samples': 'min_data_in_leaf',
        'feature_fraction': 'rsm',  # Random subspace method
        'bagging_fraction': 'subsample',
        'bagging_freq': None,  # CatBoost doesn't have exact equivalent
        'reg_alpha': 'l2_leaf_reg',  # CatBoost uses L2 primarily
        'reg_lambda': 'l2_leaf_reg',
        'random_state': 'random_seed',
        'verbosity': 'verbose',
        'early_stopping_rounds': 'early_stopping_rounds',
    }
    
    # Convert parameters
    for lgbm_key, lgbm_value in lgbm_params.items():
        if lgbm_key in param_mapping:
            catboost_key = param_mapping[lgbm_key]
            if catboost_key is not None:
                # Special handling for specific parameters
                if lgbm_key == 'reg_alpha' or lgbm_key == 'reg_lambda':
                    # Combine both into l2_leaf_reg
                    if 'l2_leaf_reg' not in catboost_params:
                        catboost_params['l2_leaf_reg'] = lgbm_value
                    else:
                        catboost_params['l2_leaf_reg'] = max(catboost_params['l2_leaf_reg'], lgbm_value)
                elif lgbm_key == 'verbosity':
                    # Convert LightGBM verbosity (-1, 0, 1) to CatBoost (False/True)
                    catboost_params['verbose'] = False if lgbm_value <= 0 else True
                else:
                    catboost_params[catboost_key] = lgbm_value
        else:
            # Keep unknown parameters as is (might be CatBoost-specific)
            if lgbm_key not in ['boosting_type', 'objective', 'metric']:
                catboost_params[lgbm_key] = lgbm_value
    
    # Set CatBoost defaults
    if 'loss_function' not in catboost_params:
        catboost_params['loss_function'] = 'RMSE'
    if 'verbose' not in catboost_params:
        catboost_params['verbose'] = False
    if 'allow_writing_files' not in catboost_params:
        catboost_params['allow_writing_files'] = False
    
    return catboost_params


class ReturnPredictor:
    """
    Return prediction model using LightGBM or CatBoost.
    
    Features:
    - Time series cross-validation
    - OOF predictions
    - Feature importance tracking
    - Model ensembling
    """
    
    def __init__(
        self,
        model_type: str = 'lightgbm',
        config_path: str = "conf/params.yaml"
    ):
        """
        Initialize return predictor.
        
        Parameters
        ----------
        model_type : str
            Model type: 'lightgbm' or 'catboost'
        config_path : str
            Path to configuration file
        """
        self.model_type = model_type
        self.config = load_config(config_path)
        
        # Get model configuration
        model_config_key = f'model_return'
        self.model_config = self.config.get(model_config_key, {})
        
        if model_type == 'lightgbm':
            self.params = self.model_config.get('lightgbm', {})
        elif model_type == 'catboost':
            self.params = self.model_config.get('catboost', {})
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Models storage (one per fold)
        self.models = []
        
        # Feature importance
        self.feature_importance = None
        
        # OOF predictions
        self.oof_predictions = None
        self.oof_indices = None
        
        # CV strategy
        self.cv_strategy = create_cv_strategy(config_path)
        
        # Metric calculator
        metric_config = self.config.get('metric', {})
        self.metric_calc = CompetitionMetric(
            vol_threshold=metric_config.get('vol_threshold', 1.2),
            min_periods=metric_config.get('min_periods', 30)
        )
        
        logger.info(f"ReturnPredictor initialized with {model_type}")
        logger.info(f"Model params: {self.params}")
    
    def train_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        fold_idx: int
    ) -> Tuple[Any, float, np.ndarray]:
        """
        Train model on single fold.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation target
        fold_idx : int
            Fold index
            
        Returns
        -------
        Tuple[model, val_score, val_predictions]
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_idx + 1}")
        logger.info(f"{'='*60}")
        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Val samples: {len(X_val)}")
        
        if self.model_type == 'lightgbm':
            model, val_score, val_preds = self._train_lgbm(
                X_train, y_train, X_val, y_val, fold_idx
            )
        elif self.model_type == 'catboost':
            model, val_score, val_preds = self._train_catboost(
                X_train, y_train, X_val, y_val, fold_idx
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Fold {fold_idx + 1} validation score: {val_score:.6f}")
        
        return model, val_score, val_preds
    
    def _train_lgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        fold_idx: int
    ) -> Tuple[lgb.Booster, float, np.ndarray]:
        """Train LightGBM model."""
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Callbacks
        callbacks = [
            lgb.log_evaluation(period=100), # 100 번의 iteration마다 로그 출력
            lgb.early_stopping(
                stopping_rounds=self.params.get('early_stopping_rounds', 50), # 50번 학습동안 개선 없으면 중단
                verbose=True
            )
        ]
        
        # Train
        with Timer(f"Training LGBM Fold {fold_idx + 1}", logger=logger):
            model = lgb.train(
                self.params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=callbacks
            )
        
        # Predict
        val_preds = model.predict(X_val)
        
        # Calculate score (RMSE)
        val_score = np.sqrt(np.mean((y_val - val_preds) ** 2))
        
        return model, val_score, val_preds
    
    def _train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        fold_idx: int
    ) -> Tuple[Any, float, np.ndarray]:
        """Train CatBoost model."""
        try:
            from catboost import CatBoostRegressor, Pool
        except ImportError:
            raise ImportError("CatBoost not installed. Install with: pip install catboost")
        
        # Convert LightGBM params to CatBoost params if needed
        catboost_params = convert_lgbm_to_catboost_params(self.params)
        
        # Extract early_stopping_rounds separately (used in fit, not init)
        early_stopping_rounds = catboost_params.pop('early_stopping_rounds', 50)
        
        # Create model
        model = CatBoostRegressor(**catboost_params)
        
        # Create pools
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)
        
        # Train
        with Timer(f"Training CatBoost Fold {fold_idx + 1}", logger=logger):
            model.fit(
                train_pool,
                eval_set=val_pool,
                use_best_model=True,
                early_stopping_rounds=early_stopping_rounds,
                verbose=100
            )
        
        # Predict
        val_preds = model.predict(X_val)
        
        # Calculate score (RMSE)
        val_score = np.sqrt(np.mean((y_val - val_preds) ** 2))
        
        return model, val_score, val_preds
    
    def train_cv(
        self,
        df: pd.DataFrame,
        target_col: str = 'forward_returns',
        date_col: str = 'date_id'
    ) -> Dict[str, Any]:
        """
        Train model with cross-validation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with features and target
        target_col : str
            Target column name
        date_col : str
            Date column name
            
        Returns
        -------
        dict
            Training results including OOF predictions and scores
        """
        logger.info("="*80)
        logger.info("Starting Cross-Validation Training")
        logger.info("="*80)
        
        # Prepare data
        feature_cols = [
            col for col in df.columns 
            if col not in [date_col, target_col, 'risk_free_rate', 
                          'market_forward_excess_returns']
        ]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        dates = df[date_col].values
        
        logger.info(f"\nFeatures: {len(feature_cols)}")
        logger.info(f"Samples: {len(X)}")
        logger.info(f"Date range: {dates.min()} to {dates.max()}")
        
        # Initialize OOF predictions
        oof_predictions = np.zeros(len(X))
        oof_indices = np.zeros(len(X), dtype=bool)
        
        # CV splits
        cv_splits = list(self.cv_strategy.get_folds(df))
        
        # Store fold scores
        fold_scores = []
        
        # Train each fol || 실제 학습데이터 검증데이터를 나누는 부분
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            # Train fold
            model, val_score, val_preds = self.train_fold(
                X_train, y_train, X_val, y_val, fold_idx
            )
            
            # Store model
            self.models.append(model)
            fold_scores.append(val_score)
            
            # Store OOF predictions  || val_preds : 검증 데이터에 대한 각각의 모든 예측값
            oof_predictions[val_idx] = val_preds
            oof_indices[val_idx] = True
        
        # Store OOF predictions
        self.oof_predictions = oof_predictions
        self.oof_indices = oof_indices
        
        # Calculate overall OOF score , 모든 날짜의 예측값과 실제값의 RMSE 계산
        oof_score = np.sqrt(np.mean((y[oof_indices] - oof_predictions[oof_indices]) ** 2))
        
        logger.info("\n" + "="*80)
        logger.info("Cross-Validation Results")
        logger.info("="*80)
        logger.info(f"\nFold Scores (RMSE):")
        for i, score in enumerate(fold_scores):
            logger.info(f"  Fold {i+1}: {score:.6f}")
        logger.info(f"\nMean CV Score: {np.mean(fold_scores):.6f} (+/- {np.std(fold_scores):.6f})")
        logger.info(f"OOF Score: {oof_score:.6f}")
        
        # Feature importance
        self._calculate_feature_importance(feature_cols)
        
        results = {
            'fold_scores': fold_scores,
            'mean_score': np.mean(fold_scores),
            'std_score': np.std(fold_scores),
            'oof_score': oof_score,
            'oof_predictions': oof_predictions,
            'oof_indices': oof_indices,
            'feature_importance': self.feature_importance,
            'n_features': len(feature_cols),
            'n_samples': len(X)
        }
        
        return results
    
    def _calculate_feature_importance(self, feature_names: List[str]) -> None:
        """Calculate average feature importance across folds."""
        if self.model_type == 'lightgbm':
            importance_list = []
            
            for model in self.models:
                importance = model.feature_importance(importance_type='gain')
                importance_list.append(importance)
            
            avg_importance = np.mean(importance_list, axis=0)
            
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)
            
        elif self.model_type == 'catboost':
            importance_list = []
            
            for model in self.models:
                importance = model.get_feature_importance()
                importance_list.append(importance)
            
            avg_importance = np.mean(importance_list, axis=0)
            
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)
        
        logger.info("\n" + "="*80)
        logger.info("Feature Importance (Top 20)")
        logger.info("="*80)
        
        for idx, row in self.feature_importance.head(20).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.2f}")
    
    def predict(
        self,
        X: pd.DataFrame,
        use_average: bool = True
    ) -> np.ndarray:
        """
        Make predictions using trained models.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to predict
        use_average : bool
            Whether to average predictions from all folds
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        if len(self.models) == 0:
            raise ValueError("No models trained. Call train_cv first.")
        
        if use_average:
            # Average predictions from all folds
            predictions = []
            
            for model in self.models:
                if self.model_type == 'lightgbm':
                    pred = model.predict(X)
                elif self.model_type == 'catboost':
                    pred = model.predict(X)
                predictions.append(pred)
            
            return np.mean(predictions, axis=0)
        else:
            # Use first fold model
            model = self.models[0]
            if self.model_type == 'lightgbm':
                return model.predict(X)
            elif self.model_type == 'catboost':
                return model.predict(X)
    
    def save_models(self, output_dir: str = "artifacts/models") -> None:
        """
        Save trained models.
        
        Parameters
        ----------
        output_dir : str
            Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_path = output_path / f"{self.model_type}_fold_{i}.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Model fold {i} saved to {model_path}")
        
        # Save feature importance
        if self.feature_importance is not None:
            importance_path = output_path / f"{self.model_type}_feature_importance.csv"
            self.feature_importance.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")
    
    def save_oof_predictions(
        self,
        output_path: str = "artifacts/oof_r_hat.csv"
    ) -> None:
        """
        Save OOF predictions.
        
        Parameters
        ----------
        output_path : str
            Output file path
        """
        if self.oof_predictions is None:
            raise ValueError("No OOF predictions. Call train_cv first.")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        oof_df = pd.DataFrame({
            'oof_prediction': self.oof_predictions, #  샘플(날짜)의 예측된 수익률 (forward_returns 예측값)
            'is_oof': self.oof_indices # 해당 샘플이 OOF 예측이 있는지 여부
        })
        
        oof_df.to_csv(output_file, index=False)  # 초반 몇천개 샘플은 is_oof=False
                                                 # (시계열 특성상 과거 데이터가 부족하여 예측 불가
        logger.info(f"OOF predictions saved to {output_file}")


def create_return_predictor(
    model_type: str = 'lightgbm',
    config_path: str = "conf/params.yaml"
) -> ReturnPredictor:
    """
    Factory function to create return predictor.
    
    Parameters
    ----------
    model_type : str
        Model type: 'lightgbm' or 'catboost'
    config_path : str
        Path to configuration file
        
    Returns
    -------
    ReturnPredictor
        Configured predictor instance
    """
    return ReturnPredictor(model_type, config_path)
