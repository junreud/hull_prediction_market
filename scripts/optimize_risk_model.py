"""
Complete optimization pipeline for risk prediction model.

This script performs:
1. Data loading and preprocessing
2. Feature engineering
3. Feature selection
4. Risk label creation (volatility prediction)
5. Hyperparameter tuning
6. Model training with best parameters
7. Model evaluation and interpretation
8. Results saving

Usage:
    python scripts/optimize_risk_model.py
"""

import sys
from pathlib import Path

# ========== DEBUG: Print current state ==========
print("="*80)
print("RISK MODEL - SCRIPT PATH SETUP - DEBUG INFO")
print("="*80)
print(f"__file__ = {__file__}")

# Add project root to Python path (Kaggle and Local compatible)
project_root = Path(__file__).parent.parent
print(f"project_root = {project_root}")
print(f"project_root exists? {project_root.exists()}")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"✓ Added project_root to sys.path")

# Kaggle-specific: Add dataset path if exists
kaggle_dataset_paths = [
    Path('/kaggle/input/mydata'),
    Path('/kaggle/input/my-hull-models'),
]
for kaggle_path in kaggle_dataset_paths:
    print(f"\nChecking {kaggle_path}...")
    if kaggle_path.exists():
        print(f"  ✓ Exists! Adding to sys.path")
        if str(kaggle_path) not in sys.path:
            sys.path.insert(0, str(kaggle_path))
        break

print(f"\nsys.path[0] = {sys.path[0]}")
print("="*80)
print()

import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error

from src.data import DataLoader
from src.features import FeatureEngineering
from src.tuner import OptunaLightGBMTuner
from src.risk import RiskLabeler, RiskForecaster
from src.interpretability import ModelInterpreter
from src.metric import CompetitionMetric
from src.utils import get_logger, Timer, load_config

logger = get_logger(log_file="logs/risk_optimization.log", level="INFO")


class RiskModelOptimizer:
    """
    End-to-end optimization pipeline for risk prediction model.
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
        self.feature_engineer = FeatureEngineering(config_path)
        
        # Risk labeler
        risk_config = self.config.get('risk', {}).get('label', {})
        self.risk_labeler = RiskLabeler(
            window=risk_config.get('window', 20),
            min_periods=risk_config.get('min_periods', 10),
            clip_threshold=risk_config.get('clip_threshold', 4.0)
        )
        
        # Results storage
        self.results = {}
        
        logger.info("="*80)
        logger.info("RISK MODEL OPTIMIZATION PIPELINE")
        logger.info("="*80)
    
    def step1_load_and_preprocess_data(
        self,
        handle_outliers: bool = True,
        normalize: bool = True,
        scale: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Step 1: Load and preprocess data.
        
        Parameters
        ----------
        handle_outliers : bool
            Winsorize outliers
        normalize : bool
            Normalize features
        scale : bool
            Scale features
            
        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            (Preprocessed dataframe, metadata)
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA LOADING AND PREPROCESSING")
        logger.info("="*80)
        
        with Timer("Data Loading and Preprocessing", logger):
            # Load data
            logger.info("\n1.1 Loading raw data...")
            train_df, _ = self.data_loader.load_data()
            logger.info(f"✓ Loaded {len(train_df)} samples")
            
            # Preprocess (data cleaning only)
            logger.info("\n1.2 Preprocessing data...")
            train_processed, metadata = self.data_loader.preprocess_timeseries(
                train_df,
                train_df=None,  # First time, fit on itself
                handle_outliers=handle_outliers,
                winsorize_limits=(0.001, 0.001),  # 0.1% clipping
                winsorize_method='rolling',
                normalize=normalize,
                normalize_method='rank_gauss',
                scale=scale,
                scale_method='robust',
                window=60
            )
            
            logger.info(f"✓ Preprocessing complete: {train_processed.shape}")
            
            # Store results
            self.results['preprocessing'] = {
                'shape': train_processed.shape,
                'metadata': metadata
            }
            
            return train_processed, metadata
    
    def step2_feature_engineering(
        self,
        df: pd.DataFrame,
        add_time_features: bool = True,
        add_regime_features: bool = True
    ) -> pd.DataFrame:
        """
        Step 2: Feature engineering.
        
        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed dataframe
        add_time_features : bool
            Add time period features
        add_regime_features : bool
            Add market regime features
            
        Returns
        -------
        pd.DataFrame
            Dataframe with engineered features
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("="*80)
        
        with Timer("Feature Engineering", logger):
            df_engineered = df.copy()
            
            # Add time period features
            if add_time_features:
                logger.info("\n2.1 Adding time period features...")
                df_engineered = self.feature_engineer.create_time_period_features(df_engineered)
                logger.info(f"✓ Time period features added")
            
            # Add market regime features
            if add_regime_features:
                logger.info("\n2.2 Adding market regime features...")
                df_engineered = self.feature_engineer.create_market_regime_features(
                    df_engineered,
                    auto_detect=True,
                    vol_threshold=2.0
                )
                logger.info(f"✓ Market regime features added")
            
            # Standard feature engineering
            logger.info("\n2.3 Creating engineered features...")
            df_engineered = self.feature_engineer.fit_transform(df_engineered)
            
            logger.info(f"\n✓ Feature engineering complete")
            logger.info(f"  Original features: {len(self.feature_engineer.original_features)}")
            logger.info(f"  Engineered features: {len(self.feature_engineer.engineered_features)}")
            logger.info(f"  Total features: {df_engineered.shape[1] - 2}")
            
            # Store results
            self.results['feature_engineering'] = {
                'original_features': len(self.feature_engineer.original_features),
                'engineered_features': len(self.feature_engineer.engineered_features),
                'total_features': df_engineered.shape[1] - 2,
                'time_features_added': add_time_features,
                'regime_features_added': add_regime_features
            }
            
            return df_engineered
    
    def step3_create_risk_labels(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Step 3: Create risk labels (future volatility).
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with engineered features
            
        Returns
        -------
        pd.DataFrame
            Dataframe with risk labels
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 3: CREATING RISK LABELS")
        logger.info("="*80)
        
        with Timer("Risk Label Creation", logger):
            # Create risk labels
            logger.info("\n3.1 Calculating future volatility...")
            df_labeled = self.risk_labeler.fit_transform(
                df,
                target_col='forward_returns' # 얘측할 타겟이 아닌 risk_label을 만들때 사용할 원본 데이터
            )
            
            # Check risk labels
            valid_labels = df_labeled['risk_label'].notna().sum()
            logger.info(f"\n✓ Risk label creation complete")
            logger.info(f"  Valid labels: {valid_labels} / {len(df_labeled)}")
            logger.info(f"  Mean: {df_labeled['risk_label'].mean():.6f}")
            logger.info(f"  Std: {df_labeled['risk_label'].std():.6f}")
            logger.info(f"  Min: {df_labeled['risk_label'].min():.6f}")
            logger.info(f"  Max: {df_labeled['risk_label'].max():.6f}")
            
            # Store results
            self.results['risk_labels'] = {
                'valid_count': int(valid_labels),
                'total_count': len(df_labeled),
                'mean': float(df_labeled['risk_label'].mean()),
                'std': float(df_labeled['risk_label'].std()),
                'min': float(df_labeled['risk_label'].min()),
                'max': float(df_labeled['risk_label'].max())
            }
            
            return df_labeled
    
    def step4_feature_selection(
        self,
        df: pd.DataFrame,
        method: str = 'correlation',
        top_n: int = 200,
        remove_correlated: bool = True,
        corr_threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Step 4: Feature selection.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with all features
        method : str
            Selection method: 'correlation', 'variance', 'mutual_info'
        top_n : int
            Number of top features to select
        remove_correlated : bool
            Remove highly correlated features
        corr_threshold : float
            Correlation threshold for removal
            
        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            (Selected dataframe, selected feature names)
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 4: FEATURE SELECTION")
        logger.info("="*80)
        
        with Timer("Feature Selection", logger):
            # Select features by importance (using risk_label as target)
            logger.info(f"\n4.1 Selecting top {top_n} features by {method}...")
            df_selected, selected_features = self.feature_engineer.select_features_by_importance(
                df,
                target_col='risk_label',  # Risk label as target
                method=method,
                top_n=top_n
            )
            
            # Remove correlated features 상관계수가 높은 특징 제거
            if remove_correlated:
                logger.info(f"\n4.2 Removing correlated features (threshold={corr_threshold})...")
                df_selected, removed_features = self.feature_engineer.remove_correlated_features(
                    df_selected,
                    threshold=corr_threshold,
                    target_col='risk_label'
                )
            
            final_features = [
                col for col in df_selected.columns 
                if col not in ['date_id', 'forward_returns', 'risk_free_rate', 
                              'market_forward_excess_returns', 'risk_label']
            ]
            
            logger.info(f"\n✓ Feature selection complete")
            logger.info(f"  Final features: {len(final_features)}")
            
            # Store results
            self.results['feature_selection'] = {
                'method': method,
                'top_n': top_n,
                'corr_threshold': corr_threshold if remove_correlated else None,
                'final_features': len(final_features),
                'selected_features': final_features
            }
            
            # Save selected features
            output_dir = Path("results/feature_selection")
            output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'feature': final_features}).to_csv(
                output_dir / 'selected_features_risk_optimized.csv',
                index=False
            )
            
            return df_selected, final_features
    
    def step5_hyperparameter_tuning(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        n_trials: int = 50,
        timeout: Optional[int] = None
    ) -> Dict:
        """
        Step 5: Hyperparameter tuning with Optuna.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training dataframe
        feature_cols : List[str]
            Feature column names
        n_trials : int
            Number of Optuna trials
        timeout : int, optional
            Time limit in seconds
            
        Returns
        -------
        Dict
            Best hyperparameters
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 5: HYPERPARAMETER TUNING")
        logger.info("="*80)
        
        with Timer("Hyperparameter Tuning", logger):
            # Create tuner (use 'risk' config section)
            tuner = OptunaLightGBMTuner(
                config_path=self.config_path,
                config_section='risk',  # Use risk config section
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=1,
                random_state=42
            )
            
            # Run optimization
            logger.info(f"\n5.1 Running Optuna optimization ({n_trials} trials)...")
            best_params = tuner.optimize(
                df=df,
                feature_cols=feature_cols,
                target_col='risk_label',  # Risk label as target
                study_name='risk_lgbm_tuning'
            )
            
            # Get best score from tuner
            best_score = tuner.best_score
            
            logger.info(f"\n✓ Hyperparameter tuning complete")
            logger.info(f"  Best score (RMSE): {best_score:.6f}")
            logger.info(f"  Best params:")
            for key, value in best_params.items():
                logger.info(f"    {key}: {value}")
            
            # Store results
            self.results['hyperparameter_tuning'] = {
                'n_trials': n_trials,
                'best_score': best_score,
                'best_params': best_params
            }
            
            # Save best parameters
            output_dir = Path("artifacts")
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / 'lightgbm_best_params_risk_optimized.json', 'w') as f:
                json.dump(best_params, f, indent=2)
            
            return best_params
    
    def step6_train_final_model(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        best_params: Dict
    ) -> Tuple[RiskForecaster, np.ndarray, float]:
        """
        Step 6: Train final model with best parameters.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training dataframe
        feature_cols : List[str]
            Feature column names
        best_params : Dict
            Best hyperparameters from tuning
            
        Returns
        -------
        Tuple[RiskForecaster, np.ndarray, float]
            (Trained risk model, OOF predictions, OOF score)
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 6: FINAL MODEL TRAINING")
        logger.info("="*80)
        
        with Timer("Final Model Training", logger):
            # Create risk model with best params
            risk_model = RiskForecaster(
                model_params=best_params,
                config_path=self.config_path
            )
            
            # Train with cross-validation
            logger.info("\n6.1 Training with cross-validation...")
            oof_preds, models = risk_model.train(
                df=df,
                feature_cols=feature_cols,
                risk_col='risk_label',
                n_folds=5  # Default CV folds
            )
            
            # Calculate OOF score
            y_true = df['risk_label'].values
            valid_idx = ~(np.isnan(oof_preds) | np.isnan(y_true))
            oof_score = np.sqrt(mean_squared_error(y_true[valid_idx], oof_preds[valid_idx]))
            
            logger.info(f"\n✓ Final model training complete")
            logger.info(f"  OOF Score (RMSE): {oof_score:.6f}")
            logger.info(f"  Number of models: {len(risk_model.predictor.models)}")
            
            # Store results
            self.results['final_model'] = {
                'oof_score': oof_score,
                'n_folds': len(risk_model.predictor.models)
            }
            
            # Save models
            risk_model.save_models(output_dir="artifacts/models_risk_optimized")
            
            # Save feature names for inference
            feature_names_path = Path("artifacts/models_risk_optimized/feature_names.json")
            with open(feature_names_path, 'w') as f:
                json.dump(feature_cols, f, indent=2)
            logger.info(f"✓ Feature names saved: {len(feature_cols)} features")
            
            # Save OOF predictions for position optimization
            oof_pred_path = Path("artifacts/oof_risk_predictions.npy")
            oof_pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(oof_pred_path, oof_preds)
            logger.info(f"\n✓ OOF predictions saved to {oof_pred_path}")
            
            return risk_model, oof_preds, oof_score
    
    def step7_model_interpretation(
        self,
        risk_model: RiskForecaster,
        df: pd.DataFrame,
        feature_cols: List[str],
        calculate_shap: bool = False
    ) -> ModelInterpreter:
        """
        Step 7: Model interpretation and analysis.
        
        Parameters
        ----------
        risk_model : RiskModel
            Trained risk model
        df : pd.DataFrame
            Training dataframe
        feature_cols : List[str]
            Feature column names
        calculate_shap : bool
            Whether to calculate SHAP values (slow)
            
        Returns
        -------
        ModelInterpreter
            Model interpreter with analysis results
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 7: MODEL INTERPRETATION")
        logger.info("="*80)
        
        with Timer("Model Interpretation", logger):
            # Create interpreter
            interpreter = ModelInterpreter(
                models=risk_model.predictor.models,
                feature_names=feature_cols,
                model_type='lightgbm'
            )
            
            # Calculate feature importance
            logger.info("\n7.1 Calculating feature importance...")
            try:
                importance_df = interpreter.calculate_feature_importance(importance_type='gain')
            except Exception as e:
                logger.warning(f"Feature importance calculation failed: {e}")
                # Create simple importance from first model using LightGBM API
                first_model = risk_model.predictor.models[0]
                importances = first_model.feature_importance(importance_type='gain')
                importance_df = pd.DataFrame({
                    'feature': feature_cols[:len(importances)],
                    'importance': importances[:len(feature_cols)]
                })
                logger.info(f"✓ Fallback: Created importance from first model ({len(importance_df)} features)")
            
            # Calculate SHAP values (optional, slow)
            if calculate_shap:
                logger.info("\n7.2 Calculating SHAP values...")
                X = df[feature_cols].sample(n=1000, random_state=42)
                try:
                    shap_values = interpreter.calculate_shap_values(X, sample_size=1000)
                except Exception as e:
                    logger.warning(f"SHAP calculation failed: {e}")
            
            # Save analysis
            logger.info("\n7.3 Saving interpretability results...")
            try:
                interpreter.save_analysis(output_dir="results/interpretability_risk_optimized")
            except Exception as e:
                logger.warning(f"Save analysis failed: {e}")
                # Manual save of importance
                output_dir = Path("results/interpretability_risk_optimized")
                output_dir.mkdir(parents=True, exist_ok=True)
                importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
            
            logger.info(f"\n✓ Model interpretation complete")
            
            # Store results
            self.results['interpretation'] = {
                'top_10_features': importance_df.head(10)['feature'].tolist(),
                'shap_calculated': calculate_shap
            }
            
            return interpreter
    
    def step8_save_results(self) -> None:
        """
        Step 8: Save optimization results.
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 8: SAVING RESULTS")
        logger.info("="*80)
        
        # Create output directory
        output_dir = Path("results/optimization")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results summary
        with open(output_dir / 'risk_optimization_summary.json', 'w') as f:
            # Convert non-serializable objects
            results_serializable = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    results_serializable[key] = {}
                    for k, v in value.items():
                        # Skip non-serializable objects
                        if isinstance(v, (pd.DataFrame, np.ndarray, list)):
                            if isinstance(v, list) and len(v) > 0:
                                # Check if list contains serializable items
                                try:
                                    json.dumps(v)
                                    results_serializable[key][k] = v
                                except (TypeError, ValueError):
                                    results_serializable[key][k] = f"<{type(v).__name__}>"
                            elif isinstance(v, (pd.DataFrame, np.ndarray)):
                                results_serializable[key][k] = f"<{type(v).__name__}>"
                            else:
                                results_serializable[key][k] = v
                        elif isinstance(v, (str, int, float, bool, type(None))):
                            results_serializable[key][k] = v
                        elif isinstance(v, tuple):
                            results_serializable[key][k] = str(v)
                        else:
                            # Try to serialize, if fails, convert to string
                            try:
                                json.dumps(v)
                                results_serializable[key][k] = v
                            except (TypeError, ValueError):
                                results_serializable[key][k] = f"<{type(v).__name__}>"
                else:
                    results_serializable[key] = value
            
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"✓ Results saved to {output_dir}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("RISK MODEL OPTIMIZATION SUMMARY")
        logger.info("="*80)
        
        if 'preprocessing' in self.results:
            logger.info(f"\nPreprocessing:")
            logger.info(f"  Shape: {self.results['preprocessing']['shape']}")
        
        if 'feature_engineering' in self.results:
            logger.info(f"\nFeature Engineering:")
            logger.info(f"  Original: {self.results['feature_engineering']['original_features']}")
            logger.info(f"  Engineered: {self.results['feature_engineering']['engineered_features']}")
            logger.info(f"  Total: {self.results['feature_engineering']['total_features']}")
        
        if 'risk_labels' in self.results:
            logger.info(f"\nRisk Labels:")
            logger.info(f"  Valid count: {self.results['risk_labels']['valid_count']}")
            logger.info(f"  Mean volatility: {self.results['risk_labels']['mean']:.6f}")
            logger.info(f"  Std: {self.results['risk_labels']['std']:.6f}")
        
        if 'feature_selection' in self.results:
            logger.info(f"\nFeature Selection:")
            logger.info(f"  Method: {self.results['feature_selection']['method']}")
            logger.info(f"  Final features: {self.results['feature_selection']['final_features']}")
        
        if 'hyperparameter_tuning' in self.results:
            logger.info(f"\nHyperparameter Tuning:")
            logger.info(f"  Trials: {self.results['hyperparameter_tuning']['n_trials']}")
            logger.info(f"  Best score (RMSE): {self.results['hyperparameter_tuning']['best_score']:.6f}")
        
        if 'final_model' in self.results:
            logger.info(f"\nFinal Model:")
            logger.info(f"  OOF Score (RMSE): {self.results['final_model']['oof_score']:.6f}")
            logger.info(f"  Folds: {self.results['final_model']['n_folds']}")
        
        if 'interpretation' in self.results:
            logger.info(f"\nTop 10 Most Important Features:")
            for i, feat in enumerate(self.results['interpretation']['top_10_features'], 1):
                logger.info(f"  {i}. {feat}")
    
    def run_full_optimization(
        self,
        # Step 1: Preprocessing
        handle_outliers: bool = True,
        normalize: bool = True,
        scale: bool = True,
        # Step 2: Feature Engineering
        add_time_features: bool = True,
        add_regime_features: bool = True,
        # Step 4: Feature Selection
        selection_method: str = 'correlation',
        top_n_features: int = 200,
        remove_correlated: bool = True,
        corr_threshold: float = 0.95,
        # Step 5: Hyperparameter Tuning
        n_trials: int = 50,
        timeout: Optional[int] = None,
        # Step 7: Interpretation
        calculate_shap: bool = False
    ) -> Dict:
        """
        Run complete optimization pipeline.
        
        Parameters
        ----------
        See individual step methods for parameter descriptions.
        
        Returns
        -------
        Dict
            Optimization results
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING FULL RISK MODEL OPTIMIZATION PIPELINE")
        logger.info("="*80)
        
        # Step 1: Load and preprocess data
        train_df, metadata = self.step1_load_and_preprocess_data(
            handle_outliers=handle_outliers,
            normalize=normalize,
            scale=scale
        )
        
        # Step 2: Feature engineering
        train_engineered = self.step2_feature_engineering(
            train_df,
            add_time_features=add_time_features,
            add_regime_features=add_regime_features
        )
        
        # Step 3: Create risk labels
        train_labeled = self.step3_create_risk_labels(train_engineered)
        
        # Step 4: Feature selection
        train_selected, selected_features = self.step4_feature_selection(
            train_labeled,
            method=selection_method,
            top_n=top_n_features,
            remove_correlated=remove_correlated,
            corr_threshold=corr_threshold
        )
        
        # Step 5: Hyperparameter tuning
        best_params = self.step5_hyperparameter_tuning(
            train_selected,
            selected_features,
            n_trials=n_trials,
            timeout=timeout
        )
        
        # Step 6: Train final model
        risk_model, oof_preds, oof_score = self.step6_train_final_model(
            train_selected,
            selected_features,
            best_params
        )
        
        # Step 7: Model interpretation
        interpreter = self.step7_model_interpretation(
            risk_model,
            train_selected,
            selected_features,
            calculate_shap=calculate_shap
        )
        
        # Step 8: Save results
        self.step8_save_results()
        
        logger.info("\n" + "="*80)
        logger.info("✓ RISK MODEL OPTIMIZATION PIPELINE COMPLETE!")
        logger.info("="*80)
        
        return self.results


def main():
    """Main optimization pipeline."""
    
    # Create optimizer
    optimizer = RiskModelOptimizer(config_path="conf/params.yaml")
    
    # Run full optimization
    results = optimizer.run_full_optimization(
        # Preprocessing (data cleaning)
        handle_outliers=True,
        normalize=True,
        scale=True,
        # Feature Engineering (new features)
        add_time_features=True,
        add_regime_features=True,
        # Feature Selection
        selection_method='correlation',  # or 'mutual_info'
        top_n_features=200,
        remove_correlated=True,
        corr_threshold=0.95,
        # Hyperparameter Tuning
        n_trials=1,  # Increase for better results
        timeout=None,  # Or set time limit in seconds
        # Interpretation
        calculate_shap=False  # Set True for SHAP analysis (slow)
    )
    
    logger.info("\n✓ Risk model optimization complete! Check results/ and artifacts/ directories.")


if __name__ == "__main__":
    main()
