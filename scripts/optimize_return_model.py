"""
Complete optimization pipeline for return prediction model.

This script performs:
1. Data loading and preprocessing
2. Feature engineering
3. Feature selection
4. Hyperparameter tuning
5. Model training with best parameters
6. Model evaluation and interpretation
7. Results saving

Usage:
    python scripts/optimize_return_model.py
"""

import sys
from pathlib import Path

# ========== DEBUG: Print current state ==========
print("="*80)
print("SCRIPT PATH SETUP - DEBUG INFO")
print("="*80)
print(f"__file__ = {__file__}")
print(f"sys.path BEFORE modification:")
for p in sys.path[:5]:
    print(f"  - {p}")

# Add project root to Python path (Kaggle and Local compatible)
project_root = Path(__file__).parent.parent
print(f"\nproject_root = {project_root}")
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
    print(f"  Exists? {kaggle_path.exists()}")
    if kaggle_path.exists():
        print(f"  Contents:")
        for item in kaggle_path.iterdir():
            print(f"    - {item.name}")
        if str(kaggle_path) not in sys.path:
            sys.path.insert(0, str(kaggle_path))
            print(f"  ✓ Added {kaggle_path} to sys.path")
        break

print(f"\nsys.path AFTER modification:")
for p in sys.path[:5]:
    print(f"  - {p}")

# Check if src module is findable
src_path = Path(sys.path[0]) / "src"
print(f"\nLooking for src at: {src_path}")
print(f"src exists? {src_path.exists()}")
if src_path.exists():
    print(f"src contents:")
    for item in src_path.iterdir():
        if item.suffix == '.py':
            print(f"  - {item.name}")

print("="*80)
print()

import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, List, Tuple, Optional

from src.data import DataLoader
from src.features import FeatureEngineering
from src.tuner import OptunaLightGBMTuner
from src.models import ReturnPredictor
from src.interpretability import ModelInterpreter
from src.metric import CompetitionMetric, evaluate_return_model
from src.utils import get_logger, Timer, load_config

logger = get_logger(log_file="logs/optimization.log", level="INFO")


class ReturnModelOptimizer:
    """
    End-to-end optimization pipeline for return prediction model.
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
        
        # Results storage
        self.results = {}
        
        logger.info("="*80)
        logger.info("RETURN MODEL OPTIMIZATION PIPELINE")
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
    
    def step3_feature_selection(
        self,
        df: pd.DataFrame,
        method: str = 'correlation',
        top_n: int = 200,
        remove_correlated: bool = True,
        corr_threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Step 3: Feature selection.
        
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
        logger.info("STEP 3: FEATURE SELECTION")
        logger.info("="*80)
        
        with Timer("Feature Selection", logger):
            # Select features by importance
            logger.info(f"\n3.1 Selecting top {top_n} features by {method}...")
            df_selected, selected_features = self.feature_engineer.select_features_by_importance(
                df,
                target_col='forward_returns',
                method=method,
                top_n=top_n
            )
            
            # Remove correlated features
            if remove_correlated:
                logger.info(f"\n3.2 Removing correlated features (threshold={corr_threshold})...")
                df_selected, removed_features = self.feature_engineer.remove_correlated_features(
                    df_selected,
                    threshold=corr_threshold,
                    target_col='forward_returns'
                )
            
            final_features = [
                col for col in df_selected.columns 
                if col not in ['date_id', 'forward_returns', 'risk_free_rate', 
                              'market_forward_excess_returns']
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
                output_dir / 'selected_features_optimized.csv',
                index=False
            )
            
            return df_selected, final_features
    
    def step4_hyperparameter_tuning(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        n_trials: int = 50,
        timeout: Optional[int] = None
    ) -> Dict:
        """
        Step 4: Hyperparameter tuning with Optuna.
        
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
        logger.info("STEP 4: HYPERPARAMETER TUNING")
        logger.info("="*80)
        
        with Timer("Hyperparameter Tuning", logger):
            # Create tuner
            tuner = OptunaLightGBMTuner(
                config_path=self.config_path,
                config_section='tuning',
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=1,
                random_state=42
            )
            
            # Run optimization
            logger.info(f"\n4.1 Running Optuna optimization ({n_trials} trials)...")
            best_params = tuner.optimize(
                df=df,
                feature_cols=feature_cols,
                target_col='forward_returns',
                study_name='return_lgbm_tuning'
            )
            
            # Get best score from tuner
            best_score = tuner.best_score
            
            logger.info(f"\n✓ Hyperparameter tuning complete")
            logger.info(f"  Best score: {best_score:.6f}")
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
            with open(output_dir / 'lightgbm_best_params_optimized.json', 'w') as f:
                json.dump(best_params, f, indent=2)
            
            return best_params
    
    def step5_train_final_model(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        best_params: Dict
    ) -> Tuple[ReturnPredictor, np.ndarray, float]:
        """
        Step 5: Train final model with best parameters.
        
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
        Tuple[ReturnPredictor, np.ndarray, float]
            (Trained predictor, OOF predictions, OOF score)
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 5: FINAL MODEL TRAINING")
        logger.info("="*80)
        
        with Timer("Final Model Training", logger):
            # Create predictor with best params
            predictor = ReturnPredictor(
                model_type='lightgbm',
                config_path=self.config_path
            )
            
            # Update params with best params
            predictor.params.update(best_params)
            
            # Train with cross-validation
            logger.info("\n5.1 Training with cross-validation...")
            results = predictor.train_cv(
                df=df,
                target_col='forward_returns',
                date_col='date_id'
            )
            
            # Extract results
            oof_preds = results['oof_predictions']
            oof_score = results['oof_score']
            
            # ========== RETURN MODEL EVALUATION ==========
            # Evaluate return model performance (독립적으로 평가)
            logger.info("\n" + "-"*80)
            logger.info("RETURN MODEL EVALUATION (Independent Assessment)")
            logger.info("-"*80)
            
            # Get actual returns for OOF indices
            actual_returns = df[results['oof_indices']]['forward_returns'].values
            oof_preds_valid = oof_preds[results['oof_indices']]
            
            # Evaluate return model
            return_metrics = evaluate_return_model(oof_preds_valid, actual_returns)
            
            logger.info("\n📊 Return Model Performance Metrics:")
            logger.info(f"  🎯 Directional Accuracy: {return_metrics['directional_accuracy']:.2%}")
            logger.info(f"     (> 0.52 is profitable)")
            logger.info(f"  📈 Correlation: {return_metrics['correlation']:.4f}")
            logger.info(f"  📊 Rank Correlation: {return_metrics['rank_correlation']:.4f}")
            logger.info(f"  ⭐ Information Coefficient: {return_metrics['information_coefficient']:.4f}")
            logger.info(f"     (> 0.05 is significant, > 0.10 is excellent)")
            logger.info(f"  📉 RMSE: {return_metrics['rmse']:.6f}")
            logger.info(f"  📉 MAE: {return_metrics['mae']:.6f}")
            logger.info(f"\n💰 Quintile Analysis:")
            logger.info(f"  Top 20% Predictions → Avg Return: {return_metrics['top_quintile_return']:.4%}")
            logger.info(f"  Bottom 20% Predictions → Avg Return: {return_metrics['bottom_quintile_return']:.4%}")
            logger.info(f"  Long-Short Spread: {return_metrics['long_short_spread']:.4%}")
            logger.info(f"     (Higher is better - shows prediction power)")
            
            logger.info(f"\n✓ Final model training complete")
            logger.info(f"  OOF Score (RMSE): {oof_score:.6f}")
            logger.info(f"  Number of models: {len(predictor.models)}")
            
            # Store results (including return model metrics)
            self.results['final_model'] = {
                'oof_score': oof_score,
                'n_folds': len(predictor.models),
                'return_model_metrics': return_metrics
            }
            
            # Save models
            predictor.save_models(output_dir="artifacts/models_optimized")
            
            # Save feature names for inference
            feature_names_path = Path("artifacts/models_optimized/feature_names.json")
            with open(feature_names_path, 'w') as f:
                json.dump(feature_cols, f, indent=2)
            logger.info(f"✓ Feature names saved: {len(feature_cols)} features")
            
            # Save OOF predictions for position optimization
            oof_pred_path = Path("artifacts/oof_return_predictions.npy")
            oof_pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(oof_pred_path, oof_preds)
            logger.info(f"\n✓ OOF predictions saved to {oof_pred_path}")
            
            return predictor, oof_preds, oof_score
    
    def step6_model_interpretation(
        self,
        predictor: ReturnPredictor,
        df: pd.DataFrame,
        feature_cols: List[str],
        calculate_shap: bool = False
    ) -> ModelInterpreter:
        """
        Step 6: Model interpretation and analysis.
        
        Parameters
        ----------
        predictor : ReturnPredictor
            Trained predictor
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
        logger.info("STEP 6: MODEL INTERPRETATION")
        logger.info("="*80)
        
        with Timer("Model Interpretation", logger):
            # Create interpreter
            interpreter = ModelInterpreter(
                models=predictor.models,
                feature_names=feature_cols,
                model_type='lightgbm'
            )
            
            # Calculate feature importance
            logger.info("\n6.1 Calculating feature importance...")
            importance_df = interpreter.calculate_feature_importance(importance_type='gain')
            
            # Calculate SHAP values (optional, slow)
            if calculate_shap:
                logger.info("\n6.2 Calculating SHAP values...")
                X = df[feature_cols].sample(n=1000, random_state=42)
                shap_values = interpreter.calculate_shap_values(X, sample_size=1000)
            
            # Save analysis
            logger.info("\n6.3 Saving interpretability results...")
            interpreter.save_analysis(output_dir="results/interpretability_optimized")
            
            logger.info(f"\n✓ Model interpretation complete")
            
            # Store results
            self.results['interpretation'] = {
                'top_10_features': importance_df.head(10)['feature'].tolist(),
                'shap_calculated': calculate_shap
            }
            
            return interpreter
    
    def step7_save_results(self) -> None:
        """
        Step 7: Save optimization results.
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 7: SAVING RESULTS")
        logger.info("="*80)
        
        # Create output directory
        output_dir = Path("results/optimization")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results summary
        with open(output_dir / 'optimization_summary.json', 'w') as f:
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
        logger.info("OPTIMIZATION SUMMARY")
        logger.info("="*80)
        
        if 'preprocessing' in self.results:
            logger.info(f"\nPreprocessing:")
            logger.info(f"  Shape: {self.results['preprocessing']['shape']}")
        
        if 'feature_engineering' in self.results:
            logger.info(f"\nFeature Engineering:")
            logger.info(f"  Original: {self.results['feature_engineering']['original_features']}")
            logger.info(f"  Engineered: {self.results['feature_engineering']['engineered_features']}")
            logger.info(f"  Total: {self.results['feature_engineering']['total_features']}")
        
        if 'feature_selection' in self.results:
            logger.info(f"\nFeature Selection:")
            logger.info(f"  Method: {self.results['feature_selection']['method']}")
            logger.info(f"  Final features: {self.results['feature_selection']['final_features']}")
        
        if 'hyperparameter_tuning' in self.results:
            logger.info(f"\nHyperparameter Tuning:")
            logger.info(f"  Trials: {self.results['hyperparameter_tuning']['n_trials']}")
            logger.info(f"  Best score: {self.results['hyperparameter_tuning']['best_score']:.6f}")
        
        if 'final_model' in self.results:
            logger.info(f"\nFinal Model:")
            logger.info(f"  OOF Score (RMSE): {self.results['final_model']['oof_score']:.6f}")
            logger.info(f"  Folds: {self.results['final_model']['n_folds']}")
            
            # Return Model Metrics
            if 'return_model_metrics' in self.results['final_model']:
                metrics = self.results['final_model']['return_model_metrics']
                logger.info(f"\n  📊 Return Model Performance:")
                logger.info(f"    Directional Accuracy: {metrics['directional_accuracy']:.2%}")
                logger.info(f"    Information Coefficient: {metrics['information_coefficient']:.4f}")
                logger.info(f"    Correlation: {metrics['correlation']:.4f}")
                logger.info(f"    Long-Short Spread: {metrics['long_short_spread']:.4%}")
        
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
        # Step 3: Feature Selection
        selection_method: str = 'correlation',
        top_n_features: int = 200,
        remove_correlated: bool = True,
        corr_threshold: float = 0.95,
        # Step 4: Hyperparameter Tuning
        n_trials: int = 50,
        timeout: Optional[int] = None,
        # Step 6: Interpretation
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
        logger.info("STARTING FULL OPTIMIZATION PIPELINE")
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
        
        # Step 3: Feature selection
        train_selected, selected_features = self.step3_feature_selection(
            train_engineered,
            method=selection_method,
            top_n=top_n_features,
            remove_correlated=remove_correlated,
            corr_threshold=corr_threshold
        )
        
        # Step 4: Hyperparameter tuning
        best_params = self.step4_hyperparameter_tuning(
            train_selected,
            selected_features,
            n_trials=n_trials,
            timeout=timeout
        )
        
        # Step 5: Train final model
        predictor, oof_preds, oof_score = self.step5_train_final_model(
            train_selected,
            selected_features,
            best_params
        )
        
        # Step 6: Model interpretation
        interpreter = self.step6_model_interpretation(
            predictor,
            train_selected,
            selected_features,
            calculate_shap=calculate_shap
        )
        
        # Step 7: Save results
        self.step7_save_results()
        
        logger.info("\n" + "="*80)
        logger.info("✓ OPTIMIZATION PIPELINE COMPLETE!")
        logger.info("="*80)
        
        return self.results


def main():
    """Main optimization pipeline."""
    
    # Create optimizer
    optimizer = ReturnModelOptimizer(config_path="conf/params.yaml")
    
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
        calculate_shap=True  # Set True for SHAP analysis (slow)
    )
    
    logger.info("\n✓ Optimization complete! Check results/ and artifacts/ directories.")


if __name__ == "__main__":
    main()