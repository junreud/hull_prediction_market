"""
Model interpretability module using SHAP and feature importance analysis.

This module provides:
- SHAP values calculation
- Feature importance analysis
- Partial dependence plots
- Feature interaction analysis
- Model behavior visualization
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
import pickle
import warnings
warnings.filterwarnings('ignore')

from src.utils import get_logger, Timer

logger = get_logger(log_file="logs/interpretability.log", level="INFO")


class ModelInterpreter:
    """
    Model interpretability analyzer using SHAP.
    
    Features:
    - SHAP values calculation
    - Feature importance ranking
    - Feature interaction detection
    - Visualization tools
    """
    
    def __init__(
        self,
        models: List[Any],
        feature_names: List[str],
        model_type: str = 'lightgbm'
    ):
        """
        Initialize model interpreter.
        
        Parameters
        ----------
        models : list
            List of trained models
        feature_names : list
            List of feature names
        model_type : str
            Type of model ('lightgbm' or 'catboost')
        """
        self.models = models
        self.feature_names = feature_names
        self.model_type = model_type
        
        # SHAP values storage
        self.shap_values = None
        self.shap_interaction_values = None
        
        # Feature importance
        self.feature_importance_gain = None
        self.feature_importance_split = None
        
        logger.info("ModelInterpreter initialized")
        logger.info(f"Models: {len(models)}, Features: {len(feature_names)}")
    
    def calculate_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Calculate feature importance from models.
        
        Parameters
        ----------
        importance_type : str
            Type of importance: 'gain' or 'split'
            
        Returns
        -------
        pd.DataFrame
            Feature importance dataframe
        """
        logger.info("="*80)
        logger.info("Calculating Feature Importance")
        logger.info("="*80)
        logger.info(f"Importance type: {importance_type}")
        
        with Timer("Feature Importance Calculation"):
            importance_list = []
            
            for fold_idx, model in enumerate(self.models):
                if self.model_type == 'lightgbm':
                    # Get importance from LightGBM
                    try:
                        importance = model.feature_importances_
                    except AttributeError:
                        # For Booster objects
                        importance = model.feature_importance(importance_type=importance_type)
                elif self.model_type == 'catboost':
                    # Get importance from CatBoost
                    importance = model.get_feature_importance()
                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")
                
                importance_list.append(importance)
            
            # Average across folds
            avg_importance = np.mean(importance_list, axis=0)
            std_importance = np.std(importance_list, axis=0)
            
            # Debug: check lengths
            logger.info(f"Feature names: {len(self.feature_names)}")
            logger.info(f"Avg importance: {len(avg_importance)}")
            logger.info(f"Std importance: {len(std_importance)}")
            
            # Ensure they match
            if len(avg_importance) != len(self.feature_names):
                logger.warning(f"Length mismatch! Using first {len(avg_importance)} features")
                feature_names_to_use = self.feature_names[:len(avg_importance)]
            else:
                feature_names_to_use = self.feature_names
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names_to_use,
                'importance': avg_importance,
                'importance_std': std_importance
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            importance_df['rank'] = range(1, len(importance_df) + 1)
            
            # Normalize importance to sum to 1
            importance_df['importance_normalized'] = (
                importance_df['importance'] / importance_df['importance'].sum()
            )
            
            # Store
            if importance_type == 'gain':
                self.feature_importance_gain = importance_df
            else:
                self.feature_importance_split = importance_df
            
            # Log top features
            logger.info(f"\nTop 20 Features by {importance_type}:")
            for idx, row in importance_df.head(20).iterrows():
                logger.info(
                    f"  {row['rank']:2d}. {row['feature']:30s}: "
                    f"{row['importance']:.4f} (±{row['importance_std']:.4f})"
                )
            
            return importance_df
    
    def calculate_shap_values(
        self,
        X: pd.DataFrame,
        sample_size: Optional[int] = None,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Calculate SHAP values for model predictions.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to explain
        sample_size : int, optional
            Number of samples to use (for speed)
        random_state : int
            Random seed for sampling
            
        Returns
        -------
        np.ndarray
            SHAP values (n_samples, n_features)
        """
        logger.info("="*80)
        logger.info("Calculating SHAP Values")
        logger.info("="*80)
        logger.info(f"Samples: {len(X)}")
        logger.info(f"Features: {len(X.columns)}")
        
        # Sample if requested
        if sample_size is not None and sample_size < len(X):
            logger.info(f"Sampling {sample_size} samples for SHAP calculation")
            X_sample = X.sample(n=sample_size, random_state=random_state)
        else:
            X_sample = X
        
        with Timer("SHAP Values Calculation"):
            try:
                import shap
                
                # Calculate SHAP values for each fold
                shap_values_list = []
                
                for fold_idx, model in enumerate(self.models):
                    logger.info(f"\nCalculating SHAP for Fold {fold_idx + 1}/{len(self.models)}")
                    
                    if self.model_type == 'lightgbm':
                        # Use TreeExplainer for LightGBM
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_sample)
                    elif self.model_type == 'catboost':
                        # Use TreeExplainer for CatBoost
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_sample)
                    else:
                        raise ValueError(f"Unknown model type: {self.model_type}")
                    
                    shap_values_list.append(shap_values)
                
                # Average SHAP values across folds
                self.shap_values = np.mean(shap_values_list, axis=0)
                
                logger.info(f"\nSHAP values shape: {self.shap_values.shape}")
                logger.info(f"Mean absolute SHAP value: {np.abs(self.shap_values).mean():.6f}")
                
                # Calculate feature importance from SHAP
                shap_importance = np.abs(self.shap_values).mean(axis=0)
                shap_importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'shap_importance': shap_importance
                }).sort_values('shap_importance', ascending=False)
                
                logger.info(f"\nTop 10 Features by SHAP Importance:")
                for idx, row in shap_importance_df.head(10).iterrows():
                    logger.info(f"  {row['feature']:30s}: {row['shap_importance']:.6f}")
                
                return self.shap_values
                
            except ImportError:
                logger.error("SHAP library not installed. Install with: pip install shap")
                raise
    
    def get_feature_interactions(
        self,
        X: pd.DataFrame,
        top_n: int = 10,
        sample_size: Optional[int] = 1000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Detect feature interactions using SHAP interaction values.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
        top_n : int
            Number of top interactions to return
        sample_size : int, optional
            Number of samples (for speed)
        random_state : int
            Random seed
            
        Returns
        -------
        pd.DataFrame
            Top feature interactions
        """
        logger.info("="*80)
        logger.info("Analyzing Feature Interactions")
        logger.info("="*80)
        
        # Sample if requested
        if sample_size is not None and sample_size < len(X):
            logger.info(f"Sampling {sample_size} samples")
            X_sample = X.sample(n=sample_size, random_state=random_state)
        else:
            X_sample = X
        
        with Timer("Feature Interaction Analysis"):
            try:
                import shap
                
                # Use first model for interaction analysis
                model = self.models[0]
                
                logger.info("Calculating SHAP interaction values (this may take a while)...")
                
                if self.model_type == 'lightgbm':
                    explainer = shap.TreeExplainer(model)
                    shap_interaction_values = explainer.shap_interaction_values(X_sample)
                else:
                    logger.warning("Interaction values only supported for LightGBM")
                    return pd.DataFrame()
                
                self.shap_interaction_values = shap_interaction_values
                
                # Extract top interactions
                n_features = len(self.feature_names)
                interaction_scores = []
                
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        # Mean absolute interaction effect
                        interaction_value = np.abs(shap_interaction_values[:, i, j]).mean()
                        interaction_scores.append({
                            'feature_1': self.feature_names[i],
                            'feature_2': self.feature_names[j],
                            'interaction_value': interaction_value
                        })
                
                # Create dataframe and sort
                interactions_df = pd.DataFrame(interaction_scores)
                interactions_df = interactions_df.sort_values(
                    'interaction_value', ascending=False
                ).head(top_n)
                
                logger.info(f"\nTop {top_n} Feature Interactions:")
                for idx, row in interactions_df.iterrows():
                    logger.info(
                        f"  {row['feature_1']} × {row['feature_2']}: "
                        f"{row['interaction_value']:.6f}"
                    )
                
                return interactions_df
                
            except ImportError:
                logger.error("SHAP library not installed")
                raise
    
    def save_analysis(
        self,
        output_dir: str = "results/interpretability"
    ):
        """
        Save interpretability analysis results.
        
        Parameters
        ----------
        output_dir : str
            Directory to save results
        """
        logger.info("="*80)
        logger.info("Saving Interpretability Results")
        logger.info("="*80)
        
        # Create directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save feature importance
        if self.feature_importance_gain is not None:
            importance_path = output_path / "feature_importance_gain.csv"
            self.feature_importance_gain.to_csv(importance_path, index=False)
            logger.info(f"✓ Feature importance saved to {importance_path}")
        
        # Save SHAP values
        if self.shap_values is not None:
            shap_path = output_path / "shap_values.npy"
            np.save(shap_path, self.shap_values)
            logger.info(f"✓ SHAP values saved to {shap_path}")
            
            # Save SHAP summary
            shap_importance = np.abs(self.shap_values).mean(axis=0)
            shap_df = pd.DataFrame({
                'feature': self.feature_names,
                'mean_abs_shap': shap_importance
            }).sort_values('mean_abs_shap', ascending=False)
            
            shap_summary_path = output_path / "shap_importance.csv"
            shap_df.to_csv(shap_summary_path, index=False)
            logger.info(f"✓ SHAP importance saved to {shap_summary_path}")
        
        logger.info(f"\n✓ All results saved to {output_dir}")
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance.
        
        Parameters
        ----------
        top_n : int
            Number of top features to plot
        save_path : str, optional
            Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            if self.feature_importance_gain is None:
                logger.warning("Calculate feature importance first")
                return
            
            # Get top features
            top_features = self.feature_importance_gain.head(top_n)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_features['importance'])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features['feature'])
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            ax.set_title(f'Top {top_n} Features by Importance')
            
            plt.tight_layout()
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✓ Feature importance plot saved to {save_path}")
            
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib not installed. Skipping plot.")
    
    def plot_shap_summary(
        self,
        X: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Plot SHAP summary.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features used for SHAP calculation
        save_path : str, optional
            Path to save plot
        """
        try:
            import shap
            import matplotlib.pyplot as plt
            
            if self.shap_values is None:
                logger.warning("Calculate SHAP values first")
                return
            
            # Create SHAP summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values,
                X,
                feature_names=self.feature_names,
                show=False
            )
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✓ SHAP summary plot saved to {save_path}")
            
            plt.close()
            
        except ImportError:
            logger.warning("SHAP or matplotlib not installed. Skipping plot.")


if __name__ == "__main__":
    # Example usage
    logger.info("Testing ModelInterpreter")
    
    # Load models
    model_paths = [
        "artifacts/models/lightgbm_fold_0.pkl",
        "artifacts/models/lightgbm_fold_1.pkl",
        "artifacts/models/lightgbm_fold_2.pkl",
        "artifacts/models/lightgbm_fold_3.pkl",
    ]
    
    models = []
    for path in model_paths:
        if Path(path).exists():
            with open(path, 'rb') as f:
                models.append(pickle.load(f))
    
    if len(models) == 0:
        logger.error("No models found. Run model training first.")
    else:
        # Load feature names from importance file
        importance_path = "artifacts/models/lightgbm_feature_importance.csv"
        if Path(importance_path).exists():
            importance_df = pd.read_csv(importance_path)
            feature_names = importance_df['feature'].tolist()
            
            # Create interpreter
            interpreter = ModelInterpreter(
                models=models,
                feature_names=feature_names,
                model_type='lightgbm'
            )
            
            # Calculate feature importance
            importance_df = interpreter.calculate_feature_importance()
            
            # Save results
            interpreter.save_analysis()
            
            logger.info("✓ Interpretability test complete")
        else:
            logger.error(f"Feature importance file not found: {importance_path}")
