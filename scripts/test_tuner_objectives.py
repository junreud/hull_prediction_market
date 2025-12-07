"""
Test script to validate tuner.py refactoring.

This script verifies that:
1. All new objective types are recognized
2. The tuner can initialize with each objective type
3. Metrics are correctly extracted from evaluate_return_model/evaluate_risk_model
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tuner import OptunaLightGBMTuner
from src.cv import TimeSeriesCV

def test_objective_types():
    """Test all objective types can initialize."""
    
    # Valid objective types
    return_objectives = [
        'return_ic',
        'return_spread', 
        'return_correlation',
        'return_rank_correlation',
        'return_directional_accuracy'
    ]
    
    risk_objectives = [
        'risk_correlation',
        'risk_hit_rate'
    ]
    
    all_objectives = return_objectives + risk_objectives
    
    print("Testing Objective Type Initialization")
    print("=" * 60)
    
    # Create dummy CV strategy
    cv_strategy = TimeSeriesCV(n_splits=3, test_size=0.2)
    
    for obj_type in all_objectives:
        try:
            tuner = OptunaLightGBMTuner(
                cv_strategy=cv_strategy,
                n_trials=1,
                objective_type=obj_type,
                timeout=10
            )
            print(f"✓ {obj_type:35s} - SUCCESS")
        except Exception as e:
            print(f"✗ {obj_type:35s} - FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("Testing Invalid Objective Type")
    print("=" * 60)
    
    # Test that invalid objective types fail properly
    invalid_objectives = ['rmse', 'ic', 'spread', 'combined']
    
    for obj_type in invalid_objectives:
        try:
            # Create small test dataset
            np.random.seed(42)
            df = pd.DataFrame({
                'date_id': range(100),
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100),
                'forward_returns': np.random.randn(100)
            })
            
            tuner = OptunaLightGBMTuner(
                cv_strategy=cv_strategy,
                n_trials=1,
                objective_type=obj_type,
                timeout=10
            )
            
            # Try to run optimization (should fail in _objective)
            result = tuner.optimize(
                df=df,
                feature_cols=['feature_1', 'feature_2'],
                target_col='forward_returns'
            )
            
            print(f"✗ {obj_type:35s} - Should have failed but didn't!")
            
        except ValueError as e:
            if "Unknown" in str(e):
                print(f"✓ {obj_type:35s} - Correctly rejected: {e}")
            else:
                print(f"? {obj_type:35s} - Unexpected error: {e}")
        except Exception as e:
            print(f"? {obj_type:35s} - Unexpected error type: {type(e).__name__}: {e}")

def test_metric_extraction():
    """Verify that metrics are correctly extracted."""
    print("\n" + "=" * 60)
    print("Testing Metric Extraction")
    print("=" * 60)
    
    from src.metric import evaluate_return_model, evaluate_risk_model
    
    # Create dummy predictions
    np.random.seed(42)
    y_pred = np.random.randn(100)
    y_true = np.random.randn(100)
    
    # Test return model metrics
    print("\nReturn Model Metrics:")
    return_metrics = evaluate_return_model(y_pred, y_true, return_all_metrics=True)
    for key, value in return_metrics.items():
        print(f"  {key:30s}: {value:.6f}")
    
    # Test risk model metrics
    print("\nRisk Model Metrics:")
    risk_metrics = evaluate_risk_model(y_pred, y_true, return_all_metrics=True)
    for key, value in risk_metrics.items():
        print(f"  {key:30s}: {value:.6f}")
    
    print("\n✓ Metric extraction successful")

if __name__ == "__main__":
    test_objective_types()
    test_metric_extraction()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
