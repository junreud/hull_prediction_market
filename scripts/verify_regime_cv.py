import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data import DataLoader
from src.cv import RegimeAwareCV

def verify_regimes():
    print("Loading data...")
    loader = DataLoader("conf/params.yaml")
    df, _ = loader.load_data()
    
    print("\nInitializing RegimeAwareCV...")
    cv = RegimeAwareCV(n_splits=5, lookback_window=60)
    
    print("Detecting regimes...")
    regimes = cv._detect_regimes(df)
    df['regime'] = regimes
    
    # Regime Mapping
    regime_names = {
        0: 'Low Vol / Bear',
        1: 'Low Vol / Neutral',
        2: 'Low Vol / Bull',
        3: 'High Vol / Bear',
        4: 'High Vol / Neutral',
        5: 'High Vol / Bull'
    }
    
    print("\nRegime Distribution:")
    counts = df['regime'].value_counts().sort_index()
    for r, count in counts.items():
        print(f"  {r} ({regime_names.get(r, 'Unknown')}): {count} samples ({count/len(df):.1%})")
        
    # Check splits
    print("\nChecking CV Splits:")
    for i, (train_idx, val_idx) in enumerate(cv.split(df)):
        train_regimes = df.iloc[train_idx]['regime']
        val_regimes = df.iloc[val_idx]['regime']
        
        print(f"\nFold {i+1}:")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Val:   {len(val_idx)} samples")
        
        # Check diversity in validation
        val_counts = val_regimes.value_counts()
        print(f"  Validation Regime Diversity: {len(val_counts)}/6 regimes present")
        if len(val_counts) < 3:
            print("  ⚠️  Warning: Low regime diversity in validation set!")

if __name__ == "__main__":
    verify_regimes()
