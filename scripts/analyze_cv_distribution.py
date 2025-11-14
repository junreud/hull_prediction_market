"""
Analyze CV fold distribution similarity.

This script:
1. Checks train/valid distribution similarity for each fold
2. Identifies distribution shifts
3. Suggests improvements
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from src.data import DataLoader
from src.cv import CVStrategy
from src.utils import get_logger

logger = get_logger(log_file="logs/cv_analysis.log", level="INFO")


def analyze_fold_distributions():
    """Analyze distribution similarity across CV folds."""
    logger.info("="*80)
    logger.info("CV Fold Distribution Analysis")
    logger.info("="*80)
    
    # Load data
    loader = DataLoader()
    train_df, _ = loader.load_data()
    
    # Get CV folds
    cv = CVStrategy()
    
    # Collect statistics
    fold_stats = []
    ks_tests = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.get_folds(train_df)):
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        # Target statistics
        train_mean = train_fold['forward_returns'].mean()
        train_std = train_fold['forward_returns'].std()
        train_skew = train_fold['forward_returns'].skew()
        train_kurt = train_fold['forward_returns'].kurtosis()
        
        val_mean = val_fold['forward_returns'].mean()
        val_std = val_fold['forward_returns'].std()
        val_skew = val_fold['forward_returns'].skew()
        val_kurt = val_fold['forward_returns'].kurtosis()
        
        # KS test (distribution similarity)
        ks_stat, ks_pvalue = stats.ks_2samp(
            train_fold['forward_returns'].dropna(),
            val_fold['forward_returns'].dropna()
        )
        
        # T-test (mean difference)
        t_stat, t_pvalue = stats.ttest_ind(
            train_fold['forward_returns'].dropna(),
            val_fold['forward_returns'].dropna()
        )
        
        fold_stats.append({
            'fold': fold_idx + 1,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'train_mean': train_mean,
            'val_mean': val_mean,
            'mean_diff': abs(train_mean - val_mean),
            'mean_diff_pct': abs(train_mean - val_mean) / abs(train_mean) * 100 if train_mean != 0 else 0,
            'train_std': train_std,
            'val_std': val_std,
            'std_diff': abs(train_std - val_std),
            'std_diff_pct': abs(train_std - val_std) / train_std * 100,
            'train_skew': train_skew,
            'val_skew': val_skew,
            'train_kurt': train_kurt,
            'val_kurt': val_kurt,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_pvalue,
            't_pvalue': t_pvalue
        })
        
        ks_tests.append({
            'fold': fold_idx + 1,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'is_similar': ks_pvalue > 0.05  # p > 0.05 means distributions are similar
        })
    
    # Create DataFrames
    df_stats = pd.DataFrame(fold_stats)
    df_ks = pd.DataFrame(ks_tests)
    
    # Print detailed statistics
    logger.info("\n" + "="*80)
    logger.info("Fold-by-Fold Statistics")
    logger.info("="*80)
    
    for _, row in df_stats.iterrows():
        logger.info(f"\nFold {int(row['fold'])}:")
        logger.info(f"  Samples: Train={row['train_samples']}, Val={row['val_samples']}")
        logger.info(f"  Mean: Train={row['train_mean']:.6f}, Val={row['val_mean']:.6f}, Diff={row['mean_diff_pct']:.2f}%")
        logger.info(f"  Std: Train={row['train_std']:.6f}, Val={row['val_std']:.6f}, Diff={row['std_diff_pct']:.2f}%")
        logger.info(f"  Skew: Train={row['train_skew']:.4f}, Val={row['val_skew']:.4f}")
        logger.info(f"  Kurt: Train={row['train_kurt']:.4f}, Val={row['val_kurt']:.4f}")
        logger.info(f"  KS test: stat={row['ks_stat']:.4f}, p-value={row['ks_pvalue']:.4f}")
        logger.info(f"  T-test p-value: {row['t_pvalue']:.4f}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("Summary")
    logger.info("="*80)
    logger.info(f"Average mean difference: {df_stats['mean_diff'].mean():.6f} ({df_stats['mean_diff_pct'].mean():.2f}%)")
    logger.info(f"Average std difference: {df_stats['std_diff'].mean():.6f} ({df_stats['std_diff_pct'].mean():.2f}%)")
    logger.info(f"Max mean difference: {df_stats['mean_diff'].max():.6f} (Fold {df_stats['mean_diff'].idxmax() + 1})")
    logger.info(f"Max std difference: {df_stats['std_diff'].max():.6f} (Fold {df_stats['std_diff'].idxmax() + 1})")
    
    # Distribution similarity
    logger.info("\n" + "="*80)
    logger.info("Distribution Similarity (KS Test)")
    logger.info("="*80)
    
    similar_count = df_ks['is_similar'].sum()
    logger.info(f"Folds with similar distributions (p > 0.05): {similar_count}/{len(df_ks)}")
    
    for _, row in df_ks.iterrows():
        status = "✓ SIMILAR" if row['is_similar'] else "✗ DIFFERENT"
        logger.info(f"  Fold {int(row['fold'])}: {status} (p={row['ks_pvalue']:.4f})")
    
    # Recommendations
    logger.info("\n" + "="*80)
    logger.info("Recommendations")
    logger.info("="*80)
    
    if similar_count >= len(df_ks) * 0.8:
        logger.info("✓ Overall distribution similarity is GOOD (≥80% folds similar)")
    elif similar_count >= len(df_ks) * 0.6:
        logger.info("⚠ Distribution similarity is MODERATE (60-80% folds similar)")
    else:
        logger.info("✗ Distribution similarity is POOR (<60% folds similar)")
    
    # Specific recommendations
    if df_stats['std_diff_pct'].max() > 50:
        logger.info("\n⚠ High volatility difference detected in some folds:")
        logger.info("  Recommendation 1: Consider using Regime-based CV (stratify by volatility)")
        logger.info("  Recommendation 2: Add volatility features to capture regime changes")
    
    if df_stats['mean_diff_pct'].max() > 50:
        logger.info("\n⚠ High mean difference detected in some folds:")
        logger.info("  Recommendation: Check for structural breaks or regime shifts")
    
    if not df_ks['is_similar'].all():
        logger.info("\n⚠ Some folds have significantly different distributions:")
        logger.info("  Recommendation: Increase n_splits or use expanding window CV")
    
    # Save results
    df_stats.to_csv('results/cv_fold_statistics.csv', index=False)
    logger.info("\n✓ Results saved to results/cv_fold_statistics.csv")
    
    return df_stats, df_ks


def plot_fold_distributions(train_df):
    """Plot distributions for each fold."""
    cv = CVStrategy()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.get_folds(train_df)):
        if fold_idx >= 4:
            break
        
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        ax = axes[fold_idx]
        
        # Plot distributions
        ax.hist(train_fold['forward_returns'].dropna(), bins=50, alpha=0.5, 
                label=f'Train (μ={train_fold["forward_returns"].mean():.4f})', density=True)
        ax.hist(val_fold['forward_returns'].dropna(), bins=50, alpha=0.5, 
                label=f'Val (μ={val_fold["forward_returns"].mean():.4f})', density=True)
        
        ax.set_title(f'Fold {fold_idx + 1}')
        ax.set_xlabel('Forward Returns')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/cv_fold_distributions.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Distribution plot saved to results/cv_fold_distributions.png")
    plt.close()


if __name__ == "__main__":
    # Analyze distributions
    df_stats, df_ks = analyze_fold_distributions()
    
    # Plot distributions
    loader = DataLoader()
    train_df, _ = loader.load_data()
    plot_fold_distributions(train_df)
    
    logger.info("\n" + "="*80)
    logger.info("Analysis Complete!")
    logger.info("="*80)
