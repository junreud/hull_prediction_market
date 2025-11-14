"""
Benchmark evaluation script.

This script:
1. Loads train data
2. Creates simple benchmark strategies (allocation=1.0 always)
3. Evaluates using custom metric
4. Generates performance report
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.data import DataLoader
from src.metric import CompetitionMetric, calculate_additional_metrics
from src.utils import get_logger, Timer, set_seed

# Initialize logger
logger = get_logger(log_file="logs/benchmark.log", level="INFO")


def run_benchmark(config_path: str = "conf/params.yaml"):
    """
    Run benchmark evaluation.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    """
    logger.info("="*80)
    logger.info("Starting Benchmark Evaluation")
    logger.info("="*80)
    
    # Set random seed
    set_seed(42)
    
    # Load data
    with Timer("Loading data", logger=logger):
        data_loader = DataLoader(config_path)
        train_df, _ = data_loader.load_data()
        logger.info(f"Train data shape: {train_df.shape}")
        logger.info(f"Date range: {train_df['date_id'].min()} to {train_df['date_id'].max()}")
    
    # Extract required columns
    forward_returns = train_df['forward_returns'].values
    risk_free_rate = train_df['risk_free_rate'].values if 'risk_free_rate' in train_df.columns else None
    
    # Remove NaN values
    valid_mask = ~np.isnan(forward_returns)
    forward_returns_clean = forward_returns[valid_mask]
    if risk_free_rate is not None:
        risk_free_rate_clean = risk_free_rate[valid_mask]
    else:
        risk_free_rate_clean = None
    
    logger.info(f"Valid samples: {len(forward_returns_clean)} / {len(forward_returns)}")
    
    # Initialize metric calculator
    metric_calc = CompetitionMetric(
        vol_threshold=1.2,
        underperformance_penalty=False,
        min_periods=30
    )
    
    # Benchmark 1: Always fully invested (allocation = 1.0)
    logger.info("\n" + "="*80)
    logger.info("Benchmark 1: Always Fully Invested (allocation = 1.0)")
    logger.info("="*80)
    
    allocations_1 = np.ones_like(forward_returns_clean)
    
    with Timer("Calculating metrics", logger=logger):
        result_1 = metric_calc.calculate_score(
            allocations_1,
            forward_returns_clean,
            market_returns=forward_returns_clean,
            risk_free_rate=risk_free_rate_clean
        )
    
    print_metrics(result_1, "Benchmark 1")
    
    # Calculate additional metrics
    additional_1 = calculate_additional_metrics(
        allocations_1,
        forward_returns_clean
    )
    print_additional_metrics(additional_1, "Benchmark 1")
    
    # Benchmark 2: Conservative (allocation = 0.5)
    logger.info("\n" + "="*80)
    logger.info("Benchmark 2: Conservative Strategy (allocation = 0.5)")
    logger.info("="*80)
    
    allocations_2 = np.ones_like(forward_returns_clean) * 0.5
    
    result_2 = metric_calc.calculate_score(
        allocations_2,
        forward_returns_clean,
        market_returns=forward_returns_clean,
        risk_free_rate=risk_free_rate_clean
    )
    
    print_metrics(result_2, "Benchmark 2")
    
    additional_2 = calculate_additional_metrics(
        allocations_2,
        forward_returns_clean
    )
    print_additional_metrics(additional_2, "Benchmark 2")
    
    # Benchmark 3: Aggressive (allocation = 1.5)
    logger.info("\n" + "="*80)
    logger.info("Benchmark 3: Aggressive Strategy (allocation = 1.5)")
    logger.info("="*80)
    
    allocations_3 = np.ones_like(forward_returns_clean) * 1.5
    
    result_3 = metric_calc.calculate_score(
        allocations_3,
        forward_returns_clean,
        market_returns=forward_returns_clean,
        risk_free_rate=risk_free_rate_clean
    )
    
    print_metrics(result_3, "Benchmark 3")
    
    additional_3 = calculate_additional_metrics(
        allocations_3,
        forward_returns_clean
    )
    print_additional_metrics(additional_3, "Benchmark 3")
    
    # Compare benchmarks
    logger.info("\n" + "="*80)
    logger.info("Benchmark Comparison")
    logger.info("="*80)
    
    comparison = pd.DataFrame({
        'Allocation': [1.0, 0.5, 1.5],
        'Score': [result_1['score'], result_2['score'], result_3['score']],
        'Sharpe': [result_1['sharpe'], result_2['sharpe'], result_3['sharpe']],
        'Vol_Penalty': [result_1['vol_penalty'], result_2['vol_penalty'], result_3['vol_penalty']],
        'Vol_Ratio': [result_1['vol_ratio'], result_2['vol_ratio'], result_3['vol_ratio']],
        'Mean_Return': [result_1['mean_return'], result_2['mean_return'], result_3['mean_return']],
        'Max_Drawdown': [additional_1['max_drawdown'], additional_2['max_drawdown'], additional_3['max_drawdown']],
        'Calmar': [additional_1['calmar_ratio'], additional_2['calmar_ratio'], additional_3['calmar_ratio']]
    })
    
    logger.info(f"\n{comparison.to_string(index=False)}")
    
    # Save results
    output_dir = Path("results/benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison.to_csv(output_dir / "benchmark_comparison.csv", index=False)
    logger.info(f"\nResults saved to {output_dir / 'benchmark_comparison.csv'}")
    
    logger.info("\n" + "="*80)
    logger.info("Benchmark Evaluation Complete")
    logger.info("="*80)


def print_metrics(result: dict, name: str):
    """Print metric results."""
    logger.info(f"\n{name} Results:")
    logger.info(f"  Score (penalized):  {result['score']:.6f}")
    logger.info(f"  Sharpe Ratio:       {result['sharpe']:.6f}")
    logger.info(f"  Vol Penalty:        {result['vol_penalty']:.6f}")
    logger.info(f"  Strategy Vol:       {result['strategy_vol']:.6f}")
    logger.info(f"  Market Vol:         {result['market_vol']:.6f}")
    logger.info(f"  Vol Ratio:          {result['vol_ratio']:.6f}")
    logger.info(f"  Mean Return:        {result['mean_return']:.6f}")
    logger.info(f"  Std Return:         {result['std_return']:.6f}")
    logger.info(f"  Valid Samples:      {result['n_valid']}")


def print_additional_metrics(metrics: dict, name: str):
    """Print additional metrics."""
    logger.info(f"\n{name} Additional Metrics:")
    logger.info(f"  Max Drawdown:       {metrics['max_drawdown']:.6f}")
    logger.info(f"  Calmar Ratio:       {metrics['calmar_ratio']:.6f}")
    logger.info(f"  Turnover:           {metrics['turnover']:.6f}")
    logger.info(f"  2x Leverage Rate:   {metrics['leverage_2x_rate']:.4%}")
    logger.info(f"  Avg Allocation:     {metrics['avg_allocation']:.6f}")


if __name__ == "__main__":
    run_benchmark()
