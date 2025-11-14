"""
Backtesting and simulation for strategy evaluation.

This module provides:
- Forward-looking bias detection
- Transaction cost modeling
- Performance simulation
- Detailed backtest reports
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.metric import CompetitionMetric, calculate_additional_metrics
from src.utils import get_logger, load_config

logger = get_logger(log_file="logs/backtest.log", level="INFO")


class BacktestSimulator:
    """
    Backtest simulator with transaction costs and bias detection.
    
    Features:
    - Forward-looking bias detection
    - Transaction cost modeling
    - Rolling performance metrics
    - Detailed performance reports
    """
    
    def __init__(
        self,
        transaction_cost_bps: float = 5.0,
        slippage_bps: float = 2.0,
        config_path: str = "conf/params.yaml"
    ):
        """
        Initialize backtest simulator.
        
        Parameters
        ----------
        transaction_cost_bps : float
            Transaction cost in basis points (default: 5 bps)
        slippage_bps : float
            Slippage cost in basis points (default: 2 bps)
        config_path : str
            Path to configuration file
        """
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.total_cost_bps = transaction_cost_bps + slippage_bps
        
        self.config = load_config(config_path)
        self.metric_calc = CompetitionMetric(
            vol_threshold=self.config.get('metric', {}).get('vol_threshold', 1.2),
            min_periods=30
        )
        
        logger.info("Backtest Simulator initialized")
        logger.info(f"Transaction cost: {transaction_cost_bps} bps")
        logger.info(f"Slippage: {slippage_bps} bps")
        logger.info(f"Total cost: {self.total_cost_bps} bps")
    
    def calculate_transaction_costs(
        self,
        allocations: np.ndarray
    ) -> np.ndarray:
        """
        Calculate transaction costs based on position changes.
        
        Parameters
        ----------
        allocations : np.ndarray
            Array of allocations over time
            
        Returns
        -------
        np.ndarray
            Transaction costs for each period (as percentage %)
        """
        # Calculate position changes
        position_changes = np.abs(np.diff(allocations, prepend=allocations[0]))
        
        # Cost = total_cost_bps * |position_change| / 10000
        # e.g., changing from 1.0 to 1.5 (0.5 change) costs 0.5 * 7bps = 3.5bps
        costs = position_changes * (self.total_cost_bps / 10000.0)
        
        return costs
    
    def apply_transaction_costs(
        self,
        allocations: np.ndarray,
        forward_returns: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply transaction costs to strategy returns.
        
        Parameters
        ----------
        allocations : np.ndarray
            Strategy allocations
        forward_returns : np.ndarray
            Market forward returns
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (net_returns, transaction_costs)
        """
        # Calculate gross returns
        gross_returns = allocations * forward_returns
        
        # Calculate transaction costs
        transaction_costs = self.calculate_transaction_costs(allocations)
        
        # Net returns = gross returns - transaction costs
        net_returns = gross_returns - transaction_costs
        
        return net_returns, transaction_costs
    
    def check_forward_looking_bias(
        self,
        data: pd.DataFrame,
        prediction_col: str = 'prediction',
        date_col: str = 'date_id'
    ) -> Dict[str, any]:
        """
        Check for forward-looking bias in predictions.
        
        This checks if predictions at time t use information from time t+1 or later.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with predictions and features
        prediction_col : str
            Column name for predictions
        date_col : str
            Column name for dates
            
        Returns
        -------
        dict
            Bias detection results
        """
        if prediction_col not in data.columns:
            logger.warning(f"Column {prediction_col} not found, skipping bias check")
            return {'bias_detected': False, 'reason': 'no_predictions'}
        
        # Sort by date
        data_sorted = data.sort_values(date_col).reset_index(drop=True)
        
        # Check 1: Are predictions available for all dates?
        # (If future info leaked, we might have predictions for all periods)
        n_predictions = data_sorted[prediction_col].notna().sum()
        n_total = len(data_sorted)
        prediction_rate = n_predictions / n_total
        
        # Check 2: Rolling correlation with future returns
        # If predictions are correlated with far-future returns, bias likely
        if 'forward_returns' in data_sorted.columns:
            data_sorted['future_return_5d'] = data_sorted['forward_returns'].shift(-5)
            data_sorted['future_return_10d'] = data_sorted['forward_returns'].shift(-10)
            
            # Calculate correlations
            valid_mask = data_sorted[[prediction_col, 'forward_returns', 
                                     'future_return_5d', 'future_return_10d']].notna().all(axis=1)
            
            if valid_mask.sum() > 30:
                corr_t0 = np.corrcoef(
                    data_sorted.loc[valid_mask, prediction_col],
                    data_sorted.loc[valid_mask, 'forward_returns']
                )[0, 1]
                
                corr_t5 = np.corrcoef(
                    data_sorted.loc[valid_mask, prediction_col],
                    data_sorted.loc[valid_mask, 'future_return_5d']
                )[0, 1]
                
                corr_t10 = np.corrcoef(
                    data_sorted.loc[valid_mask, prediction_col],
                    data_sorted.loc[valid_mask, 'future_return_10d']
                )[0, 1]
                
                # Suspicious if future correlations are higher than current
                bias_suspicious = (abs(corr_t5) > abs(corr_t0) * 1.2) or \
                                (abs(corr_t10) > abs(corr_t0) * 1.2)
            else:
                corr_t0, corr_t5, corr_t10 = np.nan, np.nan, np.nan
                bias_suspicious = False
        else:
            corr_t0, corr_t5, corr_t10 = np.nan, np.nan, np.nan
            bias_suspicious = False
        
        result = {
            'bias_detected': bias_suspicious,
            'prediction_rate': prediction_rate,
            'corr_t0': corr_t0,
            'corr_t5': corr_t5,
            'corr_t10': corr_t10,
            'n_predictions': n_predictions,
            'n_total': n_total
        }
        
        if bias_suspicious:
            logger.warning("⚠️  Forward-looking bias suspected!")
            logger.warning(f"   Correlation t+0: {corr_t0:.4f}")
            logger.warning(f"   Correlation t+5: {corr_t5:.4f}")
            logger.warning(f"   Correlation t+10: {corr_t10:.4f}")
        else:
            logger.info("✓ No obvious forward-looking bias detected")
        
        return result
    
    def run_backtest(
        self,
        allocations: np.ndarray,
        forward_returns: np.ndarray,
        dates: Optional[np.ndarray] = None,
        risk_free_rate: Optional[np.ndarray] = None,
        apply_costs: bool = True
    ) -> Dict[str, any]:
        """
        Run complete backtest with all metrics.
        
        Parameters
        ----------
        allocations : np.ndarray
            Strategy allocations
        forward_returns : np.ndarray
            Market forward returns
        dates : np.ndarray, optional
            Date identifiers
        risk_free_rate : np.ndarray, optional
            Risk-free rate
        apply_costs : bool
            Whether to apply transaction costs (default: True)
            
        Returns
        -------
        dict
            Comprehensive backtest results
        """
        logger.info("="*80)
        logger.info("Starting Backtest Simulation")
        logger.info("="*80)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(allocations) | np.isnan(forward_returns))
        allocations_clean = allocations[valid_mask]
        forward_returns_clean = forward_returns[valid_mask]
        
        if risk_free_rate is not None:
            risk_free_rate_clean = risk_free_rate[valid_mask]
        else:
            risk_free_rate_clean = None
        
        if dates is not None:
            dates_clean = dates[valid_mask]
        else:
            dates_clean = None
        
        logger.info(f"Valid samples: {len(allocations_clean)}")
        
        # Calculate returns with and without costs
        gross_returns = allocations_clean * forward_returns_clean
        
        if apply_costs:
            net_returns, transaction_costs = self.apply_transaction_costs(
                allocations_clean, forward_returns_clean
            )
            total_costs = np.sum(transaction_costs)
            avg_cost_per_trade = np.mean(transaction_costs[transaction_costs > 0]) \
                                 if np.any(transaction_costs > 0) else 0.0
        else:
            net_returns = gross_returns
            transaction_costs = np.zeros_like(gross_returns)
            total_costs = 0.0
            avg_cost_per_trade = 0.0
        
        # Calculate metrics without costs
        logger.info("\n--- Performance WITHOUT Transaction Costs ---")
        gross_metrics = self.metric_calc.calculate_score(
            allocations_clean,
            forward_returns_clean,
            market_returns=forward_returns_clean,
            risk_free_rate=risk_free_rate_clean
        )
        self._print_metrics(gross_metrics, "Gross")
        
        gross_additional = calculate_additional_metrics(
            allocations_clean,
            forward_returns_clean
        )
        
        # Calculate metrics with costs
        if apply_costs:
            logger.info("\n--- Performance WITH Transaction Costs ---")
            # For net metrics, we need to recalculate using net returns
            net_allocations = np.ones_like(net_returns)  # Net returns already include allocation
            net_metrics = self.metric_calc.calculate_score(
                net_allocations,
                net_returns,
                market_returns=forward_returns_clean,
                risk_free_rate=risk_free_rate_clean
            )
            self._print_metrics(net_metrics, "Net")
            
            # Transaction cost impact
            cost_impact = {
                'total_cost': total_costs,
                'avg_cost_per_trade': avg_cost_per_trade,
                'n_trades': np.sum(transaction_costs > 0),
                'score_impact': gross_metrics['score'] - net_metrics['score'],
                'sharpe_impact': gross_metrics['sharpe'] - net_metrics['sharpe']
            }
            
            logger.info("\n--- Transaction Cost Impact ---")
            logger.info(f"Total cost: {total_costs:.6f} ({total_costs*100:.4f}%)")
            logger.info(f"Number of trades: {cost_impact['n_trades']}")
            logger.info(f"Avg cost per trade: {avg_cost_per_trade:.6f}")
            logger.info(f"Score impact: {cost_impact['score_impact']:.6f}")
            logger.info(f"Sharpe impact: {cost_impact['sharpe_impact']:.6f}")
        else:
            net_metrics = gross_metrics
            cost_impact = None
        
        # Compile results
        results = {
            'gross_metrics': gross_metrics,
            'net_metrics': net_metrics,
            'additional_metrics': gross_additional,
            'cost_impact': cost_impact,
            'summary': {
                'n_samples': len(allocations_clean),
                'date_range': (dates_clean[0], dates_clean[-1]) if dates_clean is not None else None,
                'avg_allocation': np.mean(allocations_clean),
                'allocation_std': np.std(allocations_clean),
                'turnover': gross_additional['turnover'],
                'leverage_2x_rate': gross_additional['leverage_2x_rate']
            }
        }
        
        logger.info("\n" + "="*80)
        logger.info("Backtest Complete")
        logger.info("="*80)
        
        return results
    
    def _print_metrics(self, metrics: dict, prefix: str):
        """Print metrics in formatted way."""
        logger.info(f"\n{prefix} Performance:")
        logger.info(f"  Score:           {metrics['score']:.6f}")
        logger.info(f"  Sharpe:          {metrics['sharpe']:.6f}")
        logger.info(f"  Vol Penalty:     {metrics['vol_penalty']:.6f}")
        logger.info(f"  Vol Ratio:       {metrics['vol_ratio']:.6f}")
        logger.info(f"  Mean Return:     {metrics['mean_return']:.6f}")
    
    def generate_report(
        self,
        results: dict,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate detailed backtest report.
        
        Parameters
        ----------
        results : dict
            Backtest results from run_backtest
        output_path : str, optional
            Path to save report CSV
            
        Returns
        -------
        pd.DataFrame
            Report DataFrame
        """
        report_data = []
        
        # Gross metrics
        gross = results['gross_metrics']
        report_data.append({
            'Metric': 'Score (Gross)',
            'Value': gross['score']
        })
        report_data.append({
            'Metric': 'Sharpe (Gross)',
            'Value': gross['sharpe']
        })
        
        # Net metrics
        net = results['net_metrics']
        report_data.append({
            'Metric': 'Score (Net)',
            'Value': net['score']
        })
        report_data.append({
            'Metric': 'Sharpe (Net)',
            'Value': net['sharpe']
        })
        
        # Cost impact
        if results['cost_impact'] is not None:
            report_data.append({
                'Metric': 'Total Transaction Cost (%)',
                'Value': results['cost_impact']['total_cost'] * 100
            })
            report_data.append({
                'Metric': 'Number of Trades',
                'Value': results['cost_impact']['n_trades']
            })
        
        # Additional metrics
        add = results['additional_metrics']
        report_data.append({
            'Metric': 'Max Drawdown',
            'Value': add['max_drawdown']
        })
        report_data.append({
            'Metric': 'Calmar Ratio',
            'Value': add['calmar_ratio']
        })
        report_data.append({
            'Metric': 'Turnover',
            'Value': add['turnover']
        })
        report_data.append({
            'Metric': '2x Leverage Rate (%)',
            'Value': add['leverage_2x_rate'] * 100
        })
        
        # Summary
        summ = results['summary']
        report_data.append({
            'Metric': 'Avg Allocation',
            'Value': summ['avg_allocation']
        })
        report_data.append({
            'Metric': 'Vol Ratio',
            'Value': gross['vol_ratio']
        })
        
        report_df = pd.DataFrame(report_data)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            report_df.to_csv(output_path, index=False)
            logger.info(f"Report saved to {output_path}")
        
        return report_df


def create_backtest_simulator(
    config_path: str = "conf/params.yaml"
) -> BacktestSimulator:
    """
    Factory function to create backtest simulator.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
        
    Returns
    -------
    BacktestSimulator
        Configured simulator instance
    """
    config = load_config(config_path)
    
    # Get transaction cost settings
    backtest_config = config.get('backtest', {})
    transaction_cost = backtest_config.get('transaction_cost_bps', 5.0)
    slippage = backtest_config.get('slippage_bps', 2.0)
    
    return BacktestSimulator(
        transaction_cost_bps=transaction_cost,
        slippage_bps=slippage,
        config_path=config_path
    )
