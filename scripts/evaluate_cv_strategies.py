"""
CV Strategy Evaluation Script

이 스크립트는 4가지 CV 전략의 성능을 비교 평가합니다:
1. Time-based Split (기본)
2. Expanding Window
3. Purged Walk-Forward (현재 사용 중)
4. Regime-aware CV

각 전략마다 완전한 파이프라인을 실행:
- Return 모델 학습 → r_hat
- Risk 모델 학습 → sigma_hat  
- Position 전략 적용 → allocations
- Competition metric 계산 → final score

실제 캐글 제출과 동일한 방식으로 점수를 계산하여
CV 전략의 신뢰도를 평가합니다.

Usage:
    python scripts/evaluate_cv_strategies.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import json

from src.data import DataLoader
from src.features import FeatureEngineering
from src.cv import (
    create_cv_strategy,
    TimeBasedSplitCV,
    ExpandingWindowCV,
    PurgedWalkForwardCV,
    RegimeAwareCV
)
from src.metric import CompetitionMetric
from src.position import SharpeScalingMapper, QuantileBinningMapper
from src.utils import get_logger, Timer

logger = get_logger(log_file="logs/cv_evaluation.log", level="INFO")


class CVStrategyEvaluator:
    """
    CV 전략 평가 클래스.
    """
    
    def __init__(self, config_path: str = "conf/params.yaml"):
        self.config_path = config_path
        self.data_loader = DataLoader(config_path)
        self.feature_engineer = FeatureEngineering(config_path)
        self.metric_calc = CompetitionMetric()
        self.position_mapper = SharpeScalingMapper(config_path)
        
        # Simple model params
        self.return_model_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'verbosity': -1,
            'n_jobs': 1
        }
        
        self.risk_model_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 15,  # Simpler for risk
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'verbosity': -1,
            'n_jobs': 1
        }
        
        # Results storage
        self.results = {}
        
    def prepare_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """데이터 로드 및 간단한 전처리."""
        logger.info("="*80)
        logger.info("DATA PREPARATION")
        logger.info("="*80)
        
        # Load data
        train_df, _ = self.data_loader.load_data()
        logger.info(f"Loaded {len(train_df)} samples")
        
        # Basic preprocessing
        train_processed, _ = self.data_loader.preprocess_timeseries(
            train_df,
            handle_outliers=True,
            normalize=True,
            scale=True,
            window=60
        )
        
        # Simple feature engineering
        train_engineered = self.feature_engineer.fit_transform(train_processed)
        
        # Select top features (간소화)
        train_selected, selected_features = self.feature_engineer.select_features_by_importance(
            train_engineered,
            target_col='forward_returns',
            method='correlation',
            top_n=50  # 빠른 실험을 위해 50개만
        )
        
        logger.info(f"Selected {len(selected_features)} features")
        
        return train_selected, selected_features
    
    def train_return_model(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        train_idx: np.ndarray,
        val_idx: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Return 예측 모델 학습.
        
        Returns
        -------
        Tuple[np.ndarray, float]
            (Validation predictions for returns, RMSE)
        """
        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx]['forward_returns']
        X_val = df.iloc[val_idx][feature_cols]
        y_val = df.iloc[val_idx]['forward_returns']
        
        # Train return model
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            self.return_model_params,
            train_data,
            num_boost_round=100,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        # Predict
        r_hat = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - r_hat) ** 2))
        
        return r_hat, rmse
    
    def train_risk_model(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        train_idx: np.ndarray,
        val_idx: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Risk 예측 모델 학습 (realized volatility).
        
        Returns
        -------
        Tuple[np.ndarray, float]
            (Validation predictions for risk/volatility, RMSE)
        """
        # Calculate realized volatility as target
        # Use rolling std of forward_returns as proxy
        df_copy = df.copy()
        df_copy['realized_vol'] = (
            df_copy.groupby('date_id')['forward_returns']
            .transform(lambda x: x.rolling(window=20, min_periods=5).std())
        )
        df_copy['realized_vol'] = df_copy['realized_vol'].fillna(
            df_copy['realized_vol'].mean()
        )
        
        X_train = df_copy.iloc[train_idx][feature_cols]
        y_train = df_copy.iloc[train_idx]['realized_vol']
        X_val = df_copy.iloc[val_idx][feature_cols]
        y_val = df_copy.iloc[val_idx]['realized_vol']
        
        # Train risk model
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            self.risk_model_params,
            train_data,
            num_boost_round=100,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        # Predict
        sigma_hat = model.predict(X_val)
        # Ensure positive predictions
        sigma_hat = np.maximum(sigma_hat, 1e-6)
        rmse = np.sqrt(np.mean((y_val - sigma_hat) ** 2))
        
        return sigma_hat, rmse
    
    def calculate_allocations(
        self,
        r_hat: np.ndarray,
        sigma_hat: np.ndarray,
        k: float = 1.0,
        b: float = 2.0
    ) -> np.ndarray:
        """
        Position 전략 적용하여 allocations 계산.
        
        Parameters
        ----------
        r_hat : np.ndarray
            Return predictions
        sigma_hat : np.ndarray
            Risk predictions
        k : float
            Sharpe scaling factor
        b : float
            Risk aversion parameter
            
        Returns
        -------
        np.ndarray
            Allocations (0 to 2)
        """
        allocations = self.position_mapper.map_positions(
            r_hat=r_hat,
            sigma_hat=sigma_hat,
            k=k,
            b=b
        )
        
        return allocations
    
    def evaluate_strategy(
        self,
        strategy_name: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        n_runs: int = 5
    ) -> Dict:
        """
        특정 CV 전략 평가.
        
        Parameters
        ----------
        strategy_name : str
            CV 전략 이름
        df : pd.DataFrame
            데이터프레임
        feature_cols : List[str]
            특징 컬럼
        n_runs : int
            실험 반복 횟수
            
        Returns
        -------
        Dict
            평가 결과
        """
        logger.info("\n" + "="*80)
        logger.info(f"EVALUATING: {strategy_name.upper()}")
        logger.info("="*80)
        
        run_results = []
        
        for run_idx in range(n_runs):
            logger.info(f"\nRun {run_idx + 1}/{n_runs}")
            
            # Create CV strategy
            cv_strategy = create_cv_strategy(
                config_path=self.config_path,
                strategy=strategy_name
            )
            
            # Get folds
            folds = list(cv_strategy.get_folds(df))
            
            # Train on each fold
            fold_results = []
            oof_allocations = np.zeros(len(df))
            oof_r_hat = np.zeros(len(df))
            oof_sigma_hat = np.zeros(len(df))
            oof_mask = np.zeros(len(df), dtype=bool)
            
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                # Step 1: Train return model
                r_hat, return_rmse = self.train_return_model(
                    df, feature_cols, train_idx, val_idx
                )
                
                # Step 2: Train risk model
                sigma_hat, risk_rmse = self.train_risk_model(
                    df, feature_cols, train_idx, val_idx
                )
                
                # Step 3: Calculate allocations
                allocations = self.calculate_allocations(r_hat, sigma_hat)
                
                # Store OOF predictions
                oof_r_hat[val_idx] = r_hat
                oof_sigma_hat[val_idx] = sigma_hat
                oof_allocations[val_idx] = allocations
                oof_mask[val_idx] = True
                
                # Calculate fold metrics
                val_returns = df.iloc[val_idx]['forward_returns'].values
                val_rfr = df.iloc[val_idx]['risk_free_rate'].values if 'risk_free_rate' in df.columns else None
                
                fold_metric = self.metric_calc.calculate_score(
                    allocations=allocations,
                    forward_returns=val_returns,
                    risk_free_rate=val_rfr
                )
                
                fold_results.append({
                    'return_rmse': return_rmse,
                    'risk_rmse': risk_rmse,
                    'score': fold_metric['score'],
                    'sharpe': fold_metric['sharpe'],
                    'vol_penalty': fold_metric['vol_penalty'],
                    'return_penalty': fold_metric['return_penalty']
                })
                
                logger.info(
                    f"  Fold {fold_idx + 1}: "
                    f"Return RMSE={return_rmse:.6f}, "
                    f"Risk RMSE={risk_rmse:.6f}, "
                    f"Score={fold_metric['score']:.6f}"
                )
            
            # Calculate OOF metrics (complete pipeline)
            oof_returns = df['forward_returns'].values[oof_mask]
            oof_allocations_valid = oof_allocations[oof_mask]
            oof_rfr = df['risk_free_rate'].values[oof_mask] if 'risk_free_rate' in df.columns else None
            
            oof_metric = self.metric_calc.calculate_score(
                allocations=oof_allocations_valid,
                forward_returns=oof_returns,
                risk_free_rate=oof_rfr
            )
            
            # Calculate return/risk model performance
            oof_return_rmse = np.sqrt(
                np.mean((oof_returns - oof_r_hat[oof_mask]) ** 2)
            )
            
            run_results.append({
                'run': run_idx + 1,
                'n_folds': len(folds),
                'mean_fold_return_rmse': np.mean([f['return_rmse'] for f in fold_results]),
                'mean_fold_risk_rmse': np.mean([f['risk_rmse'] for f in fold_results]),
                'mean_fold_score': np.mean([f['score'] for f in fold_results]),
                'std_fold_score': np.std([f['score'] for f in fold_results]),
                'oof_return_rmse': oof_return_rmse,
                'oof_coverage': oof_mask.sum() / len(df),
                'oof_score': oof_metric['score'],
                'oof_sharpe': oof_metric['sharpe'],
                'oof_vol_penalty': oof_metric['vol_penalty'],
                'oof_return_penalty': oof_metric['return_penalty'],
                'oof_strategy_vol': oof_metric['strategy_vol'],
                'oof_market_vol': oof_metric['market_vol'],
                'oof_vol_ratio': oof_metric['vol_ratio']
            })
            
            logger.info(f"\n  📊 Run {run_idx + 1} Summary:")
            logger.info(f"     OOF Coverage: {oof_mask.sum() / len(df):.2%}")
            logger.info(f"     OOF Return RMSE: {oof_return_rmse:.6f}")
            logger.info(f"     OOF Competition Score: {oof_metric['score']:.6f}")
            logger.info(f"       → Sharpe: {oof_metric['sharpe']:.6f}")
            logger.info(f"       → Vol Penalty: {oof_metric['vol_penalty']:.4f}")
            logger.info(f"       → Return Penalty: {oof_metric['return_penalty']:.4f}")
            logger.info(f"     Strategy Vol / Market Vol: {oof_metric['vol_ratio']:.4f}")
        
        # Aggregate results
        summary = {
            'strategy': strategy_name,
            'n_runs': n_runs,
            # Return model metrics
            'avg_return_rmse': np.mean([r['oof_return_rmse'] for r in run_results]),
            'std_return_rmse': np.std([r['oof_return_rmse'] for r in run_results]),
            # Competition score metrics (가장 중요!)
            'avg_oof_score': np.mean([r['oof_score'] for r in run_results]),
            'std_oof_score': np.std([r['oof_score'] for r in run_results]),
            'avg_oof_sharpe': np.mean([r['oof_sharpe'] for r in run_results]),
            'avg_oof_vol_penalty': np.mean([r['oof_vol_penalty'] for r in run_results]),
            'avg_oof_return_penalty': np.mean([r['oof_return_penalty'] for r in run_results]),
            'avg_vol_ratio': np.mean([r['oof_vol_ratio'] for r in run_results]),
            # Fold-level metrics
            'avg_fold_score': np.mean([r['mean_fold_score'] for r in run_results]),
            'avg_coverage': np.mean([r['oof_coverage'] for r in run_results]),
            'runs': run_results
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"{strategy_name.upper()} - Final Summary")
        logger.info(f"{'='*80}")
        logger.info(f"OOF Competition Score: {summary['avg_oof_score']:.6f} ± {summary['std_oof_score']:.6f}")
        logger.info(f"  → Sharpe: {summary['avg_oof_sharpe']:.6f}")
        logger.info(f"  → Vol Penalty: {summary['avg_oof_vol_penalty']:.4f}")
        logger.info(f"  → Return Penalty: {summary['avg_oof_return_penalty']:.4f}")
        logger.info(f"Return RMSE: {summary['avg_return_rmse']:.6f} ± {summary['std_return_rmse']:.6f}")
        
        return summary
    
    def compare_all_strategies(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        n_runs: int = 5
    ) -> Dict:
        """
        모든 CV 전략 비교.
        
        Parameters
        ----------
        df : pd.DataFrame
            데이터프레임
        feature_cols : List[str]
            특징 컬럼
        n_runs : int
            각 전략당 실험 횟수
            
        Returns
        -------
        Dict
            비교 결과
        """
        strategies = [
            'time_based',
            'expanding_window',
            'purged_walk_forward',
            'regime_aware'
        ]
        
        results = {}
        
        for strategy in strategies:
            with Timer(f"{strategy} evaluation", logger):
                results[strategy] = self.evaluate_strategy(
                    strategy, df, feature_cols, n_runs
                )
        
        # Create comparison summary
        comparison = pd.DataFrame([
            {
                'Strategy': name,
                'OOF Score': f"{res['avg_oof_score']:.6f} ± {res['std_oof_score']:.6f}",
                'Sharpe': f"{res['avg_oof_sharpe']:.6f}",
                'Vol Penalty': f"{res['avg_oof_vol_penalty']:.4f}",
                'Return Penalty': f"{res['avg_oof_return_penalty']:.4f}",
                'Vol Ratio': f"{res['avg_vol_ratio']:.4f}",
                'Return RMSE': f"{res['avg_return_rmse']:.6f} ± {res['std_return_rmse']:.6f}",
                'Coverage': f"{res['avg_coverage']:.2%}"
            }
            for name, res in results.items()
        ])
        
        logger.info("\n" + "="*80)
        logger.info("CV STRATEGY COMPARISON - COMPLETE PIPELINE")
        logger.info("="*80)
        logger.info("\n📊 각 전략별 완전한 파이프라인 평가:")
        logger.info("   Return Model → Risk Model → Position Strategy → Competition Score")
        logger.info("\n" + comparison.to_string(index=False))
        
        # Find best strategy by OOF score
        best_strategy = max(results.items(), key=lambda x: x[1]['avg_oof_score'])
        logger.info(f"\n🏆 Best Strategy: {best_strategy[0].upper()}")
        logger.info(f"   Score: {best_strategy[1]['avg_oof_score']:.6f} ± {best_strategy[1]['std_oof_score']:.6f}")
        
        # Save results
        output_dir = Path("results/cv_evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        comparison.to_csv(output_dir / 'cv_strategy_comparison.csv', index=False)
        
        with open(output_dir / 'cv_strategy_details.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Results saved to {output_dir}")
        
        return results


def main():
    """Main evaluation pipeline."""
    
    evaluator = CVStrategyEvaluator()
    
    # Step 1: Prepare data
    df, feature_cols = evaluator.prepare_data()
    
    # Step 2: Compare all CV strategies
    results = evaluator.compare_all_strategies(
        df=df,
        feature_cols=feature_cols,
        n_runs=5  # 각 전략당 5번 실험
    )
    
    logger.info("\n✓ CV strategy evaluation complete!")


if __name__ == "__main__":
    main()
