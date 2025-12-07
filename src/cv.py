"""
Cross-validation strategies for time series prediction.

This module provides:
- Time-aware cross-validation with purge and embargo
- Walk-forward validation
- Expanding window validation
- Regime-aware cross-validation
- Evaluation metrics computation
"""

from typing import Tuple, List, Iterator, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from src.utils import get_logger, load_config

logger = get_logger(log_file="logs/prediction_market.log", level="INFO")


class PurgedWalkForwardCV(BaseCrossValidator):
    """
    Time series cross-validator with purge and embargo.
    
    This implements a walk-forward validation strategy that:
    1. Respects temporal ordering (no data leakage)
    2. Purges overlapping samples between train and validation
    3. Applies embargo period after validation to prevent look-ahead bias
    
    Parameters
    ----------
    n_splits : int
        Number of folds
    embargo : int
        Number of periods to embargo after validation set
    purge : bool
        Whether to purge overlapping periods
    purge_period : int
        Number of days to remove from train end (default: 15)
    train_ratio : float
        Ratio of training data per fold (remaining goes to validation)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo: int = 5,
        purge: bool = True,
        purge_period: int = 5,
        train_ratio: float = 0.8
    ):
        self.n_splits = n_splits
        self.embargo = embargo
        self.purge = purge
        self.purge_period = purge_period
        self.train_ratio = train_ratio
        
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data with date_id index or column
        y : pd.Series, optional
            Target variable
        groups : pd.Series, optional
            Group labels (date_id) for the samples
            
        Yields
        ------
        train : np.ndarray
            Training set indices
        test : np.ndarray
            Testing set indices
        """
        # Get date_id information
        if 'date_id' in X.columns:
            dates = X['date_id'].values
        elif isinstance(X.index, pd.Index) and X.index.name == 'date_id':
            dates = X.index.values
        elif groups is not None:
            dates = groups.values
        else:
            raise ValueError("date_id must be in columns, index, or provided as groups")
        
        # Get unique dates sorted
        unique_dates = np.sort(np.unique(dates)) # 고유 날짜 개수
        n_dates = len(unique_dates)
        
        # Calculate fold size
        fold_size = n_dates // self.n_splits
        
        if fold_size < 2: # 폴드 수를 너무 적게 하지 않는 코드
            raise ValueError(
                f"n_splits={self.n_splits} is too large for {n_dates} unique dates. "
                f"Reduce n_splits or use more data."
            )
        
        logger.info(f"Starting Purged Walk-Forward CV with {self.n_splits} splits")
        logger.info(f"Total dates: {n_dates}, Fold size: {fold_size}")
        logger.info(f"Embargo: {self.embargo}, Purge: {self.purge}, Purge period: {self.purge_period}, Train ratio: {self.train_ratio}")
        
        # Reserve initial dates for minimum training data
        min_train_dates = fold_size  # Use one fold size as minimum training data
        
        for fold_idx in range(self.n_splits):
            # Define validation period for this fold
            # Start validation after minimum training period
            val_start_idx = min_train_dates + fold_idx * fold_size
            val_end_idx = min(val_start_idx + fold_size, n_dates)
            
            # Skip if not enough data
            if val_start_idx >= n_dates - 1 or val_end_idx > n_dates:
                break
                
            # Calculate training period
            train_start_idx = 0
            # For walk-forward CV, use all data before validation period
            train_end_idx = val_start_idx
            
            # Ensure we have enough training data
            if train_end_idx <= train_start_idx:
                logger.warning(f"Fold {fold_idx + 1}: Not enough training data, skipping")
                continue
            
            # Get date ranges
            train_dates = unique_dates[train_start_idx:train_end_idx]
            val_dates = unique_dates[val_start_idx:val_end_idx]
            
            # Apply purge: remove dates near the boundary between train and validation
            purge_buffer = 0
            if self.purge:
                # Remove last few dates from training to avoid overlap
                # This prevents data leakage when samples might span multiple dates
                if len(train_dates) > self.purge_period:
                    train_dates = train_dates[:-self.purge_period]
                    purge_buffer = self.purge_period
                elif len(train_dates) > 0:
                    # If train_dates is smaller than purge_period, remove at least 1
                    train_dates = train_dates[:-1]
                    purge_buffer = 1
            
            # Apply embargo: remove dates immediately after validation from future training
            # This is handled implicitly in walk-forward CV since we only use past data
            # But we log it for transparency
            if self.embargo > 0:
                embargo_end_idx = min(val_end_idx + self.embargo, n_dates)
                embargo_dates = unique_dates[val_end_idx:embargo_end_idx]
            else:
                embargo_dates = np.array([])
            
            # Convert dates to indices
            train_idx = np.where(np.isin(dates, train_dates))[0]
            val_idx = np.where(np.isin(dates, val_dates))[0]
            
            # Log fold information
            embargo_info = f", Embargo: {len(embargo_dates)} dates" if len(embargo_dates) > 0 else ""
            purge_info = f", Purged: {purge_buffer} dates from train end" if purge_buffer > 0 else ""
            logger.info(
                f"Fold {fold_idx + 1}/{self.n_splits}: "
                f"Train dates: {len(train_dates)} ({train_dates[0]} to {train_dates[-1]}), "
                f"Val dates: {len(val_dates)} ({val_dates[0]} to {val_dates[-1]}), "
                f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}"
                f"{purge_info}{embargo_info}"
            )
            
            yield train_idx, val_idx
    
    def get_n_splits(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None
    ) -> int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits


class TimeBasedSplitCV(BaseCrossValidator):
    """
    Simple time-based split CV (기본).
    
    데이터를 시간 순서대로 n_splits개로 나누고,
    각 fold마다 이전 데이터는 train, 해당 구간은 validation으로 사용.
    
    가장 단순하고 빠른 방식. Purge/Embargo 없음.
    
    Parameters
    ----------
    n_splits : int
        Number of folds
    min_train_size : int
        Minimum number of dates for training (default: 252 = 1 year)
    """
    
    def __init__(self, n_splits: int = 5, min_train_size: int = 252):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation splits."""
        # Get dates
        if 'date_id' in X.columns:
            dates = X['date_id'].values
        elif isinstance(X.index, pd.Index) and X.index.name == 'date_id':
            dates = X.index.values
        elif groups is not None:
            dates = groups.values
        else:
            raise ValueError("date_id must be in columns, index, or provided as groups")
        
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        
        # Calculate fold size
        fold_size = (n_dates - self.min_train_size) // self.n_splits
        
        if fold_size < 10:
            raise ValueError(
                f"Not enough data for {self.n_splits} splits. "
                f"Total dates: {n_dates}, min_train: {self.min_train_size}"
            )
        
        logger.info(f"TimeBasedSplitCV: {self.n_splits} splits, fold_size={fold_size}")
        
        for fold_idx in range(self.n_splits):
            # Validation period
            val_start_idx = self.min_train_size + fold_idx * fold_size
            val_end_idx = min(val_start_idx + fold_size, n_dates)
            
            if val_start_idx >= n_dates:
                break
            
            # Training period: all data before validation
            train_dates = unique_dates[:val_start_idx]
            val_dates = unique_dates[val_start_idx:val_end_idx]
            
            # Convert to indices
            train_idx = np.where(np.isin(dates, train_dates))[0]
            val_idx = np.where(np.isin(dates, val_dates))[0]
            
            logger.info(
                f"Fold {fold_idx + 1}/{self.n_splits}: "
                f"Train: {len(train_dates)} dates ({train_dates[0]}-{train_dates[-1]}), "
                f"Val: {len(val_dates)} dates ({val_dates[0]}-{val_dates[-1]})"
            )
            
            yield train_idx, val_idx
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class ExpandingWindowCV(BaseCrossValidator):
    """
    Expanding window CV (확장 윈도우).
    
    매 fold마다 validation은 고정 크기로 이동하고,
    training은 과거 데이터를 계속 누적 (expanding).
    
    Parameters
    ----------
    n_splits : int
        Number of folds
    val_size : int
        Validation window size in dates (default: 252 = 1 year)
    min_train_size : int
        Minimum training size in dates (default: 252 = 1 year)
    step_size : int
        Step size for moving validation window (default: val_size, no overlap)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        val_size: int = 252,
        min_train_size: int = 252,
        step_size: Optional[int] = None
    ):
        self.n_splits = n_splits
        self.val_size = val_size
        self.min_train_size = min_train_size
        self.step_size = step_size if step_size is not None else val_size
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate expanding window splits."""
        # Get dates
        if 'date_id' in X.columns:
            dates = X['date_id'].values
        elif isinstance(X.index, pd.Index) and X.index.name == 'date_id':
            dates = X.index.values
        elif groups is not None:
            dates = groups.values
        else:
            raise ValueError("date_id must be in columns, index, or provided as groups")
        
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        
        logger.info(
            f"ExpandingWindowCV: {self.n_splits} splits, "
            f"val_size={self.val_size}, step_size={self.step_size}"
        )
        
        fold_count = 0
        for fold_idx in range(self.n_splits):
            # Validation period
            val_start_idx = self.min_train_size + fold_idx * self.step_size
            val_end_idx = min(val_start_idx + self.val_size, n_dates)
            
            if val_start_idx >= n_dates or val_end_idx - val_start_idx < 10:
                break
            
            # Training: from start to val_start (expanding)
            train_dates = unique_dates[:val_start_idx]
            val_dates = unique_dates[val_start_idx:val_end_idx]
            
            # Convert to indices
            train_idx = np.where(np.isin(dates, train_dates))[0]
            val_idx = np.where(np.isin(dates, val_dates))[0]
            
            logger.info(
                f"Fold {fold_count + 1}: "
                f"Train: {len(train_dates)} dates ({train_dates[0]}-{train_dates[-1]}), "
                f"Val: {len(val_dates)} dates ({val_dates[0]}-{val_dates[-1]})"
            )
            
            yield train_idx, val_idx
            fold_count += 1
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class RegimeAwareCV(BaseCrossValidator):
    """
    Regime-aware CV (시장 regime 고려).
    
    각 fold가 다양한 시장 regime을 포함하도록 분할.
    Regime은 volatility 기반으로 자동 탐지:
    - High volatility regime
    - Normal volatility regime  
    - Low volatility regime
    
    Parameters
    ----------
    n_splits : int
        Number of folds
    regime_col : str
        Column name for pre-computed regime labels (optional)
        If None, will auto-detect using forward_returns volatility
    lookback_window : int
        Window for volatility calculation (default: 20)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        regime_col: Optional[str] = None,
        lookback_window: int = 20
    ):
        self.n_splits = n_splits
        self.regime_col = regime_col
        self.lookback_window = lookback_window
    
    def _detect_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """
        Detect market regimes based on volatility and trend.
        
        Regimes:
        0: Low Vol + Bear
        1: Low Vol + Neutral
        2: Low Vol + Bull
        3: High Vol + Bear
        4: High Vol + Neutral
        5: High Vol + Bull
        """
        if self.regime_col is not None and self.regime_col in df.columns:
            return df[self.regime_col].values
        
        # Use forward_returns for regime detection
        if 'forward_returns' not in df.columns:
            raise ValueError("forward_returns column required for regime detection")
        
        # 1. Calculate Volatility
        returns = df.groupby('date_id')['forward_returns'].mean()
        rolling_vol = returns.rolling(window=self.lookback_window, min_periods=5).std()
        
        # 2. Calculate Trend (Cumulative return over window)
        rolling_return = returns.rolling(window=self.lookback_window, min_periods=5).sum()
        
        # Get values
        vol_values = rolling_vol.values
        ret_values = rolling_return.values
        
        # Volatility Thresholds (Median split for simplicity, or 33/67)
        # Using median split for Volatility (Low/High) to keep total regimes manageable (2x3=6)
        vol_median = np.nanmedian(vol_values)
        
        # Trend Thresholds (33/67 split for Bear/Neutral/Bull)
        ret_q33 = np.nanpercentile(ret_values, 33)
        ret_q67 = np.nanpercentile(ret_values, 67)
        
        # Assign Regimes
        # Structure: Vol (0=Low, 1=High) * 3 + Trend (0=Bear, 1=Neutral, 2=Bull)
        regimes = np.zeros(len(rolling_vol), dtype=int)
        
        # Low Volatility (0-2)
        low_vol_mask = vol_values <= vol_median
        regimes[low_vol_mask & (ret_values <= ret_q33)] = 0      # Low Vol + Bear
        regimes[low_vol_mask & (ret_values > ret_q33) & (ret_values <= ret_q67)] = 1  # Low Vol + Neutral
        regimes[low_vol_mask & (ret_values > ret_q67)] = 2      # Low Vol + Bull
        
        # High Volatility (3-5)
        high_vol_mask = vol_values > vol_median
        regimes[high_vol_mask & (ret_values <= ret_q33)] = 3     # High Vol + Bear
        regimes[high_vol_mask & (ret_values > ret_q33) & (ret_values <= ret_q67)] = 4 # High Vol + Neutral
        regimes[high_vol_mask & (ret_values > ret_q67)] = 5     # High Vol + Bull
        
        # Map back to original dataframe
        regime_map = dict(zip(returns.index, regimes))
        # Fill NaN (start of series) with Neutral (Low Vol + Neutral = 1)
        df_regimes = df['date_id'].map(regime_map).fillna(1).astype(int).values
        
        return df_regimes
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate regime-aware splits."""
        # Get dates
        if 'date_id' in X.columns:
            dates = X['date_id'].values
        else:
            raise ValueError("date_id must be in columns for RegimeAwareCV")
        
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        
        # Detect regimes
        regimes = self._detect_regimes(X)
        
        # Get regime for each date
        date_regimes = {}
        for date in unique_dates:
            date_mask = dates == date
            date_regimes[date] = regimes[date_mask][0]
        
        # Count regime distribution
        regime_counts = pd.Series(list(date_regimes.values())).value_counts()
        logger.info(f"RegimeAwareCV: Regime distribution - {regime_counts.to_dict()}")
        
        # Create folds ensuring each has diverse regimes
        fold_size = n_dates // self.n_splits
        
        logger.info(f"RegimeAwareCV: {self.n_splits} splits, fold_size={fold_size}")
        
        for fold_idx in range(self.n_splits):
            # Validation period
            val_start_idx = fold_idx * fold_size
            val_end_idx = min(val_start_idx + fold_size, n_dates)
            
            if fold_idx == self.n_splits - 1:
                val_end_idx = n_dates  # Last fold takes remaining data
            
            # Training: all other folds
            train_dates = np.concatenate([
                unique_dates[:val_start_idx],
                unique_dates[val_end_idx:]
            ])
            val_dates = unique_dates[val_start_idx:val_end_idx]
            
            # Skip if not enough data
            if len(train_dates) < 100 or len(val_dates) < 10:
                continue
            
            # Check regime diversity
            train_regimes = [date_regimes[d] for d in train_dates]
            val_regimes = [date_regimes[d] for d in val_dates]
            train_regime_dist = pd.Series(train_regimes).value_counts()
            val_regime_dist = pd.Series(val_regimes).value_counts()
            
            # Convert to indices
            train_idx = np.where(np.isin(dates, train_dates))[0]
            val_idx = np.where(np.isin(dates, val_dates))[0]
            
            logger.info(
                f"Fold {fold_idx + 1}/{self.n_splits}: "
                f"Train: {len(train_dates)} dates, Val: {len(val_dates)} dates | "
                f"Train regimes: {train_regime_dist.to_dict()}, "
                f"Val regimes: {val_regime_dist.to_dict()}"
            )
            
            yield train_idx, val_idx
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class CVStrategy:
    """
    Cross-validation strategy manager.
    
    Supports multiple CV strategies:
    - 'purged_walk_forward': PurgedWalkForwardCV (default)
    - 'time_based': TimeBasedSplitCV
    - 'expanding_window': ExpandingWindowCV
    - 'regime_aware': RegimeAwareCV
    
    This class handles:
    - CV configuration from params.yaml
    - Fold generation
    - Metric aggregation across folds
    """
    
    def __init__(
        self,
        config_path: str = "conf/params.yaml",
        strategy: str = "purged_walk_forward"
    ):
        """
        Initialize CV strategy with configuration.
        
        Parameters
        ----------
        config_path : str
            Path to configuration file
        strategy : str
            CV strategy name:
            - 'purged_walk_forward' (default)
            - 'time_based'
            - 'expanding_window'
            - 'regime_aware'
        """
        self.config = load_config(config_path)
        self.cv_config = self.config.get('cv', {})
        self.strategy = strategy
        
        # CV parameters
        self.n_splits = self.cv_config.get('n_splits', 5)
        
        # Initialize CV splitter based on strategy
        if strategy == "purged_walk_forward":
            self.embargo = self.cv_config.get('embargo', 5)
            self.purge = self.cv_config.get('purge', True)
            self.purge_period = self.cv_config.get('purge_period', 5)
            self.train_ratio = self.cv_config.get('train_ratio', 0.8)
            
            self.cv_splitter = PurgedWalkForwardCV(
                n_splits=self.n_splits,
                embargo=self.embargo,
                purge=self.purge,
                purge_period=self.purge_period,
                train_ratio=self.train_ratio
            )
            
        elif strategy == "time_based":
            min_train_size = self.cv_config.get('min_train_size', 252)
            self.cv_splitter = TimeBasedSplitCV(
                n_splits=self.n_splits,
                min_train_size=min_train_size
            )
            
        elif strategy == "expanding_window":
            val_size = self.cv_config.get('val_size', 252)
            min_train_size = self.cv_config.get('min_train_size', 252)
            step_size = self.cv_config.get('step_size', None)
            self.cv_splitter = ExpandingWindowCV(
                n_splits=self.n_splits,
                val_size=val_size,
                min_train_size=min_train_size,
                step_size=step_size
            )
            
        elif strategy == "regime_aware":
            regime_col = self.cv_config.get('regime_col', None)
            lookback_window = self.cv_config.get('lookback_window', 20)
            self.cv_splitter = RegimeAwareCV(
                n_splits=self.n_splits,
                regime_col=regime_col,
                lookback_window=lookback_window
            )
            
        else:
            raise ValueError(
                f"Unknown CV strategy: {strategy}. "
                f"Choose from: purged_walk_forward, time_based, expanding_window, regime_aware"
            )
        
        logger.info(f"CV Strategy initialized: {strategy}")
        logger.info(f"Configuration: {self.cv_config}")
    
    def get_folds(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate CV folds.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with date_id
        y : pd.Series, optional
            Target variable
            
        Yields
        ------
        train_idx : np.ndarray
            Training indices
        val_idx : np.ndarray
            Validation indices
        """
        return self.cv_splitter.split(X, y)
    
def create_cv_strategy(
    config_path: str = "conf/params.yaml",
    strategy: str = "purged_walk_forward"
) -> CVStrategy:
    """
    Factory function to create CV strategy.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    strategy : str
        CV strategy name:
        - 'purged_walk_forward' (default, most reliable)
        - 'time_based' (simple, fast)
        - 'expanding_window' (累積 training data)
        - 'regime_aware' (diverse market conditions)
        
    Returns
    -------
    CVStrategy
        Configured CV strategy instance
    """
    return CVStrategy(config_path, strategy)
