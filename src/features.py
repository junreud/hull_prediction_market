"""
Feature engineering module for Hull Tactical Market Prediction.

This module provides comprehensive feature engineering including:
- Group-based derived features (rolling stats, deviations, volatility)
- Lag features (1-5 days)
- Difference features (changes, acceleration)
- Interaction features (M*×V*, I*×P*)
- Domain-specific features (momentum, RSI-like, Bollinger Bands)
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

from src.utils import get_logger, load_config, Timer

logger = get_logger(log_file="logs/features.log", level="INFO")


class FeatureEngineering:
    """
    Feature engineering pipeline for market prediction.
    
    Features:
    - Rolling statistics (mean, std, min, max)
    - Lag features with leakage prevention
    - Difference and acceleration features
    - Interaction features between groups
    - Domain-specific technical indicators
    """
    
    def __init__(self, config_path: str = "conf/params.yaml"):
        """
        Initialize feature engineering pipeline.
        
        Parameters
        ----------
        config_path : str
            Path to configuration file
        """
        self.config = load_config(config_path)
        self.feature_config = self.config.get('features', {})
        
        # Feature groups
        self.feature_groups = self.feature_config.get('groups', {})
        
        # Rolling windows
        self.rolling_windows = self.feature_config.get('rolling_windows', [5, 10, 20, 40, 60])
        
        # Lag periods
        self.lag_periods = self.feature_config.get('lag_periods', [1, 2, 3, 5, 10])
        
        # Scaler
        scaler_type = self.config.get('scaling', {}).get('scaler_type', 'robust')
        self.scaler = self._get_scaler(scaler_type)
        
        # Feature names tracking
        self.original_features = []
        self.engineered_features = []
        
        logger.info("Feature Engineering initialized")
        logger.info(f"Rolling windows: {self.rolling_windows}")
        logger.info(f"Lag periods: {self.lag_periods}")
        logger.info(f"Scaler type: {scaler_type}")
    
    def _get_scaler(self, scaler_type: str):
        """Get scaler instance based on type."""
        scalers = {
            'robust': RobustScaler(),
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        return scalers.get(scaler_type, RobustScaler())
    
    def _get_feature_columns(self, df: pd.DataFrame, pattern: str) -> List[str]:
        """
        Get column names matching a pattern.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        pattern : str
            Pattern to match (e.g., "M*", "E*")
            
        Returns
        -------
        List[str]
            Matching column names
        """
        if pattern.endswith('*'):
            prefix = pattern[:-1]
            return [col for col in df.columns if col.startswith(prefix)]
        return []
    
    def create_availability_features(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Create availability flags for features with structural missingness.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Columns to check for availability
            
        Returns
        -------
        pd.DataFrame
            Dataframe with availability features added
        """
        new_features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            # Create binary flag: 1 if data exists, 0 if NaN (pre-existence)
            # This helps model distinguish between "value 0" and "value unknown"
            new_features[f'{col}_is_available'] = df[col].notna().astype(int)
            
        if new_features:
            df_new = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
            logger.info(f"Created availability features for {len(new_features)} columns")
            return df_new
        
        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe (must be sorted by date)
        columns : List[str]
            Columns to create rolling features for
        windows : List[int], optional
            Rolling window sizes (default: from config)
            
        Returns
        -------
        pd.DataFrame
            Dataframe with rolling features added
        """
        if windows is None:
            windows = self.rolling_windows
        
        # Collect all new features in a list for efficient concatenation
        new_features = []
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            col_features = {}
            for window in windows:
                # Calculate all rolling statistics
                # Use min_periods=1 to get values as soon as data starts (structural missingness handling)
                roll_mean = df[col].rolling(window=window, min_periods=1).mean()
                roll_std = df[col].rolling(window=window, min_periods=1).std()
                roll_min = df[col].rolling(window=window, min_periods=1).min()
                roll_max = df[col].rolling(window=window, min_periods=1).max()
                
                # Store features
                col_features[f'{col}_roll_mean_{window}'] = roll_mean
                col_features[f'{col}_roll_std_{window}'] = roll_std
                col_features[f'{col}_roll_min_{window}'] = roll_min
                col_features[f'{col}_roll_max_{window}'] = roll_max
                
                # Deviation from rolling mean
                dev = df[col] - roll_mean
                col_features[f'{col}_dev_{window}'] = dev
                
                # Z-score (vectorized)
                col_features[f'{col}_zscore_{window}'] = np.where(
                    roll_std > 0,
                    dev / roll_std,
                    0
                )
            
            # Add all features for this column
            new_features.append(pd.DataFrame(col_features, index=df.index))
        
        # Concatenate all new features at once
        if new_features:
            df_new = pd.concat([df] + new_features, axis=1)
        else:
            df_new = df.copy()
        
        logger.info(f"Created rolling features for {len(columns)} columns")
        return df_new
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Create lag features with leakage prevention.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe (must be sorted by date)
        columns : List[str]
            Columns to create lag features for
        lags : List[int], optional
            Lag periods (default: from config)
            
        Returns
        -------
        pd.DataFrame
            Dataframe with lag features added
        """
        if lags is None:
            lags = self.lag_periods
        
        # Collect all lag features for efficient concatenation
        lag_features = {}
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            for lag in lags:
                lag_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Concatenate all lag features at once
        if lag_features:
            df_new = pd.concat([df, pd.DataFrame(lag_features, index=df.index)], axis=1)
        else:
            df_new = df.copy()
        
        logger.info(f"Created lag features for {len(columns)} columns")
        return df_new
    
    def create_difference_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Create difference and acceleration features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe (must be sorted by date)
        columns : List[str]
            Columns to create difference features for
        periods : List[int], optional
            Difference periods (default: [1, 5, 10])
            
        Returns
        -------
        pd.DataFrame
            Dataframe with difference features added
        """
        if periods is None:
            periods = [1, 5, 10]
        
        # Collect all difference features for efficient concatenation
        diff_features = {}
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            for period in periods:
                # First difference (change)
                diff_features[f'{col}_diff_{period}'] = df[col].diff(period)
                
                # Percent change
                pct = df[col].pct_change(period)
                # Handle infinite values from division by zero
                diff_features[f'{col}_pct_{period}'] = pct.replace([np.inf, -np.inf], np.nan)
                
                # Second difference (acceleration)
                if period == 1:
                    diff_features[f'{col}_accel'] = df[col].diff(1).diff(1)
        
        # Concatenate all difference features at once
        if diff_features:
            df_new = pd.concat([df, pd.DataFrame(diff_features, index=df.index)], axis=1)
        else:
            df_new = df.copy()
        
        logger.info(f"Created difference features for {len(columns)} columns")
        return df_new
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        group_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> pd.DataFrame:
        """
        Create interaction features between feature groups.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        group_pairs : List[Tuple[str, str]], optional
            Pairs of feature group patterns to create interactions
            (default: [("M*", "V*"), ("I*", "P*")])
            
        Returns
        -------
        pd.DataFrame
            Dataframe with interaction features added
        """
        if group_pairs is None:
            group_pairs = [("M*", "V*"), ("I*", "P*"), ("E*", "S*")]
        
        df_new = df.copy()
        
        for pattern1, pattern2 in group_pairs:
            cols1 = self._get_feature_columns(df, pattern1)
            cols2 = self._get_feature_columns(df, pattern2)
            
            # Sample a few interactions to avoid explosion
            # Take first 3 from each group
            cols1_sample = cols1[:3]
            cols2_sample = cols2[:3]
            
            for col1 in cols1_sample:
                for col2 in cols2_sample:
                    # Multiplication
                    df_new[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                    
                    # Ratio (with safety check)
                    df_new[f'{col1}_div_{col2}'] = np.where(
                        np.abs(df[col2]) > 1e-6,
                        df[col1] / df[col2],
                        0
                    )
        
        logger.info(f"Created interaction features for {len(group_pairs)} group pairs")
        return df_new
    
    def create_technical_features(
        self,
        df: pd.DataFrame,
        price_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create domain-specific technical indicators.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe (must be sorted by date)
        price_cols : List[str], optional
            Price-like columns to calculate indicators for
            
        Returns
        -------
        pd.DataFrame
            Dataframe with technical features added
        """
        df_new = df.copy()
        
        # If no price columns specified, use P* group
        if price_cols is None:
            price_cols = self._get_feature_columns(df, "P*")[:5]  # Sample 5
        
        for col in price_cols:
            if col not in df.columns:
                continue
            
            # RSI-like indicator
            delta = df[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-6)
            df_new[f'{col}_rsi'] = 100 - (100 / (1 + rs))
            
            # Momentum (rate of change)
            df_new[f'{col}_momentum_10'] = df[col].pct_change(10)
            df_new[f'{col}_momentum_20'] = df[col].pct_change(20)
            
            # Bollinger Bands
            rolling_mean = df[col].rolling(window=20, min_periods=1).mean()
            rolling_std = df[col].rolling(window=20, min_periods=1).std()
            
            df_new[f'{col}_bb_upper'] = rolling_mean + 2 * rolling_std
            df_new[f'{col}_bb_lower'] = rolling_mean - 2 * rolling_std
            df_new[f'{col}_bb_width'] = (df_new[f'{col}_bb_upper'] - df_new[f'{col}_bb_lower']) / (rolling_mean + 1e-6)
            df_new[f'{col}_bb_position'] = (df[col] - df_new[f'{col}_bb_lower']) / (df_new[f'{col}_bb_upper'] - df_new[f'{col}_bb_lower'] + 1e-6)
        
        logger.info(f"Created technical features for {len(price_cols)} columns")
        return df_new
    
    def create_regime_features(
        self,
        df: pd.DataFrame,
        volatility_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create regime classification features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        volatility_cols : List[str], optional
            Volatility columns to use for regime detection
            
        Returns
        -------
        pd.DataFrame
            Dataframe with regime features added
        """
        df_new = df.copy()
        
        # If no volatility columns specified, use V* group
        if volatility_cols is None:
            volatility_cols = self._get_feature_columns(df, "V*")[:3]  # Sample 3
        
        for col in volatility_cols:
            if col not in df.columns:
                continue
            
            # Calculate percentiles
            # Use min_periods=30 to ensure some reliability (at least 30 days of data)
            # This prevents very unreliable regime detection in the first few days
            rolling_pct = df[col].rolling(window=60, min_periods=30).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
            
            # High/Low volatility regime
            df_new[f'{col}_high_vol'] = (rolling_pct > 0.75).astype(float)
            df_new[f'{col}_low_vol'] = (rolling_pct < 0.25).astype(float)
            
            # Fill NaN with 0 (neutral regime for first 30 days)
            # This ensures we have valid features even at the start of validation period
            df_new[f'{col}_high_vol'] = df_new[f'{col}_high_vol'].fillna(0)
            df_new[f'{col}_low_vol'] = df_new[f'{col}_low_vol'].fillna(0)
        
        logger.info(f"Created regime features for {len(volatility_cols)} columns")
        return df_new
    
    def handle_feature_nans(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Handle NaN values with feature-type-specific strategies.
        
        IMPORTANT: Only handles NaN values in ENGINEERED features.
        Original features' NaN values are preserved as-is.
        
        Different feature types have different optimal NaN handling strategies:
        - Lag features: forward fill (most recent value is most relevant)
        - Rolling stats: backward fill then forward fill (avoid future leakage)
        - Difference/Pct: 0 (no change)
        - Ratio/Division: median or 1.0 (avoid extremes)
        - Technical indicators: neutral values (RSI=50, BB_position=0.5)
        - Regime flags: 0 (not in special regime)
        - Interaction features: 0 (safe default)
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with potential NaN values
            
        Returns
        -------
        pd.DataFrame
            Dataframe with NaN values handled appropriately (only for engineered features)
        """
        df_filled = df.copy()
        
        # First, handle inf values in ALL columns (inf is always invalid)
        df_filled = df_filled.replace([np.inf, -np.inf], np.nan)
        
        # Only process ENGINEERED features (not original features)
        engineered_cols = [col for col in df_filled.columns if col not in self.original_features]
        
        nan_count_before = df_filled[engineered_cols].isna().sum().sum()
        
        logger.info(f"Processing NaN values for {len(engineered_cols)} engineered features only")
        logger.info(f"Original features ({len(self.original_features)}) will preserve their NaN values")
        
        for col in engineered_cols:
            if df_filled[col].isna().sum() == 0:
                continue
            
            # 1. Lag features: forward fill (limit 1) then median (neutral)
            if '_lag_' in col:
                df_filled[col] = df_filled[col].fillna(method='ffill', limit=1)
                # Structural missingness: fill with median (neutral) instead of 0
                median_val = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(median_val if pd.notna(median_val) else 0)
            
            # 2. Rolling statistics: min_periods=1 handled most, but for remaining:
            elif any(x in col for x in ['_roll_mean_', '_roll_std_', '_roll_min_', '_roll_max_']):
                # Forward fill first (short gaps)
                df_filled[col] = df_filled[col].fillna(method='ffill', limit=5)
                # Structural missingness (pre-existence): fill with median
                median_val = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(median_val if pd.notna(median_val) else 0)
            
            # 3. Z-score: 0 (mean) is safe
            elif '_zscore_' in col:
                df_filled[col] = df_filled[col].fillna(0)
            
            # 4. Deviation: 0 (no deviation) is safe
            elif '_dev_' in col:
                df_filled[col] = df_filled[col].fillna(0)
            
            # 5. Difference/Percent change: 0 (no change) is safe
            elif any(x in col for x in ['_diff_', '_pct_', '_accel']):
                df_filled[col] = df_filled[col].fillna(0)
            
            # 6. Ratio/Division: median
            elif '_div_' in col:
                median_val = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(median_val if pd.notna(median_val) else 1.0)
            
            # 7. RSI: 50 (neutral)
            elif '_rsi' in col:
                df_filled[col] = df_filled[col].fillna(50)
            
            # 8. Bollinger Bands position: 0.5 (middle)
            elif '_bb_position' in col:
                df_filled[col] = df_filled[col].fillna(0.5)
            
            # 9. Bollinger Bands width: median
            elif '_bb_width' in col:
                median_val = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(median_val if pd.notna(median_val) else 0)
            
            # 10. Bollinger Bands upper/lower: median
            elif any(x in col for x in ['_bb_upper', '_bb_lower']):
                median_val = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(median_val if pd.notna(median_val) else 0)
            
            # 11. Momentum: 0
            elif '_momentum_' in col:
                df_filled[col] = df_filled[col].fillna(0)
            
            # 12. Regime flags: 0
            elif any(x in col for x in ['_high_vol', '_low_vol']):
                df_filled[col] = df_filled[col].fillna(0)
            
            # 13. Interaction features: 0
            elif '_x_' in col:
                df_filled[col] = df_filled[col].fillna(0)
            
            # 14. Availability flags: 0 (already handled but just in case)
            elif '_is_available' in col:
                df_filled[col] = df_filled[col].fillna(0)
            
            # 15. Others: median (safer than 0 for unknown features)
            else:
                median_val = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(median_val if pd.notna(median_val) else 0)
        
        nan_count_after = df_filled[engineered_cols].isna().sum().sum()
        
        logger.info(f"NaN handling (engineered features only): {nan_count_before} → {nan_count_after} (removed {nan_count_before - nan_count_after})")
        
        if nan_count_after > 0:
            logger.warning(f"Still have {nan_count_after} NaN values in engineered features after handling!")
            nan_cols = [col for col in engineered_cols if df_filled[col].isna().any()]
            logger.warning(f"Columns with NaN: {nan_cols[:10]}...")
        
        return df_filled
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        date_col: str = 'date_id'
    ) -> pd.DataFrame:
        """
        Full feature engineering pipeline.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        date_col : str
            Date column name for sorting
            
        Returns
        -------
        pd.DataFrame
            Dataframe with all engineered features
        """
        logger.info("="*80)
        logger.info("Starting Feature Engineering Pipeline")
        logger.info("="*80)
        
        # Sort by date
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        
        # Store original feature names
        self.original_features = [col for col in df_sorted.columns if col not in [date_col, 'forward_returns']]
        
        with Timer("Feature Engineering", logger=logger):
            # Get feature columns by group
            m_cols = self._get_feature_columns(df_sorted, "M*")
            e_cols = self._get_feature_columns(df_sorted, "E*")
            p_cols = self._get_feature_columns(df_sorted, "P*")
            v_cols = self._get_feature_columns(df_sorted, "V*")
            s_cols = self._get_feature_columns(df_sorted, "S*")
            
            logger.info(f"\nOriginal features by group:")
            logger.info(f"  M (Market): {len(m_cols)} features")
            logger.info(f"  E (Economic): {len(e_cols)} features")
            logger.info(f"  P (Price): {len(p_cols)} features")
            logger.info(f"  V (Volatility): {len(v_cols)} features")
            logger.info(f"  S (Sentiment): {len(s_cols)} features")
            
            # 1. Rolling features (for M, V groups)
            logger.info("\n1. Creating rolling features...")
            df_sorted = self.create_rolling_features(df_sorted, m_cols + v_cols)
            
            # 2. Lag features (all groups)
            logger.info("\n2. Creating lag features...")
            df_sorted = self.create_lag_features(df_sorted, m_cols + e_cols + p_cols)
            
            # 3. Difference features (M, P groups)
            logger.info("\n3. Creating difference features...")
            df_sorted = self.create_difference_features(df_sorted, m_cols + p_cols)
            
            # 4. Interaction features
            logger.info("\n4. Creating interaction features...")
            df_sorted = self.create_interaction_features(df_sorted)
            
            # 5. Technical features
            logger.info("\n5. Creating technical features...")
            df_sorted = self.create_technical_features(df_sorted)
            
            # 6. Regime features
            logger.info("\n6. Creating regime features...")
            df_sorted = self.create_regime_features(df_sorted)
            
            # 7. Handle NaN values
            logger.info("\n7. Handling NaN values...")
            df_sorted = self.handle_feature_nans(df_sorted)
        
        # Track engineered features
        self.engineered_features = [
            col for col in df_sorted.columns 
            if col not in self.original_features and col not in [date_col, 'forward_returns']
        ]
        
        logger.info(f"\n✓ Feature engineering complete!")
        logger.info(f"  Original features: {len(self.original_features)}")
        logger.info(f"  Engineered features: {len(self.engineered_features)}")
        logger.info(f"  Total features: {len(df_sorted.columns) - 2}")  # Exclude date and target
        
        return df_sorted
    
    def transform(
        self,
        df: pd.DataFrame,
        date_col: str = 'date_id'
    ) -> pd.DataFrame:
        """
        Apply feature engineering to new data (same as fit_transform for now).
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        date_col : str
            Date column name
            
        Returns
        -------
        pd.DataFrame
            Transformed dataframe
        """
        return self.fit_transform(df, date_col)
    
    def select_features_by_importance(
        self,
        df: pd.DataFrame,
        target_col: str = 'forward_returns',
        method: str = 'correlation',
        top_n: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features based on importance.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with features
        target_col : str
            Target column name (forward_returns or risk_label)
        method : str
            Selection method: 'correlation', 'variance', 'mutual_info'
        top_n : int, optional
            Select top N features
        threshold : float, optional
            Select features above threshold
            
        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            (Filtered dataframe, Selected feature names)
        """
        logger.info("="*80)
        logger.info("Feature Selection")
        logger.info("="*80)
        logger.info(f"Method: {method}")
        logger.info(f"Top N: {top_n}")
        logger.info(f"Threshold: {threshold}")
        
        # Get feature columns (exclude date and target)
        feature_cols = [
            col for col in df.columns 
            if col not in ['date_id', target_col, 'risk_free_rate', 
                          'market_forward_excess_returns']
        ]
        
        logger.info(f"\nTotal features to evaluate: {len(feature_cols)}")
        
        # Calculate importance scores
        if method == 'correlation':
            # Absolute correlation with target
            scores = df[feature_cols + [target_col]].corr()[target_col].abs()
            scores = scores[scores.index != target_col].sort_values(ascending=False)
        
        elif method == 'variance':
            # Variance-based (remove low-variance features)
            variances = df[feature_cols].var()
            scores = variances.sort_values(ascending=False)
        
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            
            # Handle NaN values: Fill features with median, drop only if target is NaN
            # This preserves data when features have different start dates (structural missingness)
            df_target_clean = df[feature_cols + [target_col]].dropna(subset=[target_col])
            
            if len(df_target_clean) < 100:
                logger.warning("Too few samples, using correlation instead")
                scores = df_target_clean[feature_cols + [target_col]].corr()[target_col].abs()
                scores = scores[scores.index != target_col].sort_values(ascending=False)
            else:
                # Fill feature NaNs with median
                X = df_target_clean[feature_cols].fillna(df_target_clean[feature_cols].median())
                # If still NaN (all NaN column), fill with 0
                X = X.fillna(0)
                y = df_target_clean[target_col]
                
                mi_scores = mutual_info_regression(
                    X, 
                    y,
                    random_state=42
                )
                scores = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Select features
        if top_n is not None:
            selected_features = scores.head(top_n).index.tolist()
            logger.info(f"\n✓ Selected top {top_n} features")
        elif threshold is not None:
            selected_features = scores[scores >= threshold].index.tolist()
            logger.info(f"\n✓ Selected {len(selected_features)} features above threshold {threshold}")
        else:
            # Default: select top 100 or all if less
            top_n = min(100, len(scores))
            selected_features = scores.head(top_n).index.tolist()
            logger.info(f"\n✓ Selected top {top_n} features (default)")
        
        # Log top features
        logger.info(f"\nTop 20 features by {method}:")
        for feat, score in scores.head(20).items():
            logger.info(f"  {feat}: {score:.4f}")
        
        # Keep date, target, and selected features
        keep_cols = ['date_id', target_col] + selected_features
        if 'risk_free_rate' in df.columns:
            keep_cols.insert(2, 'risk_free_rate')
        if 'market_forward_excess_returns' in df.columns:
            keep_cols.insert(2, 'market_forward_excess_returns')
        
        df_selected = df[keep_cols].copy()
        
        logger.info(f"\n✓ Feature selection complete")
        logger.info(f"  Selected features: {len(selected_features)}")
        logger.info(f"  Dataframe shape: {df_selected.shape}")
        
        return df_selected, selected_features
    
    def select_features_voting(
        self,
        df: pd.DataFrame,
        target_col: str = 'forward_returns',
        methods: Optional[List[str]] = None,
        top_n_per_method: int = 150,
        min_votes: int = 2,
        final_top_n: Optional[int] = None
    ) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
        """
        Select features using voting across multiple methods.
        
        각 방법에서 상위 N개 feature를 선택하고, 최소 min_votes 이상의
        방법에서 선택된 feature만 최종 선택합니다.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with features
        target_col : str
            Target column name
        methods : List[str], optional
            Methods to use: ['correlation', 'variance', 'mutual_info']
            Default: all three methods
        top_n_per_method : int
            Top N features to select per method (default: 150)
        min_votes : int
            Minimum votes required (default: 2)
        final_top_n : int, optional
            Final number of features to select (default: all with min_votes)
            
        Returns
        -------
        Tuple[pd.DataFrame, List[str], pd.DataFrame]
            (Filtered dataframe, Selected feature names, Voting summary)
        """
        logger.info("="*80)
        logger.info("Feature Selection - Voting Method")
        logger.info("="*80)
        
        if methods is None:
            methods = ['correlation', 'variance', 'mutual_info']
        
        logger.info(f"Methods: {methods}")
        logger.info(f"Top N per method: {top_n_per_method}")
        logger.info(f"Min votes required: {min_votes}")
        
        # Get feature columns
        feature_cols = [
            col for col in df.columns 
            if col not in ['date_id', target_col, 'risk_free_rate', 
                          'market_forward_excess_returns']
        ]
        
        logger.info(f"\nTotal features to evaluate: {len(feature_cols)}")
        
        # Store selected features from each method
        method_selections = {}
        
        for method in methods:
            logger.info(f"\nRunning method: {method}...")
            
            # Calculate scores
            if method == 'correlation':
                scores = df[feature_cols + [target_col]].corr()[target_col].abs()
                scores = scores[scores.index != target_col].sort_values(ascending=False)
            
            elif method == 'variance':
                variances = df[feature_cols].var()
                scores = variances.sort_values(ascending=False)
            
            elif method == 'mutual_info':
                from sklearn.feature_selection import mutual_info_regression
                
                # Handle NaN values: Fill features with median, drop only if target is NaN
                df_target_clean = df[feature_cols + [target_col]].dropna(subset=[target_col])
                
                if len(df_target_clean) < 100:
                    logger.warning(f"Too few samples for {method}, skipping")
                    continue
                
                # Fill feature NaNs with median
                X = df_target_clean[feature_cols].fillna(df_target_clean[feature_cols].median())
                X = X.fillna(0)
                y = df_target_clean[target_col]
                
                mi_scores = mutual_info_regression(
                    X, 
                    y,
                    random_state=42
                )
                scores = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
            
            # Select top N
            selected = scores.head(top_n_per_method).index.tolist()
            method_selections[method] = set(selected)
            
            logger.info(f"  Selected {len(selected)} features")
        
        # Count votes for each feature
        from collections import Counter
        
        all_features = []
        for features in method_selections.values():
            all_features.extend(features)
        
        vote_counts = Counter(all_features)
        
        # Create voting summary
        voting_summary = pd.DataFrame([
            {'feature': feat, 'votes': count}
            for feat, count in vote_counts.items()
        ]).sort_values('votes', ascending=False)
        
        # Add which methods voted for each feature
        voting_summary['methods'] = voting_summary['feature'].apply(
            lambda f: ', '.join([m for m, feats in method_selections.items() if f in feats])
        )
        
        logger.info(f"\nVoting Summary:")
        logger.info(f"  Features with 3 votes: {sum(voting_summary['votes'] == 3)}")
        logger.info(f"  Features with 2 votes: {sum(voting_summary['votes'] == 2)}")
        logger.info(f"  Features with 1 vote: {sum(voting_summary['votes'] == 1)}")
        
        # Select features with minimum votes
        selected_features = voting_summary[
            voting_summary['votes'] >= min_votes
        ]['feature'].tolist()
        
        # If final_top_n specified, take top N
        if final_top_n is not None and len(selected_features) > final_top_n:
            selected_features = voting_summary.head(final_top_n)['feature'].tolist()
            logger.info(f"\n✓ Selected top {final_top_n} features from voting")
        else:
            logger.info(f"\n✓ Selected {len(selected_features)} features with {min_votes}+ votes")
        
        # Log top features
        logger.info(f"\nTop 20 features by votes:")
        for _, row in voting_summary.head(20).iterrows():
            logger.info(f"  {row['feature']}: {int(row['votes'])} votes ({row['methods']})")
        
        # Keep date, target, and selected features
        keep_cols = ['date_id', target_col] + selected_features
        if 'risk_free_rate' in df.columns:
            keep_cols.insert(2, 'risk_free_rate')
        if 'market_forward_excess_returns' in df.columns:
            keep_cols.insert(2, 'market_forward_excess_returns')
        
        df_selected = df[keep_cols].copy()
        
        logger.info(f"\n✓ Voting-based selection complete")
        logger.info(f"  Selected features: {len(selected_features)}")
        logger.info(f"  Dataframe shape: {df_selected.shape}")
        
        return df_selected, selected_features, voting_summary
    
    def select_features_weighted_ensemble(
        self,
        df: pd.DataFrame,
        target_col: str = 'forward_returns',
        weights: Optional[Dict[str, float]] = None,
        top_n: int = 100
    ) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
        """
        Select features using weighted ensemble of multiple methods.
        
        각 방법의 중요도 점수를 정규화한 후 가중치를 적용하여 합산하고,
        최종 점수가 높은 상위 N개 feature를 선택합니다.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with features
        target_col : str
            Target column name
        weights : Dict[str, float], optional
            Weights for each method
            Default: {'correlation': 0.3, 'mutual_info': 0.5, 'variance': 0.2}
            (금융 데이터는 비선형 관계가 많아 mutual_info 가중치 높임)
        top_n : int
            Number of features to select (default: 100)
            
        Returns
        -------
        Tuple[pd.DataFrame, List[str], pd.DataFrame]
            (Filtered dataframe, Selected feature names, Score summary)
        """
        logger.info("="*80)
        logger.info("Feature Selection - Weighted Ensemble Method")
        logger.info("="*80)
        
        if weights is None:
            weights = {
                'correlation': 0.3,
                'mutual_info': 0.5,  # 금융 데이터는 비선형 관계 많음
                'variance': 0.2
            }
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        logger.info(f"Normalized weights:")
        for method, weight in weights.items():
            logger.info(f"  {method}: {weight:.3f}")
        logger.info(f"Top N: {top_n}")
        
        # Get feature columns
        feature_cols = [
            col for col in df.columns 
            if col not in ['date_id', target_col, 'risk_free_rate', 
                          'market_forward_excess_returns']
        ]
        
        logger.info(f"\nTotal features to evaluate: {len(feature_cols)}")
        
        # Store normalized scores from each method
        method_scores = {}
        
        for method, weight in weights.items():
            logger.info(f"\nCalculating {method} scores...")
            
            # Calculate raw scores
            if method == 'correlation':
                scores = df[feature_cols + [target_col]].corr()[target_col].abs()
                scores = scores[scores.index != target_col]
            
            elif method == 'variance':
                scores = df[feature_cols].var()
            
            elif method == 'mutual_info':
                from sklearn.feature_selection import mutual_info_regression
                
                # Handle NaN values: Fill features with median, drop only if target is NaN
                df_target_clean = df[feature_cols + [target_col]].dropna(subset=[target_col])
                
                if len(df_target_clean) < 100:
                    logger.warning(f"Too few samples for {method}, skipping")
                    continue
                
                # Fill feature NaNs with median
                X = df_target_clean[feature_cols].fillna(df_target_clean[feature_cols].median())
                X = X.fillna(0)
                y = df_target_clean[target_col]
                
                mi_scores = mutual_info_regression(
                    X, 
                    y,
                    random_state=42
                )
                scores = pd.Series(mi_scores, index=feature_cols)
            
            else:
                logger.warning(f"Unknown method: {method}, skipping")
                continue
            
            # Normalize scores to [0, 1]
            # Min-Max normalization
            scores_min = scores.min()
            scores_max = scores.max()
            
            if scores_max > scores_min:
                normalized = (scores - scores_min) / (scores_max - scores_min)
            else:
                normalized = pd.Series(0, index=scores.index)
            
            method_scores[method] = normalized
            
            logger.info(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
            logger.info(f"  Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
        
        # Combine scores with weights
        logger.info(f"\nCombining scores with weights...")
        
        combined_scores = pd.Series(0.0, index=feature_cols)
        
        for method, normalized_scores in method_scores.items():
            weight = weights[method]
            combined_scores += weight * normalized_scores
        
        # Sort by combined score
        combined_scores = combined_scores.sort_values(ascending=False)
        
        # Select top N
        selected_features = combined_scores.head(top_n).index.tolist()
        
        # Create score summary
        score_summary = pd.DataFrame({
            'feature': combined_scores.index,
            'combined_score': combined_scores.values
        })
        
        # Add individual method scores
        for method, scores in method_scores.items():
            score_summary[f'{method}_score'] = score_summary['feature'].map(scores)
        
        score_summary = score_summary.sort_values('combined_score', ascending=False)
        
        logger.info(f"\n✓ Selected top {top_n} features by weighted ensemble")
        
        # Log top features
        logger.info(f"\nTop 20 features by combined score:")
        for _, row in score_summary.head(20).iterrows():
            feat = row['feature']
            score = row['combined_score']
            logger.info(f"  {feat}: {score:.4f}")
            
            # Show contribution from each method
            for method in method_scores.keys():
                method_score = row.get(f'{method}_score', 0)
                weight = weights[method]
                contribution = method_score * weight
                logger.info(f"    └─ {method}: {method_score:.4f} × {weight:.3f} = {contribution:.4f}")
        
        # Keep date, target, and selected features
        keep_cols = ['date_id', target_col] + selected_features
        if 'risk_free_rate' in df.columns:
            keep_cols.insert(2, 'risk_free_rate')
        if 'market_forward_excess_returns' in df.columns:
            keep_cols.insert(2, 'market_forward_excess_returns')
        
        df_selected = df[keep_cols].copy()
        
        logger.info(f"\n✓ Weighted ensemble selection complete")
        logger.info(f"  Selected features: {len(selected_features)}")
        logger.info(f"  Dataframe shape: {df_selected.shape}")
        
        return df_selected, selected_features, score_summary
    
class FeatureDeleter:
    """
    Multi-stage feature deletion with smart strategies
    """
    def __init__(
        self, 
        target_col: str = 'forward_returns',
        variance_threshold: float = 1e-6,
        correlation_threshold: float = 0.95,
        vif_threshold: float = 10.0,
        importance_percentile: int = 20
    ):
        self.target_col = target_col
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.importance_percentile = importance_percentile
        self.deletion_log = []
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute multi-stage deletion pipeline
        """
        logger.info("="*80)
        logger.info("FEATURE DELETION PIPELINE")
        logger.info("="*80)
        logger.info(f"Initial features: {len(df.columns) - 1}")  # -1 for target
        
        df_clean = df.copy()
        
        # Stage 1: Remove useless features
        df_clean = self._stage1_remove_useless(df_clean)
        
        # Stage 2: Remove redundant features (smart correlation)
        df_clean = self._stage2_remove_redundant(df_clean)
        
        # Stage 3: Remove multicollinear features (VIF)
        df_clean = self._stage3_remove_multicollinear(df_clean)
        
        # Stage 4: Remove low-importance features (optional)
        # df_clean = self._stage4_remove_low_importance(df_clean)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"DELETION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Final features: {len(df_clean.columns) - 1}")
        logger.info(f"Deleted: {len(df.columns) - len(df_clean.columns)} features")
        
        # Print deletion log
        self._print_deletion_log()
        
        return df_clean
    
    def _stage1_remove_useless(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 1: Remove obviously useless features
        - Zero variance
        - Near-zero variance
        - Too many missing values
        - Constant values
        """
        logger.info(f"\n{'='*40}")
        logger.info("Stage 1: Remove Useless Features")
        logger.info(f"{'='*40}")
        
        feature_cols = [col for col in df.columns if col != self.target_col]
        to_drop = []
        
        for col in feature_cols:
            reasons = []
            
            # 1. Zero variance
            if df[col].var() == 0:
                reasons.append("zero variance")
            
            # 2. Near-zero variance
            # Financial data (returns) can be very small, so use a safe threshold
            elif df[col].var() < self.variance_threshold:
                reasons.append(f"near-zero variance ({df[col].var():.2e} < {self.variance_threshold})")
            
            # 3. Too many missing (>50%)
            missing_pct = df[col].isnull().sum() / len(df) * 100
            if missing_pct > 50:
                reasons.append(f"too many missing ({missing_pct:.1f}%)")
            
            # 4. Too many identical values (>95%)
            if len(df[col].dropna()) > 0:
                most_common_pct = df[col].value_counts().iloc[0] / len(df) * 100
                if most_common_pct > 95:
                    reasons.append(f"constant-like ({most_common_pct:.1f}% same value)")
            
            # 5. All NaN
            if df[col].isnull().all():
                reasons.append("all NaN")
            
            if reasons:
                to_drop.append(col)
                self.deletion_log.append({
                    'stage': 1,
                    'feature': col,
                    'reason': ', '.join(reasons)
                })
        
        logger.info(f"Removing {len(to_drop)} useless features:")
        for col in to_drop[:10]:  # Show first 10
            logger.info(f"  - {col}")
        if len(to_drop) > 10:
            logger.info(f"  ... and {len(to_drop) - 10} more")
        
        return df.drop(columns=to_drop)
    
    def _stage2_remove_redundant(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 2: Remove redundant features (smart correlation-based)
        - Consider target correlation when choosing which to keep
        - Keep the feature more correlated with target
        """
        logger.info(f"\n{'='*40}")
        logger.info("Stage 2: Remove Redundant Features")
        logger.info(f"{'='*40}")
        
        feature_cols = [col for col in df.columns if col != self.target_col]
        
        # Calculate correlation matrix
        corr_matrix = df[feature_cols + [self.target_col]].corr()
        feature_corr = corr_matrix.loc[feature_cols, feature_cols].abs()
        target_corr = corr_matrix[self.target_col].abs()
        
        # Find pairs with high correlation
        to_drop = set()
        threshold = self.correlation_threshold
        
        for i in range(len(feature_cols)):
            if feature_cols[i] in to_drop:
                continue
            
            for j in range(i+1, len(feature_cols)):
                if feature_cols[j] in to_drop:
                    continue
                
                col_i = feature_cols[i]
                col_j = feature_cols[j]
                
                # Check if highly correlated
                if feature_corr.loc[col_i, col_j] > threshold:
                    # Keep the one more correlated with target
                    if target_corr[col_i] >= target_corr[col_j]:
                        to_drop.add(col_j)
                        reason = f"corr with {col_i}: {feature_corr.loc[col_i, col_j]:.3f}, " \
                                f"target_corr: {target_corr[col_j]:.3f} < {target_corr[col_i]:.3f}"
                    else:
                        to_drop.add(col_i)
                        reason = f"corr with {col_j}: {feature_corr.loc[col_i, col_j]:.3f}, " \
                                f"target_corr: {target_corr[col_i]:.3f} < {target_corr[col_j]:.3f}"
                    
                    self.deletion_log.append({
                        'stage': 2,
                        'feature': list(to_drop)[-1],
                        'reason': reason
                    })
        
        logger.info(f"Removing {len(to_drop)} redundant features (correlation > {threshold})")
        
        return df.drop(columns=list(to_drop))
    
    def _stage3_remove_multicollinear(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 3: Remove multicollinear features using VIF
        - VIF (Variance Inflation Factor) > 10 indicates high multicollinearity
        - Iteratively remove highest VIF until all < threshold
        """
        logger.info(f"\n{'='*40}")
        logger.info("Stage 3: Remove Multicollinear Features (VIF)")
        logger.info(f"{'='*40}")
        
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        feature_cols = [col for col in df.columns if col != self.target_col]
        df_features = df[feature_cols].fillna(df[feature_cols].median())
        
        # Add constant for VIF calculation
        from statsmodels.tools.tools import add_constant
        df_with_const = add_constant(df_features)
        
        vif_threshold = self.vif_threshold
        to_drop = []
        iteration = 0
        max_iterations = 50  # Prevent infinite loop
        
        # Optimization: Skip VIF if too many features (too slow)
        if len(df_with_const.columns) > 50:
            logger.info(f"  ⚠️  Skipping VIF calculation: too many features ({len(df_with_const.columns)} > 50)")
            logger.info("      Use correlation filtering (Stage 2) with stricter threshold instead.")
            return df
        
        while iteration < max_iterations:
            iteration += 1
            
            # Calculate VIF for all features
            vif_data = pd.DataFrame()
            vif_data["feature"] = df_with_const.columns[1:]  # Skip constant
            vif_data["VIF"] = [
                variance_inflation_factor(df_with_const.values, i+1)  # +1 to skip constant
                for i in range(len(df_with_const.columns)-1)
            ]
            
            # Find max VIF
            max_vif = vif_data["VIF"].max()
            
            if max_vif < vif_threshold:
                logger.info(f"  ✓ All VIF < {vif_threshold} after {iteration} iterations")
                break
            
            # Remove feature with highest VIF
            feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
            
            logger.info(f"  Iteration {iteration}: Removing {feature_to_remove} (VIF={max_vif:.2f})")
            
            to_drop.append(feature_to_remove)
            self.deletion_log.append({
                'stage': 3,
                'feature': feature_to_remove,
                'reason': f"VIF={max_vif:.2f} > {vif_threshold}"
            })
            
            # Remove from dataframe
            df_with_const = df_with_const.drop(columns=[feature_to_remove])
        
        if iteration >= max_iterations:
            logger.warning(f"  ⚠️  Reached max iterations ({max_iterations})")
        
        logger.info(f"Removed {len(to_drop)} multicollinear features")
        
        return df.drop(columns=to_drop)
    
    def _stage4_remove_low_importance(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Stage 4: Remove low-importance features (optional, model-based)
        - Train a quick model to get feature importance
        - Remove bottom X percentile
        """
        logger.info(f"\n{'='*40}")
        logger.info("Stage 4: Remove Low-Importance Features")
        logger.info(f"{'='*40}")
        
        from sklearn.ensemble import RandomForestRegressor
        
        feature_cols = [col for col in df.columns if col != self.target_col]
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[self.target_col].dropna()
        
        # Align X and y
        mask = df[self.target_col].notna()
        X = X[mask]
        
        # Quick model for importance
        # IMPORTANT: Use only past data to avoid look-ahead bias
        train_size = int(len(X) * 0.8)
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        
        logger.info(f"  Training on first {train_size} samples (80%) to avoid look-ahead bias")
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Get importance
        importance = pd.Series(model.feature_importances_, index=feature_cols)
        threshold_val = np.percentile(importance, self.importance_percentile)
        
        to_drop = importance[importance < threshold_val].index.tolist()
        
        logger.info(f"Removing {len(to_drop)} low-importance features "
                   f"(bottom {self.importance_percentile}%)")
        logger.info(f"Importance threshold: {threshold_val:.6f}")
        
        for col in to_drop:
            self.deletion_log.append({
                'stage': 4,
                'feature': col,
                'reason': f"low importance ({importance[col]:.6f} < {threshold_val:.6f})"
            })
        
        return df.drop(columns=to_drop)
    
    def _print_deletion_log(self):
        """
        Print deletion log by stage
        """
        logger.info(f"\nDeletion Log by Stage:")
        for stage in [1, 2, 3, 4]:
            stage_deletions = [log for log in self.deletion_log if log['stage'] == stage]
            if stage_deletions:
                logger.info(f"\n  Stage {stage}: {len(stage_deletions)} features")
                for log in stage_deletions[:5]:  # Show first 5
                    logger.info(f"    - {log['feature']}: {log['reason']}")
                if len(stage_deletions) > 5:
                    logger.info(f"    ... and {len(stage_deletions) - 5} more")

    def remove_correlated_features(
        self,
        df: pd.DataFrame,
        threshold: float = 0.95,
        target_col: str = 'forward_returns'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        다중공선성(Multicollinearity) 제거를 위한 고급 Correlation Feature Selection
        - 다중공선성 : 내용이 중복되는 Feature가 여러개 존재하는 특성

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        threshold : float
            Correlation threshold (default: 0.95)
        target_col : str
            Target column name
        
        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            (Filtered dataframe, Removed feature names)
        """
        logger.info("="*80)
        logger.info("Removing Correlated Features")
        logger.info("="*80)
        logger.info(f"Threshold: {threshold}")
        
        # Get feature columns
        feature_cols = [
            col for col in df.columns 
            if col not in ['date_id', target_col, 'risk_free_rate', 
                          'market_forward_excess_returns']
        ]
        
        # Feature 간 Correlation Matrix 계산
        corr_matrix = df[feature_cols].corr().abs()
        
        # Find highly correlated pairs
        # -> 상관행렬은 대각선을 기준으로 위쪽/아래쪽 동일한 데이터이므로 하나만 보면 되기때문에 masking 작업이라고 생각하면됨.
        upper_tri = corr_matrix.where( # False인 위치(반절)는 값이 NaN 들어가게됨.
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool) 
        )
        
        # 타깃과 feature간 상관 미리 계산 (고비용 반복 방지)
        target_corr = None
        if target_col is not None and target_col in df.columns:
            target_corr = df[feature_cols + [target_col]].corr()[target_col].abs()

        # Features to remove
        to_remove = set()
        
        for column in upper_tri.columns:
            if any(upper_tri[column] > threshold):
                # Get correlation with target
                if target_col in df.columns:
                    target_corr = abs(df[[column, target_col]].corr().iloc[0, 1])
                    
                    # Find correlated features
                    correlated = upper_tri.index[upper_tri[column] > threshold].tolist()
                    
                    for corr_feat in correlated:
                        corr_target_corr = abs(df[[corr_feat, target_col]].corr().iloc[0, 1])
                        
                        # Remove the one with lower target correlation
                        if target_corr < corr_target_corr:
                            if column not in to_remove:
                                to_remove.add(column)
                        else:
                            if corr_feat not in to_remove:
                                to_remove.add(corr_feat)
                else:
                    # No target, just remove one of the pair
                    if column not in to_remove:
                        to_remove.add(column)
        
        # Remove duplicates
        to_remove = list(to_remove)
        
        logger.info(f"\n✓ Found {len(to_remove)} highly correlated features to remove")
        
        if len(to_remove) > 0:
            logger.info(f"\nRemoving features (sample):")
            for feat in to_remove[:10]:
                logger.info(f"  {feat}")
        
        # Remove features
        df_filtered = df.drop(columns=to_remove, errors='ignore')
        
        logger.info(f"\n✓ Correlation filtering complete")
        logger.info(f"  Removed features: {len(to_remove)}")
        logger.info(f"  Remaining features: {len(df.columns) - 2}")  # Exclude date and target
        
        return df_filtered, to_remove
    

def create_feature_engineering(
    config_path: str = "conf/params.yaml"
) -> FeatureEngineering:
    """
    Factory function to create feature engineering instance.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
        
    Returns
    -------
    FeatureEngineering
        Configured instance
    """
    return FeatureEngineering(config_path)
