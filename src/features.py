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
        
        df_new = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            for window in windows:
                # Rolling mean
                df_new[f'{col}_roll_mean_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Rolling std
                df_new[f'{col}_roll_std_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).std()
                
                # Rolling min/max
                df_new[f'{col}_roll_min_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).min()
                
                df_new[f'{col}_roll_max_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).max()
                
                # Deviation from rolling mean
                df_new[f'{col}_dev_{window}'] = (
                    df[col] - df_new[f'{col}_roll_mean_{window}']
                )
                
                # Z-score
                std_col = df_new[f'{col}_roll_std_{window}']
                df_new[f'{col}_zscore_{window}'] = np.where(
                    std_col > 0,
                    df_new[f'{col}_dev_{window}'] / std_col,
                    0
                )
        
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
        
        df_new = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            for lag in lags:
                df_new[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
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
        
        df_new = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            for period in periods:
                # First difference (change)
                df_new[f'{col}_diff_{period}'] = df[col].diff(period)
                
                # Percent change
                df_new[f'{col}_pct_{period}'] = df[col].pct_change(period)
                
                # Second difference (acceleration)
                if period == 1:
                    df_new[f'{col}_accel'] = df_new[f'{col}_diff_1'].diff(1)
        
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
            
            # Handle NaN values
            df_clean = df[feature_cols + [target_col]].dropna()
            
            if len(df_clean) < 100:
                logger.warning("Too few samples after removing NaN, using correlation instead")
                scores = df[feature_cols + [target_col]].corr()[target_col].abs()
                scores = scores[scores.index != target_col].sort_values(ascending=False)
            else:
                mi_scores = mutual_info_regression(
                    df_clean[feature_cols], 
                    df_clean[target_col],
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
                
                df_clean = df[feature_cols + [target_col]].dropna()
                
                if len(df_clean) < 100:
                    logger.warning(f"Too few samples for {method}, skipping")
                    continue
                
                mi_scores = mutual_info_regression(
                    df_clean[feature_cols], 
                    df_clean[target_col],
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
                
                df_clean = df[feature_cols + [target_col]].dropna()
                
                if len(df_clean) < 100:
                    logger.warning(f"Too few samples for {method}, skipping")
                    continue
                
                mi_scores = mutual_info_regression(
                    df_clean[feature_cols], 
                    df_clean[target_col],
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
    
    def remove_correlated_features(
        self,
        df: pd.DataFrame,
        threshold: float = 0.95,
        target_col: str = 'forward_returns'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        다중공선성(Multicollinearity) 제거를 위한 고급 Correlation Feature Selection
        - 다중공선성(Multicollinearity) : 내용이 중복되는 Feature가 여러개 존재하는 특성

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
        if target_col is not None:
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
    
    def create_time_period_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        시간 구간 기반 특성 생성 (Feature Engineering).
        
        date_id를 기반으로 명시적인 시간 구간을 생성합니다.
        결측 패턴이 date_id와 완벽히 상관되어 있으므로,
        missing indicator 대신 명시적인 시간 구간을 사용하는 것이 더 해석 가능합니다.
        
        Parameters
        ----------
        df : pd.DataFrame
            'date_id' 컬럼이 있는 데이터프레임
            
        Returns
        -------
        pd.DataFrame
            시간 구간 특성이 추가된 데이터프레임
            
        Examples
        --------
        >>> fe = FeatureEngineering()
        >>> df = fe.create_time_period_features(df)
        >>> # 추가되는 컬럼: early_period, mid_period, recent_period
        """
        df = df.copy()
        
        if 'date_id' not in df.columns:
            logger.warning("date_id column not found, skipping time period features")
            return df
        
        # 시간 구간 정의 (데이터 분포 기반)
        # early: 처음 25% (많은 특성 결측)
        # mid: 25-75% (일부 특성 결측)
        # recent: 마지막 25% (대부분 특성 존재)
        max_date_id = df['date_id'].max()
        
        early_threshold = max_date_id * 0.25
        mid_threshold = max_date_id * 0.75
        
        # 시간 구간 더미 변수
        df['time_early_period'] = (df['date_id'] < early_threshold).astype(int)
        df['time_mid_period'] = ((df['date_id'] >= early_threshold) & 
                                  (df['date_id'] < mid_threshold)).astype(int)
        df['time_recent_period'] = (df['date_id'] >= mid_threshold).astype(int)
        
        logger.info(f"Created time period features:")
        logger.info(f"  Early period (< {early_threshold:.0f}): {df['time_early_period'].sum()} samples")
        logger.info(f"  Mid period ({early_threshold:.0f}-{mid_threshold:.0f}): {df['time_mid_period'].sum()} samples")
        logger.info(f"  Recent period (>= {mid_threshold:.0f}): {df['time_recent_period'].sum()} samples")
        
        return df
    
    def create_market_regime_features(
        self,
        df: pd.DataFrame,
        crisis_periods: Optional[List[Tuple[int, int]]] = None,
        auto_detect: bool = True,
        vol_threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        시장 국면 특성 생성 (Feature Engineering).
        
        위기/고변동성 기간에 대한 국면 더미 변수를 추가합니다.
        위기 기간은 다른 시장 역학을 가지므로, 국면 지표를 추가하면
        모델이 이러한 구조적 변화에 적응하는 데 도움이 됩니다.
        
        방법:
        1. 수동: crisis_periods 매개변수를 통해 알려진 위기 기간 지정
        2. 자동: 고변동성 국면을 자동으로 감지
        
        Parameters
        ----------
        df : pd.DataFrame
            'forward_returns'와 'date_id' 컬럼이 있는 데이터프레임
        crisis_periods : List[Tuple[int, int]], optional
            위기 기간에 대한 (시작_date_id, 종료_date_id) 튜플 리스트
            예시: [(2008, 2009), (2020, 2020)]
        auto_detect : bool, default=True
            고변동성 국면을 자동 감지할지 여부
        vol_threshold : float, default=2.0
            자동 감지를 위한 변동성 임계값 (중앙값의 배수)
            
        Returns
        -------
        pd.DataFrame
            국면 지표 컬럼이 추가된 데이터프레임
            
        Examples
        --------
        >>> fe = FeatureEngineering()
        >>> df = fe.create_market_regime_features(
        ...     df,
        ...     crisis_periods=[(2008, 2009), (2020, 2020)],
        ...     auto_detect=True
        ... )
        >>> # 추가되는 컬럼: regime_crisis_2008_2009, regime_crisis_2020_2020, regime_high_vol
        """
        df = df.copy()
        
        # Manual crisis periods
        if crisis_periods:
            for start, end in crisis_periods:
                col_name = f"regime_crisis_{start}_{end}"
                df[col_name] = 0
                
                # Mark crisis period
                crisis_mask = (df['date_id'] >= start) & (df['date_id'] <= end)
                df.loc[crisis_mask, col_name] = 1
                
                logger.info(f"Added regime indicator: {col_name} ({crisis_mask.sum()} periods)")
        
        # Auto-detect high volatility regimes
        if auto_detect and 'forward_returns' in df.columns:
            # Calculate rolling volatility
            rolling_vol = df['forward_returns'].rolling(window=60, min_periods=20).std()
            vol_median = rolling_vol.median()
            
            # High volatility regime
            df['regime_high_vol'] = 0
            high_vol_mask = rolling_vol > (vol_threshold * vol_median)
            df.loc[high_vol_mask, 'regime_high_vol'] = 1
            
            logger.info(f"Added auto-detected regime: regime_high_vol ({high_vol_mask.sum()} periods)")
        
        return df


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
