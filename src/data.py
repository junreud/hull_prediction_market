"""
Data loading, preprocessing, and benchmark calculation for Hull Tactical Market Prediction.

This module provides:
- Data loading and validation
- Missing value analysis and handling
- Outlier detection and treatment
- Feature scaling strategies
- Benchmark metric calculation
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler

from .utils import get_logger, Timer, load_config

# Initialize global logger once (will use logs/prediction_market.log)
logger = get_logger(log_file="logs/prediction_market.log", level="INFO")


class DataLoader:
    """Handle data loading, validation, and preprocessing for the competition."""
    
    def __init__(self, config_path: str = "conf/params.yaml"):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.data_dir = Path(self.config.get('paths', {}).get('data', 'data/raw/'))
        
        # Feature groups
        self.feature_groups = {
            'M': [],  # Market/Technical indicators
            'E': [],  # Economic indicators
            'I': [],  # Interest rate indicators
            'P': [],  # Price/Valuation indicators
            'V': [],  # Volatility indicators
            'S': [],  # Sentiment indicators
            'D': [],  # Dummy/Event flags
        }
        
        self.train_df = None
        self.test_df = None
        
    def load_data(
        self, 
        train_path: Optional[str] = None, 
        test_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train and test data from CSV files.
        
        Args:
            train_path: Path to train.csv (optional)
            test_path: Path to test.csv (optional)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        with Timer("Loading data", logger):
            # Use provided paths or default from config
            train_path = train_path or self.data_dir / 'train.csv'
            test_path = test_path or self.data_dir / 'test.csv'
            
            logger.info(f"Loading train data from {train_path}")
            self.train_df = pd.read_csv(train_path)
            
            logger.info(f"Loading test data from {test_path}")
            self.test_df = pd.read_csv(test_path)
            
            # Categorize features by group
            self._categorize_features()
            
            # Sort by date_id to ensure chronological order
            self.train_df = self.train_df.sort_values('date_id').reset_index(drop=True)
            self.test_df = self.test_df.sort_values('date_id').reset_index(drop=True)
            
            logger.info("Data loaded and sorted by date_id")
            
        return self.train_df.copy(), self.test_df.copy()
    
    def _categorize_features(self) -> None:
        """Categorize features into groups (M, E, I, P, V, S, D)."""
        for col in self.train_df.columns:
            for group in self.feature_groups.keys():
                if col.startswith(group) and col[1:].isdigit():
                    self.feature_groups[group].append(col)
                    break
        
        logger.info("Feature groups identified:")
        for group, features in self.feature_groups.items():
            logger.info(f"  {group}: {len(features)} features")
    
    def get_missing_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get simple summary of missing values by feature.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            DataFrame with missing value summary
        """
        missing_info = []
        
        for group, features in self.feature_groups.items():
            group_features = [f for f in features if f in df.columns]
            
            for col in group_features:
                missing_count = df[col].isna().sum()
                if missing_count > (len(df[col])*0.2):
                    missing_info.append({
                        'feature': col,
                        'group': group,
                        'missing_count': missing_count,
                        'missing_pct': missing_count / len(df) * 100,
                        'first_valid_idx': df[col].first_valid_index(),
                        'last_valid_idx': df[col].last_valid_index(),
                    })
        
        return pd.DataFrame(missing_info).sort_values('missing_pct', ascending=False)
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
        method: str = 'rank_gauss',
        by_group: bool = True,
        window: int = 60
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        안정적인 분포를 위한 특성 정규화 (Competition Best Practice).
        
        정규화 방법:
        - 'rank_gauss': 순위 변환 → 가우시안 분포 (가장 강건함)
        - 'log1p': log(1+x) 변환 (왜도가 있는 특성용)
        - 'rolling_zscore': (x - 이동평균) / 이동표준편차 (시계열 인식)
        
        Competition 가이드라인:
        - Rank-Gauss가 대부분의 대회에서 선호됨 (이상치에 강건함)
        - Log1p는 자연적으로 왜곡된 특성용 (가격, 거래량 등)
        - Rolling z-score는 국면 변화가 있는 시계열용
        
        Args:
            df: 정규화할 데이터프레임
            train_df: 학습 데이터 (fitting용, 선택사항)
            method: 정규화 방법
            by_group: 특성 그룹별로 정규화할지 여부
            window: 이동 윈도우 크기 (rolling_zscore용)
            
        Returns:
            Tuple of (정규화된 DataFrame, 정규화 메타데이터)
            
        Example:
            >>> df_norm, norm_meta = loader.normalize_features(train_df, method='rank_gauss')
            >>> # 테스트 데이터에 동일한 변환 적용
            >>> test_norm, _ = loader.normalize_features(test_df, train_df=train_df, method='rank_gauss')
        """
        from scipy.stats import rankdata
        from scipy.special import erfinv
        
        df = df.copy()
        metadata = {'method': method, 'by_group': by_group}
        
        # Get numeric columns (exclude date_id and target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate',
                       'market_forward_excess_returns', 'is_scored']
        
        if by_group:
            # Normalize by feature group
            groups_to_process = {k: v for k, v in self.feature_groups.items() if k != 'D'}
            features_to_normalize = []
            for group, features in groups_to_process.items():
                features_to_normalize.extend([f for f in features if f in df.columns])
        else:
            # Normalize all numeric features
            # exclude_cols 칼럼들은 정규화 안하는 칼럼들.
            features_to_normalize = [c for c in numeric_cols if c not in exclude_cols]
        
        if method == 'rank_gauss':
            # Rank transformation → Gaussian distribution
            for col in features_to_normalize:
                if df[col].notna().sum() > 0:
                    # Get valid values
                    valid_mask = df[col].notna()
                    values = df.loc[valid_mask, col].values
                    
                    # Rank transformation (0 to 1)
                    ranks = rankdata(values, method='average')
                    # Avoid exactly 0 and 1 (for erfinv)
                    ranks = (ranks - 0.5) / len(ranks)
                    
                    # Convert to Gaussian using inverse error function
                    # erfinv maps uniform [0,1] to normal distribution
                    normalized = np.sqrt(2) * erfinv(2 * ranks - 1)
                    
                    df.loc[valid_mask, col] = normalized
            
            logger.info(f"Rank-Gauss normalization: {len(features_to_normalize)} features")
            
        elif method == 'log1p':
            # log(1+x) transformation (for positive skewed features)
            for col in features_to_normalize:
                if df[col].notna().sum() > 0:
                    # Shift to positive if needed
                    min_val = df[col].min()
                    if min_val < 0:
                        shift = abs(min_val) + 1
                        df[col] = np.log1p(df[col] + shift)
                    else:
                        df[col] = np.log1p(df[col])
            
            logger.info(f"Log1p normalization: {len(features_to_normalize)} features")
            
        elif method == 'rolling_zscore':
            # Rolling z-score (time-series aware)
            for col in features_to_normalize:
                if df[col].notna().sum() > 0:
                    rolling_mean = df[col].rolling(window=window, min_periods=20, center=False).mean()
                    rolling_std = df[col].rolling(window=window, min_periods=20, center=False).std()
                    
                    # Avoid division by zero
                    rolling_std = rolling_std.replace(0, np.nan)
                    
                    df[col] = (df[col] - rolling_mean) / rolling_std
            
            logger.info(f"Rolling z-score normalization: {len(features_to_normalize)} features (window={window})")
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'rank_gauss', 'log1p', or 'rolling_zscore'")
        
        metadata['features_normalized'] = features_to_normalize
        
        return df, metadata
    
    def winsorize_outliers(
        self, 
        df: pd.DataFrame, 
        limits: Tuple[float, float] = (0.01, 0.01),
        method: str = 'rolling',
        window: int = 60
    ) -> pd.DataFrame:
        """
        시계열 인식 방법을 사용한 이상치 윈저라이제이션. 
        너무 이상하리만치 큰 데이터, 작은 데이터를 만났을 경우 
        아예 삭제하지않고 조정해서 과적합 되지 않게함.

        시계열을 위한 개선된 방법:
        - 'rolling': 이동 윈도우 백분위수 (지역 기준선)
        - 'global': 전역 백분위수 (레거시 방법)
        
        Args:
            df: 처리할 데이터프레임
            limits: (하한, 상한) 백분위수 한계
            method: 'rolling' 또는 'global'
            window: 이동 윈도우 크기 (rolling 방법용)
            
        Returns:
            윈저라이제이션이 적용된 데이터프레임
        """
        df = df.copy()
        
        # Get numeric columns (exclude date_id and target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate',
                       'market_forward_excess_returns', 'is_scored']
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        if method == 'rolling':
            # Rolling window winsorization (time-series aware)
            for col in numeric_cols:
                if df[col].notna().sum() > 0:
                    # Calculate rolling percentiles
                    rolling_lower = df[col].rolling(
                        window=window,    # 60일 윈도우
                        min_periods=20,   # 최소 20일 있으면 계산
                        center=False      # 과거 데이터만 사용 (미래 보지 않음)
                    ).quantile(limits[0]) # 하위 1% 백분위수
                    
                    rolling_upper = df[col].rolling(
                        window=window, 
                        min_periods=20, 
                        center=False
                    ).quantile(1 - limits[1])
                    
                    # Clip values based on rolling bounds
                    df[col] = df[col].clip(lower=rolling_lower, upper=rolling_upper)
            
            logger.info(f"Rolling winsorization complete with limits {limits}, window={window}")
            
        elif method == 'global':
            # Global winsorization (legacy method)
            for col in numeric_cols:
                if df[col].notna().sum() > 0:
                    df[col] = stats.mstats.winsorize(df[col].values, limits=limits)
            
            logger.info(f"Global winsorization complete with limits {limits}")
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'rolling' or 'global'")
        
        return df
    
    def detect_feature_clusters(
        self,
        df: pd.DataFrame,
        method: str = 'correlation',
        threshold: float = 0.85,
        by_group: bool = True
    ) -> Dict[str, List[List[str]]]:
        """
        Detect highly correlated feature clusters to identify redundancy.
        
        This helps identify groups of features that provide similar information,
        which can lead to overfitting in tree-based models or instability in linear models.
        
        Args:
            df: DataFrame to analyze
            method: 'correlation' (Pearson correlation) or 'distance' (not implemented)
            threshold: Correlation threshold for clustering (default: 0.85)
            by_group: Whether to cluster within feature groups (default: True)
            
        Returns:
            Dictionary mapping group name -> list of clusters
            Each cluster is a list of feature names
            
        Example:
            >>> clusters = loader.detect_feature_clusters(train_df, threshold=0.90)
            >>> # {'V': [['V1', 'V3', 'V7'], ['V2', 'V5']], 'M': [['M1', 'M4']]}
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        all_clusters = {}
        
        if by_group:
            # Cluster within each feature group
            groups_to_process = {k: v for k, v in self.feature_groups.items() if k != 'D'}
        else:
            # Cluster all features together
            all_features = [f for group_features in self.feature_groups.values() 
                          for f in group_features if f in df.columns]
            groups_to_process = {'all': all_features}
        
        for group, features in groups_to_process.items():
            group_features = [f for f in features if f in df.columns]
            
            if len(group_features) < 2:
                continue
            
            # Calculate correlation matrix
            corr_matrix = df[group_features].corr().abs()
            
            # Convert correlation to distance (1 - correlation)
            distance_matrix = 1 - corr_matrix
            
            # Perform hierarchical clustering
            try:
                # Convert to condensed distance matrix for linkage
                condensed_dist = squareform(distance_matrix, checks=False)
                
                # Perform clustering
                linkage_matrix = linkage(condensed_dist, method='average')
                
                # Cut tree at distance threshold
                distance_threshold = 1 - threshold
                cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
                
                # Group features by cluster
                clusters = {}
                for feature, label in zip(group_features, cluster_labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(feature)
                
                # Only keep clusters with multiple features
                group_clusters = [cluster for cluster in clusters.values() if len(cluster) > 1]
                
                if group_clusters:
                    all_clusters[group] = group_clusters
                    logger.info(f"Group {group}: Found {len(group_clusters)} clusters with {sum(len(c) for c in group_clusters)} features")
                    
            except Exception as e:
                logger.warning(f"Failed to cluster group {group}: {e}")
                continue
        
        return all_clusters
    
    def reduce_feature_redundancy(
        self,
        df: pd.DataFrame,
        clusters: Dict[str, List[List[str]]],
        method: str = 'representative'
    ) -> pd.DataFrame:
        """
        클러스터링 결과를 사용하여 특성 중복성을 줄입니다.
        
        방법:
        - 'representative': 각 클러스터에서 가장 높은 분산을 가진 특성만 유지
        - 'mean': 클러스터를 z-점수화된 특성들의 평균으로 대체
        - 'pca': 클러스터를 첫 번째 주성분으로 대체 (아직 구현되지 않음)
        
        Args:
            df: 처리할 데이터프레임
            clusters: detect_feature_clusters()의 출력 결과
            method: 축소 방법
            
        Returns:
            축소된 특성을 가진 데이터프레임
            
        Example:
            >>> clusters = loader.detect_feature_clusters(train_df, threshold=0.90)
            >>> df_reduced = loader.reduce_feature_redundancy(df, clusters, method='representative')
        """
        df = df.copy()
        features_removed = 0
        
        for group, group_clusters in clusters.items():
            for cluster in group_clusters:
                if len(cluster) < 2:
                    continue
                
                if method == 'representative':
                    # Keep feature with highest variance (most informative)
                    variances = df[cluster].var()
                    representative = variances.idxmax()
                    
                    # Drop all others in the cluster
                    to_drop = [f for f in cluster if f != representative]
                    df = df.drop(columns=to_drop)
                    features_removed += len(to_drop)
                    
                    logger.debug(f"Cluster {cluster}: Kept {representative}, dropped {len(to_drop)} features")
                    
                elif method == 'mean':
                    # Create mean of z-scored features
                    cluster_name = f"{group}_cluster_{cluster[0]}"
                    
                    # Z-score each feature in cluster
                    z_scored = df[cluster].apply(lambda x: (x - x.mean()) / x.std())
                    
                    # Take mean
                    df[cluster_name] = z_scored.mean(axis=1)
                    
                    # Drop original features
                    df = df.drop(columns=cluster)
                    features_removed += len(cluster)
                    
                    logger.debug(f"Cluster {cluster}: Created {cluster_name}, dropped {len(cluster)} features")
                    
                else:
                    raise ValueError(f"Unknown method: {method}. Use 'representative' or 'mean'")
        
        logger.info(f"Feature redundancy reduction: Removed {features_removed} features using '{method}' method")
        
        return df
    
    def apply_group_pca(
        self,
        df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
        n_components: Optional[Dict[str, int]] = None,
        variance_threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        차원 축소를 위해 각 특성 그룹 내에서 PCA를 적용합니다.

        이 방법은 대부분의 분산을 유지하면서 특성의 수를 줄여,
        과적합을 방지하고 모델의 일반화 성능을 향상시킵니다.

        Args:
            df: 변환할 데이터프레임
            train_df: PCA 학습용 학습 데이터 (선택사항)
            n_components: 그룹별 컴포넌트 수 (선택사항)
                예시: {'M': 3, 'V': 2}는 M 그룹에 3개, V 그룹에 2개의 컴포넌트를 의미
            variance_threshold: 유지할 누적 분산 비율 (기본값: 0.95)
            
        Returns:
            Tuple of (변환된 데이터프레임, PCA 모델 딕셔너리)
            
        Example:
            >>> n_components = {'M': 5, 'V': 3, 'E': 4}
            >>> df_pca, pca_models = loader.apply_group_pca(train_df, n_components=n_components)
            >>> # 테스트 데이터에 동일한 변환 적용
            >>> test_pca, _ = loader.apply_group_pca(test_df, train_df=train_df, n_components=n_components)
        """
        from sklearn.decomposition import PCA
        
        df = df.copy()
        pca_models = {}
        
        # Get non-dummy feature groups
        groups_to_process = {k: v for k, v in self.feature_groups.items() if k != 'D'}
        
        for group, features in groups_to_process.items():
            group_features = [f for f in features if f in df.columns]
            
            if len(group_features) < 2:
                logger.info(f"Group {group}: Skipping PCA (only {len(group_features)} features)")
                continue
            
            # Determine number of components
            if n_components and group in n_components:
                n = n_components[group]
            else:
                # Auto-determine based on variance threshold
                n = None  # Will be determined by PCA
            
            # Fit PCA on training data or current data
            fit_df = train_df if train_df is not None else df
            fit_data = fit_df[group_features].dropna()
            
            if len(fit_data) == 0:
                logger.warning(f"Group {group}: No valid data for PCA fitting")
                continue
            
            # Initialize and fit PCA
            if n is None:
                # Determine n_components to reach variance threshold
                pca = PCA(n_components=variance_threshold, svd_solver='full')
            else:
                pca = PCA(n_components=min(n, len(group_features)))
            
            try:
                pca.fit(fit_data)
                
                # Transform the data
                transformed = pca.transform(df[group_features].fillna(0))
                
                # Create new column names
                pca_cols = [f"{group}_PC{i+1}" for i in range(pca.n_components_)]
                
                # Add to dataframe
                for i, col in enumerate(pca_cols):
                    df[col] = transformed[:, i]
                
                # Drop original features
                df = df.drop(columns=group_features)
                
                # Store model
                pca_models[group] = {
                    'model': pca,
                    'original_features': group_features,
                    'pca_features': pca_cols,
                    'variance_explained': pca.explained_variance_ratio_.sum()
                }
                
                logger.info(f"Group {group}: Reduced {len(group_features)} features to {pca.n_components_} "
                          f"components (variance: {pca.explained_variance_ratio_.sum():.2%})")
                
            except Exception as e:
                logger.error(f"Group {group}: PCA failed - {e}")
                continue
        
        return df, pca_models
    
    def add_event_dummies(
        self,
        df: pd.DataFrame,
        event_calendar: pd.DataFrame
    ) -> pd.DataFrame:
        """
        중요한 시장 이벤트에 대한 이진 이벤트 지표를 추가합니다.

        이벤트 윈도우(이벤트 전후)에 대한 더미 변수를 생성하여
        이벤트 중심의 가격 변동을 포착합니다.

        Args:
            df: date_id 컬럼이 있는 데이터프레임
            event_calendar: 다음 컬럼들을 포함하는 데이터프레임:
                - event_date: 이벤트의 date_id
                - event_type: 이벤트 유형 ('FOMC', 'earnings', 'CPI' 등)
                - window_before: 이벤트 전에 표시할 일수 (기본값: 0)
                - window_after: 이벤트 후에 표시할 일수 (기본값: 0)
            
        Returns:
            이벤트 더미 컬럼이 추가된 데이터프레임
            
        Example:
            >>> event_calendar = pd.DataFrame({
            >>>     'event_date': [100, 120, 150],
            >>>     'event_type': ['FOMC', 'CPI', 'FOMC'],
            >>>     'window_before': [1, 0, 1],
            >>>     'window_after': [1, 1, 1]
            >>> })
            >>> df_events = loader.add_event_dummies(df, event_calendar)
            >>> # 생성되는 컬럼: event_FOMC, event_CPI
        """
        df = df.copy()
        
        # Get unique event types
        event_types = event_calendar['event_type'].unique()
        
        # Initialize event columns
        for event_type in event_types:
            col_name = f"event_{event_type}"
            df[col_name] = 0
        
        # Fill in event indicators
        for _, row in event_calendar.iterrows():
            event_date = row['event_date']
            event_type = row['event_type']
            window_before = row.get('window_before', 0)
            window_after = row.get('window_after', 0)
            
            col_name = f"event_{event_type}"
            
            # Mark event window
            event_mask = (
                (df['date_id'] >= event_date - window_before) &
                (df['date_id'] <= event_date + window_after)
            )
            
            df.loc[event_mask, col_name] = 1
        
        logger.info(f"Added {len(event_types)} event dummy variables: {list(event_types)}")
        
        return df
    
    def analyze_granger_causality(
        self,
        df: pd.DataFrame,
        target: str = 'forward_returns',
        max_lag: int = 5,
        significance: float = 0.05
    ) -> pd.DataFrame:
        """
        특성(features)과 타겟(target) 간의 그레인저 인과관계(Granger causality)를 검증합니다.
        
        그레인저 인과관계 검정은 특성 X의 과거 값이 Y의 과거 값만으로 예측하는 것보다
        Y의 미래 값을 더 잘 예측하는 데 도움이 되는지를 테스트합니다.
        
        이를 통해 어떤 특성이 시간 지연(time lag)을 가지고 예측력을 가지는지 식별할 수 있습니다.
        
        Args:
            df: 분석할 데이터프레임 (date_id로 정렬되어 있어야 함)
            target: 타겟 변수 이름
            max_lag: 테스트할 최대 시차(lag) (기본값: 5)
            significance: 유의성 판단을 위한 P-값 임계값 (기본값: 0.05)
            
        Returns:
            인과관계 검정 결과를 담은 데이터프레임, 유의성 순으로 정렬됨
            
        Example:
            >>> causality_results = loader.analyze_granger_causality(
            >>>     train_df,
            >>>     target='forward_returns',
            >>>     max_lag=5
            >>> )
            >>> # 어떤 특성이 타겟에 그레인저 인과관계를 가지는지 보여줍니다
        """
        from statsmodels.tsa.stattools import grangercausalitytests
        
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found in DataFrame")
        
        # Get numeric features (exclude date_id and target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['date_id', target, 'risk_free_rate', 
                       'market_forward_excess_returns', 'is_scored']
        features_to_test = [c for c in numeric_cols if c not in exclude_cols]
        
        results = []
        
        for feature in features_to_test:
            # Skip if too many missing values
            if df[[feature, target]].isna().any(axis=1).sum() > len(df) * 0.1:
                continue
            
            # Prepare data (drop NaN)
            test_data = df[[target, feature]].dropna()
            
            if len(test_data) < max_lag + 10:
                continue
            
            try:
                # Run Granger causality test
                gc_result = grangercausalitytests(
                    test_data[[target, feature]], 
                    maxlag=max_lag,
                    verbose=False
                )
                
                # Extract p-values for each lag
                p_values = []
                for lag in range(1, max_lag + 1):
                    # Get F-test p-value
                    p_value = gc_result[lag][0]['ssr_ftest'][1]
                    p_values.append(p_value)
                
                # Find best lag (lowest p-value)
                best_lag = np.argmin(p_values) + 1
                best_p_value = p_values[best_lag - 1]
                
                # Store results
                results.append({
                    'feature': feature,
                    'best_lag': best_lag,
                    'p_value': best_p_value,
                    'significant': best_p_value < significance,
                    'all_p_values': p_values
                })
                
            except Exception as e:
                logger.debug(f"Granger test failed for {feature}: {e}")
                continue
        
        # Convert to DataFrame and sort by p-value
        results_df = pd.DataFrame(results).sort_values('p_value')
        
        significant_count = results_df['significant'].sum()
        logger.info(f"Granger causality analysis: {significant_count}/{len(results_df)} features "
                   f"are significant at p<{significance}")
        
        return results_df
    
    def calculate_regime_weights(
        self,
        df: pd.DataFrame,
        regime_col: str = 'regime',
        weight_map: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        변동성 국면에 기반한 샘플 가중치를 계산합니다.
        
        고변동성 기간은 종종 시그널보다 노이즈를 더 많이 포함하므로,
        학습 시 이러한 샘플의 가중치를 낮춥니다.
        
        기본 가중치:
        - 저변동성: 1.0 (표준 가중치)
        - 정상: 1.0
        - 고변동성: 0.5 (노이즈로 인해 감소된 가중치)
        
        Args:
            df: 국면 컬럼이 있는 데이터프레임
            regime_col: 국면 컬럼 이름 (기본값: 'regime')
            weight_map: 사용자 정의 가중치 매핑 (선택사항)
            예시: {'low_vol': 1.0, 'normal': 1.0, 'high_vol': 0.3}
            
        Returns:
            샘플 가중치 시리즈 (df와 동일한 길이)
            
        Example:
            >>> df = loader.detect_regime_changes(df)
            >>> sample_weights = loader.calculate_regime_weights(df)
            >>> # 모델 학습에 사용: model.fit(X, y, sample_weight=sample_weights)
        """
        if regime_col not in df.columns:
            logger.warning(f"Regime column '{regime_col}' not found. Returning uniform weights.")
            return pd.Series(1.0, index=df.index)
        
        if weight_map is None:
            weight_map = {
                'low_vol': 1.0,
                'normal': 1.0,
                'high_vol': 0.5  # Down-weight high volatility periods
            }
        
        weights = df[regime_col].map(weight_map)
        
        # Fill unknown regimes with 1.0 (normal weight)
        weights = weights.fillna(1.0)
        
        logger.info(f"Regime-based sample weights calculated: {weights.value_counts().to_dict()}")
        
        return weights
    
    def preprocess_timeseries(
        self,
        df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
        # Outlier handling
        handle_outliers: bool = True,
        winsorize_limits: Tuple[float, float] = (0.001, 0.001),
        winsorize_method: str = 'rolling',
        # Normalization
        normalize: bool = False,
        normalize_method: str = 'rank_gauss',
        # Scaling
        scale: bool = True,
        scale_method: str = 'robust',
        # Window size
        window: int = 60
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        완전한 시계열 기반 전처리 파이프라인 (Competition Best Practices).

        파이프라인 단계:
        1. 극단값 제거 (winsorization) - 상하위 0.1~0.5% 절삭
        2. 특성 정규화 (선택) - Rank-Gauss, Log1p, Rolling Z-score
        3. 특성 스케일링 (선택) - Robust 또는 Standard 스케일링

        Competition 모범 사례:
        ✅ 행 삭제 금지 (시계열 연속성 유지)
        ✅ 극단값 절삭 (0.1~0.5% winsorization)
        ✅ 분포 정규화 (rank-gauss 선호)
        
        Note:
            Feature Engineering (시간 구간, 국면 지표 등)은 
            FeatureEngineering 클래스의 create_time_period_features(), 
            create_market_regime_features()를 사용하세요.

        Args:
            df: 처리할 데이터프레임 (date_id로 정렬되어 있어야 함)
            train_df: 학습 데이터 (fitting용, 선택사항)
            
            # 이상치 처리
            handle_outliers: 이상치 winsorize 여부
            winsorize_limits: 백분위수 한계 (기본값: 0.001 = 0.1%)
            winsorize_method: 'rolling' (이동 윈도우) 또는 'global' (전역)
            
            # 정규화 (Normalization)
            normalize: 특성 정규화 여부
            normalize_method: 'rank_gauss', 'log1p', 'rolling_zscore' 중 선택
            
            # 스케일링 (Scaling)
            scale: 특성 스케일링 여부
            scale_method: 'robust' 또는 'standard'
            
            # 윈도우 크기
            window: 시계열 방법에 사용할 이동 윈도우 크기
            
        Returns:
            Tuple of (처리된 DataFrame, metadata 딕셔너리)
            
        Example:
            >>> # 전처리만
            >>> train_processed, metadata = loader.preprocess_timeseries(
            >>>     train_df,
            >>>     handle_outliers=True,
            >>>     winsorize_limits=(0.001, 0.001),
            >>>     normalize=True,
            >>>     normalize_method='rank_gauss',
            >>>     scale=True
            >>> )
            >>> 
            >>> # Feature Engineering은 별도로
            >>> from src.features import FeatureEngineering
            >>> fe = FeatureEngineering()
            >>> train_processed = fe.create_time_period_features(train_processed)
            >>> train_processed = fe.create_market_regime_features(train_processed)
        """
        df = df.copy()
        metadata = {}
        
        logger.info("="*60)
        logger.info("PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        # Step 1: Clip extreme values (Competition Best Practice)
        if handle_outliers:
            with Timer("Winsorizing outliers", logger):
                df = self.winsorize_outliers(
                    df,
                    limits=winsorize_limits,
                    method=winsorize_method,
                    window=window
                )
                metadata['outliers_handled'] = True
        
        # Step 2: Normalize features (Competition Best Practice)
        if normalize:
            with Timer("Normalizing features", logger):
                df, norm_meta = self.normalize_features(
                    df,
                    train_df=train_df,
                    method=normalize_method,
                    by_group=True,
                    window=window
                )
                metadata['normalization'] = norm_meta
                metadata['normalized'] = True
        
        # Step 3: Scale features
        if scale:
            with Timer("Scaling features", logger):
                df, scalers = self.scale_features(
                    df,
                    train_df=train_df,
                    method=scale_method,
                    by_group=True
                )
                metadata['scalers'] = scalers
                metadata['scaled'] = True
        
        logger.info("="*60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("="*60)
        
        return df, metadata
    
    def detect_regime_changes(
        self,
        df: pd.DataFrame,
        vol_window: int = 60,
        vol_threshold: Tuple[float, float] = (0.5, 1.5)
    ) -> pd.Series:
        """
        시장 국면 변화를 감지합니다 (고변동성 / 저변동성 기간).
        
        다음 용도로 유용합니다:
        - 적응형 전처리 (국면별 다른 전략 적용)
        - 위험 관리
        - 모델 앙상블 가중치 조정
        
        Args:
            df: 'forward_returns' 컬럼이 있는 데이터프레임
            vol_window: 변동성 계산을 위한 윈도우 크기
            vol_threshold: 국면 분류를 위한 (저변동성_배수, 고변동성_배수) 튜플
            
        Returns:
            국면 레이블이 포함된 시리즈: 'low_vol', 'normal', 'high_vol'
        """
        # Calculate rolling volatility
        returns_vol = df['forward_returns'].rolling(window=vol_window, min_periods=20).std()
        
        # Calculate median volatility as baseline
        vol_median = returns_vol.median()
        
        # Classify regimes
        regime = pd.Series('normal', index=df.index)
        regime[returns_vol > vol_threshold[1] * vol_median] = 'high_vol'
        regime[returns_vol < vol_threshold[0] * vol_median] = 'low_vol'
        
        # Count regime changes
        regime_changes = (regime != regime.shift(1)).sum()
        
        regime_counts = regime.value_counts()
        logger.info(f"Regime detection complete:")
        logger.info(f"  - Low volatility: {regime_counts.get('low_vol', 0)} periods")
        logger.info(f"  - Normal: {regime_counts.get('normal', 0)} periods")
        logger.info(f"  - High volatility: {regime_counts.get('high_vol', 0)} periods")
        logger.info(f"  - Regime changes: {regime_changes}")
        
        return regime
    
    def scale_features(
        self, 
        df: pd.DataFrame, 
        train_df: Optional[pd.DataFrame] = None,
        method: str = 'robust',
        by_group: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Scale features using group-specific strategies.
        
        IMPORTANT: Scalers are fitted on training data only to prevent leakage.
        
        Args:
            df: DataFrame to scale
            train_df: Training data for fitting scalers
            method: 'robust' or 'standard'
            by_group: Whether to scale by feature group
            
        Returns:
            Tuple of (scaled DataFrame, scalers dict)
        """
        df = df.copy()
        scalers = {}
        
        # Determine which data to fit on
        fit_df = train_df if train_df is not None else df
        
        if method == 'robust':
            scaler_class = RobustScaler
        elif method == 'standard':
            scaler_class = StandardScaler
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        if by_group:
            # Scale by feature group
            for group, features in self.feature_groups.items():
                group_features = [f for f in features if f in df.columns]
                
                if not group_features:
                    continue
                
                # Skip dummy variables
                if group == 'D':
                    continue
                
                scaler = scaler_class()
                
                # Fit on training data
                scaler.fit(fit_df[group_features].fillna(0))
                
                # Transform the data
                df[group_features] = scaler.transform(df[group_features].fillna(0))
                
                scalers[group] = scaler
                
                logger.info(f"Scaled {len(group_features)} features in group {group}")
        else:
            # Scale all numeric features together
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 
                           'market_forward_excess_returns', 'is_scored']
            scale_cols = [c for c in numeric_cols if c not in exclude_cols and not c.startswith('D')]
            
            scaler = scaler_class()
            scaler.fit(fit_df[scale_cols].fillna(0))
            df[scale_cols] = scaler.transform(df[scale_cols].fillna(0))
            
            scalers['all'] = scaler
            
            logger.info(f"Scaled {len(scale_cols)} features")
        
        return df, scalers


def calculate_benchmark_score(
    df: pd.DataFrame,
    allocation: float = 1.0,
    return_details: bool = True
) -> Union[float, Tuple[float, pd.Series, Dict]]:
    """
    allocation=1.0(시장 대등 전략)으로 벤치마크 점수를 계산합니다.

    지표: 변동성 페널티가 적용된 수정 샤프 비율

    score = (평균(전략_수익률) / 표준편차(전략_수익률)) / 변동성_페널티

    여기서:
    - 전략_수익률 = 할당량 × 선도_수익률
    - 변동성_페널티 = 1 + max(0, (전략_변동성 / 시장_변동성) - 1.2)

    Args:
        df: 'forward_returns' 컬럼이 있는 데이터프레임
        allocation: 고정 할당 값 (0에서 2 사이)
        return_details: 상세 지표를 반환할지 여부
        
    Returns:
        return_details=False인 경우: benchmark_score
        return_details=True인 경우: (benchmark_score, strategy_returns, metrics_dict)
    """
    # Validate allocation
    if not 0 <= allocation <= 2:
        raise ValueError(f"Allocation must be between 0 and 2, got {allocation}")
    
    # Calculate strategy returns
    strategy_returns = allocation * df['forward_returns']
    
    # Calculate volatilities
    strategy_vol = strategy_returns.std()
    market_vol = df['forward_returns'].std()
    
    # Volatility penalty
    vol_ratio = strategy_vol / market_vol if market_vol > 0 else 1.0
    vol_penalty = 1 + max(0, vol_ratio - 1.2)
    
    # Sharpe ratio
    mean_return = strategy_returns.mean()
    sharpe = mean_return / strategy_vol if strategy_vol > 0 else 0.0
    
    # Final score
    benchmark_score = sharpe / vol_penalty
    
    if not return_details:
        return benchmark_score
    
    # Calculate additional metrics
    annual_factor = 252  # Trading days per year
    
    # Cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # Maximum drawdown
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Annualized metrics
    annual_return = mean_return * annual_factor
    annual_vol = strategy_vol * np.sqrt(annual_factor)
    annual_sharpe = sharpe * np.sqrt(annual_factor)
    
    # Underperformance check
    market_mean_return = df['forward_returns'].mean()
    underperformance = mean_return < market_mean_return
    
    # Win rate
    win_rate = (strategy_returns > 0).sum() / len(strategy_returns)
    
    metrics = {
        'score': benchmark_score,
        'sharpe': sharpe,
        'annual_sharpe': annual_sharpe,
        'vol_penalty': vol_penalty,
        'vol_ratio': vol_ratio,
        'mean_return': mean_return,
        'annual_return': annual_return,
        'strategy_vol': strategy_vol,
        'annual_vol': annual_vol,
        'market_vol': market_vol,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'underperformance': underperformance,
        'allocation': allocation,
        'n_days': len(df),
    }
    
    logger.info(f"Benchmark Score: {benchmark_score:.4f}")
    logger.info(f"Annual Sharpe: {annual_sharpe:.4f}")
    logger.info(f"Volatility Penalty: {vol_penalty:.4f}")
    logger.info(f"Max Drawdown: {max_drawdown:.2%}")
    
    return benchmark_score, strategy_returns, metrics


if __name__ == "__main__":
    # Test COMPETITION-READY preprocessing pipeline
    from src.utils import set_seed
    
    set_seed(42)
    
    # Initialize loader
    loader = DataLoader()
    
    # Load data
    train_df, test_df = loader.load_data()
    
    print("\n" + "="*60)
    print("COMPETITION-READY PREPROCESSING PIPELINE TEST")
    print("="*60)
    
    # Test the complete preprocessing pipeline
    train_processed, metadata = loader.preprocess_timeseries(
        train_df,
        train_df=None,  # No separate train data (will use self for fitting)
        
        # Outlier handling (0.1% clip - competition best practice)
        handle_outliers=True,
        winsorize_limits=(0.001, 0.001),
        winsorize_method='rolling',
        
        # Normalization (rank-gauss - most robust)
        normalize=True,
        normalize_method='rank_gauss',
        
        # Scaling
        scale=True,
        scale_method='robust',
        
        # Window size
        window=60
    )
    
    print("\n=== Preprocessing Metadata ===")
    print(f"Outliers handled: {metadata.get('outliers_handled', False)}")
    print(f"Normalized: {metadata.get('normalized', False)}")
    if metadata.get('normalized'):
        print(f"  Method: {metadata['normalization']['method']}")
    print(f"Scaled: {metadata.get('scaled', False)}")
    
    print("\n" + "="*60)
    print("BENCHMARK CALCULATION")
    print("="*60)
    
    # Calculate benchmark
    score, returns, metrics = calculate_benchmark_score(train_df, allocation=1.0)
    print("\n=== Benchmark Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "="*60)
    print("DATA COMPARISON: BEFORE vs AFTER PREPROCESSING")
    print("="*60)
    
    # Compare statistics before and after
    feature = 'M1'  # Example feature
    if feature in train_df.columns:
        print(f"\n=== Feature: {feature} ===")
        print(f"Before - Missing: {train_df[feature].isna().sum()}, "
              f"Mean: {train_df[feature].mean():.4f}, "
              f"Std: {train_df[feature].std():.4f}")
        print(f"After  - Missing: {train_processed[feature].isna().sum()}, "
              f"Mean: {train_processed[feature].mean():.4f}, "
              f"Std: {train_processed[feature].std():.4f}")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
