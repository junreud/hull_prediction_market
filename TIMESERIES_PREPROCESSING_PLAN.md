# 📋 Time-Series Preprocessing Implementation Plan

**작성일**: 2025-11-12  
**목표**: 금융 시계열 데이터에 특화된 전처리 파이프라인 구축

---

## 🎯 핵심 원칙

1. **미래정보 금지**: 모든 연산은 `t-1` 시점까지의 데이터만 사용
2. **발표주기 존중**: 월/분기 지표는 발표일 기준 forward-fill만
3. **변동성 적응**: 레짐별 차등 처리 (고변동기 ≠ 저변동기)
4. **정보 압축**: 중복 피처는 PCA/클러스터링으로 축소

---

## 📊 현재 상태 분석

### ✅ 이미 구현된 기능

| 기능 | 파일 | 상태 |
|------|------|------|
| 시계열 인지 결측치 보간 | `data.py::handle_missing_values` | ✅ interpolate/EWMA |
| 롤링 윈도우 이상치 탐지 | `data.py::detect_outliers` | ✅ rolling_mad/rolling_iqr/ewma |
| 롤링 winsorization | `data.py::winsorize_outliers` | ✅ rolling method |
| 레짐 감지 | `data.py::detect_regime_changes` | ✅ 변동성 기반 분류 |
| 통합 파이프라인 | `data.py::preprocess_timeseries` | ✅ 전체 흐름 |

### ❌ 누락된 기능 (Critical)

| 기능 | 중요도 | 현재 문제 |
|------|--------|-----------|
| 발표일 정렬 | 🔴 Critical | E 그룹 선형보간 시 미래정보 유출 가능 |
| 결측 마스크 피처 | 🔴 Critical | 결측 자체가 신호인데 정보 손실 |
| 그룹별 최대 허용 공백 | 🟡 Important | 장기 결측 시 무리한 보간 |
| 공선성 처리 (PCA) | 🟡 Important | V 그룹 13개가 유사하면 과적합 |
| 레짐 기반 가중치 | 🟢 Nice-to-have | 고변동기 노이즈 완화 |
| 이벤트 더미 변수 | 🟢 Nice-to-have | 발표일 급등락 보존 |

---

## 🛠️ 구현 계획

### Phase 1: Critical 기능 (즉시 구현)

#### 1.1 발표일 정렬 기능 추가

**목표**: 거시지표(E 그룹)의 발표 시차를 반영

```python
def align_announcement_dates(
    self,
    df: pd.DataFrame,
    announcement_calendar: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Align economic indicators with their announcement dates.
    
    Args:
        df: DataFrame with date_id
        announcement_calendar: DataFrame with columns:
            - feature: Feature name (e.g., 'E1')
            - announcement_date: Actual announcement date_id
            - reference_period: Period the value refers to
    
    Returns:
        DataFrame with properly aligned values
    """
    pass
```

**구현 위치**: `DataLoader` 클래스 내  
**적용 그룹**: E (Economic), 일부 P (Valuation)

#### 1.2 결측 마스크 피처 생성

**목표**: 결측 여부와 결측 지속 기간을 피처로 추가

```python
def create_missing_masks(
    self,
    df: pd.DataFrame,
    suffix: str = '_is_missing'
) -> pd.DataFrame:
    """
    Create binary missing indicators and gap duration features.
    
    For each feature with missing values, creates:
    1. {feature}_is_missing: Binary indicator
    2. {feature}_missing_days: Days since last valid observation
    
    Args:
        df: DataFrame to process
        suffix: Suffix for mask columns
    
    Returns:
        DataFrame with additional mask features
    """
    df = df.copy()
    
    for group, features in self.feature_groups.items():
        if group == 'D':  # Skip dummy variables
            continue
            
        for col in features:
            if col not in df.columns:
                continue
                
            # Binary missing indicator
            mask_col = f"{col}{suffix}"
            df[mask_col] = df[col].isna().astype(int)
            
            # Missing duration (days since last valid value)
            duration_col = f"{col}_missing_days"
            is_missing = df[col].isna()
            
            # Calculate cumulative days missing
            missing_counter = 0
            duration_values = []
            
            for missing in is_missing:
                if missing:
                    missing_counter += 1
                else:
                    missing_counter = 0
                duration_values.append(missing_counter)
            
            df[duration_col] = duration_values
    
    return df
```

#### 1.3 그룹별 보간 전략 강화

**현재 문제**: E/I/P 그룹에 `interpolate` 사용 중 → 미래정보 위험

**수정안**:
```python
# data.py::handle_missing_values 수정
default_strategy = {
    'E': 'locf',        # ❌ interpolate → ✅ LOCF only
    'I': 'locf_median', # Interest rates: LOCF + fallback median
    'P': 'locf',        # Price/Valuation: LOCF only
    'M': 'ewma',        # Market: EWMA (현재와 동일)
    'V': 'ewma',        # Volatility: EWMA
    'S': 'ewma',        # Sentiment: EWMA
    'D': 'zero',
}

# max_gap 파라미터 추가
if group_strategy == 'locf':
    for col in group_features:
        # LOCF with max gap limit
        df[col] = df[col].fillna(method='ffill', limit=max_gap)
        
        # Fallback: training median (미래정보 없음)
        if train_df is not None:
            fallback = train_df[col].median()
        else:
            fallback = df[col].median()
        df[col] = df[col].fillna(fallback)
```

---

### Phase 2: Important 기능 (1주일 내)

#### 2.1 공선성 처리 - 상관 클러스터링

**목표**: 유사한 피처를 클러스터링하고 대표 피처만 선택

```python
def detect_feature_clusters(
    self,
    df: pd.DataFrame,
    method: str = 'correlation',
    threshold: float = 0.85,
    by_group: bool = True
) -> Dict[str, List[List[str]]]:
    """
    Detect highly correlated feature clusters.
    
    Args:
        df: DataFrame to analyze
        method: 'correlation' or 'distance'
        threshold: Correlation threshold for clustering
        by_group: Whether to cluster within feature groups
    
    Returns:
        Dictionary mapping group -> list of clusters
    """
    pass

def reduce_feature_redundancy(
    self,
    df: pd.DataFrame,
    clusters: Dict[str, List[List[str]]],
    method: str = 'representative'
) -> pd.DataFrame:
    """
    Reduce feature redundancy using clustering results.
    
    Methods:
    - 'representative': Keep only the feature with highest variance
    - 'mean': Replace cluster with mean of z-scores
    - 'pca': Replace cluster with first principal component
    
    Args:
        df: DataFrame to process
        clusters: Output from detect_feature_clusters
        method: Reduction method
    
    Returns:
        DataFrame with reduced features
    """
    pass
```

#### 2.2 그룹별 PCA

**목표**: 각 피처 그룹 내에서 차원 축소

```python
def apply_group_pca(
    self,
    df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    n_components: Dict[str, int] = None,
    variance_threshold: float = 0.95
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply PCA within each feature group.
    
    Args:
        df: DataFrame to transform
        train_df: Training data for fitting PCA
        n_components: Number of components per group
        variance_threshold: Cumulative variance to retain
    
    Returns:
        Tuple of (transformed DataFrame, PCA models dict)
    
    Example:
        >>> n_components = {
        >>>     'M': 3,  # 18 features -> 3 components
        >>>     'V': 2,  # 13 features -> 2 components
        >>> }
    """
    from sklearn.decomposition import PCA
    
    df = df.copy()
    pca_models = {}
    
    for group, features in self.feature_groups.items():
        if group == 'D' or not features:
            continue
            
        # Determine n_components
        if n_components and group in n_components:
            n = n_components[group]
        else:
            # Auto-determine based on variance threshold
            n = 'auto'
        
        # Fit PCA on training data
        fit_df = train_df if train_df is not None else df
        # ... implementation ...
    
    return df, pca_models
```

#### 2.3 레짐 기반 샘플 가중치

**목표**: 고변동성 구간의 샘플에 낮은 가중치 부여

```python
def calculate_regime_weights(
    self,
    df: pd.DataFrame,
    regime_col: str = 'regime',
    weight_map: Optional[Dict[str, float]] = None
) -> pd.Series:
    """
    Calculate sample weights based on volatility regime.
    
    Default weights:
    - Low Vol: 1.0 (standard weight)
    - Normal: 1.0
    - High Vol: 0.5 (reduced weight due to noise)
    
    Args:
        df: DataFrame with regime column
        regime_col: Name of regime column
        weight_map: Custom weight mapping
    
    Returns:
        Series of sample weights
    """
    if weight_map is None:
        weight_map = {
            'low_vol': 1.0,
            'normal': 1.0,
            'high_vol': 0.5
        }
    
    weights = df[regime_col].map(weight_map)
    weights = weights.fillna(1.0)  # Unknown regime -> normal weight
    
    return weights
```

---

### Phase 3: Nice-to-have 기능 (2주일 내)

#### 3.1 이벤트 더미 변수

**목표**: FOMC, 실적발표 등 이벤트 구간 표시

```python
def add_event_dummies(
    self,
    df: pd.DataFrame,
    event_calendar: pd.DataFrame
) -> pd.DataFrame:
    """
    Add binary event indicators.
    
    Args:
        df: DataFrame with date_id
        event_calendar: DataFrame with columns:
            - event_date: date_id of event
            - event_type: 'FOMC', 'earnings', 'CPI', etc.
            - window_before: Days before event
            - window_after: Days after event
    
    Returns:
        DataFrame with event dummy columns
    """
    pass
```

#### 3.2 Granger Causality 분석

**목표**: 피처 간 시차 인과관계 탐색

```python
def analyze_granger_causality(
    self,
    df: pd.DataFrame,
    target: str = 'forward_returns',
    max_lag: int = 5,
    significance: float = 0.05
) -> pd.DataFrame:
    """
    Test Granger causality between features and target.
    
    Args:
        df: DataFrame to analyze
        target: Target variable
        max_lag: Maximum lag to test
        significance: P-value threshold
    
    Returns:
        DataFrame with causality test results
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    # ... implementation ...
```

---

## 📁 파일 구조 변경

### 기존 구조
```
src/
  ├── data.py              # 전처리 전체
  ├── cv.py                # Cross-validation
  └── utils.py             # 유틸리티
```

### 제안 구조
```
src/
  ├── data.py              # 기본 데이터 로딩
  ├── preprocessing/
  │   ├── __init__.py
  │   ├── missing.py       # 결측치 처리 (Phase 1)
  │   ├── outliers.py      # 이상치 처리
  │   ├── scaling.py       # 스케일링
  │   ├── reduction.py     # 차원 축소 (Phase 2)
  │   ├── regime.py        # 레짐 분석
  │   └── events.py        # 이벤트 처리 (Phase 3)
  ├── cv.py
  └── utils.py
```

**장점**:
- 기능별 분리로 유지보수 용이
- 테스트 작성 쉬움
- 선택적 기능 사용 가능

---

## 🧪 검증 계획

### 1. Unit Tests

각 기능별로 테스트 작성:

```python
# tests/test_preprocessing_missing.py
def test_missing_mask_creation():
    """결측 마스크가 올바르게 생성되는지 확인"""
    pass

def test_locf_no_future_leakage():
    """LOCF가 미래 정보를 사용하지 않는지 확인"""
    pass

# tests/test_preprocessing_reduction.py
def test_pca_preserves_variance():
    """PCA가 지정된 분산을 보존하는지 확인"""
    pass
```

### 2. Integration Tests

전체 파이프라인 테스트:

```python
# tests/test_pipeline_integration.py
def test_full_preprocessing_pipeline():
    """전처리 파이프라인이 순서대로 작동하는지 확인"""
    
    loader = DataLoader()
    train_df, _ = loader.load_data()
    
    # Phase 1
    train_df = loader.create_missing_masks(train_df)
    train_df = loader.handle_missing_values(train_df, strategy='safe')
    
    # Phase 2
    clusters = loader.detect_feature_clusters(train_df)
    train_df = loader.reduce_feature_redundancy(train_df, clusters)
    
    # Validate
    assert train_df.isna().sum().sum() == 0, "No missing values should remain"
    # ... more assertions ...
```

### 3. Backtest Validation

OOF 성능으로 검증:

```python
# scripts/validate_preprocessing.py
"""
각 Phase의 전처리를 적용한 후 OOF Sharpe Ratio 비교:
- Baseline (현재 코드)
- Phase 1 (발표일 정렬 + 결측 마스크 + LOCF)
- Phase 2 (+ PCA + 레짐 가중치)
- Phase 3 (+ 이벤트 더미)
"""
```

---

## 📅 타임라인

| Phase | 기능 | 예상 시간 | 담당 |
|-------|------|-----------|------|
| **Phase 1** | 발표일 정렬 | 2일 | - |
| | 결측 마스크 | 1일 | - |
| | LOCF 강화 | 1일 | - |
| | **소계** | **4일** | |
| **Phase 2** | 상관 클러스터링 | 2일 | - |
| | 그룹 PCA | 2일 | - |
| | 레짐 가중치 | 1일 | - |
| | **소계** | **5일** | |
| **Phase 3** | 이벤트 더미 | 2일 | - |
| | Granger 분석 | 2일 | - |
| | **소계** | **4일** | |
| **총합** | | **13일 (약 2.5주)** | |

---

## 🎓 학습 포인트

### 금융 시계열 특화 지식

1. **발표일 정렬의 중요성**
   - CPI는 매월 중순 발표되지만, "2월 CPI"는 3월 15일에 발표됨
   - 2월 28일에 2월 CPI를 사용하면 **미래정보 유출**!

2. **결측의 의미**
   - 일반 ML: 결측 = 노이즈 → 제거/보간
   - 금융: 결측 = 신호 → "거래 중단", "데이터 미공개" 등의 정보

3. **변동성 레짐**
   - 2008 금융위기: 고변동성 → 노이즈 많음 → 낮은 가중치
   - 2010-2019: 저변동성 → 신호 명확 → 높은 가중치

4. **공선성의 위험**
   - VIX, ATR, Realized Vol이 모두 0.95 상관
   - 트리 모델: 같은 split을 여러 번 → 과적합
   - 선형 모델: 계수 불안정 → 해석 불가

---

## 🔗 참고 자료

- **Advances in Financial Machine Learning** (Marcos López de Prado)
  - Chapter 3: Labeling
  - Chapter 4: Sample Weights
  - Chapter 5: Fractional Differentiation

- **Machine Learning for Asset Managers** (Marcos López de Prado)
  - Chapter 2: Denoising and Detoning

- **Quantitative Trading** (Ernest Chan)
  - Chapter 2: Mean Reversion
  - Chapter 5: Risk Management

---

## ✅ Action Items

- [ ] **Week 1**: Phase 1 구현 및 테스트
  - [ ] `create_missing_masks()` 구현
  - [ ] `align_announcement_dates()` 구현 (간단 버전)
  - [ ] `handle_missing_values()` LOCF 강화
  - [ ] Unit tests 작성
  
- [ ] **Week 2**: Phase 2 구현 및 검증
  - [ ] `detect_feature_clusters()` 구현
  - [ ] `apply_group_pca()` 구현
  - [ ] `calculate_regime_weights()` 구현
  - [ ] OOF Sharpe 비교 실험
  
- [ ] **Week 3**: Phase 3 및 통합
  - [ ] `add_event_dummies()` 구현
  - [ ] 전체 파이프라인 통합
  - [ ] 노트북 업데이트 (EDA에 새 기능 추가)
  - [ ] 문서화 완료

---

**마지막 업데이트**: 2025-11-12  
**작성자**: AI Assistant  
**상태**: 📝 Draft (검토 필요)
