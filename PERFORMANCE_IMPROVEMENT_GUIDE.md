# 🚀 Performance Improvement Guide

## 현재 상황 요약

### 완료된 최적화
- ✅ **LightGBM 하이퍼파라미터**: Phase 3에서 Optuna로 최적화 (고정값으로 간주)
- ✅ **Position 전략 파라미터**: Phase 4에서 최적화 (Quantile 7-bin allocations)
- ✅ **Ensemble 가중치**: Phase 6에서 최적화 ([77.9%, 14.8%, 7.2%])
- ✅ **현재 성능**: Ensemble RMSE 0.010540, Correlation 0.0229 (stacking)

### 파라미터 재튜닝의 한계
위 3가지를 **재튜닝해도 성능 개선은 매우 제한적**입니다 (이미 최적화됨).
실질적 개선은 **Feature Engineering, Model Diversity, Risk Modeling**에서 나옵니다.

---

## 🎯 성능 개선 우선순위

### 1️⃣ Feature Engineering (가장 중요! 성능의 80% 결정)

#### 현재 상태
```python
# 현재 feature 확인
results/feature_selection/selected_features_optimized.csv      # 선택된 feature
artifacts/lightgbm_feature_importance.csv                      # 중요도
```

#### 개선 방향

**A. 상호작용 Feature**
```python
# features.py에 추가
def create_interaction_features(df):
    """시장지표 × 변동성 상호작용"""
    # 시장 momentum × 변동성 → 추세 강도
    df['M_V_interaction'] = df['M_momentum'] * df['V_realized_vol']
    
    # 금리 × 가격 → 밸류에이션 조정
    df['I_P_interaction'] = df['I_fed_rate'] * df['P_pe_ratio']
    
    # 거시경제 × 시장 → 경기 사이클
    df['E_M_interaction'] = df['E_gdp_growth'] * df['M_market_return']
    
    return df
```

**B. 레짐 Feature**
```python
def create_regime_features(df):
    """시장 레짐 분류 feature"""
    # 변동성 레짐 (고/중/저)
    df['regime_vol'] = pd.qcut(df['V_realized_vol'], q=3, labels=[0, 1, 2])
    
    # 추세 레짐 (상승/횡보/하락)
    rolling_return = df['M_market_return'].rolling(20).mean()
    df['regime_trend'] = pd.cut(rolling_return, bins=[-np.inf, -0.01, 0.01, np.inf], 
                                  labels=[0, 1, 2])
    
    # 공포 레짐 (VIX 대용)
    df['regime_fear'] = (df['V_vix'] > df['V_vix'].rolling(60).quantile(0.75)).astype(int)
    
    return df
```

**C. 도메인 지식 Feature**
```python
def create_technical_features(df):
    """금융 도메인 지식 기반 feature"""
    # RSI (Relative Strength Index)
    delta = df['M_market_return'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # Bollinger Bands
    rolling_mean = df['M_market_return'].rolling(20).mean()
    rolling_std = df['M_market_return'].rolling(20).std()
    df['bollinger_upper'] = rolling_mean + 2 * rolling_std
    df['bollinger_lower'] = rolling_mean - 2 * rolling_std
    df['bollinger_position'] = (df['M_market_return'] - rolling_mean) / (rolling_std + 1e-10)
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['M_market_return'].ewm(span=12).mean()
    ema_26 = df['M_market_return'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    return df
```

**D. 시계열 Feature 다양화**
```python
def create_temporal_features(df):
    """다양한 시간 윈도우 feature"""
    windows = [5, 10, 20, 40, 60, 120]
    
    for window in windows:
        # 롤링 평균
        df[f'M_return_ma_{window}'] = df['M_market_return'].rolling(window).mean()
        
        # 롤링 표준편차
        df[f'M_return_std_{window}'] = df['M_market_return'].rolling(window).std()
        
        # 롤링 최대/최소
        df[f'M_return_max_{window}'] = df['M_market_return'].rolling(window).max()
        df[f'M_return_min_{window}'] = df['M_market_return'].rolling(window).min()
        
        # 현재값 vs 롤링 평균 비율
        df[f'M_return_ratio_{window}'] = df['M_market_return'] / (df[f'M_return_ma_{window}'] + 1e-10)
    
    return df
```

#### 실험 프로세스
```bash
# 1. features.py 수정 (위 함수 추가)
vim src/features.py

# 2. Feature selection 재실행
python scripts/test_features.py

# 3. 모델 재학습
python scripts/test_model_training.py

# 4. 성능 비교
# Before: RMSE 0.010540
# After:  RMSE 0.010xxx (개선 확인)

# 5. 개선되면 keep, 아니면 drop
```

---

### 2️⃣ Model Diversity (Ensemble 다양성 확보)

#### 현재 상태
```python
# 현재: 3개 LightGBM 모델만 앙상블
# - standard (optimized params)
# - complex (deeper trees)
# - regularized (L1/L2)
```

#### 개선 방향

**A. 다른 알고리즘 추가**
```python
# scripts/optimize_ensemble.py 수정

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

def train_diverse_models_v2(df, target_col, date_col, config_path):
    """더 다양한 모델 학습"""
    
    # 1. LightGBM (기존)
    lgbm_predictor = ReturnPredictor(config_path=config_path)
    lgbm_result = lgbm_predictor.train_cv(df, target_col, date_col)
    
    # 2. CatBoost (NEW)
    catboost_params = {
        'iterations': 200,
        'depth': 6,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_seed': 42,
        'verbose': False
    }
    catboost_predictor = CatBoostPredictor(params=catboost_params)
    catboost_result = catboost_predictor.train_cv(df, target_col, date_col)
    
    # 3. XGBoost (NEW)
    xgb_params = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    xgb_predictor = XGBPredictor(params=xgb_params)
    xgb_result = xgb_predictor.train_cv(df, target_col, date_col)
    
    # 4. Ridge (Linear for diversity)
    ridge_predictor = RidgePredictor(alpha=1.0)
    ridge_result = ridge_predictor.train_cv(df, target_col, date_col)
    
    return {
        'lgbm': lgbm_result,
        'catboost': catboost_result,
        'xgb': xgb_result,
        'ridge': ridge_result
    }
```

**B. 다른 Feature Set으로 학습**
```python
def train_feature_subset_models(df, target_col, date_col):
    """서로 다른 feature 조합으로 모델 학습"""
    
    # Feature 그룹 분리
    M_features = [col for col in df.columns if col.startswith('M_')]
    V_features = [col for col in df.columns if col.startswith('V_')]
    E_features = [col for col in df.columns if col.startswith('E_')]
    I_features = [col for col in df.columns if col.startswith('I_')]
    P_features = [col for col in df.columns if col.startswith('P_')]
    
    # Model A: 시장/변동성 위주
    df_market = df[M_features + V_features + [target_col, date_col]]
    model_A = train_model(df_market, target_col, date_col)
    
    # Model B: 거시경제/금리/가격 위주
    df_macro = df[E_features + I_features + P_features + [target_col, date_col]]
    model_B = train_model(df_macro, target_col, date_col)
    
    # Model C: 전체 feature (기존)
    model_C = train_model(df, target_col, date_col)
    
    return {'market_vol': model_A, 'macro': model_B, 'all': model_C}
```

**C. 다른 CV 전략**
```python
def train_different_cv_models(df, target_col, date_col):
    """다른 CV 전략으로 모델 학습"""
    
    # Model A: 4-fold CV (현재)
    model_A = train_cv(df, target_col, date_col, n_splits=4)
    
    # Model B: 5-fold CV
    model_B = train_cv(df, target_col, date_col, n_splits=5)
    
    # Model C: Time-series split (expanding window)
    model_C = train_expanding_cv(df, target_col, date_col)
    
    return {'cv4': model_A, 'cv5': model_B, 'expanding': model_C}
```

---

### 3️⃣ Risk Model 개선

#### 현재 상태
```python
# 현재: LightGBM regression for rolling_std(forward_returns, 20)
# artifacts/oof_sigma_hat.csv
```

#### 개선 방향

**A. GARCH 모델 추가**
```python
# src/risk.py에 추가

from arch import arch_model

class GARCHRiskModel:
    """GARCH(1,1) 변동성 예측"""
    
    def predict_volatility(self, returns, horizon=1):
        """
        GARCH(1,1) 모델로 변동성 예측
        
        σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
        """
        model = arch_model(returns, vol='GARCH', p=1, q=1)
        fitted = model.fit(disp='off')
        forecast = fitted.forecast(horizon=horizon)
        
        # Annualized volatility -> Daily volatility
        sigma_hat = np.sqrt(forecast.variance.values[-1, 0])
        return sigma_hat
```

**B. EWMA (Exponentially Weighted Moving Average)**
```python
class EWMARiskModel:
    """지수가중 이동평균 변동성"""
    
    def __init__(self, span=20):
        self.span = span
    
    def predict_volatility(self, returns):
        """EWMA volatility"""
        return returns.ewm(span=self.span).std()
```

**C. Risk Ensemble (Conservative)**
```python
def ensemble_risk_predictions(lgbm_risk, garch_risk, ewma_risk):
    """보수적 리스크 앙상블 (과소추정 방지)"""
    
    # 전략 1: Max (가장 보수적)
    risk_max = np.maximum.reduce([lgbm_risk, garch_risk, ewma_risk])
    
    # 전략 2: Weighted average (편향 보정)
    risk_weighted = 0.5 * lgbm_risk + 0.3 * garch_risk + 0.2 * ewma_risk
    
    # 전략 3: 75th percentile (극단값 배제)
    risk_75th = np.percentile([lgbm_risk, garch_risk, ewma_risk], 75, axis=0)
    
    # 최종: Max와 Weighted의 평균 (안전성 + 정확성)
    final_risk = 0.6 * risk_max + 0.4 * risk_weighted
    
    return final_risk
```

---

### 4️⃣ Position Strategy 다양화 (선택)

#### 현재 상태
```python
# 현재: Quantile Binning만 사용
# allocations: [0.00023, 0.0039, 0.087, 0.74, 0.90, 1.08, 1.85]
```

#### 개선 방향

**A. Kelly Criterion 추가**
```python
class KellyCriterionMapper(BasePositionMapper):
    """켈리 기준 포지션 매핑"""
    
    def map_positions(self, r_hat, sigma_hat):
        """
        Kelly Fraction: f* = μ / σ²
        
        Full Kelly는 변동성이 너무 크므로 Half Kelly 사용
        """
        kelly_fraction = r_hat / (sigma_hat ** 2 + 1e-10)
        
        # Half Kelly (안전)
        half_kelly = 0.5 * kelly_fraction
        
        # Clip to [0, 2]
        positions = np.clip(1 + half_kelly, 0, 2)
        
        return positions
```

**B. 전략 앙상블**
```python
def ensemble_positions(r_hat, sigma_hat):
    """여러 전략 앙상블"""
    
    # Strategy 1: Quantile Binning (현재)
    quantile_mapper = QuantileBinningMapper(config_path="conf/params.yaml")
    pos_quantile = quantile_mapper.map_positions(r_hat, sigma_hat)
    
    # Strategy 2: Sharpe Scaling
    sharpe_mapper = SharpeScalingMapper(config_path="conf/params.yaml")
    pos_sharpe = sharpe_mapper.map_positions(r_hat, sigma_hat)
    
    # Strategy 3: Kelly Criterion
    kelly_mapper = KellyCriterionMapper()
    pos_kelly = kelly_mapper.map_positions(r_hat, sigma_hat)
    
    # 가중 평균 (Optuna로 최적화 가능)
    final_position = 0.5 * pos_quantile + 0.3 * pos_sharpe + 0.2 * pos_kelly
    
    return np.clip(final_position, 0, 2)
```

---

### 5️⃣ 데이터 품질 개선

#### 현재 상태
```python
# 결측치: forward-fill + median
# 이상치: MAD 기준 윈저라이제이션
# Scaling: RobustScaler
```

#### 개선 방향

**A. 고급 결측치 처리**
```python
from sklearn.impute import KNNImputer

def advanced_imputation(df):
    """그룹별 KNN imputation"""
    
    # Economic features: KNN imputation
    E_features = [col for col in df.columns if col.startswith('E_')]
    imputer = KNNImputer(n_neighbors=5)
    df[E_features] = imputer.fit_transform(df[E_features])
    
    # Market features: Linear interpolation
    M_features = [col for col in df.columns if col.startswith('M_')]
    df[M_features] = df[M_features].interpolate(method='linear', limit_direction='both')
    
    return df
```

**B. 레짐별 스케일링**
```python
def regime_based_scaling(df):
    """변동성 레짐별로 다른 scaling"""
    
    # 변동성 레짐 분류
    vol = df['V_realized_vol'].rolling(20).mean()
    df['regime'] = pd.qcut(vol, q=3, labels=['low', 'mid', 'high'])
    
    # 레짐별 scaling
    scaler_low = RobustScaler()
    scaler_mid = RobustScaler()
    scaler_high = RobustScaler()
    
    df.loc[df['regime'] == 'low', features] = scaler_low.fit_transform(
        df.loc[df['regime'] == 'low', features]
    )
    # mid, high도 동일하게...
    
    return df
```

---

## 📋 Phase 6-7 추가 작업

### Phase 6 완료 문서
```bash
# PHASE6_COMPLETE.md 작성
- Ensemble 전략 4개 비교 결과
- 최적 가중치: [77.9%, 14.8%, 7.2%]
- 성능: Single model 0.011168 → Ensemble 0.010540 (5.6% 개선)
- Correlation: 0.0145 → 0.0229 (57% 개선)
```

### Phase 7 백테스트 파이프라인
```python
# scripts/backtest.py 생성
def backtest_strategy(allocations, forward_returns):
    """
    전략 시뮬레이션 및 리포트
    
    Returns
    -------
    report : dict
        - sharpe_ratio
        - annual_return
        - annual_volatility
        - max_drawdown
        - turnover
        - constraint_violation_rate (σ_strategy/σ_market > 1.2 비율)
        - leverage_usage (2배 레버리지 사용 비율)
    """
    pass
```

### Kaggle 제출 노트북
```python
# notebooks/30_final_train_submit.ipynb
# 
# 1. 데이터 로딩 및 전처리
# 2. Feature engineering
# 3. Return model 학습 및 예측
# 4. Risk model 학습 및 예측
# 5. Ensemble 예측
# 6. Position mapping
# 7. submission.csv 생성
```

### 런타임 최적화
```bash
# 목표: 8시간 이내 완료
# 
# 1. 메모리 프로파일링
python -m memory_profiler scripts/optimize_ensemble.py

# 2. 불필요한 feature 제거
# feature importance < 0.001 제거

# 3. 모델 수 조정
# 현재: 3개 모델 앙상블
# 최적: 2개로 축소 가능 (standard + complex)

# 4. CV fold 수 조정
# 현재: 4-fold
# 고려: 3-fold (20% 시간 단축)
```

### 제약 조건 최종 검증
```python
# 반드시 확인
constraints = {
    'vol_ratio_violation': (σ_strategy / σ_market > 1.2).mean() <= 0.02,  # ≤2%
    'leverage_usage': (allocations >= 1.9).mean() <= 0.10,                # ≤10%
    'underperformance': mean(strategy_returns) >= mean(market_returns),   # No penalty
}
```

---

## 🔍 실험 추적

### MLflow 설정
```python
import mlflow

# 실험 추적
with mlflow.start_run():
    mlflow.log_params({
        'n_models': 3,
        'ensemble_strategy': 'stacking',
        'position_strategy': 'quantile_binning'
    })
    
    mlflow.log_metrics({
        'rmse': 0.010540,
        'correlation': 0.0229,
        'sharpe': 0.0308
    })
    
    mlflow.log_artifact('artifacts/ensemble_config.json')
```

---

## ⚡ Quick Wins (빠른 성능 개선)

우선순위 높은 것부터:

1. **상호작용 feature 추가** (30분) → 예상 개선: RMSE 2-3%
2. **CatBoost 모델 추가** (1시간) → 예상 개선: RMSE 1-2%
3. **GARCH risk model 추가** (1시간) → 예상 개선: Sharpe 5-10%
4. **레짐 feature 추가** (1시간) → 예상 개선: RMSE 1-2%
5. **Position strategy 앙상블** (30분) → 예상 개선: Sharpe 3-5%

---

## 📚 참고 자료

- Feature Engineering: `/notebooks/01_feature_group_detailed_analysis.ipynb`
- Current Features: `/results/feature_selection/selected_features_optimized.csv`
- Ensemble Results: `/artifacts/ensemble_comparison.csv`
- Position Strategy: `/src/position.py`
- Risk Model: `/src/risk.py`
