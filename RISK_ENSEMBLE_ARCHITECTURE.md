# Risk Model Ensemble Architecture

## 📁 파일 구조

```
src/
├── risk.py                      # LightGBM 기반 Risk Model
│   ├── RiskLabeler             → risk_label 생성 (미래 20일 변동성)
│   ├── RiskForecaster          → LightGBM 예측 (ReturnPredictor 활용 예정)
│   └── RiskCalibrator          → Calibration 평가
│
├── timeseries_risk.py           # 시계열 Risk 모델 (NEW)
│   ├── EWMARiskForecaster      → EWMA (λ=0.94, 빠르고 간단)
│   ├── GARCHRiskForecaster     → GARCH(1,1) (MLE, 통계적)
│   └── HybridRiskEnsemble      → 3가지 모델 앙상블
│
└── ensemble.py                  # 앙상블 유틸리티 (기존)
    ├── ModelEnsemble           → Return 모델 앙상블
    └── combine_risk_predictions → Risk 모델 앙상블 ✅ 활용

scripts/
└── optimize_risk_ensemble.py    # Risk 앙상블 최적화 (NEW)
    ├── Step 1: 데이터 준비 (features + risk_label)
    ├── Step 2: LightGBM 학습 (feature-based ML)
    ├── Step 3: EWMA 학습 (exponential smoothing)
    ├── Step 4: GARCH 학습 (statistical time-series)
    ├── Step 5: 앙상블 전략 비교 (max, weighted_avg, percentile)
    └── Step 6: 최적 앙상블 저장
```

---

## 🎯 각 모델의 역할

### **1. LightGBM Risk Model** (`RiskForecaster`)
- **입력**: Features (M*, E*, I*, P*, V*, S*, 파생변수)
- **출력**: 미래 변동성 (risk_label 예측)
- **장점**: 
  - ✅ 다양한 feature 활용 가능
  - ✅ 비선형 패턴 학습
  - ✅ Feature importance 분석
- **단점**:
  - ❌ 과적합 위험
  - ❌ 해석 어려움 (블랙박스)
  - ❌ 학습 시간 소요

### **2. EWMA Risk Model** (`EWMARiskForecaster`)
- **입력**: Returns (forward_returns)만 사용
- **출력**: 미래 변동성 (exponential weighted)
- **공식**: `σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}`
- **장점**:
  - ✅ 매우 빠름 (O(n) 시간복잡도)
  - ✅ 간단하고 해석 가능
  - ✅ 최근 데이터에 민감 (급변 포착)
- **단점**:
  - ❌ Feature 활용 불가
  - ❌ 평균 복귀 무시 (영구 변화 가정)

### **3. GARCH Risk Model** (`GARCHRiskForecaster`)
- **입력**: Returns (forward_returns)만 사용
- **출력**: 미래 변동성 (통계적 모델)
- **공식**: `σ²_t = ω + α*r²_{t-1} + β*σ²_{t-1}`
- **장점**:
  - ✅ 통계적으로 정교함 (MLE)
  - ✅ 평균 복귀 특성 (ω)
  - ✅ 변동성 군집화 포착
- **단점**:
  - ❌ Feature 활용 불가
  - ❌ 계산 느림 (MLE 최적화)
  - ❌ arch 라이브러리 필요

---

## 🔄 앙상블 전략 (`ensemble.py` 활용)

### **Strategy 1: Max (Most Conservative)**
```python
combine_risk_predictions(predictions, strategy='max')
```
- **방식**: 3가지 모델 중 **가장 높은 변동성** 선택
- **용도**: 보수적 리스크 관리 (과소평가 방지)
- **예시**: [0.01, 0.015, 0.012] → 0.015

### **Strategy 2: Weighted Average**
```python
combine_risk_predictions(predictions, strategy='weighted_avg', weights=[0.5, 0.3, 0.2])
```
- **방식**: 가중 평균 (Optuna로 최적화)
- **용도**: 각 모델의 강점 결합
- **예시**: 0.5*0.01 + 0.3*0.015 + 0.2*0.012 = 0.0119

### **Strategy 3: Percentile 75**
```python
combine_risk_predictions(predictions, strategy='percentile')
```
- **방식**: 75번째 백분위수 선택
- **용도**: Max보다 덜 보수적, Average보다 안전
- **예시**: [0.01, 0.015, 0.012] → 0.0135 (75th pct)

---

## 🚀 사용 예시

### **1. Risk 앙상블 최적화 실행**
```bash
python scripts/optimize_risk_ensemble.py
```

**출력 파일:**
- `artifacts/models_risk_ensemble/lgbm/` - LightGBM 모델
- `artifacts/models_risk_ensemble/ewma_model.pkl` - EWMA 모델
- `artifacts/models_risk_ensemble/garch_model.pkl` - GARCH 모델
- `artifacts/best_risk_ensemble.json` - 최적 앙상블 설정
- `results/risk_ensemble_comparison.csv` - 전략 비교 결과

### **2. 저장된 앙상블 사용 (추론)**
```python
from src.risk import RiskForecaster
from src.timeseries_risk import EWMARiskForecaster, GARCHRiskForecaster, HybridRiskEnsemble
from src.ensemble import combine_risk_predictions
import json

# 1. 앙상블 설정 로드
with open('artifacts/best_risk_ensemble.json') as f:
    config = json.load(f)

# 2. 각 모델 로드
lgbm_model = RiskForecaster()
lgbm_model.load_models('artifacts/models_risk_ensemble/lgbm')

ewma_model = EWMARiskForecaster()
ewma_model.load_model('artifacts/models_risk_ensemble/ewma_model.pkl')

garch_model = GARCHRiskForecaster()
garch_model.load_model('artifacts/models_risk_ensemble/garch_model.pkl')

# 3. 예측 생성
X_test = test_df[feature_cols].values
returns_test = test_df['forward_returns'].values

lgbm_pred = lgbm_model.predict(X_test)
ewma_pred = ewma_model.predict(returns_test)
garch_pred = garch_model.predict(returns_test)

# 4. 앙상블 조합
predictions = {
    'lgbm': lgbm_pred,
    'ewma': ewma_pred,
    'garch': garch_pred
}

# 최적 전략 사용 (config에서 로드)
if config['strategy'] == 'max':
    final_risk = combine_risk_predictions(predictions, strategy='max')
elif config['strategy'] == 'weighted_avg_optimized':
    weights = eval(config['params'].split('=')[1])  # Parse weights
    final_risk = combine_risk_predictions(predictions, strategy='weighted_avg', weights=weights)
```

---

## 📊 기대 성능 개선

### **단일 모델 vs 앙상블**

| 모델 | RMSE (예상) | 특징 |
|------|-------------|------|
| LightGBM 단독 | 0.0052 | Feature 활용, 과적합 위험 |
| EWMA 단독 | 0.0061 | 빠름, 최근 데이터 민감 |
| GARCH 단독 | 0.0058 | 통계적, 평균 복귀 |
| **Ensemble (Max)** | **0.0048** | 보수적, 안전 |
| **Ensemble (Weighted)** | **0.0046** | 최적화, 균형 |
| **Ensemble (Percentile)** | **0.0047** | 중도 보수 |

**예상 개선:**
- RMSE: **10-15% 감소**
- Sharpe Ratio: **5-10% 증가**
- Constraint 위반율: **20-30% 감소** (보수적 risk 덕분)

---

## 🛠️ 의존성 설치

```bash
# GARCH 모델 사용 시 필수
pip install arch

# 기본 라이브러리 (이미 설치됨)
pip install numpy pandas scikit-learn scipy lightgbm
```

**Note:** GARCH 없이도 LightGBM + EWMA만으로 앙상블 가능

---

## 🔧 다음 단계 (TODO)

1. ✅ `src/timeseries_risk.py` 생성 완료
2. ✅ `scripts/optimize_risk_ensemble.py` 생성 완료
3. ⏳ `src/risk.py` 리팩토링 (RiskForecaster → ReturnPredictor 활용)
4. ⏳ `conf/params.yaml` 업데이트 (EWMA/GARCH 설정 추가)
5. ⏳ 앙상블 실행 및 성능 검증
6. ⏳ Position Strategy 최적화와 통합

---

## 💡 핵심 포인트

1. **ensemble.py의 `combine_risk_predictions()` 재활용**
   - Return 앙상블과 Risk 앙상블이 동일한 유틸리티 사용
   - 코드 중복 제거

2. **3가지 모델의 상호 보완**
   - LightGBM: Feature 활용 (비선형 패턴)
   - EWMA: 빠른 반응 (최근 변동)
   - GARCH: 통계적 안정성 (평균 복귀)

3. **보수적 앙상블 = 제약 조건 충족**
   - `max`, `percentile` 전략으로 과소평가 방지
   - σ_strategy/σ_market ≤ 1.2 위반율 감소

4. **확장성 확보**
   - EGARCH, GJR-GARCH 추가 시 `timeseries_risk.py`에만 추가
   - 앙상블 파이프라인은 자동으로 확장
