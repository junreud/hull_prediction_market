# 수익률 예측 모델 최적화 가이드

## 📋 개요
이 문서는 `/src` 폴더의 모든 주요 함수를 활용하여 수익률 예측 모델을 최적화하는 방법을 설명합니다.

## 🎯 최적화 파이프라인 구조

```
Step 1: 데이터 전처리 (data.py)
   ↓
Step 2: 피처 엔지니어링 (features.py)
   ↓
Step 3: 피처 선택 (features.py)
   ↓
Step 4: 하이퍼파라미터 튜닝 (tuner.py)
   ↓
Step 5: 최종 모델 학습 (models.py)
   ↓
Step 6: 모델 해석 (interpretability.py)
   ↓
Step 7: 결과 저장 및 분석
```

## 📂 사용되는 주요 모듈 및 함수

### 1. `src/data.py` - 데이터 전처리
- **`DataLoader.load_train_data()`**: 학습 데이터 로드
- **`DataLoader.preprocess_timeseries()`**: 종합 전처리 파이프라인
  - `add_missing_indicators()`: 결측치 패턴을 신호로 활용
  - `add_regime_indicators()`: 금융 위기 구간 표시
  - `winsorize_outliers()`: 이상치 처리 (0.1% winsorization)
  - `normalize_features()`: 정규화 (rank-gauss)
  - `scale_features()`: 스케일링 (robust scaler)

### 2. `src/features.py` - 피처 엔지니어링
- **`FeatureEngineering.fit_transform()`**: 전체 피처 생성 파이프라인
  - `create_rolling_features()`: 롤링 통계량
  - `create_lag_features()`: 시차 피처
  - `create_difference_features()`: 차분 피처
  - `create_interaction_features()`: 상호작용 피처
  - `create_technical_features()`: 기술적 지표 (RSI, Bollinger Bands)
  - `create_regime_features()`: 변동성 구간 분류
- **`select_features_by_importance()`**: 중요도 기반 피처 선택
- **`remove_correlated_features()`**: 중복 피처 제거

### 3. `src/tuner.py` - 하이퍼파라미터 최적화
- **`OptunaLightGBMTuner.tune()`**: Optuna 기반 베이지안 최적화
  - TPE (Tree-structured Parzen Estimator) 샘플러
  - Median Pruner로 조기 종료
  - 시계열 교차검증 통합

### 4. `src/models.py` - 모델 학습
- **`ReturnPredictor.train()`**: 교차검증 기반 모델 학습
  - 시계열 split (PurgedGroupTimeSeriesSplit)
  - OOF (Out-of-Fold) 예측
  - 피처 중요도 추적
- **`ReturnPredictor.save_models()`**: 모델 저장

### 5. `src/interpretability.py` - 모델 해석
- **`ModelInterpreter.calculate_feature_importance()`**: 피처 중요도 분석
- **`ModelInterpreter.calculate_shap_values()`**: SHAP 값 계산
- **`ModelInterpreter.get_feature_interactions()`**: 피처 상호작용 분석
- **`ModelInterpreter.save_analysis()`**: 분석 결과 저장

### 6. `src/cv.py` - 교차검증 전략
- **`PurgedGroupTimeSeriesSplit`**: 시계열 교차검증
  - 데이터 누수 방지 (purging)
  - 그룹 기반 split

### 7. `src/metric.py` - 평가 메트릭
- **`CompetitionMetric.calculate_r_hat()`**: 대회 메트릭 계산
- **`CompetitionMetric.calculate_score()`**: 전체 스코어 계산

## 🚀 실행 방법

### 방법 1: 전체 파이프라인 한 번에 실행
```bash
cd /Users/gimjunseog/projects/kaggle/Prediction_Market
python scripts/optimize_return_model.py
```

### 방법 2: 단계별 실행 (권장)
```python
from scripts.optimize_return_model import ReturnModelOptimizer

# 1. Optimizer 생성
optimizer = ReturnModelOptimizer(config_path="conf/params.yaml")

# 2. Step 1: 데이터 전처리
train_df, metadata = optimizer.step1_load_and_preprocess_data(
    add_missing_indicators=True,   # 결측치 지표 추가
    add_regime_indicators=True,    # 레짐 지표 추가
    handle_outliers=True,          # 이상치 처리
    normalize=True,                # 정규화
    scale=True                     # 스케일링
)

# 3. Step 2: 피처 엔지니어링
train_engineered = optimizer.step2_feature_engineering(train_df)

# 4. Step 3: 피처 선택
train_selected, selected_features = optimizer.step3_feature_selection(
    train_engineered,
    method='correlation',      # 또는 'mutual_info'
    top_n=200,                # Top 200 features
    remove_correlated=True,
    corr_threshold=0.95
)

# 5. Step 4: 하이퍼파라미터 튜닝
best_params = optimizer.step4_hyperparameter_tuning(
    train_selected,
    selected_features,
    n_trials=50,              # 튜닝 시행 횟수
    timeout=None              # 시간 제한 (초)
)

# 6. Step 5: 최종 모델 학습
predictor, oof_preds, oof_score = optimizer.step5_train_final_model(
    train_selected,
    selected_features,
    best_params
)

# 7. Step 6: 모델 해석
interpreter = optimizer.step6_model_interpretation(
    predictor,
    train_selected,
    selected_features,
    calculate_shap=False      # SHAP 계산 (느림)
)

# 8. Step 7: 결과 저장
optimizer.step7_save_results()
```

## 🎛️ 파라미터 튜닝 가이드

### 전처리 단계
```python
# 이상치 처리 강도 조절
winsorize_limits=(0.001, 0.001)  # 0.1% 클리핑 (기본)
winsorize_limits=(0.005, 0.005)  # 0.5% 클리핑 (더 강하게)

# 정규화 방법
normalize_method='rank_gauss'     # 순위 기반 가우스 변환 (추천)
normalize_method='log1p'          # Log 변환
normalize_method='rolling_zscore' # 롤링 Z-score

# 스케일링 방법
scale_method='robust'    # 이상치에 강함 (추천)
scale_method='standard'  # 표준 스케일링
```

### 피처 선택 단계
```python
# 선택 방법
method='correlation'   # 빠름, 선형 관계 포착
method='mutual_info'   # 느림, 비선형 관계 포착
method='variance'      # 분산 기반

# 피처 개수
top_n=200  # 기본값
top_n=150  # 더 적게 (과적합 방지)
top_n=300  # 더 많이 (정보 손실 방지)

# 상관관계 제거
corr_threshold=0.95  # 기본값
corr_threshold=0.90  # 더 엄격하게
```

### 하이퍼파라미터 튜닝 단계
```python
# 튜닝 시행 횟수
n_trials=50   # 빠른 테스트
n_trials=100  # 기본값 (권장)
n_trials=200  # 더 정밀하게

# 시간 제한
timeout=None      # 제한 없음
timeout=3600      # 1시간
timeout=7200      # 2시간
```

## 📊 결과 확인

### 1. 로그 확인
```bash
tail -f logs/optimization.log
```

### 2. 결과 파일 위치
```
artifacts/
  ├── lightgbm_best_params_optimized.json  # 최적 하이퍼파라미터
  └── models_optimized/                     # 학습된 모델들

results/
  ├── feature_selection/
  │   └── selected_features_optimized.csv  # 선택된 피처 목록
  ├── interpretability_optimized/
  │   ├── feature_importance_gain.csv      # 피처 중요도
  │   └── shap_importance.csv              # SHAP 중요도
  └── optimization/
      └── optimization_summary.json        # 전체 요약
```

### 3. 주요 메트릭 확인
```python
import json

# 최적화 결과 로드
with open('results/optimization/optimization_summary.json', 'r') as f:
    results = json.load(f)

print(f"OOF Score: {results['final_model']['oof_score']:.6f}")
print(f"Best Tuning Score: {results['hyperparameter_tuning']['best_score']:.6f}")
print(f"Final Features: {results['feature_selection']['final_features']}")
```

## 🔄 반복 실험 전략

### 실험 1: 베이스라인 (빠른 실행)
```python
results = optimizer.run_full_optimization(
    # Preprocessing
    add_missing_indicators=True,
    handle_outliers=True,
    normalize=False,        # 빠르게
    scale=True,
    # Feature Selection
    top_n_features=150,     # 적게
    # Hyperparameter Tuning
    n_trials=20,            # 빠르게
    calculate_shap=False
)
```

### 실험 2: 중간 단계 (균형)
```python
results = optimizer.run_full_optimization(
    # Preprocessing
    add_missing_indicators=True,
    add_regime_indicators=True,
    handle_outliers=True,
    normalize=True,
    scale=True,
    # Feature Selection
    top_n_features=200,
    # Hyperparameter Tuning
    n_trials=50,
    calculate_shap=False
)
```

### 실험 3: 완전 최적화 (느림, 고성능)
```python
results = optimizer.run_full_optimization(
    # Preprocessing
    add_missing_indicators=True,
    add_regime_indicators=True,
    handle_outliers=True,
    normalize=True,
    scale=True,
    # Feature Selection
    selection_method='mutual_info',  # 더 정밀
    top_n_features=300,
    # Hyperparameter Tuning
    n_trials=100,
    timeout=7200,  # 2시간
    calculate_shap=True  # SHAP 분석
)
```

## 🐛 문제 해결

### 메모리 부족
```python
# 피처 개수 줄이기
top_n_features=100

# SHAP 계산 끄기
calculate_shap=False

# 샘플링 사용
# (tuner.py 내부에서 자동 처리됨)
```

### 학습 시간이 너무 긺
```python
# 튜닝 시행 줄이기
n_trials=20

# 시간 제한 설정
timeout=1800  # 30분

# 피처 선택 강화
top_n_features=100
```

### 과적합 문제
```python
# 피처 개수 줄이기
top_n_features=100

# 정규화 강화 (config에서 조절)
# learning_rate 낮추기
# min_data_in_leaf 높이기

# 상관관계 제거 강화
corr_threshold=0.90
```

## 📈 성능 향상 팁

1. **전처리 조합 테스트**
   - 정규화 방법 변경: rank_gauss vs log1p vs rolling_zscore
   - 윈도우 크기 조절: 30 vs 60 vs 90

2. **피처 선택 전략**
   - correlation으로 빠르게 테스트 후
   - mutual_info로 정밀하게 재선택

3. **하이퍼파라미터 탐색**
   - 첫 실행: n_trials=20 (빠른 탐색)
   - 두 번째: n_trials=50 (중간)
   - 최종: n_trials=100+ (정밀)

4. **앙상블 전략**
   - 여러 설정으로 모델 학습
   - 예측값 평균 또는 가중 평균

## 🎯 예상 성능 향상

| 단계 | 예상 개선 | 시간 |
|-----|----------|------|
| 베이스라인 (기본 설정) | - | 10분 |
| + 전처리 최적화 | +1~2% | 15분 |
| + 피처 엔지니어링 | +2~3% | 20분 |
| + 피처 선택 | +1~2% | 25분 |
| + 하이퍼파라미터 튜닝 | +2~4% | 1~2시간 |
| + SHAP 기반 재선택 | +1~2% | 추가 30분 |
| **총 예상 개선** | **7~13%** | **2~3시간** |

## 📝 체크리스트

- [ ] 데이터 로드 확인
- [ ] 전처리 설정 확인
- [ ] 피처 엔지니어링 실행
- [ ] 피처 선택 완료
- [ ] 하이퍼파라미터 튜닝 완료
- [ ] 최종 모델 학습 완료
- [ ] OOF 스코어 확인
- [ ] 피처 중요도 분석
- [ ] 결과 저장 확인
- [ ] 모델 파일 저장 확인

## 🚀 빠른 시작

```bash
# 1. 의존성 확인
pip install optuna lightgbm scikit-learn pandas numpy

# 2. 설정 파일 확인
cat conf/params.yaml

# 3. 최적화 실행
python scripts/optimize_return_model.py

# 4. 결과 확인
python -c "
import json
with open('results/optimization/optimization_summary.json', 'r') as f:
    print(json.dumps(json.load(f), indent=2))
"
```
