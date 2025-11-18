# 📊 Return 모델 종합 평가 가이드

**작성일**: 2025-11-18  
**목적**: Return 예측 모델(r_hat)의 품질을 종합적으로 평가하는 기준

---

## 🎯 평가 기준 요약

| 지표 | 최소 | 목표 | 우수 | 현재 | 상태 |
|------|------|------|------|------|------|
| **Information Coefficient (IC)** | 0.03 | 0.05 | 0.10+ | 0.0278 | ⚠️ 개선 필요 |
| **Directional Accuracy** | 52% | 55% | 60%+ | 52.87% | ⚠️ 최소 통과 |
| **Correlation** | 0.05 | 0.10 | 0.20+ | 0.0359 | ⚠️ 매우 약함 |
| **Long-Short Spread** | 0.10% | 0.20% | 0.30%+ | 0.047% | ❌ 너무 작음 |

**종합 평가**: ⚠️ **대폭 개선 필요**

---

## 📈 1. 예측 정확도 (Prediction Quality)

### 1.1 Information Coefficient (IC) ⭐⭐⭐ 가장 중요

**정의**: 예측 순위와 실제 수익률 순위의 상관관계

```python
IC = correlation(rank(r_hat), rank(actual_returns))
```

**평가 기준**:
- ❌ **< 0.03**: 예측력 거의 없음 (랜덤 수준)
- ⚠️ **0.03 ~ 0.05**: 약한 예측력
- ✅ **0.05 ~ 0.10**: 유의미한 예측력
- 🌟 **> 0.10**: 매우 우수한 예측력
- 🏆 **> 0.15**: 탁월한 예측력 (드묾)

**왜 중요한가?**:
- IC가 높다 = "어떤 날이 더 많이 오를지" 순서를 잘 예측
- 절대값 예측보다 **상대적 순위** 예측이 중요
- Long-Short 전략의 핵심 지표

**개선 방법**:
- 피처 엔지니어링 개선 (시차, 상호작용, 도메인 지식)
- 하이퍼파라미터 튜닝 (n_trials 증가)
- 앙상블 (여러 모델 조합)

---

### 1.2 Directional Accuracy ⭐⭐

**정의**: 방향(상승/하락)을 맞춘 비율

```python
Directional Accuracy = mean(sign(r_hat) == sign(actual_returns))
```

**평가 기준**:
- ❌ **< 52%**: 랜덤보다 나쁨
- ⚠️ **52% ~ 55%**: 최소 통과
- ✅ **55% ~ 60%**: 좋음
- 🌟 **> 60%**: 매우 우수

**왜 중요한가?**:
- 방향만 맞춰도 수익 가능
- 간단하고 직관적인 지표
- Position mapping의 기본

**현재 상태**: 52.87% → 겨우 통과, 개선 필요

---

### 1.3 Correlation ⭐

**정의**: 예측값과 실제값의 선형 상관관계

```python
Correlation = pearson_correlation(r_hat, actual_returns)
```

**평가 기준**:
- ❌ **< 0.05**: 거의 무관
- ⚠️ **0.05 ~ 0.10**: 약한 상관
- ✅ **0.10 ~ 0.20**: 괜찮은 상관
- 🌟 **> 0.20**: 강한 상관

**왜 중요한가?**:
- 예측 크기가 실제 크기와 비례하는지
- IC보다 이상치에 민감

**현재 상태**: 0.0359 → 매우 약함, 대폭 개선 필요

---

## 💰 2. 수익성 (Profitability)

### 2.1 Long-Short Spread ⭐⭐⭐ 실전 핵심

**정의**: 상위 20% 예측의 실제 평균 수익률 - 하위 20% 예측의 실제 평균 수익률

```python
top_20_return = mean(actual_returns[r_hat >= percentile(r_hat, 80)])
bottom_20_return = mean(actual_returns[r_hat <= percentile(r_hat, 20)])
spread = top_20_return - bottom_20_return
```

**평가 기준**:
- ❌ **< 0.10%**: 거래비용 고려 시 수익 불가능
- ⚠️ **0.10% ~ 0.20%**: 최소 수익 가능
- ✅ **0.20% ~ 0.30%**: 좋은 수익성
- 🌟 **> 0.30%**: 매우 높은 수익성

**왜 중요한가?**:
- **실제 수익과 직결**되는 지표
- IC가 높아도 Spread가 작으면 수익 안 남
- 거래비용 (약 0.01~0.02%) 고려 필수

**현재 상태**: 0.047% → 거래비용 빼면 손실! **최우선 개선 필요**

**개선 방법**:
- 극단값 예측력 향상 (상위/하위 구간 집중)
- 변동성 큰 구간 가중치 증가
- 앙상블로 확신도 높은 예측만 사용

---

### 2.2 Quintile Analysis

**Top/Bottom Quintile Returns 확인**:
```python
Top 20% 예측 → 실제 평균 수익률이 높아야 함
Bottom 20% 예측 → 실제 평균 수익률이 낮아야 함
```

**좋은 모델 예시**:
```
Top 20%: +0.25% (상위 예측이 실제로 많이 올랐음)
Bottom 20%: -0.05% (하위 예측이 실제로 내렸음)
Spread: 0.30% ✅
```

**나쁜 모델 예시** (현재):
```
Top 20%: +0.10%
Bottom 20%: +0.053%
Spread: 0.047% ❌ (거의 차이 없음)
```

---

## 🔄 3. 안정성 (Stability)

### 3.1 Fold 간 일관성

**확인 방법**:
```python
# 각 Fold별 IC 확인
Fold 1: IC = 0.08
Fold 2: IC = 0.07
Fold 3: IC = 0.06
Fold 4: IC = 0.05
평균: 0.065, 표준편차: 0.012 ✅ 안정적
```

**불안정한 예시**:
```python
Fold 1: IC = 0.12
Fold 2: IC = 0.03  ← 큰 차이
Fold 3: IC = -0.02 ← 음수!
Fold 4: IC = 0.08
평균: 0.05, 표준편차: 0.06 ❌ 매우 불안정
```

**판단 기준**:
- IC 표준편차 < 0.03: 안정적 ✅
- IC 표준편차 > 0.05: 불안정 ⚠️
- 음수 IC 있음: 심각한 문제 ❌

---

### 3.2 시간별 성능 (Temporal Stability)

**확인 방법**:
```python
# 최근 vs 과거 성능 비교
최근 1년 IC: 0.08
과거 2년 IC: 0.05
→ 최근 성능이 더 좋으면 개선 중 ✅

최근 1년 IC: 0.03
과거 2년 IC: 0.10
→ 성능 저하, 시장 변화 적응 못함 ❌
```

---

## 🎯 4. 종합 평가 체크리스트

### ✅ 최소 요구사항 (Production Ready)
- [ ] IC > 0.05
- [ ] Directional Accuracy > 55%
- [ ] Long-Short Spread > 0.15%
- [ ] Fold 간 IC 표준편차 < 0.03
- [ ] 모든 Fold에서 IC > 0.03

### 🌟 우수 모델 기준
- [ ] IC > 0.10
- [ ] Directional Accuracy > 60%
- [ ] Long-Short Spread > 0.25%
- [ ] Correlation > 0.15
- [ ] 시간별 성능 안정적

---

## 🚀 현재 모델 개선 로드맵

### Priority 1: 하이퍼파라미터 튜닝 ⚠️ **즉시**
```python
# 현재: n_trials=1 (거의 튜닝 안 됨)
# 목표: n_trials=50~100
```

**기대 효과**:
- IC: 0.028 → 0.05~0.08 (예상)
- Spread: 0.047% → 0.10~0.15% (예상)

---

### Priority 2: 피처 엔지니어링 강화
- [ ] 시차 피처 추가 (lag 1~5일)
- [ ] 상호작용 피처 (M × V, I × P)
- [ ] 변동성 regime 피처
- [ ] 모멘텀 지표 추가

**기대 효과**:
- IC: +0.02~0.04
- Spread: +0.05~0.10%

---

### Priority 3: 앙상블
- [ ] LightGBM + CatBoost
- [ ] 서로 다른 윈도우 크기
- [ ] Stacking

**기대 효과**:
- IC: +0.01~0.03
- 안정성 향상

---

## 📊 5. 최종 평가 시나리오

### Scenario A: IC만 높은 경우
```
IC: 0.12 ✅
Directional Accuracy: 51% ❌
Spread: 0.08% ❌
```
**판단**: 순위는 잘 예측하지만 실제 수익은 낮음 → **수익성 부족**

---

### Scenario B: Spread만 높은 경우
```
IC: 0.04 ❌
Directional Accuracy: 58% ✅
Spread: 0.25% ✅
```
**판단**: 방향은 잘 맞추지만 순위 예측 약함 → **일부 구간에만 과적합**

---

### Scenario C: 균형잡힌 모델 (이상적)
```
IC: 0.08 ✅
Directional Accuracy: 57% ✅
Spread: 0.20% ✅
Correlation: 0.12 ✅
```
**판단**: 모든 지표 양호 → **Production Ready** 🎉

---

## 🔍 6. 실전 팁

### IC vs Spread 중 뭐가 더 중요?

**단기 트레이딩**: Spread > IC
- 매일 거래 → 거래비용 중요
- Spread 0.20% 이상 필수

**장기 투자**: IC ≥ Spread
- 순위 기반 포트폴리오
- IC 0.08 이상 중요

**이 대회**: **둘 다 중요!**
- Daily rebalancing → Spread 중요
- Volatility penalty → IC도 중요

---

### 언제 모델을 버려야 하나?

다음 중 하나라도 해당되면 재설계:
- IC < 0.03 (50회 이상 튜닝 후에도)
- Spread < 0.10% (개선 안 됨)
- Fold 간 IC 부호 바뀜
- 최근 성능이 계속 하락

---

## 📝 7. 평가 로그 해석

### 좋은 로그 예시:
```
📊 Return Model Performance:
  🎯 Directional Accuracy: 58.23%  ← 좋음
     (> 0.52 is profitable)
  ⭐ Information Coefficient: 0.0892  ← 우수
     (> 0.05 is significant, > 0.10 is excellent)
  📈 Correlation: 0.1234  ← 괜찮음

💰 Quintile Analysis:
  Top 20% → Avg Return: 0.28%  ← 높음
  Bottom 20% → Avg Return: -0.06%  ← 낮음
  Long-Short Spread: 0.34%  ← 매우 좋음
```

---

### 나쁜 로그 예시 (현재):
```
📊 Return Model Performance:
  🎯 Directional Accuracy: 52.87%  ← 겨우 통과
  ⭐ Information Coefficient: 0.0278  ← 너무 낮음
  📈 Correlation: 0.0359  ← 거의 없음

💰 Quintile Analysis:
  Top 20% → Avg Return: 0.10%  ← 낮음
  Bottom 20% → Avg Return: 0.053%  ← 차이 거의 없음
  Long-Short Spread: 0.047%  ← 수익 불가능
```

---

## ✅ 결론

### Return 모델 평가는 종합적으로!

1. **IC**: 순위 예측력 (가장 중요)
2. **Spread**: 실제 수익성 (실전에서 가장 중요)
3. **Directional Accuracy**: 기본 예측력
4. **Correlation**: 크기 예측력
5. **Stability**: 시간/Fold 간 일관성

### 우선순위:
1. **IC > 0.05 + Spread > 0.15%** ← 최소 기준
2. **IC > 0.08 + Spread > 0.20%** ← 좋은 모델
3. **IC > 0.10 + Spread > 0.25%** ← 우수한 모델

### 현재 해야 할 일:
```bash
# n_trials를 50으로 늘려서 재실행
python scripts/optimize_return_model.py
```

**예상 소요 시간**: 2~4시간
**기대 효과**: IC 0.05~0.08, Spread 0.10~0.15% 달성 가능

---

**작성자**: AI Assistant  
**최종 업데이트**: 2025-11-18  
**다음 리뷰**: 하이퍼파라미터 튜닝 후
