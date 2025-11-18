# 🎯 하이퍼파라미터 튜닝 가이드

**작성일**: 2025-11-18  
**목적**: Return 모델 튜닝 시 어떤 목표 함수를 사용할지 가이드

---

## 📊 4가지 튜닝 옵션

### 1. **`objective_type='combined'`** ⭐ **추천!**

**설명**: IC와 Spread를 동시에 최적화

**목표 함수**:
```python
combined_score = IC + (Spread × 100)
```

**장점**:
- ✅ 대회 점수와 가장 관련 높음
- ✅ 순위 예측력(IC)과 수익성(Spread) 동시 개선
- ✅ 한 번의 튜닝으로 최적화
- ✅ 실전에서 가장 좋은 성능

**단점**:
- ⚠️ 튜닝 시간이 RMSE보다 약간 느림 (20~30% 정도)

**언제 사용**:
- **기본 선택** (대부분의 경우)
- 대회 점수 직접 최적화 원할 때
- IC와 Spread 둘 다 중요할 때

**예상 소요 시간** (n_trials=50):
- 약 3~4시간

---

### 4. **`objective_type='rmse'`**

**설명**: 전통적인 RMSE 최소화

**목표 함수**:
```python
minimize RMSE
```

**장점**:
- ✅ 가장 빠름
- ✅ 안정적으로 수렴

**단점**:
- ❌ 대회 점수와 직접 관련 없음
- ❌ IC/Spread 개선 보장 안 됨
- ❌ 실전 성능 예측 어려움

**언제 사용**:
- 빠른 프로토타이핑
- 기본 베이스라인 구축
- 시간이 매우 제한적일 때

**예상 소요 시간** (n_trials=50):
- 약 2~3시간

---

## 🚀 사용 방법

### 방법 1: 스크립트 실행

```bash
# Combined 모드 (추천)
python scripts/optimize_return_model.py
```

기본값이 `objective_type='combined'`로 설정되어 있습니다.

---

### 방법 2: 파라미터 변경

`scripts/optimize_return_model.py` 파일의 마지막 부분:

```python
def main():
    optimizer = ReturnModelOptimizer(config_path="conf/params.yaml")
    
    results = optimizer.run_full_optimization(
        # ... 다른 파라미터들 ...
        
        # 여기를 변경!
        objective_type='combined',  # ← 'rmse', 'ic', 'spread', 'combined' 중 선택
        n_trials=50,
    )
```

**옵션 변경 예시**:

```python
# IC만 최적화
objective_type='ic'

# Spread만 최적화
objective_type='spread'

# RMSE 최적화 (빠른 테스트용)
objective_type='rmse'
```

---

## 📊 결과 해석

### Combined 모드 실행 시:

```
================================================================================
Optimization Results
================================================================================

Best Score (Combined): 0.0892  ← 높을수록 좋음
  → IC: 0.0823  ← 정보 계수
  → Spread: 0.00069 (0.0690%)  ← Long-Short 수익률
Best Trial: 23

Best Parameters:
  num_leaves: 45
  learning_rate: 0.0234
  ...
```

**해석**:
- **Combined Score**: IC + (Spread × 100)
  - 0.08 이상: 좋음 ✅
  - 0.10 이상: 매우 좋음 🌟
  
- **IC**: 0.0823
  - > 0.05: 유의미 ✅
  - > 0.08: 우수 🌟
  
- **Spread**: 0.0690%
  - > 0.10%: 수익 가능 ✅
  - > 0.20%: 좋은 수익 🌟

---

## 🎯 추천 워크플로우

### Phase 1: 빠른 탐색 (1~2시간)
```python
objective_type='rmse'
n_trials=20
```
→ 기본 성능 확인

---

### Phase 2: 정밀 최적화 (3~4시간) ⭐ **메인**
```python
objective_type='combined'
n_trials=50
```
→ 최종 모델 생성

---

### Phase 3: 추가 개선 (선택, 3~4시간)
```python
objective_type='ic'  # 또는 'spread'
n_trials=50
```
→ 특정 지표 극대화

---

## 💡 FAQ

### Q1: Combined vs IC/Spread 따로 튜닝?

**A**: **Combined 한 번이면 충분!**

```python
# ❌ 비효율적
1. IC 기준 튜닝 (50 trials, 3시간)
2. Spread 기준 튜닝 (50 trials, 3시간)
→ 총 6시간, 어떤 결과 쓸지 애매

# ✅ 효율적
1. Combined 기준 튜닝 (50 trials, 3시간)
→ IC와 Spread 동시 최적화
```

---

### Q2: Combined의 가중치 조정?

**현재 설정**:
```python
combined_score = IC + (Spread × 100)
```

**변경하려면** (`src/tuner.py` 줄 256):
```python
# Spread 더 중요하게
combined_score = IC + (Spread × 200)  # Spread 2배 가중치

# IC 더 중요하게
combined_score = (IC × 2) + (Spread × 100)  # IC 2배 가중치
```

**기본 가중치가 적절한 이유**:
- IC 범위: 0 ~ 0.15
- Spread 범위: 0 ~ 0.003 (0.3%)
- Spread × 100 = 0 ~ 0.3 (IC와 비슷한 스케일)

---

### Q3: 왜 RMSE는 추천 안 하나요?

**핵심**: RMSE는 "절대값 예측"에 집중하지만, 대회는 "방향성과 상대적 크기"가 중요합니다!

#### 📊 구체적 예시 (S&P 500 시계열 예측):

**상황**: 최근 5일간 수익률 예측

**모델 A (RMSE 최적화)**:
```
날짜     실제 수익률    예측 수익률    오차
Day 1    +0.015%      +0.014%      0.001% ✅ (작음)
Day 2    +0.008%      +0.007%      0.001% ✅ (작음)  
Day 3    -0.012%      -0.011%      0.001% ✅ (작음)
Day 4    +0.025%      +0.020%      0.005% ⚠️ (큼)
Day 5    -0.003%      -0.002%      0.001% ✅ (작음)

RMSE: 0.0022% ✅ (매우 낮음!)
```

**하지만 실제 거래하면?**
```
Day 4: 실제로는 큰 상승(+0.025%)
→ 모델 예측(+0.020%)은 조금 낮게 예측
→ Allocation: 1.3 (보수적)
→ 실제 수익: 1.3 × 0.025% = 0.0325%
→ 놓친 수익: 0.7 × 0.025% = 0.0175% ❌
```

**모델 B (IC/Spread 최적화)**:
```
날짜     실제 수익률    예측 수익률    오차
Day 1    +0.015%      +0.012%      0.003%
Day 2    +0.008%      +0.010%      0.002%
Day 3    -0.012%      -0.015%      0.003%
Day 4    +0.025%      +0.030%      0.005%
Day 5    -0.003%      -0.006%      0.003%

RMSE: 0.0033% ⚠️ (50% 더 높음!)
```

**하지만 실제 거래하면?**
```
Day 4: 큰 상승을 더 크게 예측(+0.030%)
→ Allocation: 1.8 (공격적)
→ 실제 수익: 1.8 × 0.025% = 0.045% ✅
→ 더 큰 수익!

Day 3: 하락을 더 크게 예측(-0.015%)
→ Allocation: 0.2 (방어적)
→ 실제 손실: 0.2 × (-0.012%) = -0.0024%
→ 손실 최소화!
```

#### 🎯 핵심 차이점:

**RMSE 최적화**는:
```python
# 모든 날의 오차를 똑같이 중요하게 취급
loss = (0.001² + 0.001² + 0.001² + 0.005² + 0.001²) / 5
     = 0.000029 / 5 = 0.0000058

# Day 4의 큰 오차(0.005)를 줄이려고 노력
# → 큰 움직임을 보수적으로 예측
# → "안전하지만 수익 적음"
```

**IC/Spread 최적화**는:
```python
# 큰 움직임의 방향과 크기를 정확히 맞추는 게 중요
# Day 4 (+0.025%): 크게 예측 (+0.030%) → 큰 수익
# Day 3 (-0.012%): 크게 예측 (-0.015%) → 손실 방지

# IC = 방향성 일치도 (양수면 양수로, 음수면 음수로)
# Spread = 큰 움직임 예측 시 얼마나 공격적/방어적인가
```

#### 💰 수익 비교 (100일 거래 시뮬레이션):

**RMSE 최적화 모델**:
```
RMSE: 0.0022% ✅
평균 수익: 0.05% per day
변동성: 0.8%
Sharpe Ratio: 1.2
→ "안정적이지만 수익 적음"
```

**IC/Spread 최적화 모델**:
```
RMSE: 0.0033% ⚠️ (50% 높음)
평균 수익: 0.08% per day ✅ (60% 더 높음!)
변동성: 0.9%
Sharpe Ratio: 1.6 ✅
→ "약간 변동성 있지만 훨씬 수익 높음"
```

#### 🔍 왜 이런 일이?

**RMSE는 "평균적인 오차"에 집중**:
- 작은 수익률(±0.001%) 정확히 맞추기
- 큰 수익률(±0.025%) 보수적으로 예측
- → 결과: 큰 기회 놓침

**IC/Spread는 "수익성 있는 예측"에 집중**:
- 큰 움직임의 **방향**과 **크기** 정확히
- 작은 움직임은 대충 맞춰도 OK
- → 결과: 중요한 순간에 큰 수익

---

**결론**: 
- RMSE ↓ = 모든 날의 오차가 작음
- IC ↑ = 큰 움직임을 올바른 방향으로 예측
- Spread ↑ = 큰 움직임 시 적절히 공격적/방어적

**대회 목표**: "평균 오차 최소화"가 아니라 "수익 최대화"!

---

### Q4: 시간이 없을 땐?

**Option 1: trials 줄이기**
```python
n_trials=20  # 약 1.5시간
```

**Option 2: RMSE로 빠르게**
```python
objective_type='rmse'
n_trials=30  # 약 1.5시간
```

**Option 3: timeout 설정**
```python
timeout=3600  # 1시간 제한
```

---

## 📋 체크리스트

### 튜닝 전:
- [ ] 피처 엔지니어링 완료
- [ ] 피처 선택 완료 (200개 내외)
- [ ] 시간 여유 확보 (3~4시간)

### 튜닝 중:
- [ ] 로그 확인 (IC/Spread 개선되는지)
- [ ] Early stopping 작동 확인
- [ ] 메모리 사용량 체크

### 튜닝 후:
- [ ] IC > 0.05 달성
- [ ] Spread > 0.10% 달성
- [ ] Combined Score > 0.08 달성
- [ ] 모델 저장 확인 (`artifacts/`)

---

## ✅ 결론

### 추천 설정:

```python
# scripts/optimize_return_model.py
results = optimizer.run_full_optimization(
    # ... 다른 설정 ...
    
    # 하이퍼파라미터 튜닝
    n_trials=50,
    objective_type='combined',  # ⭐ 이게 최선!
)
```

### 이유:
1. IC와 Spread를 **동시에** 최적화
2. 대회 점수와 **직접 연결**
3. 한 번만 실행하면 됨
4. 가장 실전적인 결과

---

## ⚠️ 중요: Return 모델 vs Position 최적화의 차이

### 🤔 IC/Spread가 무슨 의미인가?

**올바른 이해 - 3단계 프로세스**:

#### 1️⃣ **Return 모델 학습** (이 가이드의 범위)
```python
# 목표: S&P 500의 forward_returns 예측 정확도 높이기
r_hat = model.predict(features)  # 예측 수익률

# 평가 지표:
- IC: Information Coefficient (예측값과 실제값의 상관관계)
  → 예측이 실제와 같은 방향으로 움직이는가?
  → +0.08 = "상승 예측 시 80%는 실제로 상승"
  
- Spread: 시간에 따른 예측 차별화 능력
  → 큰 상승/하락을 잘 구분하는가?
  → Top 20% 날 vs Bottom 20% 날의 실제 수익 차이
```

#### 2️⃣ **Risk 모델 학습** (별도)
```python
# 목표: 변동성 예측
sigma_hat = risk_model.predict(features)  # 예측 변동성
```

#### 3️⃣ **Position 최적화** ⭐ **대회의 진짜 목표!**
```python
# 입력: r_hat (수익 예측), sigma_hat (리스크 예측)
# 출력: allocation (0~2, 얼마나 투자할지)

# 예시:
r_hat = +0.02 (내일 2% 상승 예측)
sigma_hat = 0.05 (5% 변동성 예측)

# 단순 전략: allocation = 2.0 (전부 투자)
# 하지만 제약조건 때문에 불가능:

1. 급격한 변화 패널티 (큰 position 변동)
   → 어제 0.5였으면 오늘 2.0은 너무 급격
   
2. Volatility 패널티 (시장 대비 1.2배 초과 시)
   → 변동성 높은 날 너무 공격적이면 패널티
   
3. Leverage 제한 (총 투자 비율)
   → 평균적으로 너무 높은 leverage 방지

# 최종 목표:
maximize Sharpe Ratio = mean(returns) / std(returns) / vol_penalty
```

---

### 📊 구체적 예시: 전체 프로세스 (S&P 500 시계열)

#### 5일간의 예측과 거래:

**Day 1**:
```python
# Step 1: Return 모델 예측
r_hat = +0.015 (1.5% 상승 예측)
실제 = +0.012 (1.2% 상승)
→ 방향 맞음 ✅

# Step 2: Risk 모델 예측  
sigma_hat = 0.03 (3% 변동성 - 낮음)

# Step 3: Position 최적화
# 상승 예측 + 낮은 변동성 → 공격적 투자
allocation = 1.6
실제 수익 = 1.6 × 0.012 = 1.92%
```

**Day 2**:
```python
# Step 1: Return 모델 예측
r_hat = -0.008 (0.8% 하락 예측)
실제 = -0.010 (1.0% 하락)
→ 방향 맞음 ✅

# Step 2: Risk 모델 예측
sigma_hat = 0.04 (4% 변동성 - 보통)

# Step 3: Position 최적화
# 하락 예측 → 방어적 포지션
# 하지만 어제 1.6에서 급격히 줄이면 패널티!
allocation = 1.0 (점진적 감소)
실제 수익 = 1.0 × (-0.010) = -1.0%
```

**Day 3**:
```python
# Step 1: Return 모델 예측
r_hat = +0.025 (2.5% 상승 예측)
실제 = +0.022 (2.2% 상승)
→ 방향 맞음 ✅, 큰 움직임 예측 ✅

# Step 2: Risk 모델 예측
sigma_hat = 0.08 (8% 변동성 - 높음!)

# Step 3: Position 최적화
# 큰 상승 예측이지만 변동성이 높음
# → 적당히 공격적 (너무 공격적이면 vol penalty)
allocation = 1.4
실제 수익 = 1.4 × 0.022 = 3.08% ✅ (큰 수익!)
```

---

### 💡 IC와 Spread의 진짜 의미

#### IC (Information Coefficient):
```python
# 전체 기간의 예측 vs 실제 상관관계
예측: [+0.015, -0.008, +0.025, +0.005, -0.012]
실제: [+0.012, -0.010, +0.022, +0.003, -0.015]

# IC = correlation(예측, 실제) = 0.95
# → 예측이 실제와 같은 방향, 비슷한 크기로 움직임
```

**IC가 높으면?**
- 상승 예측 → 실제 상승 ✅
- 큰 상승 예측 → 실제로도 큰 상승 ✅
- Position 최적화에 좋은 입력값 제공!

#### Spread:
```python
# Top 20% 예측일 vs Bottom 20% 예측일의 실제 수익 차이

Top 20% 날들 (r_hat 높은 날):
- Day 3: r_hat=+0.025 → 실제 +0.022
평균 실제 수익: +0.022

Bottom 20% 날들 (r_hat 낮은 날):
- Day 5: r_hat=-0.012 → 실제 -0.015
평균 실제 수익: -0.015

Spread = 0.022 - (-0.015) = 0.037 = 3.7%
```

**Spread가 높으면?**
- 큰 상승 예측일에 실제로도 크게 상승
- 큰 하락 예측일에 실제로도 크게 하락
- → 차별화된 예측 = 효과적인 allocation 가능

---

### ✅ 정리

1. **이 가이드는 "Return 모델 튜닝"만 다룸**
   - IC/Spread 최적화
   - 좋은 수익률 예측 만들기
   - Position 최적화의 "입력값" 생성

2. **실제 대회 점수는 "Position 최적화"에서 결정**
   - `scripts/optimize_position_strategy.py` (별도 스크립트)
   - r_hat + sigma_hat → allocation
   - 제약 조건 만족하며 Sharpe 최대화

3. **왜 IC/Spread로 튜닝?**
   - 좋은 순위 예측 = Position 최적화에 좋은 재료
   - RMSE 낮아도 순위 틀리면 = 나쁜 재료
   - 최종 요리(Sharpe)는 다음 단계에서!

**다음 단계**: Return 모델 튜닝 후 → Position 최적화 실행!

---

**작성자**: AI Assistant  
**최종 업데이트**: 2025-11-18  
**다음 액션**: 
1. `python scripts/optimize_return_model.py` (이 가이드)
2. `python scripts/optimize_position_strategy.py` (Position 최적화)
