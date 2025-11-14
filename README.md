## 🏦 Hull Tactical Market Prediction — Core Summary

### 🎯 **Goal**

S&P 500의 **하루 후 수익률(`forward_returns`)을 예측**하는 모델을 만들어
이를 기반으로 **일일 투자 포지션(0~2배 레버리지)** 을 결정.

---

### 📊 **Dataset**

* **train.csv**

  * `date_id`: 거래일 식별자
  * `M*`: 시장 기술적 지표 (Market/Technical)
  * `E*`: 거시경제 지표 (Economic)
  * `I*`: 금리 지표 (Interest)
  * `P*`: 밸류에이션/가격 관련 지표 (Price/Valuation)
  * `V*`: 변동성 관련 지표 (Volatility)
  * `S*`: 투자심리 지표 (Sentiment)
  * `D*`: 더미/이벤트 플래그
  * `forward_returns`: S&P500의 다음날 수익률
  * `risk_free_rate`: 연방기금금리(당일 기준, 하루마다 약간 변함)
  * `market_forward_excess_returns`: 초과수익률(= 실제 수익률 - 5년 평균 기대수익률, MAD 4기준으로 윈저라이즈됨)

* **test.csv**

  * `lagged_forward_returns`, `lagged_risk_free_rate`, `lagged_market_forward_excess_returns`: 하루 지연된 버전 제공
  * `is_scored`: 점수 계산 포함 여부

---

### ⚖️ **Evaluation Metric**

**변형된 샤프 비율 기반 메트릭**
(= 수익률 대비 변동성 보정, + 전략 과도 변동성 페널티)

[
score = \frac{mean(strategy_returns)}{std(strategy_returns)} / vol_penalty
]

* `strategy_returns = allocation_t × forward_returns_t`
* `allocation_t ∈ [0, 2]` (0=현금, 1=시장과 동일, 2=2배 레버리지)
* **vol_penalty = 1 + max(0, (strategy_vol / market_vol) - 1.2)**
  → 전략 변동성이 시장 변동성의 1.2배를 초과하면 페널티 부여
  → 너무 공격적인 전략(레버리지 과다)은 점수 감소

---

### 💡 **Key Concepts**

| 개념                                | 의미                                       |
| --------------------------------- | ---------------------------------------- |
| **forward_returns**               | S&P500의 다음날 실제 수익률                       |
| **risk_free_rate**                | 하루 단위 무위험 이자율 (연방기금금리)                   |
| **market_forward_excess_returns** | 시장의 초과수익률: (forward_returns - 과거 5년 평균)  |
| **전략 변동성**                        | 모델이 낸 포지션(레버리지 포함)에 의해 발생하는 일일 수익률의 표준편차 |
| **시장 변동성**                        | 동일 기간 S&P500의 일일 수익률 표준편차                |

---

### 📈 **전략 해석**

* 모델은 단순히 “오를지/내릴지” 뿐 아니라 **확신도**(신호 강도)를 예측해야 함.
* 확신이 높고 시장이 안정적일 때만 **레버리지↑ (최대 2배)**
* 확신 낮거나 시장이 불안정하면 **포지션↓ (0~1배)**
* 즉, **리스크 조절(Volatility Targeting)** 이 핵심.

---

### 🧠 **전략 설계 방향**

1. `forward_returns` 예측 (Return Model)
2. 단기 변동성 or 불확실성 예측 (Risk Model)
3. 두 값을 결합해 **Sharpe 기반 포지션 스케일링**
   [
   allocation_t = clip(1 + k \cdot tanh(b \cdot \frac{r_hat}{σ_hat}), 0, 2)
   ]
4. 시장 변동성 대비 1.2배 이하로 유지되도록 제어

---

### 🏁 **승리 전략 요약**

✅ 수익률 예측 정확도 +
✅ 안정적 변동성 제어(σ_strategy ≤ 1.2×σ_market) +
✅ 과적합 없는 OOF/Leaderboard 일관성

→ **“높은 Sharpe × 낮은 리스크”** 구조로 장기 안정 수익률을 만드는 모델이 승리

---

# Phase 0 — 세팅 (오늘~내일, ~11/12)

* [x] Kaggle 노트북 템플릿 복사 및 런타임 테스트(인터넷 불가 확인, 8h 내 샘플 실행)
* [x] Kaggle API 설정 및 데이터 다운로드 자동화
* [x] 리포 구조 생성(`src/ data/ conf/ artifacts/ notebook/`)
* [x] Git 버전 관리 초기화 (.gitignore 설정: artifacts/, *.parquet, *.pkl)
* [x] 환경 설정 파일 생성 (requirements.txt, environment.yml)
* [x] `params.yaml` 기본 스켈레톤 생성(윈도우, 스케일러, CV 파라미터)
* [x] 랜덤시드/재현성 유틸 추가, 로깅/타이머 유틸 추가
* [x] 유닛테스트 프레임워크 설정 (pytest)

# Phase 1 — 데이터 정리 & EDA & 벤치마크 (D-4, ~11/15)

* [x] `data.py`: 로딩, `date_id` 정렬, 중복/누락 검사, 학습/검증 분할 기준 고정
* [x] 결측 전략 확정(그룹별: E/I/P forward-fill 제한 + 중앙값, M/V/S 롤링 계산 시 누출 방지)
* [x] 결측값 패턴 분석 (MCAR, MAR, MNAR 판별)
* [x] 이상치 탐지 및 처리 전략 (MAD, IQR 기준)
* [x] 그룹별 스케일링 설계(robust/z, 학습 구간에서만 적합)  
* [x] 기본 벤치마크 계산(allocation=1 고정 → 커스텀 메트릭)
* [x] EDA 노트 작성:
  * [x] 피처 그룹별(M/E/I/P/V/S) 분포 및 기술통계
  * [x] 타겟 변수(`forward_returns`) 시계열 특성 (자기상관, 구조적 변화점)
  * [x] 피처 그룹 간 상관관계 분석 (히트맵)
  * [x] 데이터 기간별 regime 분석 (고/저 변동성 구간 식별)
  * [x] 학습/테스트 데이터 분포 차이 분석 (distribution shift)
* [x] 피처별 시간에 따른 안정성 검증 (non-stationarity 체크)

**산출물:** `00_eda.ipynb`, 벤치마크 점수, 데이터 품질 리포트

# Phase 2 — 검증 스키마 & 커스텀 메트릭 (D-10, ~11/21)

* [x] `cv.py`: 시계열 K-Fold(예: 5), `embargo/purge` 구현
  * [x] fold 수, train/validation 기간 비율 명시 (예: 60/20/20)
  * [x] embargo 기간 설정 (예: 5일)
* [x] `metric.py`: Sharpe 변형 + vol_penalty + 언더퍼포먼스 패널티 함수 구현/유닛테스트
  * [x] Edge case 처리 (0 division, nan/inf 처리)
  * [x] 메트릭 검증 (알려진 값으로 유닛테스트)
* [x] 벤치마크에 메트릭 적용 → OOF 파이프라인 뼈대 점검
* [x] 백테스트 시뮬레이션 검증 (forward-looking bias 체크)
* [x] 트랜잭션 비용 고려 여부 결정 (turnover penalty 옵션)

**산출물:** OOF 평가 가능 스캐폴딩, 검증된 메트릭 함수, 백테스트 시뮬레이터

# Phase 3 — 수익률 예측 모델(Return Regressor) (D-14, ~11/25)

* [x] `features.py`: 피처 엔지니어링
  * [x] 그룹별 파생 피처 (롤링 리턴/乖離/vol, 레짐 플래그, D* 이벤트)
  * [x] 시차 피처 (lag 1~5일)
  * [x] 차분 피처 (변화율, 가속도)
  * [x] 상호작용 피처 (M*×V*, I*×P* 등)
  * [x] 도메인 특화 피처 (모멘텀, RSI-like, 볼린저 밴드 등)
* [x] 피처 중요도 분석 및 선택 (상위 N개 선택 또는 임계값 기반)
* [x] `models.py`: LGBM(1) + CatBoost(옵션) 기본 모델 훈련
  * [x] 하이퍼파라미터 튜닝 전략 (Optuna 또는 Grid Search)
  * [x] Early stopping 설정
* [x] 시계열 CV로 OOF 예측(`r_hat`) 생성/저장
* [x] OOF 기준 성능 리포트(최근 구간 가중 평균도 병기)
* [x] 모델 해석 가능성 확보 (SHAP values, feature importance)
* [x] 예측 불확실성 정량화 (prediction intervals)

**산출물:** `artifacts/oof_r_hat.parquet`, 피처 중요도 리포트

# Phase 4 — 리스크 예측 모델(Risk Forecaster) (D-18, ~11/29)

* [x] 리스크 라벨 정의: `roll_std(forward_returns, N=20)` 미래값을 예측하도록 라벨링
* [x] LGBM 기반 리스크 회귀(간단) + (옵션) GARCH-lite
* [x] 앙상블 분산(모델 간 편차) 추정도 비교
* [x] OOF `sigma_hat` 산출/저장, 캘리브레이션 체크
* [x] 극단 시장 상황(Black Swan) 대응 전략 설계
* [x] 실시간 리스크 모니터링 지표 (rolling Sharpe, Calmar ratio)

**산출물:** `artifacts/oof_sigma_hat.parquet`, 리스크 모델 캘리브레이션 리포트

# Phase 5 — 포지션 매핑 & 리스크 제어 (핵심) (D-22, ~12/03)

* [x] 전략1: Sharpe 스케일링 `a=clip(1+k*tanh(b*r_hat/(sigma_hat+eps)),0,2)`
* [x] 전략2: 퀀타일 계단형(z=r_hat/sigma_hat, 5~7구간 a값 최적화)
* [x] 전략3: 볼 타깃팅 + 일일 변화폭 제한(Δ=0.3)
* [x] 전략4: 시장 레짐별 포지션 조정 (VIX 등 공포지수 활용)
* [x] 전략5: 연속 손실 시 포지션 축소 로직 (drawdown-based allocation)
* [x] OOF에서 **커스텀 메트릭 최대화**로 `k,b`/구간 a/Δ 탐색
* [x] 제약 체크: `σ_strategy/σ_market ≤ 1.2` **유지율 ≥ 98%**
* [x] 2배 레버리지 사용 비중 ≤ 5~10% 모니터링
* [x] 다양한 전략 비교 및 앙상블 가능성 검토

**산출물:** `position.py` 최종 로직, 튠된 하이퍼파라미터, 전략 비교 리포트

# Phase 6 — 앙상블 & 안정화 (D-26, ~12/07)

* [ ] 앙상블 전략 설계:
  * [ ] Stacking vs Blending 비교
  * [ ] 서로 다른 모델/피처셋의 `r_hat` 수평 앙상블(가중치 합=1, 비음수)
  * [ ] 시간 가중 앙상블 (최근 데이터 가중치 증가)
  * [ ] 다양성 확보 (다른 피처셋, 다른 윈도우, 다른 알고리즘)
* [ ] 가중치 소수 파라미터로 OOF 메트릭 최대화
* [ ] `sigma_hat`는 과소추정 방지 위해 max/가중평균 비교 → 선택
* [ ] 최근 구간 성능/변동성/언더퍼포먼스 패널티 재확인
* [ ] 실험 관리 시스템 활용 (MLflow/W&B로 실험 추적)
* [ ] 모델 버저닝 전략 수립

**산출물:** 최종 OOF 점수표, 안정성 리포트, 앙상블 가중치

# Phase 7 — 제출 파이프라인 구성 (D-27~D-30, ~12/10)

* [ ] `backtest.py`: 시뮬/리포트(Sharpe, 연수익/연변동성, MDD, turnover, σ비율 초과 일수)
* [ ] `30_final_train_submit.ipynb`: 학습→추론→포지션 산출→API 제출 루틴
* [ ] 런타임 최적화(≤8h; 예측단계 9h): 모델 수/피처 수 조정
  * [ ] 메모리 사용량 프로파일링 및 최적화
  * [ ] 불필요한 피처 제거, 모델 경량화
* [ ] 코드 리팩토링 및 주석 정리
* [ ] 에러 핸들링 및 로깅 강화
* [ ] Kaggle 제출 형식 검증 (submission.csv 포맷)
* [ ] 재현성 검증 (동일 시드로 여러 번 실행)
* [ ] 리허설 제출 1~2회(퍼블릭LB 과적합 여부 점검)
* [ ] 코드 전체 dry-run (clean environment에서)

**산출물:** 제출 준비 완료된 노트북, 백테스트 리포트

# 마감 전 체크리스트

* [ ] 시계열 CV + embargo 적용 완료
* [ ] 모든 스케일/롤링 파라미터는 **학습 구간에서만 적합**
* [ ] 데이터 누출(leakage) 철저히 방지 (미래 정보 사용 금지)
* [ ] `σ_strategy/σ_market ≤ 1.2` 위반 거의 없음(≤2%)
* [ ] 언더퍼포먼스 패널티(평균 초과수익<시장) 발생 없음
* [ ] 2배 레버리지 과다 사용 없음(≤10%)
* [ ] 재현 스크립트/시드 고정, 노트북 단일 실행로직 OK
* [ ] 제출 노트북 8~9h 내 완주 확인
* [ ] 코드 전체 dry-run (clean environment에서)
* [ ] 제출 전 Public LB 과적합 체크 (Public/Private split 고려)
* [ ] 최종 모델 성능 리포트 작성 (plots, tables)
* [ ] 제출 파일 백업 (여러 버전 보관)

# Stretch (시간 여유 시)

* [ ] 작은 Transformer/TCN 시도(피처 축소 후)
* [ ] 레짐 분류기(평온/변동/침체) → 포지션 상한 동적 조정
* [ ] 비용·슬리피지 모형(가벼운 가정) 반영한 로버스트성 테스트
* [ ] AutoML 도구 실험 (AutoGluon, H2O)
* [ ] 시장 마이크로구조 피처 (bid-ask spread proxy 등)
* [ ] 대체 데이터 소스 탐색 (가능한 경우)

---

# 실험 추적 & 문서화

* [ ] MLflow 또는 Weights & Biases 설정 및 사용
* [ ] 각 실험마다 메타데이터 기록 (하이퍼파라미터, 성능, 날짜)
* [ ] 주요 의사결정 및 인사이트 문서화
* [ ] 실패한 시도 기록 (무엇이 작동하지 않았는지)
* [ ] 최종 솔루션 설명 문서 작성