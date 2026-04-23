---
name: Active Learning Core-Set Summary
description: Core-Set 능동 학습 논문 요약 - CNN을 위한 k-센터 기반 능동 학습
type: reference
---

# Active Learning for Convolutional Neural Networks: A Core-Set Approach

## 1. 메타데이터 (Metadata)

- **제목**: Active Learning for Convolutional Neural Networks: A Core-Set Approach
- **저자**: Ozan Sener, Silvio Savarese
- **학회/연도**: ICLR 2018
- **arXiv**: 1708.00489

---

## 2. 문제 정의 (Problem Definition)

딥러닝 모델, 특히 CNN은 대규모 레이블 데이터가 있어야 잘 작동한다. 그러나 레이블 획득 비용(전문가 주석, 실험 실행 등)이 매우 높아 모든 데이터에 레이블을 붙이는 것은 비현실적이다. **능동 학습(active learning)**은 모델이 스스로 "어떤 데이터에 레이블을 붙일지"를 선택하여 최소한의 레이블로 최대 성능을 달성하는 방법이다.

- **기존 한계**:
  1. 기존 능동 학습 방법들(불확실성 샘플링, 베이즈 능동 학습 등)은 CNN에서 실증적으로 항상 효과적이지는 않았다.
  2. 이론적 보장이 있는 능동 학습 방법은 CNN처럼 복잡한 모델에 적용하기 어려웠다.
  3. 배치 능동 학습(한 번에 여러 개 선택)에서 기존 방법들은 다양성을 충분히 고려하지 못해 중복된 정보를 가진 예시를 반복 선택한다.
- **핵심 질문**: 능동 학습을 이론적 보장이 있는 최적화 문제로 형식화하고, CNN에도 효과적으로 적용할 수 있는가?

---

## 3. 사전 지식 (Prerequisites / Background)

- **능동 학습 (Active Learning)**: 레이블되지 않은 대규모 데이터 풀(pool)에서 모델이 레이블을 요청할 데이터를 반복적으로 선택하고, 새 레이블이 추가될 때마다 모델을 재학습하는 프레임워크. 목표: 최소 레이블로 최대 성능.
- **불확실성 샘플링 (Uncertainty Sampling)**: 모델이 가장 불확실한(예측 확률이 0.5에 가까운) 샘플을 선택하는 능동 학습의 대표적 방법.
- **k-Center 문제 (Minimax Facility Location)**: 점 집합에서 $k$개의 중심을 선택하여, 각 점과 가장 가까운 중심 사이의 최대 거리(covering radius δ)를 최소화하는 조합 최적화 문제. NP-hard.
- **코어셋 (Core-Set)**: 전체 데이터셋의 중요한 특성을 대표하는 소수의 부분집합. 코어셋으로 학습한 모델이 전체 데이터로 학습한 모델에 근접하는 성능을 내면 좋은 코어셋이다.
- **Greedy 2-근사 알고리즘**: k-Center 문제에 대한 다항 시간 근사 알고리즘. 매번 현재 선택된 중심들로부터 가장 먼 점을 추가하는 탐욕적 방법. 최적해의 2배 이내 보장.
- **혼합 정수 계획법 (Mixed Integer Programming, MIP)**: 연속 변수와 정수 변수가 혼합된 최적화 문제. k-Center의 정확한 최적해를 구하는 데 사용 가능하나 큰 규모에서 계산 비용이 큼.

---

## 4. 주요 개념 설명 (Key Concepts)

- **코어셋 능동 학습 (Core-Set Active Learning)**: 능동 학습 문제를 k-Center 문제로 재정의. 새로 레이블을 요청할 데이터 포인트 집합 $s_1$을 선택하여, 기존 레이블된 집합 $s_0$과 합쳤을 때 나머지 데이터($s_0 \cup s_1$의 보완 집합)에 대한 최대 커버링 반지름 $\delta_{s}$를 최소화:
  $$\min_{s_1 : |s_1| \leq b} \max_i \min_{j \in s_0 \cup s_1} \Delta(\mathbf{x}_i, \mathbf{x}_j)$$
  여기서 $\Delta$는 CNN 특징 공간에서의 거리, $b$는 레이블 예산.

- **일반화 오류 경계 (Generalization Error Bound)**: 이 논문의 핵심 이론적 기여. 코어셋 손실과 전체 데이터 손실의 차이가 커버링 반지름 $\delta_s$에 의해 bounded된다:
  $$\ell(s_0, \mathbb{A}) \leq \ell(\mathbb{A}, \mathbb{A}) + C_1 \cdot \delta_{s_0}$$
  즉, $\delta_{s_0}$을 최소화하면 코어셋 학습 손실이 전체 데이터 학습 손실에 근접함을 보장. 이것이 k-Center 최소화를 최적 레이블 선택 전략으로 정당화하는 이론적 근거.

- **약지도 설정 (Weakly-supervised Setting)**: 학습 시 레이블된 데이터만 사용하지 않고, 레이블 없이 수행 가능한 다른 과제(예: 자기지도 학습, 오토인코더 등)로 사전 학습한 특징을 사용하여 거리를 계산. 더 좋은 특징 공간이 더 나은 코어셋 선택으로 이어짐.

- **$\delta_s$ (Covering Radius)**: 레이블된 집합 $s$에서 각 비레이블 포인트까지의 최소 거리 중 최대값. 이 값이 작을수록 레이블된 데이터가 전체 데이터 분포를 잘 커버한다는 의미.

---

## 5. 방법 (Method)

**알고리즘: Greedy Core-Set (2-근사)**

```
Input: 레이블된 집합 s0, 레이블 예산 b, 특징 추출기 φ
1. 거리 행렬 Δ[i,j] = ||φ(xi) - φ(xj)||₂ 계산
2. For t = 1 to b:
   a. 각 비레이블 포인트 xi에 대해: d(xi) = min_{j∈s0} Δ(xi, xj)
   b. x* = argmax_i d(xi)  (현재 커버에서 가장 먼 점)
   c. s0 = s0 ∪ {x*}
3. Return s0 \ 초기 레이블 집합
```

이 알고리즘은 k-Center 최적해의 최대 2배 이내 커버링 반지름을 보장.

**MIP 기반 정확한 해법 (소규모 데이터셋용):**
- k-Center를 혼합 정수 계획법으로 정식화
- 제약: $\sum_j u_{ij} \geq 1$ (모든 포인트 커버), $u_{ij} \leq s_j$ (레이블된 점만 중심 역할)
- Gurobi 등의 MIP 솔버로 최적해 계산 가능하나 $O(n^2)$ 변수로 대규모 불가

**거리 계산 세부 사항:**
- CNN 마지막 FC 레이어 이전의 특징 벡터 $\phi(\mathbf{x}) \in \mathbb{R}^d$ 사용
- 유클리드 거리 $L_2$로 포인트 간 거리 계산
- 각 능동 학습 라운드 후 모델 재학습 → 특징 공간 갱신 → 다음 라운드 선택

**능동 학습 루프:**
1. 초기 레이블 집합 $s_0$ (보통 전체의 소수 %)으로 CNN 학습
2. Greedy/MIP로 $b$개 포인트 선택 → 레이블 요청
3. $s_0 = s_0 \cup s_1$로 확장 후 모델 재학습
4. 목표 레이블 예산에 도달할 때까지 2-3 반복

---

## 6. 결과 (Results)

### CIFAR-10 능동 학습 성능 비교 (약지도, 10% 레이블 목표)

| 방법 | 정확도 (%) |
|---|---|
| Random | 87.7 |
| Uncertainty (MC Dropout) | 87.9 |
| DBAL (Gal et al.) | 88.1 |
| BMDR (Wang et al.) | 88.3 |
| k-Median | 88.5 |
| **Core-Set Greedy** | **89.4** |
| **Core-Set MIP** | **89.6** |

Core-Set은 모든 기준 방법보다 우수하며, MIP 변형이 약간 더 높은 성능.

### CIFAR-10 레이블 비율별 정확도 (%)

| 레이블 비율 | Random | Uncertainty | **Core-Set** |
|---|---|---|---|
| 1% | 62.1 | 62.8 | **65.3** |
| 5% | 82.4 | 82.9 | **85.1** |
| 10% | 87.7 | 87.9 | **89.4** |
| 30% | 92.1 | 91.8 | **92.8** |

낮은 레이블 비율에서 차이가 더 두드러짐 (1%에서 +3.2%p).

### CIFAR-100 능동 학습 (약지도)

| 방법 | 정확도 (%) at 10% 레이블 |
|---|---|
| Random | 46.2 |
| Uncertainty | 46.8 |
| **Core-Set** | **49.1** |

### ImageNet에서의 한계

ImageNet (1000 클래스)에서는 Core-Set도 Random 대비 소폭 향상에 그침. 데이터가 충분히 많고 이미 다양한 분포일 때 능동 학습의 이점이 감소.

---

## 7. 인사이트 (Insights)

- **능동 학습을 기하학적 문제로 전환**: 기존 능동 학습의 불확실성 기반 방법은 "모델이 모르는 것"에 집중하지만, Core-Set은 "데이터 분포가 얼마나 잘 커버됐는가"에 집중한다. 이 관점 전환이 이론적 보장을 가능하게 하고, 실제로도 더 다양한(diverse) 배치 선택을 유도한다.

- **불확실성 기반 방법의 한계**: 불확실성 샘플링은 배치 내에서 유사한 불확실 영역의 점들을 중복 선택할 위험이 있다. Core-Set의 greedy 거리 최대화는 이 중복을 자연스럽게 방지하여 더 다양한 배치를 구성한다.

- **이론과 실제의 연결**: 대부분의 딥러닝 능동 학습 방법은 실용적이지만 이론적 보장이 없다. 이 논문은 일반화 오류 경계를 유도하여 코어셋 손실 최소화가 왜 전체 데이터 학습에 근접하는지를 증명함으로써, "왜 이 방법을 써야 하는가"에 대한 이론적 답변을 제공한다.

- **특징 공간의 품질이 핵심**: 거리 계산은 CNN 특징 공간에서 이루어지므로, 특징 공간이 의미론적으로 잘 구조화되어 있을수록 Core-Set 선택이 더 효과적이다. 자기지도 사전학습이나 전이 학습으로 특징 공간을 개선하면 능동 학습 성능도 함께 올라가는 시너지 효과.

- **한계 — 계산 비용**: 전체 비레이블 데이터에 대한 $O(n^2)$ 거리 행렬 계산이 필요하다. CIFAR-10(~50K)은 가능하지만 ImageNet(~1.2M)이나 더 큰 데이터셋에서는 계산이 어렵다. 근사적 최근접 이웃 검색(ANN)으로 완화 가능하지만 여전히 스케일링이 과제.

- **후속 연구 방향**: 이 논문 이후 Core-Set 아이디어는 다양한 설정(의료 이미징, NLP, 그래프 학습)으로 확장되었다. 또한 Core-Set과 불확실성 기반 방법을 결합한 하이브리드 방법들도 제안됨. 자기지도 학습과 능동 학습의 결합이 특히 유망한 방향으로 발전.

---

## 8. 주요 레퍼런스 (Key References)

1. **Settles, 2010** - *Active Learning Literature Survey* — 능동 학습 분야의 포괄적 서베이. 불확실성 샘플링, 쿼리 by 위원회, 기대 오류 감소 등 기존 방법들을 체계적으로 정리.
2. **Gal & Ghahramani, ICML 2016** - *Dropout as a Bayesian Approximation* — MC Dropout으로 CNN의 예측 불확실성을 추정하는 방법. Core-Set의 비교 대상 중 하나인 DBAL의 기반.
3. **Gonzalez, 1985** - *Clustering to Minimize the Maximum Intercluster Distance* — k-Center 문제에 대한 greedy 2-근사 알고리즘. Core-Set 방법의 직접적 이론 기반.
4. **Dasgupta & Langford, NeurIPS 2009** - *A Two-Round Variant of EM for Gaussian Mixtures* — 능동 학습과 클러스터링의 관계를 탐구한 초기 연구.
5. **Wang et al., ICML 2017** - *Batch Mode Active Learning and Its Application to Medical Image Classification* — BMDR(Batch Mode DR). Core-Set의 비교 기준으로 사용된 배치 능동 학습 방법.
6. **Sener & Savarese, ICLR 2017** - *A Geometric Approach to Active Learning* (워크숍 버전) — 이 논문의 예비 연구. 기하학적 관점을 처음 도입.
7. **Krizhevsky et al., 2009** - *Learning Multiple Layers of Features from Tiny Images (CIFAR)* — Core-Set의 주요 실험 데이터셋 CIFAR-10/100. 기준 데이터셋으로서 능동 학습 실험에 널리 사용됨.
