---
name: Class-Balanced Loss Summary
description: Class-Balanced Loss 논문 요약 - 유효 샘플 수 기반 장기 꼬리 분포 불균형 학습
type: reference
---

# Class-Balanced Loss Based on Effective Number of Samples

## 1. 메타데이터 (Metadata)

- **제목**: Class-Balanced Loss Based on Effective Number of Samples
- **저자**: Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, Serge Belongie
- **학회/연도**: CVPR 2019
- **arXiv**: 1901.05555

---

## 2. 문제 정의 (Problem Definition)

실세계 데이터는 대부분 **장기 꼬리 분포(long-tailed distribution)**를 따른다. 즉, 소수의 클래스는 수천~수만 개의 샘플을 가지지만, 대다수의 클래스는 수십~수백 개의 샘플밖에 없다. 표준 손실로 학습한 모델은 다수 클래스에 편향되어 소수 클래스에서 크게 실패한다.

- **기존 접근법의 한계**:
  1. **역빈도 가중치 (inverse frequency weighting)**: 클래스 손실에 $1/n_j$ 비례 가중치 부여. 매우 희귀한 클래스는 극단적으로 큰 가중치를 받아 학습 불안정을 초래.
  2. **오버샘플링**: 소수 클래스를 반복 샘플링. 과적합 위험 증가 및 학습 효율 저하.
  3. **언더샘플링**: 다수 클래스를 제거. 다수 클래스의 유용한 정보가 손실.
- **핵심 질문**: "데이터 수 $n$이 증가할수록 실제로 학습에 기여하는 유효한 정보는 얼마나 증가하는가?" — 이를 정량화하여 합리적인 클래스 가중치를 설계할 수 있는가?

---

## 3. 사전 지식 (Prerequisites / Background)

- **장기 꼬리 분포 (Long-tailed Distribution)**: 파레토 법칙처럼, 소수의 클래스가 전체 데이터의 대부분을 차지하고 다수의 클래스는 극소 샘플만 존재하는 분포. iNaturalist(생물 종 분류), LSVT, ImageNet-LT 등이 대표적 실세계 장기 꼬리 데이터셋.
- **데이터 증강 (Data Augmentation)**: 회전, 크롭, 색상 변환 등을 통해 기존 샘플로부터 새로운 학습 샘플을 생성. 모델이 학습하는 특징 공간을 암묵적으로 확장.
- **Cross-Entropy 손실**: 분류에서 가장 기본적인 손실 함수. 클래스별 가중치를 곱하여 손실 기여도를 조정하는 것이 일반적.
- **Focal Loss**: 클래스 불균형을 다루는 최신 방법 중 하나. OHEM과 마찬가지로 학습이 어려운 예시에 집중. Class-Balanced Loss와 결합 가능하다.
- **ResNet, ResNeXt**: 표준 이미지 분류 백본으로, 이 논문의 실험에서 사용되는 모델.

---

## 4. 주요 개념 설명 (Key Concepts)

- **유효 샘플 수 (Effective Number of Samples)**: 이 논문의 핵심 이론적 기여. 데이터 공간을 유한한 원소들의 합집합으로 가정할 때, $n$개 샘플의 기대 고유 특징 볼륨은 기하급수적 감쇠를 따른다:
  $$E_n = \frac{1 - \beta^n}{1 - \beta}, \quad \beta = \frac{N-1}{N}$$
  여기서 $N$은 가능한 고유 원소의 총 수. $\beta \in [0, 1)$. $n$이 커질수록 $E_n$은 포화(saturation)하여 $\frac{1}{1-\beta}$에 수렴.

- **직관적 이해**: 첫 번째 샘플은 항상 새로운 정보($E_1 = 1$). 두 번째 샘플은 이미 덮인 특징과 겹칠 확률이 있어 기대 새 정보 = $\beta$. $n$번째 샘플의 한계 기여 = $\beta^{n-1}$. 따라서 전체 유효 정보는 등비급수 합: $E_n = \frac{1-\beta^n}{1-\beta}$.

- **Class-Balanced (CB) Loss**: 각 클래스의 손실에 유효 샘플 수의 역수를 가중치로 곱:
  $$\text{CB}(p, y) = \frac{1-\beta}{1-\beta^{n_y}} \cdot L(p, y)$$
  $n_y$: 실제 클래스 $y$의 샘플 수, $L$: 기반 손실 함수 (Softmax CE, Sigmoid CE, Focal Loss 등). $\beta \to 0$이면 모든 클래스에 동일 가중치, $\beta \to 1$이면 역빈도($1/n_y$) 가중치.

- **기반 손실 유연성**: CB Loss는 Softmax Cross-Entropy, Sigmoid Cross-Entropy, Focal Loss 위에 모두 적용 가능한 플러그인 방식의 재가중치 방법이다.

---

## 5. 방법 (Method)

**이론적 도출:**

데이터 공간을 유한한 $N$개 원소의 합집합으로 모델링. 각 샘플이 $N$개 원소 중 하나를 균등 확률($1/N$)로 커버한다고 가정. 이미 커버된 원소를 제외한 기대 새 원소 수:

$$E[\text{새 원소}| n \text{개 이미 커버}] = \frac{N - (N - (N/N)^{n})}{N} \approx \beta^{n}$$

$\beta = (N-1)/N$. 따라서:
$$E_n = \sum_{j=1}^{n} \beta^{j-1} = \frac{1-\beta^n}{1-\beta}$$

**클래스 가중치 계산:**
$$w_j = \frac{1}{E_{n_j}} = \frac{1-\beta}{1-\beta^{n_j}}$$

이를 정규화하여 $\bar{w}_j = C \cdot w_j$ (합이 클래스 수 $C$가 되도록).

**$\beta$ 선택:**
- $\beta$는 유효 샘플 수의 포화 속도를 결정하는 유일한 하이퍼파라미터
- 논문에서는 0.9, 0.99, 0.999, 0.9999를 실험하여 데이터셋별 최적값 결정
- iNaturalist 2018: $\beta=0.9999$가 최적, 장기꼬리가 극심할수록 $\beta$를 1에 가깝게 설정

**학습 과정:**
- 표준 SGD/Adam 최적화 사용
- 배치 샘플링은 그대로 유지 (언더/오버샘플링 없음)
- 손실 계산 시에만 클래스 가중치 적용

---

## 6. 결과 (Results)

### 장기 꼬리 CIFAR-10 분류 오류율 (%) — 불균형 비율 200

| 방법 | 오류율 |
|---|---|
| Softmax (표준) | 34.32 |
| Sigmoid | 34.04 |
| Focal Loss | 35.62 |
| CB Softmax | 31.63 |
| CB Sigmoid | 30.24 |
| **CB Focal Loss** | **31.11** |

CB를 적용하지 않은 Focal Loss는 오히려 표준 Softmax보다 나쁨에 주목 — 클래스 불균형 문제에서는 단순 Focal Loss만으로는 부족.

### 장기 꼬리 CIFAR-100 분류 오류율 (%) — 불균형 비율 200

| 방법 | 오류율 |
|---|---|
| Softmax | 61.23 |
| CB Softmax | 57.99 |
| CB Sigmoid | 57.89 |
| **CB Focal Loss** | **58.17** |

### iNaturalist 2018 분류 (Top-1 오류율 %)

| 방법 | 백본 | 오류율 |
|---|---|---|
| Softmax (기준선) | ResNet-50 | 42.86 |
| CB Sigmoid | ResNet-50 | 36.65 |
| **CB Focal Loss** | **ResNet-50** | **38.88** |
| Softmax (기준선) | ResNet-152 | 38.61 |
| **CB Focal Loss** | **ResNet-152** | **30.95** |

ResNet-152 기준 -7.66%p 오류율 감소.

### $\beta$ 값에 따른 성능 변화 (iNaturalist 2018, ResNet-50)

| $\beta$ | 오류율 |
|---|---|
| 0 (균등 가중치) | 42.86 |
| 0.9 | 41.52 |
| 0.99 | 38.48 |
| 0.999 | 37.42 |
| **0.9999** | **36.65** |
| 1 (역빈도) | 41.92 |

$\beta=1$(역빈도)은 오히려 균등 가중치보다 나쁨. 중간 지점($\beta=0.9999$)이 최적.

---

## 7. 인사이트 (Insights)

- **역빈도 가중치가 나쁜 이유의 이론적 설명**: 소수 클래스의 샘플이 10개라도 이 10개가 해당 클래스의 특징 공간을 충분히 커버하고 있을 수 있다. 역빈도($1/10$)는 이를 무시하고 과도한 가중치를 부여한다. 유효 샘플 수는 "실제 학습에 기여하는" 정보량을 더 정확하게 추정한다.

- **단일 하이퍼파라미터 $\beta$로 연속적 스펙트럼 제어**: $\beta=0$은 균등 가중치, $\beta \to 1$은 역빈도, 중간값은 그 사이 어딘가. 이 단일 파라미터로 다양한 데이터셋의 불균형 정도에 맞춰 가중치 전략을 조절 가능하여 직관적이고 적용하기 쉽다.

- **Focal Loss와의 조합이 핵심**: Focal Loss 단독으로는 클래스 불균형 문제에서 오히려 성능이 떨어질 수 있다. Focal Loss는 어려운 예시에 집중하지만, 클래스간 표본 수 차이는 보정하지 않는다. CB 가중치와 결합하면 두 문제를 동시에 해결한다.

- **데이터 증강의 암묵적 효과를 정식화**: 유효 샘플 수 이론은 왜 데이터 증강이 도움이 되는지도 설명한다. 증강은 유효 샘플 수를 늘리는 효과, 즉 특징 공간에서 새로운 원소를 추가하는 것에 해당한다.

- **한계 — 피처 공간 중립성 가정**: 이 이론은 각 샘플이 균등하게 데이터 공간을 커버한다고 가정한다. 실제로는 의미론적으로 유사한 샘플들이 클러스터를 이루어 새로운 정보를 거의 추가하지 않는 경우도 있다. 또한 소수 클래스의 적은 샘플이 실제로 그 클래스를 대표하는지(분포 내 vs 분포 외 샘플)는 고려하지 않는다.

- **실용성**: 손실 계산에 클래스 가중치 하나를 곱하는 것이 전부이므로, 기존 학습 파이프라인에 코드 몇 줄로 통합 가능하다. 언더/오버샘플링처럼 데이터 파이프라인을 변경할 필요가 없어 실무 적용이 간단하다.

---

## 8. 주요 레퍼런스 (Key References)

1. **Lin et al., ICCV 2017** - *Focal Loss for Dense Object Detection* — 예시 난이도 기반 가중치 방법. CB Loss는 클래스 수준 재가중치를 추가하여 보완 관계를 형성.
2. **He & Garcia, TKDE 2009** - *Learning from Imbalanced Data* — 클래스 불균형 학습의 포괄적 서베이. 오버/언더샘플링, 비용 민감 학습 등 기존 접근법을 정리.
3. **Van Horn et al., CVPR 2018** - *The iNaturalist Species Classification and Detection Dataset* — 생물 종 분류 데이터셋. 8,142 클래스, 최대 불균형 비율 500:1의 실세계 장기 꼬리 분포.
4. **Huang et al., CVPR 2016** - *Learning Deep Representation for Imbalanced Classification* — 클래스 불균형에 대한 딥러닝 접근법 초기 연구. 범주별 부트스트래핑 방법 제안.
5. **He et al., CVPR 2016** - *Deep Residual Learning for Image Recognition* — 이 논문의 실험에서 사용한 ResNet 백본. 장기 꼬리 분류의 표준 백본.
6. **Xie et al., CVPR 2017** - *Aggregated Residual Transformations for Deep Neural Networks (ResNeXt)* — 추가 실험에서 사용한 백본. CB Loss의 백본 독립적 적용 가능성을 보여줌.
