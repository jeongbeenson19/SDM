---
name: Focal Loss for Dense Object Detection Summary
description: Focal Loss (RetinaNet) 논문 요약 - 단일 스테이지 검출기의 클래스 불균형 해결
type: reference
---

# Focal Loss for Dense Object Detection

## 1. 메타데이터 (Metadata)

- **제목**: Focal Loss for Dense Object Detection
- **저자**: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár
- **학회/연도**: ICCV 2017 (Best Student Paper Award) / arXiv 2018
- **DOI**: 10.1109/ICCV.2017.324

---

## 2. 문제 정의 (Problem Definition)

단일 스테이지 검출기(one-stage detector)는 속도 면에서 뛰어나지만, 2단계 방법(Faster R-CNN 등)에 비해 정확도가 낮다. 이 논문은 그 근본 원인을 **학습 중 극단적인 전경-배경 클래스 불균형**으로 규명한다.

- **기존 한계**: 단일 스테이지 검출기는 이미지 전체에 걸쳐 ~100,000개의 앵커를 조밀하게 배치한다. 실제 얼굴/물체를 포함하는 앵커는 극소수이고, 나머지 대부분(예: 99%)은 배경이다. 표준 cross-entropy 손실로 학습하면 다수의 쉬운 배경 예시가 손실을 압도하여 유용한 신호를 상쇄한다.
- **OHEM의 한계**: 어려운 예시 선택(OHEM) 방법은 일부 도움이 되지만 여전히 이미 잘 분류된 쉬운 음성 예시를 대량 포함하며, 선택되지 않은 쉬운 예시들을 완전히 버려 정보 낭비가 발생한다.
- **핵심 질문**: 단순히 배경 예시를 하향 가중치(downweight)하되, 어려운 예시(잘못 분류된 예시)에는 여전히 집중할 수 있는 손실 함수를 설계할 수 있는가?

---

## 3. 사전 지식 (Prerequisites / Background)

- **이진 Cross-Entropy 손실**: $\text{CE}(p, y) = -y \log(p) - (1-y)\log(1-p)$. 쉬운 예시라도 손실이 0이 아니어서, 대규모 배경 앵커가 집합적으로 신호를 지배한다.
- **단일 스테이지 검출기 (SSD, YOLO)**: 앵커를 이미지 전체에 배치하고, 분류 + 박스 회귀를 한 번에 수행. 2단계 방법에 비해 빠르지만 정확도가 낮았다.
- **2단계 검출기 (Faster R-CNN)**: Region Proposal Network(RPN)이 먼저 관심 영역 후보를 소수(~2000개)로 걸러낸 뒤, RoI 풀링 후 정밀 분류를 수행. RPN 단계가 클래스 불균형을 자연스럽게 완화한다.
- **Feature Pyramid Network (FPN)**: Top-down 경로와 Lateral Connection으로 다중 스케일 특징 맵을 생성. 다양한 크기의 물체 검출에 효과적.
- **ResNet**: 잔차 연결(skip connection)을 사용한 딥 네트워크. ResNet-50/101/152가 특징 추출 백본으로 널리 사용됨.

---

## 4. 주요 개념 설명 (Key Concepts)

- **Focal Loss (FL)**: 표준 cross-entropy에 변조 인자 $(1-p_t)^\gamma$를 곱하여, 이미 잘 분류된 쉬운 예시의 손실 기여를 자동으로 낮추는 손실 함수:
  $$\text{FL}(p_t) = -(1-p_t)^\gamma \log(p_t)$$
  - $p_t$: 실제 클래스에 대한 모델의 예측 확률
  - $\gamma \geq 0$: 집중 파라미터 (focusing parameter). 논문에서 $\gamma=2$가 최적
  - $p_t = 0.9$인 쉬운 예시의 손실은 $\gamma=2$에서 표준 CE의 $\frac{1}{100}$로 축소
  - 실제로는 클래스 불균형 처리를 위해 $\alpha$-균형 변형 사용: $\text{FL}(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)$

- **RetinaNet**: Focal Loss를 적용한 단일 스테이지 검출기. Feature Pyramid Network(FPN) 백본 위에 분류 서브넷과 박스 회귀 서브넷을 각각 붙인 구조. 이전까지는 단일 스테이지가 따라잡지 못했던 2단계 방법의 정확도를 최초로 능가.

- **앵커 (Anchors)**: FPN의 P3~P7 레벨에 배치. 각 레벨에 3개 종횡비(1:2, 1:1, 2:1) × 3개 스케일(2⁰, 2^{1/3}, 2^{2/3}) = 9개 앵커. 총 약 100K~200K개.

- **$p_t$ 정의**: $p_t = p$ (정답 클래스=1), $p_t = 1-p$ (정답 클래스=0). 이를 통해 이진/다중 클래스 모두 하나의 수식으로 표현.

---

## 5. 방법 (Method)

**Focal Loss 설계:**

표준 CE: $\text{CE}(p_t) = -\log(p_t)$

$\gamma=0$일 때 FL = CE. $\gamma$가 커질수록 쉬운 예시($p_t$가 높은 경우)의 손실 기여가 지수적으로 감소. 어려운 예시($p_t$가 낮은 경우)는 거의 영향을 받지 않아 상대적으로 집중도가 올라간다.

**RetinaNet 아키텍처:**
1. **백본**: ResNet-50 또는 ResNet-101
2. **FPN (Feature Pyramid Network)**: P3~P7 총 5개 스케일 특징 맵 생성
3. **분류 서브넷 (Classification Subnet)**: 각 FPN 레벨에 4개의 3×3 Conv(256채널, ReLU) → 3×3 Conv(K×A 채널, 시그모이드). K=클래스 수, A=앵커 수(9)
4. **박스 회귀 서브넷 (Box Regression Subnet)**: 분류 서브넷과 동일한 구조이나 출력은 4×A채널 (클래스 무관 회귀)
5. 두 서브넷은 파라미터를 공유하지 않음 (독립적으로 학습)

**학습 세부 사항:**
- 분류 서브넷 마지막 레이어 바이어스를 $b = -\log((1-\pi)/\pi)$, $\pi=0.01$로 초기화 → 학습 초기 안정성 확보 (무작위 초기화 시 대규모 음성 앵커가 불안정한 손실을 유발하는 문제 방지)
- 최적 하이퍼파라미터: $\gamma=2$, $\alpha=0.25$
- SGD, 90K 이터레이션, 학습률 0.01 (60K/80K에서 ×0.1 감소), 배치 16
- 앵커 할당: IoU ≥ 0.5 → 양성, IoU < 0.4 → 음성, 그 사이는 무시

---

## 6. 결과 (Results)

### COCO test-dev 검출 성능 (AP %)

| 방법 | 백본 | AP | AP₅₀ | AP₇₅ | APs | APm | APl |
|---|---|---|---|---|---|---|---|
| YOLOv2 | DarkNet-19 | 21.6 | 44.0 | 19.2 | 5.0 | 22.4 | 35.5 |
| SSD513 | ResNet-101 | 31.2 | 50.4 | 33.3 | 10.2 | 34.5 | 49.8 |
| DSSD513 | ResNet-101 | 33.2 | 53.3 | 35.2 | 13.0 | 35.4 | 51.1 |
| Faster R-CNN (FPN) | ResNet-101 | 36.2 | 59.1 | 39.0 | 18.2 | 39.0 | 48.2 |
| Mask R-CNN | ResNet-101 | 38.2 | 60.3 | 41.7 | 20.1 | 41.1 | 50.2 |
| **RetinaNet** | ResNet-101 | **39.1** | **59.1** | **42.3** | **21.8** | **42.7** | **50.2** |
| **RetinaNet** | ResNet-101-FPN | **40.8** | **61.1** | **44.1** | **24.1** | **44.2** | **51.2** |

### Focal Loss vs. OHEM (COCO minival AP %)

| 방법 | AP |
|---|---|
| CE (표준 교차 엔트로피) | 30.9 |
| OHEM (1:3 neg:pos) | 32.8 |
| OHEM (최적 neg:pos) | 35.7 |
| Focal Loss ($\gamma$=2, $\alpha$=0.25) | **36.0** |

FL이 최적화된 OHEM보다 +0.3 AP 향상, 표준 CE보다 +5.1 AP 향상.

### $\gamma$ 값별 성능 (COCO minival)

| $\gamma$ | AP |
|---|---|
| 0 (= CE + $\alpha$-균형) | 31.1 |
| 0.5 | 33.6 |
| 1.0 | 35.5 |
| 2.0 | **36.0** |
| 5.0 | 35.2 |

---

## 7. 인사이트 (Insights)

- **클래스 불균형이 진짜 병목이었다**: 단일 스테이지 검출기가 2단계보다 느린 게 아니라 덜 정확했던 이유가 속도-정확도 트레이드오프가 아닌 학습 신호의 왜곡 때문임을 실증했다. 이 하나의 발견이 이후 single-stage 연구의 방향을 바꿨다.

- **단순한 아이디어, 강력한 효과**: Focal Loss는 기존 cross-entropy 손실에 $(1-p_t)^\gamma$ 인자 하나를 추가하는 매우 단순한 변경이지만, COCO AP 기준으로 표준 CE 대비 +5 AP 이상의 향상을 가져온다. 추가적인 구조 변경 없이 손실 함수 변경만으로 달성.

- **어려운 예시는 자동으로 선택된다**: OHEM은 명시적으로 예시를 선택/제외하는 반면, FL은 연속적인 가중치로 쉬운 예시를 서서히 억제한다. 이 "soft selection"이 더 안정적이고 우수한 성능을 보임. 하이퍼파라미터 민감도도 낮다.

- **초기화 전략의 중요성**: 분류 헤드 바이어스를 $\pi=0.01$로 설정하는 기법은 단독으로도 FL이 없을 때 CE의 성능을 크게 향상시킨다. 이는 초기 학습 불안정성이 별도의 문제임을 시사한다.

- **속도-정확도 프론티어 이동**: RetinaNet은 같은 속도(FPS)에서 기존 최고 정확도를 크게 능가함. 5 FPS에서 RetinaNet(ResNet-101-FPN)은 40.8 AP로, 비슷한 속도의 Faster R-CNN (36.2 AP)보다 +4.6 AP 높다.

- **범용성**: Focal Loss는 face detection, scene understanding, 3D detection 등 다양한 조밀한 예측 문제에서 광범위하게 채택됨. 클래스 불균형이 있는 모든 학습 문제에 적용 가능한 일반적인 기법이다.

---

## 8. 주요 레퍼런스 (Key References)

1. **Girshick et al., ICCV 2015** - *Fast R-CNN* — 분류 및 박스 회귀의 멀티태스크 학습을 확립. RetinaNet의 손실 구조의 기반이 된 2단계 검출기.
2. **Ren et al., NeurIPS 2015** - *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks* — RPN으로 후보 영역을 사전 필터링하여 클래스 불균형을 암묵적으로 해결한 2단계 검출기. RetinaNet이 정확도 면에서 처음으로 능가한 방법.
3. **Lin et al., CVPR 2017** - *Feature Pyramid Networks for Object Detection* — RetinaNet 백본의 핵심 구조. 멀티스케일 특징 융합을 위한 FPN.
4. **Liu et al., ECCV 2016** - *SSD: Single Shot MultiBox Detector* — 조밀한 앵커 기반 단일 스테이지 검출기의 대표. RetinaNet이 개선 대상으로 삼은 방법.
5. **Shrivastava et al., CVPR 2016** - *Training Region-based Object Detectors with Online Hard Example Mining* — 어려운 예시 선택을 통한 학습 개선. FL이 비교 대상으로 삼아 성능을 능가한 방법.
6. **He et al., CVPR 2016** - *Deep Residual Learning for Image Recognition* — RetinaNet의 백본 ResNet. 잔차 연결을 통한 매우 깊은 네트워크 학습.
