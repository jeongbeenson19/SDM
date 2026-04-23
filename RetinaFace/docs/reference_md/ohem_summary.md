---
name: OHEM Summary
description: OHEM 논문 요약 - 온라인 어려운 예시 선택을 통한 Region-based 객체 검출기 학습
type: reference
---

# Training Region-based Object Detectors with Online Hard Example Mining

## 1. 메타데이터 (Metadata)

- **제목**: Training Region-based Object Detectors with Online Hard Example Mining
- **저자**: Abhinav Shrivastava, Abhinav Gupta, Ross Girshick
- **학회/연도**: CVPR 2016
- **arXiv**: 1604.03540

---

## 2. 문제 정의 (Problem Definition)

Region-based CNN 검출기(Fast R-CNN 등)는 이미지당 수천 개의 RoI(Region of Interest) 후보를 처리하지만, 대부분은 배경이다. 표준 학습 방법은 수작업 설계된 전경/배경 비율(예: 25% 전경, 75% 배경)과 고정 배치 구성에 의존하는데, 이는 여러 문제를 일으킨다.

- **기존 한계**:
  1. 고정된 전경/배경 비율은 학습 전반에 걸쳐 최적이지 않다. 초기에는 어려운 음성이 많고 후기에는 쉬운 음성이 지배한다.
  2. 쉬운 음성 예시(이미 잘 분류된 배경)가 손실을 희석시켜 유용한 학습 신호가 감소한다.
  3. 학습률 스케줄이나 배치 크기 등 하이퍼파라미터 조정으로 이 문제를 우회하기 어렵다.
- **동기**: 부트스트래핑(bootstrapping) 아이디어 — 어려운 예시에 집중하는 학습 — 를 딥러닝에서 온라인(online, 즉 학습 중 실시간으로)으로 수행할 수 있는가?
- **핵심 기여**: 복잡한 데이터셋 재구성 없이, SGD 미니배치 내에서 자동으로 어려운 예시를 선택하는 **OHEM(Online Hard Example Mining)** 방법 제안.

---

## 3. 사전 지식 (Prerequisites / Background)

- **Fast R-CNN**: RoI 풀링으로 이미지당 여러 후보 영역의 특징을 공유 합성곱에서 추출하고, 분류 + 박스 회귀를 수행하는 2단계 검출기. 이 논문의 직접적인 개선 대상.
- **부트스트래핑 / Hard Negative Mining**: SVM 기반 검출기(DPM 등)에서 사용하던 반복적 재학습 방법. 모델이 틀린 음성(false positive)을 오프라인으로 수집하여 재학습. 이 논문은 이를 딥 네트워크에서 온라인으로 수행.
- **SGD 미니배치**: 전체 데이터 중 일부 샘플을 임의 추출하여 그래디언트를 계산하는 최적화 방법. 미니배치 내 샘플 구성이 학습에 큰 영향을 미침.
- **RoI 풀링 (Region of Interest Pooling)**: 합성곱 특징 맵에서 임의 크기의 관심 영역을 고정 크기로 풀링하는 연산. Fast R-CNN의 핵심 구성 요소로, 이미지당 수천 개의 RoI를 효율적으로 처리.
- **RPN (Region Proposal Network)**: Faster R-CNN에서 제안 영역을 생성하는 네트워크. OHEM 논문은 Selective Search와 RPN 양쪽 제안 방식에 모두 적용됨.

---

## 4. 주요 개념 설명 (Key Concepts)

- **OHEM (Online Hard Example Mining)**: SGD 미니배치 내에서 손실이 높은 RoI(즉, 모델이 현재 잘못 처리하는 예시)를 자동으로 선택하여 역전파에 사용하는 기법. 매 이터레이션마다 "현재 모델에게 어려운" 예시를 동적으로 파악.

- **읽기 전용 RoI 네트워크 (Read-only RoI Network)**: OHEM 구현의 핵심. 동일한 가중치의 두 개의 RoI 서브네트워크를 유지:
  1. **읽기 전용(readonly) 복사본**: 이미지당 모든 RoI(~2000개)에 대해 순전파(forward pass)만 수행 → 손실 계산 → 상위 B/N개 어려운 예시 선택 (N=이미지 수, B=배치 크기)
  2. **학습 가능한 복사본**: 선택된 B개 RoI에 대해서만 순전파 + 역전파 수행
  두 복사본은 가중치를 공유하므로 메모리 오버헤드는 약 2배이지만, 실제 역전파는 소수 RoI에만 적용.

- **IoU 억압 (IoU-based Suppression)**: 선택된 어려운 예시 중 서로 IoU > 임계값(0.7)인 중복 박스를 제거하여 동일 영역이 과대 표현되는 것을 방지. NMS와 유사한 역할을 수행.

- **전경/배경 비율 제거**: OHEM은 전통적인 고정 비율 샘플링을 완전히 대체한다. 전경과 배경 구분 없이 오직 손실 크기로만 예시를 선택하므로, 각 이터레이션의 배치 구성이 모델 상태에 따라 자동으로 결정된다.

---

## 5. 방법 (Method)

**OHEM 알고리즘:**

1. **입력 이미지 → 공유 합성곱 특징 추출**: 이미지 전체에 대해 공유 합성곱 레이어(VGG-16의 경우 conv1~conv5) 실행 → 특징 맵 $\mathbf{F}$ 생성.

2. **모든 RoI 순전파 (읽기 전용 네트워크)**: 생성된 특징 맵 위에서 모든 $|R|$ (~2000개)개의 RoI에 대해 RoI 풀링 → FC 레이어 → 손실 계산. 단, 그래디언트는 계산하지 않음 (읽기 전용).

3. **어려운 예시 선택**: 손실이 큰 상위 B개 RoI 선택. IoU 억압으로 중복 제거.

4. **학습 가능한 네트워크에서 역전파**: 선택된 B개 RoI만 학습 가능한 네트워크에 전달 → 그래디언트 계산 → 가중치 업데이트.

5. 두 네트워크는 가중치를 공유하므로, 각 이터레이션 후 읽기 전용 네트워크 가중치도 자동 업데이트.

**구현 세부 사항:**
- 이미지당 모든 RoI에 대해 순전파 → 상위 16개(배치당 2장 이미지, 각 8개) 선택
- 원래 Fast R-CNN은 이미지당 64개 RoI를 랜덤 샘플링. OHEM은 손실 기반 8개 선택으로 대체
- SGD, 학습률 0.001 (40K 이터레이션) → 0.0001 (20K 이터레이션), 모멘텀 0.9, 가중치 감쇠 0.0005

---

## 6. 결과 (Results)

### PASCAL VOC 2007 test mAP (%)

| 방법 | 백본 | mAP |
|---|---|---|
| Fast R-CNN (기준선) | VGG16 | 66.9 |
| Fast R-CNN + OHEM | VGG16 | **69.9** |
| Faster R-CNN | VGG16 | 69.9 |
| Fast R-CNN + OHEM + extra | VGG16 | **78.9** |

OHEM 추가만으로 기준선 대비 +3.0% mAP 향상. Faster R-CNN(RPN 포함)과 동등한 성능을 Selective Search 제안으로 달성.

### PASCAL VOC 2012 test mAP (%)

| 방법 | mAP |
|---|---|
| Fast R-CNN | 65.7 |
| Fast R-CNN + OHEM | 68.5 |
| Fast R-CNN + OHEM + extras | **76.3** |

### COCO minival AP (%)

| 방법 | AP@0.5 | AP@[0.5:0.95] |
|---|---|---|
| Fast R-CNN | 35.9 | 19.7 |
| Fast R-CNN + OHEM | **38.5** | **22.6** |

COCO의 엄격한 평가 지표에서도 +2.6% AP@0.5, +2.9% AP@[0.5:0.95] 향상.

### 배경 비율 실험 (PASCAL VOC 2007)

| 음성:양성 비율 | mAP |
|---|---|
| 1:1 | 66.1 |
| 3:1 | 66.9 |
| 7:1 | 66.4 |
| OHEM (비율 없음) | **69.9** |

어떤 고정 비율도 OHEM을 능가하지 못함. OHEM이 고정 비율의 필요성을 완전히 제거.

---

## 7. 인사이트 (Insights)

- **"언제나 어려운 예시를 써라"가 아니라 "지금 모델에게 어려운 예시를 써라"**: OHEM의 핵심 통찰은 어려운 예시의 정의가 학습 전반에 걸쳐 바뀐다는 것이다. 초기에는 거의 모든 전경이 어렵고, 후기에는 특정 클래스의 배경 패턴이 어려워진다. 온라인 방식이 이를 자동으로 따라간다.

- **전경/배경 비율이라는 하이퍼파라미터 제거**: 기존 Fast R-CNN은 fg:bg = 1:3 비율을 수작업으로 튜닝해야 했다. OHEM은 손실이 선택 기준이므로 이 하이퍼파라미터가 불필요해진다. 하이퍼파라미터 하나를 없앤 것 자체가 실용적 가치.

- **읽기 전용 복사본 트릭의 영리함**: 딥 네트워크에서 모든 RoI에 대해 순전파를 수행하면 메모리/속도 비용이 크지만, 읽기 전용 복사본은 그래디언트를 계산하지 않아 비용이 절반으로 줄어든다. 구현 복잡도를 최소화하면서 기능을 구현한 실용적인 설계.

- **스케일 가능성**: OHEM은 Fast R-CNN 학습 루프에 직접 통합되며, 추론 시 오버헤드가 전혀 없다. 더 좋은 검출기 구조(FPN, DCN 등)와 직교적으로 결합 가능하여 이후 많은 연구에서 기본 구성 요소로 채택됨 (RetinaFace의 neg:pos 7:1 OHEM도 이 방법의 직접 응용).

- **한계 — 2단계 구조 의존**: OHEM은 제안 영역(proposals)이 먼저 존재하는 2단계 구조에 적합하다. SSD 같은 앵커 기반 단일 스테이지에서는 수십만 개의 앵커에 대해 읽기 전용 순전파를 수행하는 비용이 과도하다. Focal Loss가 단일 스테이지에서 더 나은 해법으로 등장한 배경.

- **정보 효율성**: OHEM이 선택하지 않은 RoI는 학습에 기여하지 않는다. 이론적으로는 선택된 소수 예시에만 집중하므로, 전체 배치를 사용하는 방법 대비 일부 정보가 버려진다. 이를 보완하기 위해 Focal Loss처럼 "버리지 않고 가중치를 낮추는" 연속적 접근이 더 발전된 방향이 됐다.

---

## 8. 주요 레퍼런스 (Key References)

1. **Girshick, ICCV 2015** - *Fast R-CNN* — OHEM의 직접적인 개선 대상. RoI 풀링으로 효율적인 2단계 검출을 구현했으나 고정 비율 샘플링의 한계를 가짐.
2. **Ren et al., NeurIPS 2015** - *Faster R-CNN* — RPN으로 제안 영역을 통합한 완전 엔드투엔드 검출기. OHEM과 결합 시 추가 성능 향상이 가능함을 실험.
3. **Felzenszwalb et al., PAMI 2010** - *Object Detection with Discriminatively Trained Part-Based Models (DPM)* — OHEM의 아이디어 원천. 오프라인 hard negative mining을 처음 검출에 도입한 연구.
4. **Krizhevsky et al., NeurIPS 2012** - *ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)* — CNN 기반 특징 학습의 시초. Fast R-CNN이 이 기반 위에서 구축됨.
5. **Uijlings et al., IJCV 2013** - *Selective Search for Object Recognition* — 초기 2단계 검출기에서 제안 영역 생성에 사용된 방법. OHEM 실험의 제안 방식 중 하나.
6. **Lin et al., ICCV 2017** - *Focal Loss for Dense Object Detection* — OHEM의 아이디어를 단일 스테이지에서 연속적 가중치 방식으로 발전시킨 후속 연구. OHEM의 직접적 경쟁/발전 관계.
