---
name: RetinaFace Summary
description: RetinaFace 논문 요약 - 단일 스테이지 밀집 얼굴 위치 추정 (멀티태스크 학습)
type: reference
---

# RetinaFace: Single-stage Dense Face Localisation in the Wild

## 1. 메타데이터 (Metadata)

- **제목**: RetinaFace: Single-stage Dense Face Localisation in the Wild
- **저자**: Jiankang Deng, Jia Guo, Yuxiang Zhou, Jinke Yu, Irene Kotsia, Stefanos Zafeiriou
- **학회/연도**: arXiv, 2019 (CVPR 2020)
- **코드**: https://github.com/deepinsight/insightface/tree/master/RetinaFace

---

## 2. 문제 정의 (Problem Definition)

얼굴 검출(face detection)은 이후 얼굴 인식, 정렬, 파싱 등 모든 얼굴 분석 파이프라인의 전처리 단계다. 기존 단일 스테이지 검출기들은 바운딩 박스 분류와 회귀만 학습하여, 작거나 가려진 얼굴에서 성능이 제한된다는 한계가 있었다. 핵심 문제:

- **기존 한계**: 표준 검출 손실(분류 + 박스 회귀)만으로는 Hard 서브셋의 작은 얼굴에서 정확도 포화
- **동기**: 랜드마크 등 추가 감독 신호(extra supervision)가 검출 성능을 향상시킬 수 있는가?
- **자기지도 학습의 활용**: 밀집 픽셀 주석을 얻기 어려운 상황에서, 자기지도 3D 메시 디코더로 픽셀 단위 3D 형상 정보를 학습할 수 있는가?

---

## 3. 사전 지식 (Prerequisites / Background)

- **Feature Pyramid Network (FPN)**: 다양한 스케일의 특징 맵을 Top-down 경로와 Lateral Connection으로 융합하여 멀티스케일 검출을 수행하는 구조.
- **단일 스테이지 검출기 (SSD, RetinaNet)**: 앵커를 FPN 레벨별로 조밀하게 배치하고, 분류 + 박스 회귀를 한 번에 예측. 빠르지만 배경 앵커가 압도적으로 많아 클래스 불균형 문제 발생.
- **Deformable Convolutional Network (DCN)**: 기존 고정 격자 합성곱 대신, 각 위치의 오프셋을 학습하여 비정형 변환에 대한 모델링 능력 향상.
- **Graph Convolution**: 비유클리드 공간(그래프)에서의 합성곱 연산. 체비쇼프 다항식으로 근사하여 3D 메시(mesh) 상에서 효율적 특징 전파 가능.
- **OHEM (Online Hard Example Mining)**: 미니배치 내에서 손실이 큰 예시를 자동으로 선택하여 어려운 예시에 집중하는 학습 전략.

---

## 4. 주요 개념 설명 (Key Concepts)

- **Multi-task Loss**: RetinaFace의 학습 손실은 네 가지를 결합한다:
  $$L = L_{cls} + \lambda_1 p^* L_{box} + \lambda_2 p^* L_{pts} + \lambda_3 p^* L_{pixel}$$
  각 항은 (1) 얼굴 분류, (2) 바운딩 박스 회귀, (3) 5개 랜드마크 회귀, (4) 픽셀 단위 3D 밀집 회귀를 담당. $\lambda_1=0.25, \lambda_2=0.1, \lambda_3=0.01$.

- **Extra Supervision (추가 감독 신호)**: WIDER FACE 학습/검증 세트에서 5개 얼굴 랜드마크(눈 중심 2개, 코 끝, 입 꼬리 2개)를 수동으로 추가 주석. 총 84.6k(학습)개 얼굴 주석.

- **Self-supervised Mesh Decoder**: 사전 학습된 3D 메시 디코더를 자기지도 방식으로 활용. 형상·텍스처 파라미터 $P_{ST} \in \mathbb{R}^{128}$을 예측하고, 미분 가능 렌더러(differentiable renderer)로 2D 이미지에 투영한 뒤 원본과의 픽셀 차이로 손실 계산.

- **Context Module**: SSH와 PyramidBox에서 영감을 받아 FPN 각 레벨에 독립적인 Context Module을 추가. 수용 영역(receptive field) 확장 및 비정형 문맥 모델링을 위해 DCN 사용.

---

## 5. 방법 (Method)

**아키텍처 개요:**
1. **백본**: ResNet-152(고정밀) 또는 MobileNet-0.25(경량). FPN으로 P2~P6 스케일 추출.
2. **Context Module**: 각 FPN 레벨에 독립 Context Module(DCN 포함) 적용.
3. **앵커 설정**: P2~P6에 스케일별 앵커 배치 (640×640 입력 기준 총 102,300개, 75%가 P2). 비율 1:1, 서브-옥타브 스케일 $2^{1/3}$ 간격.
4. **학습**: SGD (momentum=0.9, weight decay=0.0005, batch=32). IoU≥0.5: 양성 앵커, IoU<0.3: 음성. OHEM으로 음성:양성 = 3:1 유지. 80 에폭.
5. **추론**: 멀티스케일 테스트 + 수평 뒤집기 + Box Voting.

**랜드마크 회귀**: 앵커 중심 기준으로 5개 얼굴 랜드마크의 정규화된 오프셋을 Smooth-L1 손실로 학습.

**밀집 회귀**: 그래프 합성곱 기반 메시 디코더로 128차원 형상·텍스처 파라미터 예측 → 미분 가능 렌더러로 2D 투영 → 픽셀 차이 손실:
$$L_{pixel} = \frac{1}{W \times H} \sum_{i,j} \|\mathcal{R}(\mathcal{D}_{P_{ST}}, P_{cam}, P_{ill})_{i,j} - I^*_{i,j}\|_1$$

---

## 6. 결과 (Results)

### WIDER FACE 검출 성능 (AP %)

| 방법 | Val Easy | Val Medium | Val Hard | Test Hard |
|---|---|---|---|---|
| ISRN (이전 SOTA) | 96.7 | 95.8 | 90.9 | 90.3 |
| **RetinaFace** | **96.9** | **96.1** | **91.8** | **91.4** |

Hard 서브셋에서 +1.1% AP 향상 (작은 얼굴 다수 포함).

### Ablation Study (WIDER FACE Val Hard, AP %)

| 방법 | Hard AP | mAP |
|---|---|---|
| FPN+Context | 90.714 | 50.842 |
| +DCN | 91.286 | 51.522 |
| +$L_{pts}$ (랜드마크) | 91.694 | 52.297 |
| +$L_{pixel}$ (밀집) | 91.276 | 51.492 |
| +$L_{pts}$ + $L_{pixel}$ | **91.857** | **52.318** |

### 얼굴 인식 성능 향상 (ArcFace와 결합)

| 검출기 | LFW | CFP-FP | AgeDB-30 | IJB-C (TAR@FAR=1e-6) |
|---|---|---|---|---|
| MTCNN + ArcFace | 99.83 | 98.37 | 98.15 | 88.29% |
| **RetinaFace + ArcFace** | **99.86** | **99.49** | **98.60** | **89.59%** |

### 추론 속도

| 백본 | VGA | HD | 4K |
|---|---|---|---|
| ResNet-152 (GPU) | 75ms | 443ms | 1742ms |
| MobileNet-0.25 (GPU) | 1.4ms | 6.1ms | 25.6ms |
| MobileNet-0.25 (CPU, 1스레드) | 17ms | 130ms | - |

---

## 7. 인사이트 (Insights)

- **랜드마크 감독의 힘**: 5개 랜드마크 회귀를 추가하는 것만으로 Hard AP가 +0.408%, mAP가 +0.775% 향상된다. 추가 주석 비용 대비 매우 효율적인 개선이며, 이는 landmark localization이 얼굴 feature alignment에 긍정적 영향을 미치기 때문이다.

- **자기지도 학습의 제한과 가치**: 밀집 3D 회귀 단독으로는 Hard 서브셋 성능이 오히려 소폭 저하되지만, 랜드마크 회귀와 결합하면 시너지가 발생한다. 자기지도 학습은 레이블 없이 추가 감독 효과를 얻는 실용적 방법이나, 단독으로는 불안정할 수 있다.

- **검출 품질이 인식에 직결**: MTCNN → RetinaFace로 교체하는 것만으로 ArcFace의 CFP-FP 성능이 98.37% → 99.49%로 급상승. 얼굴 인식 파이프라인에서 검출/정렬 단계가 병목임을 실증적으로 보여준다.

- **단일 스테이지의 가능성**: RetinaFace는 FPN + Context Module + DCN + 멀티태스크 손실의 조합으로 2단계 방법들을 능가. "단일 스테이지 = 낮은 정확도"라는 통념을 깼다.

- **스케일-정확도 트레이드오프**: MobileNet-0.25 버전은 1MB 모델로 ARM에서 16 FPS를 달성. 엣지 디바이스 배포 가능성을 보여준다. 반면 Hard AP는 78.2%로 ResNet-152(91.8%)에 비해 큰 격차가 있어, 극한 환경 적용에는 한계가 있다.

- **한계**: 1×1×256의 압축된 FPN 특징으로 밀집 3D 대응을 예측하는 것은 어렵다. 심한 가림(occlusion)이나 복잡한 장면에서 밀집 회귀 실패 사례가 많다.

---

## 8. 주요 레퍼런스 (Key References)

1. **Yang et al., CVPR 2016** - *WIDER FACE: A Face Detection Benchmark* — 학습/평가에 사용된 핵심 데이터셋.
2. **Lin et al., ICCV 2017** - *Focal Loss for Dense Object Detection (RetinaNet)* — 단일 스테이지 검출기 설계 및 클래스 불균형 처리의 기반.
3. **Lin et al., CVPR 2017** - *Feature Pyramid Networks for Object Detection* — 멀티스케일 특징 추출의 핵심 구조.
4. **Shrivastava et al., CVPR 2016** - *Training Region-based Object Detectors with OHEM* — 어려운 예시 자동 선택 학습 전략.
5. **Deng et al., CVPR 2019** - *ArcFace: Additive Angular Margin Loss* — 얼굴 인식 성능 향상을 검증하는 데 결합된 방법.
6. **Zhang et al., SPL 2016** - *MTCNN: Joint Face Detection and Alignment* — 비교 기준 검출기.
7. **Najibi et al., ICCV 2017** - *SSH: Single Stage Headless Face Detector* — Context Module의 영감 원천.
