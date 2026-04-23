---
name: WIDER FACE Summary
description: WIDER FACE 논문 요약 - 대규모 얼굴 검출 벤치마크 데이터셋
type: reference
---

# WIDER FACE: A Face Detection Benchmark

## 1. 메타데이터 (Metadata)

- **제목**: WIDER FACE: A Face Detection Benchmark
- **저자**: Shuo Yang, Ping Luo, Chen Change Loy, Xiaoou Tang
- **학회/연도**: CVPR 2016
- **프로젝트**: http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/

---

## 2. 문제 정의 (Problem Definition)

얼굴 검출(face detection) 연구는 기존 벤치마크(FDDB, AFW, PASCAL FACE)에서 이미 성능이 포화(saturation) 상태에 도달했다. 이 논문은 실세계의 다양하고 어려운 조건을 충분히 반영하는 새로운 대규모 벤치마크 데이터셋, **WIDER FACE**를 제안한다.

- **기존 한계**: FDDB는 약 5,000개 얼굴 / PASCAL FACE는 851개 이미지로, 조명·포즈·가림 등의 변화를 충분히 커버하지 못한다. 최신 검출기들이 90% 이상의 AP를 달성하여 벤치마크로서의 변별력을 상실.
- **동기**: 현실 세계는 군중 장면, 극단적 조명, 심한 가림, 다양한 얼굴 크기 등 복잡한 상황이 조합된다. 연구자들이 이러한 도전적 조건에서도 실질적으로 진보를 이룰 수 있는 환경이 필요하다.
- **핵심 기여**: 393,703개 얼굴이 레이블된 32,203장의 이미지와 변화 요인 기반의 세분화된 난이도 분류 체계를 제공.

---

## 3. 사전 지식 (Prerequisites / Background)

- **Precision-Recall 곡선과 AP**: 검출 결과를 신뢰도 순으로 정렬 후, IoU 임계값(일반적으로 0.5)을 기준으로 참양성/거짓양성을 계산하여 PR 곡선을 그린다. Average Precision(AP)은 PR 곡선 아래 면적.
- **EdgeBox**: 엣지 기반의 객체 검출 후보 생성 방법. 이 논문에서는 얼굴에 대한 EdgeBox의 검출률(recall)을 기반으로 난이도 분류(Easy/Medium/Hard)에 사용.
- **DPM (Deformable Part Model)**: 부분 모델을 이용한 객체 검출 방법. 이 논문의 기준 검출기로 사용됨.
- **Cascade CNN**: 여러 단계의 CNN을 단계적으로 적용하여 계산 효율과 정확도를 균형 있게 유지하는 얼굴 검출 방법.
- **IoU (Intersection over Union)**: 예측 박스와 정답 박스의 교집합 / 합집합 비율. 매칭 기준으로 사용.

---

## 4. 주요 개념 설명 (Key Concepts)

- **WIDER FACE 데이터셋**: 60개 이벤트 카테고리(퍼레이드, 스포츠, 결혼식 등)에서 수집된 32,203장의 이미지와 393,703개의 레이블된 얼굴을 포함. 훈련(40%) / 검증(10%) / 테스트(50%) 분할.

- **이벤트 기반 분류**: 60개 카테고리는 내용 유형(야외 행사, 스포츠, 일상 등)에 따라 분류되며, 각 카테고리별로 Easy/Medium/Hard 서브셋이 존재. 이벤트 다양성이 포즈·가림·조명의 자연스러운 변화를 내포.

- **세 가지 난이도 서브셋**:
  - **Easy**: EdgeBox 검출률 ≥ 20% 카테고리 (6개)
  - **Medium**: EdgeBox 검출률 ≥ 5% 카테고리 (20개)
  - **Hard**: 나머지 모든 카테고리 (34개) — 군중 장면, 작은 얼굴, 심한 가림 포함
  - Hard 서브셋은 전체 얼굴의 약 74%를 포함하며, 기존 검출기들이 크게 실패하는 조건을 모음.

- **주석 속성 (Annotation Attributes)**: 각 얼굴 바운딩 박스에 다음 7가지 속성을 부여:
  - 가림 수준 (occlusion level): 없음/부분/심함
  - 포즈 변화 (pose variation): 일반/과장
  - 흐릿함 (blur): 없음/보통/극심
  - 조명 (illumination): 일반/극단
  - 표정 (expression): 일반/과장
  - 화장 (make-up): 없음/있음
  - 유효 여부 (invalid): 유효/무효

---

## 5. 방법 (Method)

**데이터 수집 과정:**
1. 60개 이벤트 카테고리 선정 (스포츠 경기, 퍼레이드, 결혼식 등 다양한 인간 활동)
2. 인터넷 이미지 검색을 통해 카테고리별 이미지 수집
3. 중복 제거 및 품질 필터링
4. 크라우드소싱 기반 바운딩 박스 및 속성 주석 (총 393,703개 얼굴)

**난이도 분류 방법:**
- 각 이미지에서 EdgeBox로 얼굴 검출 시도
- 이벤트 카테고리별 EdgeBox 검출률(recall@0.5 IoU) 계산
- 검출률에 따라 카테고리를 세 그룹으로 분류:
  - Easy: 검출률 높음 (얼굴이 크고 명확)
  - Medium: 중간 검출률
  - Hard: 검출률 낮음 (작은 얼굴, 군중, 심한 가림)

**멀티스케일 Cascade CNN 기준 모델 제안:**
- 기존 Cascade CNN을 WIDER FACE 학습 데이터로 학습
- 멀티스케일 처리로 다양한 크기의 얼굴 검출
- 5×5픽셀 크기의 얼굴까지 검출 가능

**평가 프로토콜:**
- 표준 PASCAL VOC AP 방식 (IoU ≥ 0.5)
- Easy/Medium/Hard 각 서브셋별 AP 따로 보고
- 검증셋은 공개 레이블, 테스트셋은 레이블 비공개 (서버 제출 방식)

---

## 6. 결과 (Results)

### WIDER FACE Val/Test AP (%) — 주요 검출기 비교

| 방법 | Val Easy | Val Medium | Val Hard | Test Hard |
|---|---|---|---|---|
| ACF-WIDER | 69.2 | 59.7 | 36.0 | - |
| DPM-WIDER | 67.5 | 55.1 | 31.5 | - |
| Faceness-WIDER | 71.6 | 60.3 | 31.5 | - |
| **Cascade CNN (제안)** | **75.3** | **65.1** | **37.4** | **36.2** |

### 속성별 성능 저하 분석

| 속성 | 최고 성능 방법 | Hard 서브셋 AP |
|---|---|---|
| 가림 없음 | Cascade CNN | ~70% |
| 심한 가림 | Cascade CNN | ~15% |
| 일반 조명 | Cascade CNN | ~65% |
| 극단적 조명 | Cascade CNN | ~30% |
| 1~4 얼굴 | 모든 방법 | ~75% |
| 20개 이상 얼굴 | 모든 방법 | ~25% |

Hard 서브셋에서 모든 기존 방법들이 40% 미만의 AP를 보여, 연구 개선 여지가 큼을 확인.

### 기존 데이터셋과의 비교

| 데이터셋 | 이미지 수 | 얼굴 수 | 최소 크기 |
|---|---|---|---|
| AFW | 205 | 473 | - |
| PASCAL FACE | 851 | 1,341 | - |
| FDDB | 2,845 | 5,171 | - |
| **WIDER FACE** | **32,203** | **393,703** | **5px** |

WIDER FACE는 이전 데이터셋보다 얼굴 수 기준으로 75배 이상 크다.

---

## 7. 인사이트 (Insights)

- **포화된 벤치마크의 교체**: 기존 FDDB/AFW 등은 2016년 당시 이미 검출률 90%를 넘어서 진보를 측정하기 어려웠다. WIDER FACE의 Hard 서브셋은 당시 최고 방법도 ~37% AP에 그쳐, 실질적인 진보 여지를 보여준다. 이는 향후 5년 이상 얼굴 검출 연구의 표준 벤치마크가 된 이유다.

- **이벤트 기반 분류의 자연스러운 다양성**: 이미지를 이벤트 카테고리로 수집함으로써 자연스럽게 다양한 조건(포즈, 조명, 가림, 군중 밀도)이 커버된다. 인위적으로 조건을 제어하는 것보다 실세계 분포를 더 잘 반영한다.

- **Hard 서브셋의 중요성**: 전체 이미지의 74%를 포함하는 Hard 서브셋 성능이 실용적인 얼굴 검출 시스템의 진짜 한계를 드러낸다. RetinaFace가 Hard 서브셋에서 특히 강조한 성능 향상(+1.1% AP)이 의미 있는 이유도 여기에 있다.

- **소형 얼굴 문제의 부각**: WIDER FACE Hard 서브셋에는 10~20픽셀 수준의 매우 작은 얼굴이 다수 포함된다. 이후 작은 얼굴 전용 네트워크(HR-Net, PyramidBox 등) 연구가 활성화된 계기가 됐다.

- **데이터셋 편향의 주의점**: 60개 이벤트 카테고리는 동아시아 미디어에서 수집된 비율이 높아 인종적 다양성에 편향이 있을 수 있다. 현실 세계의 완전한 대표성 확보는 어렵다.

- **멀티태스크 학습의 필요성을 간접 제시**: 바운딩 박스 주석 외에 7가지 속성 주석을 제공함으로써, 이후 얼굴 검출을 넘어 속성 예측을 결합한 멀티태스크 방법들의 연구를 촉진했다.

---

## 8. 주요 레퍼런스 (Key References)

1. **Viola & Jones, CVPR 2001** - *Rapid Object Detection using a Boosted Cascade of Simple Features* — Haar Feature + Cascade 구조의 얼굴 검출 선구 연구. 이후 모든 얼굴 검출 연구의 출발점.
2. **Shen et al., CVPR 2013** - *The Devil is in the Details: Delving into Unbiased Data Normalization on Action Recognition* — 크라우드 씬 얼굴 검출의 어려움을 제기한 초기 연구.
3. **Yang et al., ECCV 2014** - *Aggregate Channel Features for Multi-view Face Detection (ACF-Face)* — WIDER FACE에서 비교 기준으로 사용한 주요 검출기.
4. **Li et al., ECCV 2014** - *A Convolutional Neural Network Cascade for Face Detection* — Cascade CNN으로 다단계 검출을 구현. 이 논문에서 제안한 멀티스케일 Cascade CNN 기준 모델의 기반.
5. **Yang et al., CVPR 2015** - *From Facial Parts Responses to Face Detection: A Deep Learning Approach* — Faceness-Net. WIDER FACE에서 비교 기준으로 사용.
6. **Mathias et al., ECCV 2014** - *Face Detection without Bells and Whistles* — DPM 기반의 얼굴 검출기. WIDER FACE 비교 기준.
7. **Zitnick & Dollár, ECCV 2014** - *Edge Boxes: Locating Object Proposals from Edges* — EdgeBox를 이용하여 WIDER FACE 난이도 분류 기준으로 활용.
