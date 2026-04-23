# RetinaFace용 WIDER FACE 데이터셋 절감 방법론 (Cited-by-Sentence)

## 1) 목적과 제약
이 문서는 WIDER FACE 학습 데이터를 줄이면서 RetinaFace의 검증 성능 저하를 통제하기 위한 실무 절차를 정의한다 [1][2].  
얼굴 검출은 작은 얼굴과 가림(occlusion) 구간에서 난이도 편차가 크므로 단순 랜덤 샘플링만으로는 성능 저하 위험이 커진다 [1][2].  
따라서 데이터 절감은 난이도 분포 보존, 하드 샘플 우선 유지, 학습 시 불균형 보정을 함께 적용하는 방식으로 설계한다 [1][3][5].  

## 2) 핵심 아이디어 요약
1차 절감은 층화 샘플링으로 수행해 원본 분포를 유지한다 [1].  
2차 절감은 모델 기반 hardness 점수로 쉬운 샘플부터 제거한다 [5].  
필요 시 feature-space core-set 선택을 병행해 표본 대표성을 보강한다 [4].  
학습 단계에서는 Focal Loss 또는 class-balanced 가중치를 적용해 절감 후 분포 왜곡을 완화한다 [3][6].  

## 3) 단계별 실행 방법론

### Step A. 데이터 프로파일링
각 이미지에 대해 `face_count`, `min_face_size`, `median_face_size`, `occlusion_ratio`를 계산해 메타테이블을 만든다 [1].  
`min_face_size` 구간은 예시로 `(<16px, 16~32px, 32~64px, >=64px)`로 나눠 소얼굴 비중을 추적한다 [1][2].  
WIDER FACE 평가 관행에 맞춰 난이도 그룹(easy/medium/hard) 비율을 별도 컬럼으로 저장한다 [1].  

### Step B. 층화 샘플링(1차 절감)
층(stratum)은 최소한 `difficulty × min_face_size_bin × face_count_bin`의 결합 키로 만든다 [1].  
목표 절감률이 `r`이면 각 층에서 `max(1, round(n_h * r))`를 뽑아 층별 분포를 유지한다 [1].  
하드 층은 보수적으로 유지하기 위해 hard strata에 최소 보존율 하한(예: `r_hard >= r + 0.1`)을 둔다 [1][2].  
이 단계 결과를 `subset_v1_manifest.csv`로 고정해 재현성을 확보한다 [1].  

### Step C. Hardness 기반 제거(2차 절감)
`subset_v1`로 3~5 epoch의 짧은 학습을 수행해 이미지별 난이도 로그를 수집한다 [2][5].  
난이도 점수는 예시로 `H = α·FN + β·FP + γ·(1-IoU_mean)` 형태로 정의한다 [5].  
여기서 FN/FP는 검출 실패 및 오탐 개수, IoU_mean은 매칭된 박스 평균 IoU다 [5].  
각 층 내부에서 `H`가 낮은(쉬운) 샘플부터 제거해 목표 절감률까지 내린다 [5].  
이 절차는 하드 예제를 자동으로 남기므로 단순 랜덤 절감보다 성능 유지에 유리하다 [5].  

### Step D. Core-set 보강(선택)
백본 feature 임베딩 공간에서 k-center greedy를 적용해 대표 샘플을 선택할 수 있다 [4].  
실무 적용은 `층화 샘플링으로 1차 축소 -> 층 내부 core-set 선택` 순서가 안정적이다 [4].  
이 방식은 데이터 커버리지를 유지해 절감률이 높을 때의 일반화 저하를 줄이는 데 도움이 된다 [4].  

### Step E. 학습 불균형 보정
절감 후 배경/전경 및 난이도 불균형을 줄이기 위해 RetinaFace 학습에서 Focal Loss 사용을 유지하거나 강화한다 [2][3].  
클래스/구간별 샘플 수 편차가 크면 effective number 기반 재가중을 추가한다 [6].  
온라인 하드 샘플 집중이 필요하면 OHEM 방식의 미니배치 샘플 선택을 보조적으로 적용한다 [5].  

### Step F. 성능 가드레일과 중단 규칙
절감률 실험은 `100% -> 60% -> 40% -> 30%` 순으로 단계적으로 수행한다 [1][2].  
각 단계에서 동일한 검증 파이프라인으로 easy/medium/hard 지표를 비교한다 [1].  
중단 기준 예시는 `easy 또는 medium 하락 > 1.0%p` 또는 `hard 하락 > 1.5%p`다 [1][2].  
기준을 초과하면 직전 절감률을 채택하고 하드 층 보존율을 상향 조정해 재실험한다 [1][2][5].  

## 4) 권장 산출물(Artifacts)
`subset_v1_manifest.csv`: 층화 샘플링 결과 파일 [1].  
`subset_v2_manifest.csv`: hardness 제거 반영 결과 파일 [5].  
`sampling_config.yaml`: bin 경계, 절감률, 하드 층 하한값, 랜덤 시드 [1][2].  
`eval_report.md`: 절감률별 easy/medium/hard 성능 비교표 [1][2].  

## 5) 바로 적용 가능한 체크리스트
원본 학습셋 메타테이블 생성 스크립트를 먼저 만든다 [1].  
층화 키와 bin 경계를 고정하고 `subset_v1`을 생성한다 [1].  
짧은 학습으로 hardness 로그를 만들고 `subset_v2`를 확정한다 [2][5].  
Focal Loss/OHEM/재가중 설정을 포함해 본 학습을 실행한다 [3][5][6].  
검증 성능이 가드레일을 만족하는지 확인하고 최종 절감률을 잠근다 [1][2].  

## References
[1] Yang, S., Luo, P., Loy, C. C., & Tang, X. (2016). *WIDER FACE: A Face Detection Benchmark*. arXiv:1511.06523. https://arxiv.org/abs/1511.06523  
[2] Deng, J., Guo, J., Zhou, Y., Yu, J., Kotsia, I., & Zafeiriou, S. (2019). *RetinaFace: Single-stage Dense Face Localisation in the Wild*. arXiv:1905.00641. https://arxiv.org/abs/1905.00641  
[3] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). *Focal Loss for Dense Object Detection*. arXiv:1708.02002. https://arxiv.org/abs/1708.02002  
[4] Sener, O., & Savarese, S. (2018). *Active Learning for Convolutional Neural Networks: A Core-Set Approach*. arXiv:1708.00489. https://arxiv.org/abs/1708.00489  
[5] Shrivastava, A., Gupta, A., & Girshick, R. (2016). *Training Region-Based Object Detectors With Online Hard Example Mining*. CVPR 2016. https://openaccess.thecvf.com/content_cvpr_2016/html/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.html  
[6] Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). *Class-Balanced Loss Based on Effective Number of Samples*. CVPR 2019. https://research.google/pubs/class-balanced-loss-based-on-effective-number-of-samples/  

