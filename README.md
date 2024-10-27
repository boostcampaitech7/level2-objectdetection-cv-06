 # Boostcamp AI Tech 7 CV 06
 
## 재활용 품목 분류를 위한 Object Detection
### 2024.10.02 11:00 ~ 2024.10.24 19:00


![image](https://github.com/user-attachments/assets/7dea38fd-73e4-4100-807b-179e1aac4c84)
## Description
바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

Input : 쓰레기 객체가 담긴 이미지가 모델의 인풋으로 사용됩니다. 또한 bbox 정보(좌표, 카테고리)는 model 학습 시 사용이 됩니다. bbox annotation은 COCO format으로 제공됩니다.

Output : 모델은 bbox 좌표, 카테고리, score 값을 리턴합니다. 이를 submission 양식에 맞게 csv 파일을 만들어 제출합니다.

Test set의 mAP50(Mean Average Precision)로 평가합니다.


## Result
![image](https://github.com/user-attachments/assets/e60242a5-b0ad-463a-bf05-11808a3d3caa)

최종 mAP50 0.7344로 리더보드 순위 5등 달성


## Contributor
| [![](https://avatars.githubusercontent.com/jhuni17)](https://github.com/jhuni17) | [![](https://avatars.githubusercontent.com/jung0228)](https://github.com/jung0228) | [![](https://avatars.githubusercontent.com/Jin-SukKim)](https://github.com/Jin-SukKim) | [![](https://avatars.githubusercontent.com/kimdyoc13)](https://github.com/kimdyoc13) | [![](https://avatars.githubusercontent.com/MinSeok1204)](https://github.com/MinSeok1204) | [![](https://avatars.githubusercontent.com/airacle100)](https://github.com/airacle100) |
| ---------------------------------------------------- | ------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
 | [최재훈](https://github.com/jhuni17)                  | [정현우](https://github.com/jung0228)                  | [김진석](https://github.com/Jin-SukKim)                  | [김동영](https://github.com/kimdyoc13)                  | [최민석](https://github.com/MinSeok1204)                  | [윤정우](https://github.com/airacle100)                  |


## Project Structure

```
.
├── README.md
├── dataset/                  # 데이터셋
│   ├── images/                   # YOLO 기반 모델 학습을 위한 이미지 데이터
│   ├── json/                     # COCO 형식의 JSON 어노테이션 파일
│   ├── labels/                   # YOLO 형식의 레이블 파일
│   ├── test/                     # 테스트 데이터셋 이미지
│   ├── train/                    # 학습 데이터셋 이미지
│   └── yaml/                     # YOLO 모델 설정을 위한 YAML 파일
│
├── eda_and_ensemble/         # EDA 및 앙상블 관련 코드
│   ├── csv/                      # CSV 파일 저장 디렉토리
│   ├── output/                   # 출력 결과 저장 디렉토리
│   ├── dataset_viewer.py         # 데이터셋 시각화 도구
│   ├── ensemble.py               # 여러 모델의 결과를 앙상블하는 스크립트
│   └── filter_low_confidence.py  # 낮은 신뢰도의 예측을 필터링하는 스크립트
│
├── mmdetection/              # MMDetection 프레임워크
│   ├── checkpoints/              # pretrained pth 저장 디렉토리
│   ├── custom_configs/           # 사용자 정의 config 저장 디렉토리
│   ├── scripts/
│   │   ├── train.py                    # MMDetection 모델 학습 스크립트
│   │   ├── split_coco_data.py          # COCO 데이터셋을 K-fold로 분할하는 스크립트
│   │   ├── merge_coco_jsons.py         # 여러 COCO JSON 파일을 병합하는 스크립트
│   │   ├── inference.py                # 학습된 모델을 사용한 추론 스크립트
│   │   ├── create_pseudo_labels.py     # Pseudo-label 생성 스크립트
│   │   └── create_custom_config.py     # 사용자 정의 config 파일 생성
│   ├── work_dirs/                # 모델 학습 결과 저장 디렉토리
│   └── ...                       # 기존 MMDetection library구조
│
├── yolo/                     # YOLO 관련 코드 및 모델
│   ├── check_kfold_ditribution.py            # K-fold 데이터 분포 확인 스크립트
│   ├── convert_coco_to_yolo.py               # COCO 형식을 YOLO 형식으로 변환하는 스크립트 (K-fold 적용)
│   ├── convert_coco_to_yolo_random_split.py  # COCO 형식을 YOLO 형식으로 변환하는 스크립트 (랜덤 분할)
│   ├── inference.py                          # YOLO 모델을 사용한 추론 스크립트
│   └── train.py                              # YOLO 모델 학습 스크립트
└── requirements.txt         
```

- `dataset/`: 학습 및 테스트에 사용되는 데이터셋
- `eda_and_ensemble/`: 탐색적 데이터 분석(EDA) 및 앙상블 관련 코드
- `mmdetection/`: MMDetection 프레임워크 및 관련 설정 파일
- `yolo/`: YOLO 모델 관련 코드 및 학습된 모델 파일
- `requirements.txt`: 프로젝트 실행에 필요한 Python 패키지 목록

  
## Usage

### Data Preparation
1. 데이터셋을 `dataset/` 디렉토리에 준비한다.
2. COCO 형식의 데이터를 YOLO 형식으로 변환하려면:
   ```
   cd yolo
   python convert_coco_to_yolo.py
   ```

### Training
1. MMDetection을 사용한 학습:
   ```
   cd mmdetection/scripts
   python train.py ../custom_configs/your_config.py
   ```
2. YOLO 모델 학습:
   ```
   cd yolo
   python train.py
   ```

### Inference
1. MMDetection을 사용한 추론:
   ```
   cd mmdetection/scripts
   python inference.py
   ```
2. YOLO 모델 추론:
   ```
   cd yolo
   python inference.py
   ```


## Requirements

- visdom==0.2.4
- seaborn==0.12.2
- albumentations==0.4.6
- imgaug==0.4.0
- pycocotools==2.0.6
- opencv-python==4.7.0.72
- tqdm==4.65.0
- torchnet==0.0.4
- pandas
- map-boxes==1.0.5
- jupyter==1.0.0
- openmim
- mmengine
- mmcv>=2.0.0rc4, <2.2.0
- mmdet=3.3.0
- ultralytics
- iterative-stratification
- ensemble_boxes


## Citation

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
```
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
```
