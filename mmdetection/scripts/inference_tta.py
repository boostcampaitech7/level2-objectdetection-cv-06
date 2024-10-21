import os
import pandas as pd
from mmdet.apis import init_detector, inference_detector
from mmengine.structures import InstanceData
from mmdet.utils import register_all_modules
from mmdet.evaluation import get_classes
from tqdm import tqdm

import sys
sys.path.append('..')

def main():
    # 설정 파일 및 체크포인트 파일 경로
    config_name = 'co_dino_5scale_swin_l_lsj_16xb1_1x_coco_nomask_trash'
    config_file = f'../custom_configs/{config_name}.py'  # 모델 설정 파일 경로

    checkpoint_file = '/data/ephemeral/home/Dong_yeong/level2-objectdetection-cv-06/mmdetection/work_dirs/co_dino/co_dino_swin_36.pth'

    # 이미지 경로 및 결과 저장 경로 설정
    image_folder = '../../dataset/test'  # 이미지 폴더 경로
    output_csv = f'../output/{config_name}_output_predictions.csv'  # 출력 CSV 파일 경로

    # 모듈 등록 및 모델 초기화
    register_all_modules()
    print("Initializing model...")
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # 새로운 test_cfg 설정
    model.test_cfg = dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        score_thr=0.0
    )

    # 결과 저장 리스트
    results = []

    # 이미지 파일 목록 생성
    image_files = [f for f in sorted(os.listdir(image_folder)) if f.endswith(('.jpg', '.png'))]

    # 클래스 이름 가져오기
    classes = get_classes('coco')

    # 이미지 추론
    print("Starting inference...")
    for image_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(image_folder, image_name)
        result = inference_detector(model, img_path)

        # TTA 결과 처리
        if isinstance(result, tuple):
            result = result[0]

        # DetDataSample에서 결과 추출
        prediction_string = []
        if isinstance(result, InstanceData):
            bboxes = result.bboxes
            scores = result.scores
            labels = result.labels

            # score_thr 적용
            mask = scores > model.test_cfg['score_thr']
            bboxes = bboxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            for j in range(len(bboxes)):
                prediction_string.append(
                    f"{classes[labels[j]]} {scores[j]:.4f} {bboxes[j][0]:.2f} {bboxes[j][1]:.2f} {bboxes[j][2]:.2f} {bboxes[j][3]:.2f}"
                )

        # PredictionString을 먼저 저장하고 image_id는 그 다음에 저장
        results.append({
            'PredictionString': " ".join(prediction_string),  # 한 줄에 저장
            'image_id': f'test/{image_name}'
        })

    # 데이터프레임 생성
    print("Creating DataFrame...")
    df = pd.DataFrame(results)

    # CSV 파일로 저장
    print(f"Saving results to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print(f"Inference complete. Results saved to {output_csv}")

if __name__ == '__main__':
    main()