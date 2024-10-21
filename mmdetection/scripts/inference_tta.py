import os
import pandas as pd
from mmdet.apis import init_detector, inference_detector
from mmdet.evaluation import DetTTAModel
from mmengine import ConfigDict
from tqdm import tqdm

import sys
sys.path.append('..')

def main():
    # 설정 파일 및 체크포인트 파일 경로
    config_name = 'co_dino_5scale_swin_l_lsj_16xb1_1x_coco_nomask_trash'
    config_file = f'../custom_configs/{config_name}.py'  # 모델 설정 파일 경로

    model_epoch = 11
    checkpoint_file = '/data/ephemeral/home/Dong_yeong/level2-objectdetection-cv-06/mmdetection/work_dirs/co_dino/co_dino_swin_36.pth' 
    # checkpoint_file = f'../work_dirs/{config_name}/epoch_{model_epoch}.pth'  # 체크포인트 파일 경로

    # 이미지 경로 및 결과 저장 경로 설정
    image_folder = '../../dataset/test'  # 이미지 폴더 경로
    output_csv = f'../output/{config_name}_output_predictions_tta.csv'  # 출력 CSV 파일 경로

    # 모델 초기화
    print("Initializing model...")
    base_model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # TTA 설정
    tta_model = dict(
        type='DetTTAModel',
        tta_cfg=dict(
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    tta_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='TestTimeAug',
            transforms=[
                [
                    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
                    dict(type='RandomFlip', prob=1.0),
                    dict(type='RandomFlip', prob=0.0),
                ],
                [
                    dict(
                        type='PackDetInputs',
                        meta_keys=('img_id', 'img_path', 'ori_shape',
                                   'img_shape', 'scale_factor', 'flip',
                                   'flip_direction'))
                ],
            ])
    ]

    # TTA 모델 설정
    cfg = ConfigDict(
        model=ConfigDict(**tta_model, module=base_model),
        test_pipeline=tta_pipeline
    )

    # TTA 모델 초기화
    model = DetTTAModel(cfg, test_cfg=None)

    # 결과 저장 리스트
    results = []

    # 이미지 파일 목록 생성
    image_files = [f for f in sorted(os.listdir(image_folder)) if f.endswith(('.jpg', '.png'))]

    # 이미지 추론
    print("Starting inference with TTA...")
    for image_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(image_folder, image_name)
        result = inference_detector(model, img_path)

        # DetDataSample에서 결과 추출
        prediction_string = []
        if hasattr(result, 'pred_instances'):
            det_samples = result.pred_instances  # 예측 결과의 인스턴스들
            if det_samples is not None:
                bboxes = det_samples.bboxes
                scores = det_samples.scores
                labels = det_samples.labels

                for j in range(len(bboxes)):
                    prediction_string.append(
                        f"{int(labels[j])} {scores[j]:.4f} {bboxes[j][0]:.2f} {bboxes[j][1]:.2f} {bboxes[j][2]:.2f} {bboxes[j][3]:.2f}"
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
    print(f"Inference with TTA complete. Results saved to {output_csv}")

if __name__ == '__main__':
    main()