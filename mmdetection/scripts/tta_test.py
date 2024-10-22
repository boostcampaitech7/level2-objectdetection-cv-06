import os
import pandas as pd
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm
import sys
import numpy as np
import mmcv
from mmcv import imflip, imrotate

sys.path.append('..')


def apply_tta(model, img_path):
    # 이미지 경로에서 이미지를 로드
    img = mmcv.imread(img_path)
    
    # 원본 이미지 추론
    original_result = inference_detector(model, img)

    # TTA 변환: 좌우 반전
    flipped_img = mmcv.imflip(img, direction='horizontal')
    flipped_result = inference_detector(model, flipped_img)

    # TTA 변환: 90도 회전
    rotated_img_90 = mmcv.imrotate(img, angle=90)
    rotated_result = inference_detector(model, rotated_img_90)

    # TTA 변환: 180도 회전
    rotated_img_180 = mmcv.imrotate(img, angle=180)
    rotated_result_180 = inference_detector(model, rotated_img_180)

    # 결과들을 모아서 평균화
    results = [original_result, flipped_result, rotated_result, rotated_result_180]

    combined_results = []
    for result in results:
        if hasattr(result, 'pred_instances'):
            det_samples = result.pred_instances
            if det_samples is not None:
                combined_results.append(det_samples)

    if not combined_results:
        return None

    combined_bboxes, combined_scores, combined_labels = [], [], []

    min_len = float('inf')  # 최소 객체 수를 찾기 위한 변수
    for det_samples in combined_results:
        bboxes = det_samples.bboxes.cpu().numpy()
        scores = det_samples.scores.cpu().numpy()
        labels = det_samples.labels.cpu().numpy()
        
        # 최소 개수를 기준으로 자름
        min_len = min(min_len, bboxes.shape[0])
        combined_bboxes.append(bboxes)
        combined_scores.append(scores)
        combined_labels.append(labels)

    # 모든 결과를 최소 길이에 맞춰 자르기
    combined_bboxes = [b[:min_len] for b in combined_bboxes]
    combined_scores = [s[:min_len] for s in combined_scores]
    combined_labels = [l[:min_len] for l in combined_labels]

    # 평균화
    bboxes = np.mean(np.array(combined_bboxes), axis=0)
    scores = np.mean(np.array(combined_scores), axis=0)
    labels = np.mean(np.array(combined_labels), axis=0)

    return (bboxes, scores, labels)

def main():
    config_name = 'co_dino_5scale_swin_l_lsj_16xb1_1x_coco_nomask_trash'
    config_file = f'../custom_configs/{config_name}.py'
    model_epoch = 1
    checkpoint_file = f'../checkpoint/co_dino_swin_36.pth'

    image_folder = '../../dataset/test'
    output_csv = f'../output/{config_name}_output_predictions.csv'

    print("Initializing model...")
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    results = []
    image_files = [f for f in sorted(os.listdir(image_folder)) if f.endswith(('.jpg', '.png'))]

    print("Starting inference...")
    for image_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(image_folder, image_name)
        
        combined_result = apply_tta(model, img_path)
        
        prediction_string = []
        if combined_result:
            bboxes, scores, labels = combined_result
            for j in range(len(bboxes)):
                prediction_string.append(
                    f"{int(labels[j])} {scores[j]:.4f} {bboxes[j][0]:.2f} {bboxes[j][1]:.2f} {bboxes[j][2]:.2f} {bboxes[j][3]:.2f}"
                )

        results.append({
            'PredictionString': " ".join(prediction_string),
            'image_id': f'test/{image_name}'
        })

    print("Creating DataFrame...")
    df = pd.DataFrame(results)

    print(f"Saving results to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print(f"Inference complete. Results saved to {output_csv}")

if __name__ == '__main__':
    main()
