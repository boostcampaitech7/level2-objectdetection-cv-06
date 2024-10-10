import pandas as pd
import numpy as np
from ensemble_boxes import nms
from pycocotools.coco import COCO
import os

def main():
    # ensemble할 csv 파일들
    submission_files = [
        'output/CO-DINO(r50, lsj, default setting).csv',
        'output/yolo11x_fold1.csv',
    ]
    submission_df = [pd.read_csv(file) for file in submission_files]

    # 이미지 ID 목록 가져오기
    image_ids = submission_df[0]['image_id'].tolist()

    # 테스트 데이터 JSON 파일 경로
    annotation = '../dataset/json/test.json'
    coco = COCO(annotation)

    prediction_strings = []
    file_names = []
    iou_thr = 0.4  # NMS를 위한 IoU 임계값

    for i, image_id in enumerate(image_ids):
        prediction_string = ''
        boxes_list = []
        scores_list = []
        labels_list = []
        image_info = coco.loadImgs(i)[0]

        for df in submission_df:
            predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
            predict_list = str(predict_string).split()

            if len(predict_list) == 0 or len(predict_list) == 1:
                continue

            predict_list = np.reshape(predict_list, (-1, 6))
            box_list = []

            for box in predict_list[:, 2:6].tolist():
                image_width = image_info['width']
                image_height = image_info['height']
                box[0] = float(box[0]) / image_width
                box[1] = float(box[1]) / image_height
                box[2] = float(box[2]) / image_width
                box[3] = float(box[3]) / image_height
                box_list.append(box)

            boxes_list.append(box_list)
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))

        if len(boxes_list):
            boxes, scores, labels = nms(boxes_list, scores_list, labels_list, iou_thr)
            for box, score, label in zip(boxes, scores, labels):
                prediction_string += f"{label} {score} {box[0]*image_info['width']} {box[1]*image_info['height']} {box[2]*image_info['width']} {box[3]*image_info['height']} "

        prediction_strings.append(prediction_string)
        file_names.append(image_id)

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names

    # 결과 저장
    os.makedirs('./output', exist_ok=True)
    submission.to_csv('./output/submission_ensemble.csv', index=False)
    print("Ensemble result saved to ./output/submission_ensemble.csv")

if __name__ == "__main__":
    main()