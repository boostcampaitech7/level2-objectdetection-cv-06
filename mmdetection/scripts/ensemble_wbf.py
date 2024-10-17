import pandas as pd
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from pycocotools.coco import COCO

# ensemble csv files
submission_files = ['../csv/output_codino_36.csv',
                    '../csv/output_yolo11x.csv']

submission_df = [pd.read_csv(file) for file in submission_files]

image_ids = submission_df[0]['image_id'].tolist()

# ensemble 할 file의 image 정보를 불러오기 위한 json
annotation = '../../dataset/json/test.json'
coco = COCO(annotation)

prediction_strings = []
file_names = []

# ensemble 시 설정할 iou threshold, 이 부분을 바꿔가며 대회 metric에 알맞게 적용해봅니다
iou_thr = 0.8
skip_box_thr = 0.001  # WBF에서 무시할 box의 score threshold

# 각 image id 별로 submission file에서 box 좌표 추출
for i, image_id in enumerate(image_ids):
    prediction_string = ''
    boxes_list = []
    scores_list = []
    labels_list = []
    image_info = coco.loadImgs(i)[0]

    # 각 submission file 별로 prediction box 좌표 불러오기
    for df in submission_df:
        predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
        predict_list = str(predict_string).split()

        if len(predict_list) == 0 or len(predict_list) == 1:
            continue

        predict_list = np.reshape(predict_list, (-1, 6))
        box_list = []

        # box의 각 좌표를 float형으로 변환한 후 image의 넓이와 높이로 각각 정규화
        for box in predict_list[:, 2:6].tolist():
            image_width = image_info['width']
            image_height = image_info['height']
            box[0] = float(box[0]) / image_width  # xmin
            box[1] = float(box[1]) / image_height  # ymin
            box[2] = float(box[2]) / image_width  # xmax
            box[3] = float(box[3]) / image_height  # ymax
            box_list.append(box)

        boxes_list.append(box_list)
        scores_list.append(list(map(float, predict_list[:, 1].tolist())))
        labels_list.append(list(map(int, predict_list[:, 0].tolist())))

    # 예측 box가 있다면 이를 WBF 방식으로 ensemble 수행
    if len(boxes_list):
        # WBF (Weighted Boxes Fusion) 계산 수행
        # weighted_boxes_fusion(boxes, scores, labels, iou_thr, skip_box_thr)
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )

        # WBF 후의 box 정보로 prediction string 생성
        for box, score, label in zip(boxes, scores, labels):
            prediction_string += str(label) + ' ' + str(score) + ' ' + \
                                 str(box[0] * image_info['width']) + ' ' + \
                                 str(box[1] * image_info['height']) + ' ' + \
                                 str(box[2] * image_info['width']) + ' ' + \
                                 str(box[3] * image_info['height']) + ' '

    prediction_strings.append(prediction_string)
    file_names.append(image_id)

# 결과 저장
submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv('../csv/submission_ensemble_wbf_0.8.csv', index=False)
submission.head()
