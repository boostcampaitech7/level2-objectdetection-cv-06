from ultralytics import YOLO
import csv
import os

def xywh_to_xyxy(x_center, y_center, width, height):
    """
    중심 좌표와 너비, 높이를 xmin, ymin, xmax, ymax로 변환
    
    :param x_center: 바운딩 박스의 중심 x 좌표
    :param y_center: 바운딩 박스의 중심 y 좌표
    :param width: 바운딩 박스의 너비
    :param height: 바운딩 박스의 높이
    :return: xmin, ymin, xmax, ymax 좌표
    """
    xmin = x_center - width / 2
    ymin = y_center - height / 2
    xmax = x_center + width / 2
    ymax = y_center + height / 2
    return xmin, ymin, xmax, ymax

def main():
    # YOLO 모델 로드
    model = YOLO('CV Object Detection/yolo11x_fold4/weights/best.pt')  # 학습된 모델 경로
    
    # 테스트 데이터 디렉토리 경로
    test_dir = '../dataset/test'

    # 결과를 저장할 리스트 초기화
    results_list = []

    # 테스트 이미지에 대해 추론 수행
    for img_name in sorted(os.listdir(test_dir)):
        img_path = os.path.join(test_dir, img_name)
        # 모델을 사용하여 이미지에 대한 예측 수행 (augment=True로 설정하여 테스트 시 augmentation 적용)
        results = model(img_path, augment=True)

        # 결과 처리
        predictions = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 바운딩 박스 좌표 추출 및 변환
                x_center, y_center, width, height = box.xywh[0]  # xywh 포맷으로 변환
                xmin, ymin, xmax, ymax = xywh_to_xyxy(x_center, y_center, width, height)
                conf = box.conf[0]  # 신뢰도 점수
                cls = int(box.cls[0])  # 예측된 클래스
                # 예측 결과를 문자열로 포맷팅
                predictions.append(f"{cls} {conf:.16f} {xmin:.16f} {ymin:.16f} {xmax:.16f} {ymax:.16f}")
        
        # 이미지 ID 생성 (test/0001.jpg 형식)
        image_id = f"test/{img_name}"

        # 결과 리스트에 추가
        results_list.append([' '.join(predictions), image_id])

    # 결과를 CSV 파일로 저장
    with open('submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PredictionString', 'image_id'])  # CSV 헤더 작성
        writer.writerows(results_list)  # 예측 결과 작성

    print("Inference complete. Results saved to submission.csv")

if __name__ == '__main__':
    main()