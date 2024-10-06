from ultralytics import YOLO

def main():
    # 학습 설정
    data_yaml = '../dataset/dataset.yaml'
    epochs = 100
    batch_size = 16
    img_size = 640
    device = '0'  # '0'은 첫 번째 GPU를 의미합니다. CPU를 사용하려면 'cpu'로 설정하세요.

    # 모델 로드
    model = YOLO('yolov10s.pt')  # 또는 'yolov10.yaml'을 사용하여 처음부터 학습

    # 학습 시작
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        name='yolov10_custom'
    )

    # 학습된 모델 저장
    model.save('yolov10_custom.pt')

if __name__ == '__main__':
    main()