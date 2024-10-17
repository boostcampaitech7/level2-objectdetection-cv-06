from ultralytics import YOLO
import wandb

def main():
    # 학습 설정
    data_yaml = '../dataset/yaml/dataset_fold4.yaml'
    epochs = 100
    batch_size = 8
    img_size = 1024
    device = '0'

    # 모델 로드
    model = YOLO('yolo11x.pt')

    # 학습 시작
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project='CV Object Detection',
        name='yolo11x_augmix',
        auto_augment='augmix'
    )

    # 학습된 모델 저장
    model.save('yolo11x_augmix.pt')

if __name__ == '__main__':
    main()