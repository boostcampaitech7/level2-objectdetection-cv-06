from mmdet.apis import DetInferencer
# 학습된 모델과 config 파일을 경로로 설정
config_file = './co_dino_trash.py'  # 학습에 사용된 config 파일 경로
checkpoint_file = './work_dirs/co_dino_trash/epoch_36.pth'  # 학습된 모델의 checkpoint 파일 경로

# DetInferencer 초기화 (학습된 모델과 config 파일 적용)
inferencer = DetInferencer(model=config_file, weights=checkpoint_file)

# 추론 수행 (이미지 파일 경로와 함께 show=True 옵션을 사용하여 결과 시각화)
inferencer('../../dataset/test/0000.jpg', show=True)
