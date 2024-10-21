from ultralytics import YOLO
import wandb
from ultralytics.yolo.utils.callbacks.raytune import RaytuneCallback
from ultralytics.yolo.data.dataloaders.v5augmentations import LetterBox
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

class RandomCenterCropPad(A.DualTransform):
    def __init__(self, min_scale=0.8, max_scale=1.2, always_apply=False, p=1.0):
        super(RandomCenterCropPad, self).__init__(always_apply, p)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def apply(self, img, scale=1.0, **params):
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        if scale > 1:
            # Padding
            pad_h, pad_w = new_h - h, new_w - w
            img = cv2.copyMakeBorder(img, pad_h//2, pad_h - pad_h//2, 
                                     pad_w//2, pad_w - pad_w//2, 
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
        else:
            # Cropping
            crop_h, crop_w = h - new_h, w - new_w
            img = img[crop_h//2:h-crop_h+crop_h//2, crop_w//2:w-crop_w+crop_w//2]
        
        return img

    def get_params(self):
        return {"scale": np.random.uniform(self.min_scale, self.max_scale)}

    def apply_to_bbox(self, bbox, scale=1.0, **params):
        x_min, y_min, x_max, y_max = bbox
        if scale > 1:
            x_min, x_max = x_min / scale, x_max / scale
            y_min, y_max = y_min / scale, y_max / scale
        else:
            x_min, x_max = (x_min - 0.5 + 0.5/scale) * scale, (x_max - 0.5 + 0.5/scale) * scale
            y_min, y_max = (y_min - 0.5 + 0.5/scale) * scale, (y_max - 0.5 + 0.5/scale) * scale
        return [x_min, y_min, x_max, y_max]

def custom_transforms():
    return A.Compose([
        RandomCenterCropPad(min_scale=0.8, max_scale=1.2, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def main():
    # 학습 설정
    data_yaml = '/data/ephemeral/home/Dong_yeong/level2-objectdetection-cv-06/dataset/yaml/dataset_fold4.yaml'
    epochs = 100
    batch_size = 8
    img_size = 1024
    device = '0'

    # 모델 로드
    model = YOLO('yolo11x.pt')

    # 커스텀 변환 설정
    model.add_callback('on_train_start', lambda: setattr(model.trainer.train_loader.dataset, 'transform', custom_transforms()))

    # 학습 시작
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project='CV Object Detection',
        name='yolo11x_autoaugment_randomcentercroppad',
        augment=True,
        auto_augment='autoaugment'
    )

    # 학습된 모델 저장
    model.save('yolo11x_fold_4_autoaugment_randomcentercroppad.pt')

if __name__ == '__main__':
    main()