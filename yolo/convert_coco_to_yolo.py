import json
import os
from tqdm import tqdm
import yaml
import random
import shutil
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    COCO 형식의 바운딩 박스를 YOLO 형식으로 변환합니다.
    COCO bbox: [x_min, y_min, width, height]
    YOLO bbox: [x_center, y_center, width, height] (모두 0~1로 정규화)
    """
    x_min, y_min, width, height = bbox

    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width = width / img_width
    height = height / img_height

    return [x_center, y_center, width, height]

def split_coco_data(json_file, n_splits=5):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 데이터프레임으로 변환
    df = pd.DataFrame(data['images'])

    # 이미지 id를 기준으로 annotation에서 레이블을 추출
    image_labels = {image['id']: [] for image in data['images']}

    # 각 이미지에 해당하는 레이블 추가
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        image_labels[image_id].append(category_id)

    # 각 이미지의 멀티라벨 리스트를 데이터프레임에 추가
    df['labels'] = df['id'].map(image_labels)

    # MultiLabelBinarizer를 사용하여 멀티라벨을 이진 매트릭스로 변환
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['labels'])

    # MultilabelStratifiedKFold 사용
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 각 fold에 대한 데이터 생성
    fold_data = []
    for fold, (train_idx, val_idx) in enumerate(mskf.split(df, y), 1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_annotations = [ann for ann in data['annotations'] if ann['image_id'] in train_df['id'].values]
        val_annotations = [ann for ann in data['annotations'] if ann['image_id'] in val_df['id'].values]

        train_data = {
            'images': train_df.to_dict(orient='records'),
            'annotations': train_annotations,
            'categories': data['categories']
        }
        val_data = {
            'images': val_df.to_dict(orient='records'),
            'annotations': val_annotations,
            'categories': data['categories']
        }

        fold_data.append((train_data, val_data))

    return fold_data

def coco2yolo(coco_data, output_path):
    """
    COCO 데이터를 YOLO 형식으로 변환합니다.
    """
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_path, exist_ok=True)

    # 이미지 ID를 파일 이름에 매핑
    image_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}

    # 카테고리 ID를 인덱스에 매핑 (YOLO는 0부터 시작하는 정수 클래스를 사용)
    category_id_to_index = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}

    # 각 이미지에 대한 주석을 처리
    for img in tqdm(coco_data['images'], desc="Converting annotations"):
        img_id = img['id']
        img_name = image_id_to_name[img_id]
        img_width = img['width']
        img_height = img['height']

        # YOLO 형식의 레이블 파일 이름 생성
        label_name = os.path.splitext(os.path.basename(img_name))[0] + '.txt'
        label_path = os.path.join(output_path, label_name)

        # 이 이미지에 대한 모든 주석 찾기
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

        with open(label_path, 'w') as f:
            for ann in annotations:
                category_index = category_id_to_index[ann['category_id']]
                bbox = convert_bbox_coco2yolo(img_width, img_height, ann['bbox'])
                f.write(f"{category_index} {' '.join([str(x) for x in bbox])}\n")

    print(f"Conversion complete. YOLO format labels saved in {output_path}")
    return coco_data['categories']

def copy_images(coco_data, src_dir, dst_dir):
    """
    이미지를 소스 디렉토리에서 대상 디렉토리로 복사합니다.
    """
    os.makedirs(dst_dir, exist_ok=True)
    for img in tqdm(coco_data['images'], desc=f"Copying images to {dst_dir}"):
        file_name = img['file_name']
        if file_name.startswith('train/') or file_name.startswith('test/'):
            file_name = os.path.basename(file_name)
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: Source file not found: {src_path}")

# 실행
fold_data = split_coco_data('../dataset/train.json', n_splits=5)

for fold, (train_data, val_data) in enumerate(fold_data, 1):
    # train_yolo.json과 val_yolo.json 저장
    with open(f'../dataset/json/train_yolo_fold{fold}.json', 'w') as f:
        json.dump(train_data, f)
    with open(f'../dataset/json/val_yolo_fold{fold}.json', 'w') as f:
        json.dump(val_data, f)

    # 이미지 복사
    copy_images(train_data, '../dataset/train', f'../dataset/images/train_yolo_fold{fold}')
    copy_images(val_data, '../dataset/train', f'../dataset/images/val_yolo_fold{fold}')

    # COCO 형식을 YOLO 형식으로 변환
    train_categories = coco2yolo(train_data, f'../dataset/labels/train_yolo_fold{fold}')
    val_categories = coco2yolo(val_data, f'../dataset/labels/val_yolo_fold{fold}')

    # YOLO 데이터셋 설정 파일 생성
    dataset_config = {
        'path': '../../dataset',
        'train': f'images/train_yolo_fold{fold}',
        'val': f'images/val_yolo_fold{fold}',
        'test': 'images/test',
        'nc': len(train_categories),  # 클래스 수
        'names': [cat['name'] for cat in train_categories]
    }

    with open(f'../dataset/yaml/dataset_fold{fold}.yaml', 'w') as f:
        yaml.dump(dataset_config, f)

    print(f"Dataset configuration for fold {fold} saved as dataset_fold{fold}.yaml")

# test 데이터 처리
with open('../dataset/json/test.json', 'r') as f:
    test_data = json.load(f)
test_categories = coco2yolo(test_data, '../dataset/labels/test')
copy_images(test_data, '../dataset/test', '../dataset/images/test')

print("All folds processed and saved.")