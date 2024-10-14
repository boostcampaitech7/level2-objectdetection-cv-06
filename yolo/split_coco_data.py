import json
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

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
        if category_id not in image_labels[image_id]:
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

def save_coco_split(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f)

# 실행
if __name__ == "__main__":
    input_json = '../dataset/json/train.json'
    output_dir = '../dataset/splits'
    os.makedirs(output_dir, exist_ok=True)

    fold_data = split_coco_data(input_json, n_splits=5)

    for fold, (train_data, val_data) in enumerate(fold_data, 1):
        train_output = os.path.join(output_dir, f'train_fold{fold}.json')
        val_output = os.path.join(output_dir, f'val_fold{fold}.json')

        save_coco_split(train_data, train_output)
        save_coco_split(val_data, val_output)

        print(f"Fold {fold} saved: {train_output}, {val_output}")

    print("All folds processed and saved.")