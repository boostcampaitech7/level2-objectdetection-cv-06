import json

def merge_coco_jsons(json_file1, json_file2, output_file):
    """
    두 개의 COCO 형식 JSON 파일을 병합
    
    :param json_file1: 첫 번째 JSON 파일 경로 (기본 데이터셋)
    :param json_file2: 두 번째 JSON 파일 경로 (추가할 데이터셋)
    :param output_file: 병합된 결과를 저장할 JSON 파일 경로
    """
    # JSON 파일 로드
    with open(json_file1, 'r') as f1, open(json_file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # 이미지 병합
    max_image_id = max(img['id'] for img in data1['images'])
    image_id_map = {}
    for img in data2['images']:
        old_id = img['id']
        new_id = max_image_id + 1
        image_id_map[old_id] = new_id  # 이미지 ID 매핑 저장
        img['id'] = new_id  # 새 이미지 ID 할당
        data1['images'].append(img)
        max_image_id = new_id

    # 어노테이션 병합
    max_ann_id = max(ann['id'] for ann in data1['annotations'])
    for ann in data2['annotations']:
        ann['id'] = max_ann_id + 1  # 새 어노테이션 ID 할당
        ann['image_id'] = image_id_map[ann['image_id']]  # 이미지 ID 업데이트
        data1['annotations'].append(ann)
        max_ann_id += 1

    # 카테고리는 동일하다고 가정하므로 병합하지 않음

    # 병합된 데이터를 새 JSON 파일로 저장
    with open(output_file, 'w') as f:
        json.dump(data1, f)

# 사용 예시
train_json = '../../dataset/json/splits/train_fold4.json'  # 기본 학습 데이터셋
pseudo_json = '../../dataset/json/pseudo_labels.json'  # 의사 레이블 데이터셋
merged_json = '../../dataset/json/merged_train_pseudo.json'  # 병합된 결과 파일

# JSON 파일 병합 실행
merge_coco_jsons(train_json, pseudo_json, merged_json)
print(f"Merged JSON saved to {merged_json}")