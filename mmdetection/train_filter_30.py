import json
from tqdm import tqdm

def filter_coco_data(json_file, output_file, max_annotations=30):
    # train.json 파일 열기
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 이미지별 annotation 수를 계산하기 위한 딕셔너리 생성
    annotation_count_per_image = {image['id']: 0 for image in data['images']}

    # 각 이미지의 annotation 수 계산
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        annotation_count_per_image[image_id] += 1

    # annotation이 30개 미만인 이미지를 필터링
    filtered_images = [image for image in data['images'] if annotation_count_per_image[image['id']] < max_annotations]

    # 해당 이미지들의 id를 저장
    filtered_image_ids = {image['id'] for image in filtered_images}

    # 필터링된 이미지 id에 해당하는 annotation만 저장
    filtered_annotations = [annotation for annotation in data['annotations'] if annotation['image_id'] in filtered_image_ids]

    # 필터링된 데이터를 새로운 json 구조로 저장
    filtered_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': data['categories'],
        'licenses': data.get('licenses', []),
        'info': data.get('info', {}),
    }

    # 새로운 파일로 저장
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)

    print(f"Filtered dataset saved to {output_file}")

# train.json 파일에서 30개 이상의 annotation을 가진 이미지를 필터링하여 filtered_30_train.json 파일로 저장
filter_coco_data('../../dataset/json/train.json', 'filtered_30_train.json')