import json

def merge_coco_jsons(json_file1, json_file2, output_file):
    with open(json_file1, 'r') as f1, open(json_file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # Merge images
    max_image_id = max(img['id'] for img in data1['images'])
    image_id_map = {}
    for img in data2['images']:
        old_id = img['id']
        new_id = max_image_id + 1
        image_id_map[old_id] = new_id
        img['id'] = new_id
        data1['images'].append(img)
        max_image_id = new_id

    # Merge annotations
    max_ann_id = max(ann['id'] for ann in data1['annotations'])
    for ann in data2['annotations']:
        ann['id'] = max_ann_id + 1
        ann['image_id'] = image_id_map[ann['image_id']]
        data1['annotations'].append(ann)
        max_ann_id += 1

    # Categories should be the same, so we don't need to merge them

    with open(output_file, 'w') as f:
        json.dump(data1, f)

# Usage
train_json = '../../dataset/json/splits/train_fold4.json'
pseudo_json = '../../dataset/json/pseudo_labels.json'
merged_json = '../../dataset/json/merged_train_pseudo.json'

merge_coco_jsons(train_json, pseudo_json, merged_json)
print(f"Merged JSON saved to {merged_json}")