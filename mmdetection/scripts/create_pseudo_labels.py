import pandas as pd
import json
from tqdm import tqdm

def create_pseudo_labels(csv_file, confidence_threshold=0.7):
    df = pd.read_csv(csv_file)
    
    pseudo_annotations = []
    images = []
    image_id_map = {}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing predictions"):
        image_id = row['image_id']
        file_name = f"test/{image_id.split('/')[-1]}"
        if file_name not in image_id_map:
            image_id_map[file_name] = len(image_id_map)
            images.append({
                'id': image_id_map[file_name],
                'file_name': file_name,
                'height': 1024,
                'width': 1024
            })
        
        predictions = row['PredictionString'].split()
        for i in range(0, len(predictions), 6):
            label, conf, x1, y1, x2, y2 = predictions[i:i+6]
            conf = float(conf)
            if conf >= confidence_threshold:
                x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
                pseudo_annotations.append({
                    'image_id': image_id_map[file_name],
                    'category_id': int(label),
                    'bbox': [x1, y1, x2-x1, y2-y1],
                    'area': (x2-x1) * (y2-y1),
                    'iscrowd': 0,
                    'id': len(pseudo_annotations)
                })
    
    categories = [
        {'id': i, 'name': name} for i, name in enumerate([
            "General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styroform", "Plastic bag", "Battery", "Clothing"
        ])
    ]
    
    pseudo_coco = {
        'images': images,
        'annotations': pseudo_annotations,
        'categories': categories
    }
    
    return pseudo_coco

def save_pseudo_labels(pseudo_coco, output_file):
    with open(output_file, 'w') as f:
        json.dump(pseudo_coco, f)

if __name__ == "__main__":
    csv_file = '../../eda_and_ensemble/csv/NMW Ensemble 0.6 (1.2, 0.8) (0.7397, cosine_6).csv'
    output_json = '../../dataset/json/pseudo_labels.json'
    
    pseudo_coco = create_pseudo_labels(csv_file)
    save_pseudo_labels(pseudo_coco, output_json)
    print(f"Pseudo labels saved to {output_json}")