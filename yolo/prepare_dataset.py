
import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_PATH, "antrenare")
OUTPUT_PATH = os.path.join(BASE_PATH, "yolo", "dataset")

CHAR_TO_ID = {
    "fred": 0,
    "daphne": 1,
    "shaggy": 2,
    "velma": 3,
}

CHARACTERS = ["fred", "daphne", "shaggy", "velma"]


def parse_annotations():
    annotations = {} 
    
    for char in CHARACTERS:
        ann_file = os.path.join(TRAIN_PATH, f"{char}_annotations.txt")
        img_folder = os.path.join(TRAIN_PATH, char)
        
        if not os.path.exists(ann_file):
            print(f"Warning: {ann_file} not found")
            continue
            
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    img_name, x1, y1, x2, y2, label = parts[:6]
                    img_path = os.path.join(img_folder, img_name)
                    
                    if img_path not in annotations:
                        annotations[img_path] = []
                    
                    annotations[img_path].append({
                        'char': label,
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2)
                    })
    
    return annotations


def convert_to_yolo_format(img_path, boxes):
    try:
        img = Image.open(img_path)
        img_w, img_h = img.size
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None
    
    yolo_annotations = []
    for box in boxes:
        char = box['char']
        if char not in CHAR_TO_ID:
            continue
            
        class_id = CHAR_TO_ID[char]
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations


def create_dataset():
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_PATH, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, 'labels', split), exist_ok=True)
    
    print("Parsing annotations...")
    annotations = parse_annotations()
    print(f"Found {len(annotations)} images with annotations")
    
    if len(annotations) == 0:
        print("No annotations found! Make sure image folders exist in antrenare/")
        return
    
    img_paths = list(annotations.keys())
    train_imgs, val_imgs = train_test_split(img_paths, test_size=0.2, random_state=42)
    
    print(f"Train images: {len(train_imgs)}, Val images: {len(val_imgs)}")
    
    for split, img_list in [('train', train_imgs), ('val', val_imgs)]:
        for img_path in img_list:
            if not os.path.exists(img_path):
                continue
                
            yolo_anns = convert_to_yolo_format(img_path, annotations[img_path])
            if yolo_anns is None:
                continue
            
            img_name = os.path.basename(img_path)
            folder_name = os.path.basename(os.path.dirname(img_path))
            unique_name = f"{folder_name}_{img_name}"
            
            dst_img = os.path.join(OUTPUT_PATH, 'images', split, unique_name)
            shutil.copy2(img_path, dst_img)
            
            label_name = os.path.splitext(unique_name)[0] + '.txt'
            dst_label = os.path.join(OUTPUT_PATH, 'labels', split, label_name)
            with open(dst_label, 'w') as f:
                f.write('\n'.join(yolo_anns))
    
    yaml_content = f"""# Scooby-Doo Face Detection Dataset
path: {OUTPUT_PATH}
train: images/train
val: images/val

# Classes
names:
  0: fred
  1: daphne
  2: shaggy
  3: velma

nc: 4
"""
    
    yaml_path = os.path.join(OUTPUT_PATH, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Dataset created at {OUTPUT_PATH}")
    print(f"YAML config: {yaml_path}")


if __name__ == "__main__":
    create_dataset()
