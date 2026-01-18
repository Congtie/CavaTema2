
import os
import numpy as np
from ultralytics import YOLO
from glob import glob
from tqdm import tqdm

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, "yolo", "runs", "scooby_doo", "weights", "best.pt")

OUTPUT_PATH = os.path.join(BASE_PATH, "evaluare", "fisiere_solutie", "solution_yolo")
OUTPUT_TASK1 = os.path.join(OUTPUT_PATH, "task1")
OUTPUT_TASK2 = os.path.join(OUTPUT_PATH, "task2")

ID_TO_CHAR = {0: "fred", 1: "daphne", 2: "shaggy", 3: "velma"}
CHARACTERS = ["fred", "daphne", "shaggy", "velma"]


def run_inference(input_folder, model_path=None, conf_threshold=0.25):
    if model_path is None:
        model_path = MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using: python train_yolo.py")
        return
    
    os.makedirs(OUTPUT_TASK1, exist_ok=True)
    os.makedirs(OUTPUT_TASK2, exist_ok=True)
    
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(input_folder, ext)))
    
    print(f"Found {len(image_files)} images in {input_folder}")
    
    all_detections = []
    all_scores = []
    all_file_names = []
    
    char_detections = {char: [] for char in CHARACTERS}
    char_scores = {char: [] for char in CHARACTERS}
    char_file_names = {char: [] for char in CHARACTERS}
    
    for img_path in tqdm(image_files, desc="Processing images"):
        img_name = os.path.basename(img_path)
        
        results = model(img_path, conf=conf_threshold, verbose=False)
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                
                all_detections.append(bbox)
                all_scores.append(conf)
                all_file_names.append(img_name)
                
                if cls_id in ID_TO_CHAR:
                    char = ID_TO_CHAR[cls_id]
                    char_detections[char].append(bbox)
                    char_scores[char].append(conf)
                    char_file_names[char].append(img_name)
    
    print("\nSaving Task 1 results...")
    np.save(os.path.join(OUTPUT_TASK1, "detections_all_faces.npy"), 
            np.array(all_detections))
    np.save(os.path.join(OUTPUT_TASK1, "scores_all_faces.npy"), 
            np.array(all_scores))
    np.save(os.path.join(OUTPUT_TASK1, "file_names_all_faces.npy"), 
            np.array(all_file_names))
    
    print(f"  Total detections: {len(all_detections)}")
    
    print("\nSaving Task 2 results...")
    for char in CHARACTERS:
        np.save(os.path.join(OUTPUT_TASK2, f"detections_{char}.npy"), 
                np.array(char_detections[char]))
        np.save(os.path.join(OUTPUT_TASK2, f"scores_{char}.npy"), 
                np.array(char_scores[char]))
        np.save(os.path.join(OUTPUT_TASK2, f"file_names_{char}.npy"), 
                np.array(char_file_names[char]))
        
        print(f"  {char}: {len(char_detections[char])} detections")
    
    print(f"\nResults saved to {OUTPUT_PATH}")
    print("Run evalueaza_solutie.py with solution_yolo path to evaluate")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run YOLO inference")
    parser.add_argument("--input", type=str, 
                        default=os.path.join(BASE_PATH, "validare", "validare"),
                        help="Input folder with test images")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    
    args = parser.parse_args()
    
    run_inference(args.input, args.model, args.conf)
