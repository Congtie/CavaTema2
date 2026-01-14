# YOLO Solution for Scooby-Doo Face Detection (Bonus 25%)

This is a state-of-the-art solution using YOLOv8 for face detection and character recognition.

## Requirements

```bash
pip install ultralytics
```

## Usage

### Step 1: Prepare Dataset
Convert annotations to YOLO format:
```bash
python prepare_dataset.py
```

### Step 2: Train Model
```bash
python train_yolo.py --epochs 100 --model s
```

Options:
- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size (default: 16)
- `--imgsz`: Image size (default: 640)
- `--model`: Model size - n/s/m/l/x (default: s)
  - `n` = nano (fastest, least accurate)
  - `s` = small (good balance)
  - `m` = medium
  - `l` = large
  - `x` = xlarge (slowest, most accurate)

### Step 3: Run Inference
```bash
python run_yolo.py --input ../validare/validare
```

### Step 4: Evaluate
Modify `evalueaza_solutie.py` to use `solution_yolo` path:
```python
solution_path_root = "../fisiere_solutie/solution_yolo/"
```

Then run:
```bash
cd ../evaluare/cod_evaluare
python evalueaza_solutie.py
```

## Why YOLO?

- **State-of-the-art**: One of the best object detection models
- **Fast**: Real-time inference
- **Accurate**: High mAP on detection tasks
- **Easy to use**: Ultralytics provides excellent API
- **Transfer learning**: Pretrained on COCO, fine-tuned on our data

## Expected Improvements

YOLO should significantly outperform HOG+SVM because:
1. Deep learning features vs hand-crafted HOG
2. End-to-end training for detection
3. Better handling of scale/pose variations
4. Strong data augmentation built-in
