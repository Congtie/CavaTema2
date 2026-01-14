"""
Train YOLOv8 model for Scooby-Doo face detection and recognition.
"""

import os
from ultralytics import YOLO

# Paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_YAML = os.path.join(BASE_PATH, "yolo", "dataset", "dataset.yaml")
OUTPUT_PATH = os.path.join(BASE_PATH, "yolo", "runs")


def train_yolo(epochs=100, imgsz=640, batch=16, model_size='n'):
    """
    Train YOLOv8 model.
    
    Args:
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        model_size: Model size ('n'=nano, 's'=small, 'm'=medium, 'l'=large, 'x'=xlarge)
    """
    # Load pretrained YOLOv8 model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train on our dataset
    results = model.train(
        data=DATASET_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=OUTPUT_PATH,
        name='scooby_doo',
        patience=20,  # Early stopping
        save=True,
        plots=True,
        verbose=True,
        
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )
    
    print(f"\nTraining complete!")
    print(f"Best model saved to: {OUTPUT_PATH}/scooby_doo/weights/best.pt")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Scooby-Doo")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--model", type=str, default='s', 
                        choices=['n', 's', 'm', 'l', 'x'],
                        help="Model size (n=nano, s=small, m=medium, l=large, x=xlarge)")
    
    args = parser.parse_args()
    
    train_yolo(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_size=args.model
    )
