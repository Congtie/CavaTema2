"""
Script pentru vizualizarea detecțiilor pe imagini.
Salvează imagini cu bounding box-uri desenate.
"""

import os
import sys
import numpy as np
import cv2

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_path)

from config import VALIDATION_IMAGES, OUTPUT_TASK1, OUTPUT_TASK2, CHARACTERS


def load_results(task1_path, task2_path):
    """Încarcă rezultatele din fișierele .npy"""
    results = {
        'task1': {
            'detections': np.load(os.path.join(task1_path, "detections_all_faces.npy"), allow_pickle=True),
            'scores': np.load(os.path.join(task1_path, "scores_all_faces.npy"), allow_pickle=True),
            'file_names': np.load(os.path.join(task1_path, "file_names_all_faces.npy"), allow_pickle=True),
        },
        'task2': {}
    }
    
    for char in CHARACTERS:
        results['task2'][char] = {
            'detections': np.load(os.path.join(task2_path, f"detections_{char}.npy"), allow_pickle=True),
            'scores': np.load(os.path.join(task2_path, f"scores_{char}.npy"), allow_pickle=True),
            'file_names': np.load(os.path.join(task2_path, f"file_names_{char}.npy"), allow_pickle=True),
        }
    
    return results


def visualize_image(image_name, image_folder, results, output_folder, show_task='task2'):
    """
    Vizualizează detecțiile pentru o imagine.
    
    Args:
        image_name: Numele imaginii (ex: "0001.jpg")
        image_folder: Folderul cu imagini
        results: Dicționarul cu rezultate
        output_folder: Folderul pentru salvare
        show_task: 'task1' sau 'task2'
    """
    img_path = os.path.join(image_folder, image_name)
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"Nu am putut încărca imaginea: {img_path}")
        return
    
    # Culori pentru fiecare personaj (BGR)
    colors = {
        'fred': (0, 165, 255),    # Orange
        'daphne': (128, 0, 128),  # Purple
        'shaggy': (0, 128, 0),    # Green
        'velma': (0, 0, 255),     # Red
        'all_faces': (255, 0, 0), # Blue
    }
    
    if show_task == 'task1':
        # Desenăm toate fețele
        mask = results['task1']['file_names'] == image_name
        detections = results['task1']['detections'][mask]
        scores = results['task1']['scores'][mask]
        
        for det, score in zip(detections, scores):
            xmin, ymin, xmax, ymax = map(int, det)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors['all_faces'], 2)
            label = f"face: {score:.2f}"
            cv2.putText(image, label, (xmin, ymin - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['all_faces'], 1)
    else:
        # Desenăm detecțiile per personaj
        for char in CHARACTERS:
            mask = results['task2'][char]['file_names'] == image_name
            detections = results['task2'][char]['detections'][mask]
            scores = results['task2'][char]['scores'][mask]
            
            for det, score in zip(detections, scores):
                xmin, ymin, xmax, ymax = map(int, det)
                color = colors[char]
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                label = f"{char}: {score:.2f}"
                
                # Background pentru text
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (xmin, ymin - text_height - 5), (xmin + text_width, ymin), color, -1)
                cv2.putText(image, label, (xmin, ymin - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Adaugăm legendă
    y_offset = 20
    for char, color in colors.items():
        if show_task == 'task1' and char != 'all_faces':
            continue
        if show_task == 'task2' and char == 'all_faces':
            continue
        cv2.rectangle(image, (10, y_offset - 12), (25, y_offset), color, -1)
        cv2.putText(image, char, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 20
    
    # Salvăm imaginea
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"viz_{image_name}")
    cv2.imwrite(output_path, image)
    print(f"Salvat: {output_path}")
    
    return image


def main():
    # Încarcă rezultatele
    task1_path = os.path.join(os.path.dirname(__file__), "evaluare", "fisiere_solutie", "solution", "task1")
    task2_path = os.path.join(os.path.dirname(__file__), "evaluare", "fisiere_solutie", "solution", "task2")
    
    print("Încărcare rezultate...")
    results = load_results(task1_path, task2_path)
    
    # Folder pentru vizualizări
    output_folder = os.path.join(os.path.dirname(__file__), "vizualizari")
    
    # Alege câteva imagini reprezentative
    images_to_visualize = ["0005.jpg"]
    
    print(f"\nGenerare vizualizări pentru {len(images_to_visualize)} imagini...")
    
    for img_name in images_to_visualize:
        visualize_image(img_name, VALIDATION_IMAGES, results, output_folder, show_task='task2')
    
    print(f"\nVizualizările au fost salvate în: {output_folder}")


if __name__ == "__main__":
    main()
