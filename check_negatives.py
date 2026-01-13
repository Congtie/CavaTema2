"""Check if negative samples contain faces"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, 'src')
from data_loader import load_annotations, extract_negative_patches, compute_iou

# Load one character's annotations
char = 'fred'
folder_path = f'antrenare/{char}'
annotations_path = f'antrenare/{char}_annotations.txt'
annotations = load_annotations(annotations_path)

# Get one image with faces
img_name = list(annotations.keys())[0]
img_path = os.path.join(folder_path, img_name)
image = cv2.imread(img_path)
face_boxes = annotations[img_name]

print(f"Image: {img_name}")
print(f"Faces in image: {len(face_boxes)}")

# Extract negative patches with different IoU thresholds
fig, axes = plt.subplots(3, 6, figsize=(15, 8))
fig.suptitle(f'Negative Patches with Different Max IoU Thresholds\nImage: {img_name}', fontsize=14)

max_ious = [0.3, 0.1, 0.05]
for row, max_iou in enumerate(max_ious):
    # Extract negatives with this threshold
    neg_patches = []
    attempts = 0
    max_attempts = 100
    h, w = image.shape[:2]
    
    while len(neg_patches) < 6 and attempts < max_attempts:
        attempts += 1
        
        # Random patch
        patch_size = np.random.randint(64, min(h, w) // 2 + 1)
        xmin = np.random.randint(0, max(1, w - patch_size))
        ymin = np.random.randint(0, max(1, h - patch_size))
        xmax = xmin + patch_size
        ymax = ymin + patch_size
        
        # Check IoU with all faces
        max_face_iou = 0
        is_negative = True
        for face_box in face_boxes:
            iou = compute_iou((xmin, ymin, xmax, ymax), face_box[:4])
            max_face_iou = max(max_face_iou, iou)
            if iou > max_iou:
                is_negative = False
                break
        
        if is_negative:
            patch = image[ymin:ymax, xmin:xmax]
            patch_resized = cv2.resize(patch, (64, 64))
            neg_patches.append((patch_resized, max_face_iou))
    
    # Display patches
    for col in range(6):
        if col < len(neg_patches):
            patch, iou_val = neg_patches[col]
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(patch_rgb)
            axes[row, col].set_title(f'IoU={iou_val:.3f}', fontsize=8)
        axes[row, col].axis('off')
        
        if col == 0:
            axes[row, col].set_ylabel(f'Max IoU\n{max_iou}', fontsize=10, rotation=0, ha='right', va='center')

# Draw the original image with face boxes
fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
img_display = image.copy()
for box in face_boxes:
    xmin, ymin, xmax, ymax = box[:4]
    cv2.rectangle(img_display, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
ax.imshow(img_rgb)
ax.set_title(f'Original Image with {len(face_boxes)} Face(s) Annotated')
ax.axis('off')

plt.tight_layout()
plt.savefig('negative_samples_iou_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: negative_samples_iou_comparison.png")

fig2.savefig('original_image_with_faces.png', dpi=150, bbox_inches='tight')
print("Saved: original_image_with_faces.png")

plt.show()

print("\n=== Conclusion ===")
print(f"With MAX_NEGATIVE_IOU = 0.3: Patches can overlap 30% with faces (BAD!)")
print(f"With MAX_NEGATIVE_IOU = 0.1: Patches can overlap 10% with faces (BETTER)")
print(f"With MAX_NEGATIVE_IOU = 0.05: Patches can overlap 5% with faces (BEST)")
print("\nRecommendation: Set MAX_NEGATIVE_IOU = 0.05 or lower")
