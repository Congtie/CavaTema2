import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, 'src')
from data_loader import load_annotations, extract_patch, extract_negative_patches

# Load character data
characters = ['fred', 'daphne', 'shaggy', 'velma']
character_data = {}

print("Loading training data...")

# First load ALL annotations for all characters
all_annotations = {}
for char in characters:
    annotations_path = f'antrenare/{char}_annotations.txt'
    all_annotations[char] = load_annotations(annotations_path)

for char in characters:
    folder_path = f'antrenare/{char}'
    annotations = all_annotations[char]
    
    # Extract positive patches with padding
    patches = []
    for img_name, face_boxes in annotations.items():
        img_path = os.path.join(folder_path, img_name)
        if not os.path.exists(img_path):
            continue
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        for box in face_boxes:
            patch = extract_patch(image, box, target_size=(64, 64), padding=0.2)
            if patch is not None:
                patches.append(patch)
    
    character_data[char] = patches
    print(f"{char}: {len(patches)} positive samples")

# Load negative samples with ALL annotations to avoid faces
# Note: extract_negative_patches now uses padding=0.0 internally for negatives
negative_patches = []
for char in characters:
    folder_path = f'antrenare/{char}'
    annotations = all_annotations[char]
    
    neg_patches = extract_negative_patches(
        folder_path, annotations, num_per_image=5, target_size=(64, 64),
        all_annotations=all_annotations  # Pass ALL annotations!
    )
    negative_patches.extend(neg_patches)

print(f"negative: {len(negative_patches)} negative samples")

# Visualize positives for each character
fig, axes = plt.subplots(len(characters), 6, figsize=(15, 10))
fig.suptitle('Positive Training Samples (with 20% padding)', fontsize=16)

for i, char in enumerate(characters):
    patches = character_data[char]
    # Sample 6 random patches
    indices = np.random.choice(len(patches), min(6, len(patches)), replace=False)
    
    for j, idx in enumerate(indices):
        patch = patches[idx]
        # Convert BGR to RGB for display
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        axes[i, j].imshow(patch_rgb)
        axes[i, j].axis('off')
        if j == 0:
            axes[i, j].set_ylabel(char.upper(), fontsize=12, rotation=0, ha='right', va='center')

plt.tight_layout()
plt.savefig('training_positives.png', dpi=150, bbox_inches='tight')
print("\nSaved: training_positives.png")
plt.show()

# Visualize negative samples - 30 examples (5 rows x 6 columns)
fig, axes = plt.subplots(5, 6, figsize=(15, 12))
fig.suptitle('Negative Training Samples (Non-Faces) - Should NOT contain any faces!', fontsize=16)

# Sample 30 random negative patches
indices = np.random.choice(len(negative_patches), min(30, len(negative_patches)), replace=False)

for i in range(5):
    for j in range(6):
        idx_pos = i * 6 + j
        if idx_pos < len(indices):
            patch = negative_patches[indices[idx_pos]]
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            axes[i, j].imshow(patch_rgb)
        axes[i, j].axis('off')

plt.tight_layout()
plt.savefig('training_negatives.png', dpi=150, bbox_inches='tight')
print("Saved: training_negatives.png")
plt.show()

# Show distribution
print("\n=== Training Data Distribution ===")
total_positives = sum(len(character_data[char]) for char in characters)
print(f"Total positive samples: {total_positives}")
for char in characters:
    count = len(character_data[char])
    percentage = (count / total_positives) * 100
    print(f"  {char:8s}: {count:4d} ({percentage:5.1f}%)")
print(f"Total negative samples: {len(negative_patches)}")
print(f"Positive/Negative ratio: {total_positives / len(negative_patches):.2f}")
