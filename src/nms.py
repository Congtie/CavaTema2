"""
Non-Maximum Suppression (NMS) module.
Removes overlapping detections, keeping only the best ones.
"""

import numpy as np
from typing import List, Tuple

from config import NMS_THRESHOLD


def compute_iou(box1: Tuple, box2: Tuple) -> float:
    """
    Compute Intersection over Union between two boxes.
    
    Args:
        box1: (xmin, ymin, xmax, ymax, [score])
        box2: (xmin, ymin, xmax, ymax, [score])
        
    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def non_maximum_suppression(
    detections: List[Tuple[int, int, int, int, float]],
    iou_threshold: float = NMS_THRESHOLD
) -> List[Tuple[int, int, int, int, float]]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    
    Args:
        detections: List of (xmin, ymin, xmax, ymax, score) tuples
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Filtered list of detections
    """
    if len(detections) == 0:
        return []
    
    # Convert to numpy array for easier manipulation
    boxes = np.array([(d[0], d[1], d[2], d[3]) for d in detections])
    scores = np.array([d[4] for d in detections])
    
    # Get indices sorted by score (descending)
    indices = np.argsort(scores)[::-1]
    
    keep = []
    
    while len(indices) > 0:
        # Take the detection with highest score
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Compute IoU with all remaining detections
        remaining = indices[1:]
        ious = np.array([compute_iou(boxes[current], boxes[i]) for i in remaining])
        
        # Keep only detections with IoU below threshold
        mask = ious < iou_threshold
        indices = remaining[mask]
    
    # Return filtered detections
    return [detections[i] for i in keep]


def non_maximum_suppression_fast(
    detections: List[Tuple[int, int, int, int, float]],
    iou_threshold: float = NMS_THRESHOLD
) -> List[Tuple[int, int, int, int, float]]:
    """
    Faster vectorized implementation of Non-Maximum Suppression.
    
    Args:
        detections: List of (xmin, ymin, xmax, ymax, score) tuples
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Filtered list of detections
    """
    if len(detections) == 0:
        return []
    
    # Convert to numpy arrays
    boxes = np.array([(d[0], d[1], d[2], d[3]) for d in detections], dtype=np.float32)
    scores = np.array([d[4] for d in detections])
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Compute areas
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by score (descending)
    order = scores.argsort()[::-1]
    
    keep = []
    
    while order.size > 0:
        # Take the first (highest scoring) detection
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        # Compute intersection with all remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        
        intersection = w * h
        
        # Compute IoU
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        # Keep boxes with IoU below threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return [detections[i] for i in keep]


def apply_nms_to_results(
    results: dict,
    iou_threshold: float = NMS_THRESHOLD,
    max_detections_per_char: int = 1,
    max_total_faces: int = 10,
    min_box_size: int = 60
) -> dict:
    """
    Apply NMS to detection results (for both Task 1 and Task 2).
    
    Args:
        results: Dictionary from detect_and_classify function
        iou_threshold: IoU threshold for suppression
        max_detections_per_char: Maximum detections to keep per character
        max_total_faces: Maximum total face detections per image
        min_box_size: Minimum width/height for a detection to be valid
        
    Returns:
        Results with NMS applied
    """
    # Apply NMS to all faces and limit total detections
    if 'all_faces' in results:
        # Filter by minimum size
        results['all_faces'] = [d for d in results['all_faces'] 
                                if (d[2] - d[0]) >= min_box_size and (d[3] - d[1]) >= min_box_size]
        nms_faces = non_maximum_suppression_fast(
            results['all_faces'], iou_threshold
        )
        # Sort by score and keep only top max_total_faces
        nms_faces = sorted(nms_faces, key=lambda x: x[4], reverse=True)[:max_total_faces]
        results['all_faces'] = nms_faces
    
    # Apply PER-CHARACTER NMS (no global NMS to allow multiple characters at same location)
    if 'characters' in results:
        for char_name in results['characters']:
            # Filter by minimum size first
            char_dets = [d for d in results['characters'][char_name]
                        if (d[2] - d[0]) >= min_box_size and (d[3] - d[1]) >= min_box_size]
            if len(char_dets) > 0:
                nms_dets = non_maximum_suppression_fast(char_dets, iou_threshold)
                # Sort by COMBINED score (SVM score * box size) to favor larger boxes
                # Larger boxes are more likely to be full faces vs. face parts
                def combined_score(det):
                    box_width = det[2] - det[0]
                    box_height = det[3] - det[1]
                    box_area = box_width * box_height
                    # Normalize by typical face size (e.g., 100x100)
                    area_factor = min(box_area / (100 * 100), 2.0)  # Cap at 2x boost
                    return det[4] * area_factor
                nms_dets = sorted(nms_dets, key=combined_score, reverse=True)[:max_detections_per_char]
                results['characters'][char_name] = nms_dets
            else:
                results['characters'][char_name] = []
    
    # GLOBAL NMS: Remove smaller boxes that are contained within larger boxes from different characters
    # This helps eliminate false positives where a small detection is inside a larger correct detection
    if 'characters' in results:
        all_char_dets = []  # List of (char_name, detection, combined_score, area)
        for char_name in results['characters']:
            for det in results['characters'][char_name]:
                box_area = (det[2] - det[0]) * (det[3] - det[1])
                area_factor = min(box_area / (100 * 100), 2.0)
                comb_score = det[4] * area_factor
                all_char_dets.append((char_name, det, comb_score, box_area))
        
        # Sort by combined score (descending)
        all_char_dets.sort(key=lambda x: x[2], reverse=True)
        
        # Keep track of which detections to remove
        to_remove = set()
        
        for i, (char1, det1, score1, area1) in enumerate(all_char_dets):
            if i in to_remove:
                continue
            for j, (char2, det2, score2, area2) in enumerate(all_char_dets):
                if i == j or j in to_remove:
                    continue
                # Check if det2 is CONTAINED within det1 (smaller box inside larger)
                # A box is contained if its center is inside the larger box
                cx2 = (det2[0] + det2[2]) / 2
                cy2 = (det2[1] + det2[3]) / 2
                if det1[0] <= cx2 <= det1[2] and det1[1] <= cy2 <= det1[3]:
                    # det2's center is inside det1
                    if area2 < area1 * 0.5:  # det2 is significantly smaller
                        to_remove.add(j)
        
        # Rebuild character results without removed detections
        kept_dets = [x for i, x in enumerate(all_char_dets) if i not in to_remove]
        for char_name in results['characters']:
            results['characters'][char_name] = [det for (cn, det, _, _) in kept_dets if cn == char_name]
    
    return results


def soft_nms(
    detections: List[Tuple[int, int, int, int, float]],
    iou_threshold: float = NMS_THRESHOLD,
    sigma: float = 0.5,
    score_threshold: float = 0.001
) -> List[Tuple[int, int, int, int, float]]:
    """
    Soft Non-Maximum Suppression - reduces scores instead of completely removing overlapping boxes.
    
    Args:
        detections: List of (xmin, ymin, xmax, ymax, score) tuples
        iou_threshold: IoU threshold for score reduction
        sigma: Gaussian decay parameter
        score_threshold: Minimum score to keep detection
        
    Returns:
        Filtered list of detections with adjusted scores
    """
    if len(detections) == 0:
        return []
    
    # Convert to numpy arrays
    boxes = np.array([(d[0], d[1], d[2], d[3]) for d in detections], dtype=np.float32)
    scores = np.array([d[4] for d in detections], dtype=np.float32)
    
    N = len(detections)
    
    for i in range(N):
        # Find the detection with maximum score
        max_idx = np.argmax(scores[i:]) + i
        
        # Swap
        boxes[[i, max_idx]] = boxes[[max_idx, i]]
        scores[[i, max_idx]] = scores[[max_idx, i]]
        
        # Compute IoU with remaining boxes
        for j in range(i + 1, N):
            iou = compute_iou(boxes[i], boxes[j])
            
            if iou > iou_threshold:
                # Gaussian decay
                scores[j] *= np.exp(-(iou ** 2) / sigma)
    
    # Filter by score threshold
    keep_mask = scores > score_threshold
    
    result = []
    for i in range(N):
        if keep_mask[i]:
            result.append((
                int(boxes[i, 0]),
                int(boxes[i, 1]),
                int(boxes[i, 2]),
                int(boxes[i, 3]),
                float(scores[i])
            ))
    
    return result


if __name__ == "__main__":
    # Test NMS
    test_detections = [
        (10, 10, 50, 50, 0.9),
        (12, 12, 52, 52, 0.85),
        (100, 100, 150, 150, 0.7),
        (15, 15, 55, 55, 0.6),
        (105, 105, 155, 155, 0.5),
    ]
    
    print("Before NMS:")
    for det in test_detections:
        print(f"  {det}")
    
    filtered = non_maximum_suppression(test_detections, iou_threshold=0.3)
    print("\nAfter NMS (threshold=0.3):")
    for det in filtered:
        print(f"  {det}")
    
    filtered_fast = non_maximum_suppression_fast(test_detections, iou_threshold=0.3)
    print("\nAfter fast NMS (threshold=0.3):")
    for det in filtered_fast:
        print(f"  {det}")
