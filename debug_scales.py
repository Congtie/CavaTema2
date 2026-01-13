import cv2
import sys
sys.path.insert(0, 'src')
from config import *

img = cv2.imread('validare/validare/0100.jpg')
h, w = img.shape[:2]
win_h, win_w = WINDOW_SIZE
print(f'Image: {w}x{h}, Window: {win_w}x{win_h}')

current_window = MIN_WINDOW_SIZE
print('\nScales being used:')
while current_window <= MAX_WINDOW_SIZE:
    scale = win_w / current_window
    new_w = int(w * scale)
    new_h = int(h * scale)
    status = ""
    if new_w < win_w or new_h < win_h:
        status = " SKIPPED!"
    print(f'  Window {current_window:3d}px: scale={scale:.3f} -> resized {new_w:4d}x{new_h:4d}{status}')
    current_window = int(current_window * SCALE_FACTOR)

# GT is 167x171
print(f'\nGT face size: 167x171')
print(f'Closest window: 165px or 198px')
