1. Libraries required:

numpy>=1.19.0
opencv-python>=4.5.0
scikit-image>=0.18.0
scikit-learn>=0.24.0
tqdm>=4.60.0
matplotlib>=3.3.0
Pillow>=8.0.0

2. How to run:

Script: RunProject.py
Command: python RunProject.py

This command trains the models (if not present), runs the inference on validation images, and saves the results.
The output will be found in 'evaluare/fisiere_solutie/334_Coman_Ioan_Alexandru/task1' and 'evaluare/fisiere_solutie/334_Coman_Ioan_Alexandru/task2'.

Optional arguments:
--input <folder>: Specify input folder with images
--output <folder>: Specify output folder
--mode <train|test|validate|full>: Specify run mode
