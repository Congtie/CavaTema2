import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_PATH, "antrenare")
TRAIN_ANNOTATIONS = {
    "fred": os.path.join(TRAIN_PATH, "fred_annotations.txt"),
    "daphne": os.path.join(TRAIN_PATH, "daphne_annotations.txt"),
    "shaggy": os.path.join(TRAIN_PATH, "shaggy_annotations.txt"),
    "velma": os.path.join(TRAIN_PATH, "velma_annotations.txt"),
}
TRAIN_IMAGES = {
    "fred": os.path.join(TRAIN_PATH, "fred"),
    "daphne": os.path.join(TRAIN_PATH, "daphne"),
    "shaggy": os.path.join(TRAIN_PATH, "shaggy"),
    "velma": os.path.join(TRAIN_PATH, "velma"),
}

VALIDATION_PATH = os.path.join(BASE_PATH, "validare")
VALIDATION_IMAGES = os.path.join(VALIDATION_PATH, "validare")
VALIDATION_GT_TASK1 = os.path.join(VALIDATION_PATH, "task1_gt_validare.txt")
VALIDATION_GT_TASK2 = {
    "fred": os.path.join(VALIDATION_PATH, "task2_fred_gt_validare.txt"),
    "daphne": os.path.join(VALIDATION_PATH, "task2_daphne_gt_validare.txt"),
    "shaggy": os.path.join(VALIDATION_PATH, "task2_shaggy_gt_validare.txt"),
    "velma": os.path.join(VALIDATION_PATH, "task2_velma_gt_validare.txt"),
}

TEST_PATH = os.path.join(BASE_PATH, "testare")

OUTPUT_PATH = os.path.join(BASE_PATH, "evaluare", "fisiere_solutie", "334_Coman_Ioan_Alexandru")
OUTPUT_TASK1 = os.path.join(OUTPUT_PATH, "task1")
OUTPUT_TASK2 = os.path.join(OUTPUT_PATH, "task2")

MODELS_PATH = os.path.join(BASE_PATH, "models")

WINDOW_SIZE = (64, 64)

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = 'L2-Hys'

LBP_RADIUS = 1
LBP_N_POINTS = 8
LBP_METHOD = 'uniform'
LBP_HIST_BINS = 10

MIN_WINDOW_SIZE = 32

MAX_WINDOW_SIZE = 300

SCALE_FACTOR = 1.05

STEP_SIZE = 8

RANDOM_SEED = 1337

SVM_C = 1.0

NMS_THRESHOLD = 0.2

NUM_NEGATIVE_SAMPLES_PER_IMAGE = 10

MIN_POSITIVE_IOU = 0.3

MAX_NEGATIVE_IOU = 0.05

DETECTION_THRESHOLD = 3.0

CHAR_DETECTION_THRESHOLDS = {
    "fred": 3.0,
    "daphne": 3.5,
    "shaggy": 2.5,
    "velma": 3.5,
}

CHARACTERS = ["fred", "daphne", "shaggy", "velma"]

CHAR_TO_LABEL = {
    "fred": 0,
    "daphne": 1,
    "shaggy": 2,
    "velma": 3,
    "unknown": 4,
    "negative": 5,
}

LABEL_TO_CHAR = {v: k for k, v in CHAR_TO_LABEL.items()}
