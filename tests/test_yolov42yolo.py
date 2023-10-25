from pathlib import Path
from datasetify.tools import yolov42yolo

YOLOV4_DATASET_PATH = Path('./datasets/yolov4')

def test_yolov42yolo():
    yolov42yolo(YOLOV4_DATASET_PATH, 'yolov4-yolo')
