from pathlib import Path
from datasetify.tools import yolo2coco

YOLO_DATASET_PATH = Path('./datasets/yolo/data.yaml')

def test_yolo2coco():
    yolo2coco(YOLO_DATASET_PATH, 'yolo-coco')
