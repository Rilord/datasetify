from pathlib import Path
from datasetify.tools import coco2yolo

COCO_DATASET_PATH = Path('./datasets/coco')

def test_coco2yolo():
    coco2yolo(COCO_DATASET_PATH, 'yolo')
