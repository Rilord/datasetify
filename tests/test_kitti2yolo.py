from pathlib import Path
from datasetify.tools import kitti2yolo

KITTI_DATASET_PATH = Path('./datasets/kitti')

def test_kitti2yolo():
    kitti2yolo(KITTI_DATASET_PATH, 'kitti-yolo')
