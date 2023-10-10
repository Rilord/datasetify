from pathlib import Path
from datasetify.tools import loadcoco

COCO_DATASET_PATH = Path('datasets/coco/annotations/instances_2017.json')

def test_loadcoco():
    dataset = loadcoco(COCO_DATASET_PATH)

    assert dataset is not None
