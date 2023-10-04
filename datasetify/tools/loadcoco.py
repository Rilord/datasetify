from pathlib import Path
from datasetify.dataset.coco import build_coco_dataset
from datasetify.yolo.cfg import get_yolo_cfg
def loadcoco(dataset_path):
    """Load COCO dataset"""

    cfg = get_yolo_cfg()


    img_paths = [Path(dataset_path).parent.parent]

    dataset = build_coco_dataset(cfg, dataset_path, img_path=img_paths)

    return dataset

