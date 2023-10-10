from pathlib import Path

from datasetify.yolo.cfg import get_yolo_cfg
from datasetify.dataset.coco import build_coco_dataset


def coco2yolo(dataset_path, save_dir):
    """
    Convert COCO dataset to YOLO

    Args:
        dataset_path -- path to annotations json
        save_dir -- directory to save COCO dataset
    """
    cfg = get_yolo_cfg()

    img_paths = [Path(dataset_path)]

    dataset = build_coco_dataset(cfg, dataset_path / "annotations" / "instances_2017.json", img_paths)

    dataset.to_yolo(save_path=Path(save_dir), autosplit=(0.8, 0.1, 0.1))
