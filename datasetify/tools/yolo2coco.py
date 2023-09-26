import argparse

from pathlib import Path

from datasetify.yolo.cfg import get_yolo_cfg
from datasetify.dataset.yolo import build_yolo_dataset, check_det_dataset

def yolo2coco(dataset_path, img_path, save_dir):
    cfg = get_yolo_cfg()

    data = check_det_dataset(dataset_path)

    dataset = build_yolo_dataset(cfg, Path(img_path), data)
    dataset.to_coco(save=True, save_path=Path(save_dir))
