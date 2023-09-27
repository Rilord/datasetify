from pathlib import Path

from datasetify.yolo.cfg import get_yolo_cfg
from datasetify.dataset.yolo import build_yolo_dataset, check_det_dataset


def yolo2coco(dataset_path, save_dir):
    """
    Convert yolo dataset to COCO

    Args:
        dataset_path -- path to data.yaml config
        save_dir -- directory to save COCO dataset
    """
    cfg = get_yolo_cfg()

    data = check_det_dataset(dataset_path)

    sets = filter(lambda x: True if x in ("test", "train", "test") else False, data.keys())

    img_paths = [Path(data["path"]).parent / Path(data[k]) for k in sets]

    dataset = build_yolo_dataset(cfg, img_paths, data)

    dataset.to_coco(save=True, save_path=Path(save_dir))
