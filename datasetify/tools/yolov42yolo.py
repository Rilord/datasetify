from pathlib import Path
from datasetify.dataset.yolov4 import build_yolov4_dataset
from datasetify.dataset.utils import try_find_labels_txt
from datasetify.dataset.kitti import check_det_dataset

from datasetify.yolo.cfg import get_yolo_cfg


def yolov42yolo(dataset_path, save_dir, autosplit=(0.8, 0.1, 0.1)):
    """
    Convert KITTI dataset to YOLO

    Args:
        dataset_path -- path to dataset root containing
        save_dir -- directory to save YOLO dataset
    """
    cfg = get_yolo_cfg()



    labels_txt_path = try_find_labels_txt(dataset_path)

    if not labels_txt_path:
        raise FileNotFoundError(
            f"no files with labels description found in {dataset_path}"
        )


    data = check_det_dataset(dataset_path, labels_txt_path)


    img_paths = [data["path"]]

    dataset = build_yolov4_dataset(cfg, img_paths, labels_txt_path, data)

    dataset.to_yolo(save_path=Path(save_dir), autosplit=(0.8, 0.1, 0.1))
