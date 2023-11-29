import os
import time
import shutil
import json
from PIL import Image, ImageOps

import cv2
import numpy as np

from itertools import repeat

from pathlib import Path
from multiprocessing.pool import ThreadPool

from datasetify.utils.bboxes import segments2boxes, yolo2xyxy
from datasetify.utils.yaml import yaml_save
from .base import BaseDataset

from datasetify import __version__

from datasetify.utils import (
    IMG_FORMATS,
    TQDM,
    LOGGER,
    NUM_THREADS,
)
from datasetify.utils.fs import make_yolo_dirs, scan_txt, check_file

from .augmentation import Compose, Format, LetterBox, yolo_v8_transforms
from .utils import (
    exif_size,
    load_dataset_cache_file,
    save_dataset_cache_file,
    yolo_autosplit,
    yolo_image2label_paths,
)


from .base import BaseDataset


def check_det_dataset(dataset, labels_txt_path):
    """
    Download, verify, and/or unzip a dataset if not found locally.

    This function checks the availability of a specified dataset, and if not found, it has the option to download and
    unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also
    resolves paths related to the dataset.

    Args:
        dataset (str): Path to the dataset or dataset descriptor (like a YAML file).
        autodownload (bool, optional): Whether to automatically download the dataset if not found. Defaults to True.

    Returns:
        (dict): Parsed dataset information and paths.
    """

    data = {}
    data["names"] = dict(enumerate(scan_txt(labels_txt_path)))
    data["path"] = Path(dataset) / "images"
    data["nc"] = len(data["names"])
    return data


class YOLOv4Dataset(BaseDataset):
    def __init__(self, *args, labels_txt_path, data, **kwargs):
        self.data = data
        self.labels_txt_path = labels_txt_path
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        """Cache dataset labels, check images and read shapes.
        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = (
            0,
            0,
            0,
            0,
            [],
        )  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(len(self.data["names"])),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for (
                im_file,
                lb,
                shape,
                nm_f,
                nf_f,
                ne_f,
                nc_f,
                msg,
            ) in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            normalized=True,
                            bbox_format="yolo",
                        )
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING No labels found in {path}.")
        # x['hash'] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        save_dataset_cache_file(self.prefix, path, x)
        return x

    def get_labels(self):
        print(self.im_files)
        """Returns dictionary of labels for YOLO training."""
        self.label_files = yolo_image2label_paths(self.im_files)
        print(self.label_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = (
                load_dataset_cache_file(cache_path),
                True,
            )  # attempt to load a *.cache file
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop(
            "results"
        )  # found, missing, empty, corrupt, total
        if exists:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results

        # Read cache
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(
                f"WARNING No images found in {cache_path}, training may not work correctly."
            )
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"])) for lb in labels)
        len_cls, len_boxes = (sum(x) for x in zip(*lengths))
        if len_cls == 0:
            LOGGER.warning(
                f"WARNING No labels found in {cache_path}, training may not work correctly."
            )
        return labels

    def to_yolo(self, save_path=Path("./yolo-dataset"), autosplit=(0.9, 0.1, 0.0)):
        """Convert dataset to YOLO format"""
        train_sets = [""]
        if autosplit:
            train_sets = ["train", "val", "test"][: len(autosplit)]

        image_sets, label_sets = yolo_autosplit(
            self.img_path[0], autosplit
        )  # destination image and label paths

        make_yolo_dirs(str(save_path), train_sets)

        for train_set in TQDM(
            train_sets, desc=f"Copying files from {self.img_path[0]} to {save_path}..."
        ):
            for img, label in zip(image_sets[train_set], label_sets[train_set]):
                print(img, label)
                shutil.copy(
                    img["src"],
                    Path(save_path) / img["dist"],
                )
                shutil.copy(
                    label["src"],
                    Path(save_path) / label["dist"],
                )

        meta_data = {
            "path": str(save_path),
            "names": self.data["names"],
        }

        for train_set in train_sets:
            if train_set == "":
                meta_data["train"] = "images/" + train_set
            else:
                meta_data[train_set] = "images/" + train_set

        yaml_save(meta_data, str(save_path / "data.yaml"))

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = yolo_v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose(
                [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)]
            )
        transforms.append(
            Format(
                bbox_format="yolo",
                normalize=True,
                return_mask=False,
                return_keypoint=False,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms


def verify_image_label(args):
    """Verify one image-label pair."""
    im_file, lb_file, prefix, num_cls = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, "", []
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(
                        im_file, "JPEG", subsampling=0, quality=100
                    )
                    msg = (
                        f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"
                    )

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [
                        np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb
                    ]  # (cls, xy1...)
                    lb = np.concatenate(
                        (classes.reshape(-1, 1), segments2boxes(segments)), 1
                    )  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert (
                    lb.shape[1] == 5
                ), f"labels require 5 columns, {lb.shape[1]} columns detected"
                assert (
                    lb[:, 1:] <= 1
                ).all(), f"non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}"
                assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                # All labels
                max_cls = int(lb[:, 0].max())  # max label count
                assert max_cls <= num_cls, (
                    f"Label class {max_cls} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        lb = lb[:, :5]
        return im_file, lb, shape, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, nm, nf, ne, nc, msg]


def build_yolov4_dataset(cfg, img_path, labels_txt_path, data, mode="train"):
    return YOLOv4Dataset(
        img_path=img_path,
        cache=cfg.cache or None,
        classes=cfg.classes,
        labels_txt_path=labels_txt_path,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )
