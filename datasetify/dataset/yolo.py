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
from .base import BaseDataset

from datasetify import __version__


from datasetify.utils import (
    TQDM,
    LOGGER,
    NUM_THREADS,
    DATASETS_DIR,
    SETTINGS_YAML,
    ROOT,
    yaml_load,
)
from datasetify.utils.fs import scan_txt, check_file

from .augmentation import Compose, Format, LetterBox, yolo_v8_transforms
from .utils import (
    exif_size,
    load_dataset_cache_file,
    save_dataset_cache_file,
    image2label_paths,
)


def check_class_names(names):
    """Check class names. Map imagenet class codes to human-readable names if required. Convert lists to dicts."""
    if isinstance(names, list):  # names is a list
        names = dict(enumerate(names))  # convert to dict
    if isinstance(names, dict):
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
            )
        if isinstance(names[0], str) and names[0].startswith(
            "n0"
        ):  # imagenet class codes, i.e. 'n01440764'
            map = yaml_load(ROOT / "cfg/datasets/ImageNet.yaml")[
                "map"
            ]  # human-readable names
            names = {k: map[v] for k, v in names.items()}
    return names


def check_det_dataset(dataset, autodownload=True):
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

    file = check_file(dataset)

    extract_dir = ""

    data = yaml_load(file, append_filename=True)  # dictionary

    # Checks
    for k in "train", "val":
        if k not in data:
            if k == "val" and "validation" in data:
                LOGGER.info(
                    "WARNING renaming data YAML 'validation' key to 'val' to match YOLO format."
                )
                data["val"] = data.pop(
                    "validation"
                )  # replace 'validation' key with 'val' key
            else:
                pass
    if "names" not in data and "nc" not in data:
        pass
        raise SyntaxError(
            f"{dataset} key missing .\n either 'names' or 'nc' are required in all data YAMLs."
        )
    if "names" in data and "nc" in data and len(data["names"]) != data["nc"]:
        pass
        raise SyntaxError(
            f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."
        )
    if "names" not in data:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]
    else:
        data["nc"] = len(data["names"])

    data["names"] = check_class_names(data["names"])

    # Resolve paths
    path = Path(
        extract_dir or data.get("path") or Path(data.get("yaml_file", "")).parent
    )  # dataset root

    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()
    data["path"] = path  # download scripts
    for k in "train", "val", "test":
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Parse YAML
    train, val, test, s = (data.get(x) for x in ("train", "val", "test", "download"))
    if val:
        val = [
            Path(x).resolve() for x in (val if isinstance(val, list) else [val])
        ]  # val path
        if not all(x.exists() for x in val):
            name = dataset  # dataset name with URL auth stripped
            m = f"\nDataset '{name}' images not found missing path '{[x for x in val if not x.exists()][0]}'"
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote dataset download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_YAML}'"
                raise FileNotFoundError(m)
            t = time.time()
            r = None  # success
            if s.startswith("bash "):  # bash script
                LOGGER.info(f"Running {s} ...")
                r = os.system(s)
            else:  # python script
                exec(s, {"yaml": data})
            dt = f"({round(time.time() - t, 1)}s)"
            s = (
                f"success {dt}, saved to {DATASETS_DIR}"
                if r in (0, None)
                else f"failure {dt}"
            )
            LOGGER.info(f"Dataset download {s}\n")

    return data  # dictionary


class YoloDataset(BaseDataset):
    def __init__(self, *args, data=None, **kwargs):
        self.data = data
        self._coco_annotation_id = 0
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
        """Returns dictionary of labels for YOLO training."""
        self.label_files = image2label_paths(self.im_files)
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

    def _get_coco_image_info(self, img_path, img_id, save_img_dir):
        img_path = Path(img_path)

        new_img_name = f"{img_id:012d}.jpg"
        save_img_path = save_img_dir / new_img_name
        img_src = cv2.imread(str(img_path))
        if img_path.suffix.lower() == ".jpg":
            shutil.copyfile(img_path, save_img_path)
        else:
            cv2.imwrite(str(save_img_path), img_src)

        year = time.strftime("%Y", time.localtime(time.time()))

        height, width = img_src.shape[:2]
        image_info = {
            "date_captured": year,
            "file_name": new_img_name,
            "id": img_id,
            "height": height,
            "width": width,
        }
        return image_info

    def to_coco(self, save=False, save_path=Path("./coco-dataset")):
        """Convert dataset to COCO format"""
        images, annotations = [], []

        year = time.strftime("%Y", time.localtime(time.time()))
        date_created = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time()))

        info = {
            "year": int(year),
            "version": __version__,
            "description": "converted from YOLO by datasetify",
            "date_created": date_created,
        }

        for img_id, img_path in enumerate(TQDM(self.im_files)):
            img_meta = self._get_coco_image_info(img_path, img_id, save_path)
            images.append(img_meta)

            label_path = image2label_paths([img_path])[0]

            annotation = self._get_coco_annotation(
                Path(label_path), img_id, img_meta["height"], img_meta["width"]
            )
            annotations.extend(annotation)

        categories = [
            {"supercategory": cat, "id": i, "name": cat}
            for i, cat in (self.data["names"]).items()
        ]

        json_data = {
            "info": info,
            "images": images,
            "categories": categories,
            "annotations": annotations,
        }

        if save:
            Path(save_path).mkdir(parents=True, exist_ok=True)

            anno_dir = save_path / "annotations"
            Path(anno_dir).mkdir(parents=True, exist_ok=True)

            with open(anno_dir / f"instances_2017.json", "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False)

        return json_data

    def _get_coco_annotation(self, label_path: Path, img_id, height, width):
        def get_box_info(vertex_info, height, width):
            x, y, w, h = [float(i) for i in vertex_info]

            xyxy = yolo2xyxy(np.array([x, y, w, h]))
            x0, y0, x1, y1 = xyxy.ravel()

            box_w = w * width
            box_h = h * height

            segmentation = [[x0, y0, x1, y0, x1, y1, x0, y1]]
            bbox = [x0, y0, box_w, box_h]
            area = box_w * box_h
            return segmentation, bbox, area

        if not label_path.exists():
            annotation = [
                {
                    "segmentation": [],
                    "area": 0,
                    "iscrowd": 0,
                    "image_id": img_id,
                    "bbox": [],
                    "category_id": -1,
                    "id": self._coco_annotation_id,
                }
            ]
            self._coco_annotation_id += 1
            return annotation

        annotation = []
        label_list = scan_txt(str(label_path))
        for i, one_line in enumerate(label_list):
            label_info = one_line.split(" ")
            if len(label_info) < 5:
                LOGGER.warn(f"The {i+1} line of the {label_path} has been corrupted.")
                continue

            category_id, vertex_info = label_info[0], label_info[1:]
            segmentation, bbox, area = get_box_info(vertex_info, height, width)
            annotation.append(
                {
                    "segmentation": segmentation,
                    "area": area,
                    "iscrowd": 0,
                    "image_id": img_id,
                    "bbox": bbox,
                    "category_id": int(category_id) + 1,
                    "id": self._coco_annotation_id,
                }
            )
            self._coco_annotation_id += 1
        return annotation


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
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


def build_yolo_dataset(cfg, img_path, data, mode="train", rect=False, stride=32):
    """Build YOLO Dataset"""

    return YoloDataset(
        img_path=img_path,
        cache=cfg.cache or None,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )
