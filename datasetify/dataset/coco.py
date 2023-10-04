from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np

from PIL import Image
from datasetify.dataset.augmentation import Compose, Format, LetterBox
from datasetify.utils import IMG_FORMATS, LOGGER, NUM_THREADS, TQDM
from datasetify.utils import json_load
from .utils import exif_size, load_dataset_cache_file, save_dataset_cache_file
from .base import BaseDataset
from .augmentation import generic_transforms


class COCODataset(BaseDataset):
    def __init__(self, *args, path, **kwargs):
        self.path = path
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        """Cache dataset labels, check images and read shapes.
        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        json_data = json_load(self.path)

        images = dict([(x["id"], str(Path(self.img_path[0]) / Path(x["file_name"]))) for x in json_data["images"]])
        lbs = dict(
            [
                (x["id"], (x["bbox"], x["image_id"], x["category_id"]))
                for x in json_data["annotations"]
            ]
        )
        categories = dict([(x["id"], x["name"]) for x in json_data["categories"]])

        ims_to_lbs = [images[lb[2]] for lb in lbs.values()]

        x = {"labels": []}

        desc = f"{self.prefix}Scanning {self.path}..."
        total = len(lbs)

        nm, nf, ne, nc, msgs = (
            0,
            0,
            0,
            0,
            [],
        )  # number missing, found, empty, corrupt, messages

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(lbs.values(), ims_to_lbs, repeat(self.prefix), repeat(categories)),
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
                            cls=lb[0:1],
                            bboxes=lb[1:],
                            normalized=True,
                            bbox_format="coco",
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
        cache_path = Path(self.path).parent / "annotations.cache"

        try:
            cache, exists = (
                load_dataset_cache_file(cache_path),
                True,
            )
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False

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

        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = generic_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose(
                [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)]
            )
        transforms.append(
            Format(
                bbox_format="coco",
                normalize=True,
                return_mask=False,
                return_keypoint=False,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms

    def to_yolo(self):
        pass


def verify_image_label(args):
    lb_desc, im_file, prefix, categories = args
    bbox, im_id, cat_id = lb_desc
    nm, nf, ne, nc, msg = 0, 0, 0, 0, ""

    try:
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert cat_id in categories.keys(), f"invalid category {cat_id}"
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        bbox = np.array(bbox, dtype=np.float32)

        lb = np.array([cat_id] + bbox, dtype=np.float32)
        assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
        nf = 1

        return im_file, lb, shape, nm, nf, ne, nc, msg

    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING {im_id}: ignoring corrupt image/label: {e}"
        return [None, None, None, nm, nf, ne, nc, msg]


def build_coco_dataset(cfg, path, img_path, mode="train", rect=False, stride=32):
    """Build YOLO Dataset"""

    return COCODataset(
        img_path=img_path,
        cache=cfg.cache or None,
        classes=cfg.classes,
        path=path,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )
