import os
from pathlib import Path
import random
import contextlib
import numpy as np

import cv2

from datasetify.utils import IMG_FORMATS, DATASETS_DIR, LOGGER, TQDM
from PIL import Image, ImageOps

from datasetify.utils.fs import is_dir_writeable


def image2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = (
        f"{os.sep}images{os.sep}",
        f"{os.sep}labels{os.sep}",
    )  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def exif_size(img: Image.Image):
    """Returns exif-corrected PIL size."""
    s = img.size  # (width, height)
    if img.format == "JPEG":  # only support JPEG images
        with contextlib.suppress(Exception):
            exif = img.getexif()
            if exif:
                rotation = exif.get(
                    274, None
                )  # the EXIF key for the orientation tag is 274
                if rotation in [6, 8]:  # rotation 270 or 90
                    s = s[1], s[0]
    return s


def verify_image(args):
    """Verify one image."""
    (im_file, cls), prefix = args
    # Number (found, corrupt), message
    nf, nc, msg = 0, 0, ""
    try:
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
                    msg = f"{prefix}WARNING {im_file}: corrupt JPEG restored and saved"
        nf = 1
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
    return (im_file, cls), nf, nc, msg


def autosplit(
    path=DATASETS_DIR / "coco8/images", weights=(0.9, 0.1, 0.0), annotated_only=False
):
    """
    Automatically split a dataset into train/val/test splits and save the resulting splits into autosplit_*.txt files.

    Args:
        path (Path, optional): Path to images directory. Defaults to DATASETS_DIR / 'coco8/images'.
        weights (list | tuple, optional): Train, validation, and test split fractions. Defaults to (0.9, 0.1, 0.0).
        annotated_only (bool, optional): If True, only images with an associated txt file are used. Defaults to False.

    Example:
        ```python
        from ultralytics.data.utils import autosplit

        autosplit()
        ```
    """

    path = Path(path)  # images dir
    files = sorted(
        x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS
    )  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices(
        [0, 1, 2], weights=weights, k=n
    )  # assign each image to a split

    txt = [
        "autosplit_train.txt",
        "autosplit_val.txt",
        "autosplit_test.txt",
    ]  # 3 txt files
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # remove existing

    LOGGER.info(
        f"Autosplitting images from {path}"
        + ", using *.txt labeled images only" * annotated_only
    )
    for i, img in TQDM(zip(indices, files), total=n):
        if (
            not annotated_only or Path(image2label_paths([str(img)])[0]).exists()
        ):  # check label
            with open(path.parent / txt[i], "a") as f:
                f.write(
                    f"./{img.relative_to(path.parent).as_posix()}" + "\n"
                )  # add image to txt file


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    """
    Args:
        imgsz (tuple): The image size.
        polygons (list[np.ndarray]): [N, M], N is the number of polygons, M is the number of points(Be divided by 2).
        color (int): color
        downsample_ratio (int): downsample ratio
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # NOTE: fillPoly first then resize is trying to keep the same way of loss calculation when mask-ratio=1.
    return cv2.resize(mask, (nw, nh))


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    """
    Args:
        imgsz (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M], N is number of polygons, M is number of points (M % 2 = 0)
        color (int): color
        downsample_ratio (int): downsample ratio
    """
    return np.array(
        [
            polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio)
            for x in polygons
        ]
    )


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(
            imgsz,
            [segments[si].reshape(-1)],
            downsample_ratio=downsample_ratio,
            color=1,
        )
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index

def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc

    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache


def save_dataset_cache_file(prefix, path, x):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        np.save(str(path), x)  # save cache for next time
        path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
        LOGGER.info(f"{prefix}New cache created: {path}")
    else:
        LOGGER.warning(
            f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved."
        )
