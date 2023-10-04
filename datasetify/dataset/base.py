# from ultralytics https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/dataset.py

import os
import glob
import math

from copy import deepcopy
# from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from datasetify.utils import IMG_FORMATS, DEFAULT_CFG


class BaseDataset(Dataset):
    def __init__(
        self,
        img_path,
        augment=True,
        imgsz=640,
        cache=False,
        hyp=DEFAULT_CFG,
        prefix="",
        batch_size=16,
        stride=32,
        rect=False,
        classes=None,
        fraction=1.0,
        single_cls=False,
        pad=0.5
    ):
        super().__init__()
        self.augment = augment
        self.img_path = img_path
        self.imgsz = imgsz
        self.prefix = prefix
        self.cache = cache
        self.batch_size = batch_size
        self.single_cls = single_cls
        self.stride = stride
        self.fraction = fraction
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.ni = len(self.labels)
        self.classes = classes
        self.pad = pad
        self.rect = rect

        if self.rect:
            self.set_rectangle()

        self.buffer = []
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        self.ims, self.im_hw0, self.im_hw = (
            [None] * self.ni,
            [None] * self.ni,
            [None] * self.ni,
        )

        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            print('images in ', img_path)
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    print(p)
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [
                            x.replace("./", parent) if x.startswith("./") else x
                            for x in t
                        ]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(
                x.replace("/", os.sep)
                for x in f
                if x.split(".")[-1].lower() in IMG_FORMATS
            )
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}"
        except Exception as e:
            raise FileNotFoundError(
                f"{self.prefix}Error loading data from {img_path}"
            ) from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]
        return im_files

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def build_transforms(self, hyp=None):
        """Users can custom augmentations here
        like:
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
        """
        raise NotImplementedError

    def update_labels_info(self, label):
        """custom your label format here."""
        return label

    def get_labels(self):
        """Users can custom their own format here.
        Make sure your output is a list with each element like below:
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
        """
        raise NotImplementedError

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f"Image Not Found {f}")
            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (
                        min(math.ceil(w0 * r), self.imgsz),
                        min(math.ceil(h0 * r), self.imgsz),
                    )
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (
                h0 == w0 == self.imgsz
            ):  # resize by stretching image to square imgsz
                im = cv2.resize(
                    im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
                )

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = (
                    im,
                    (h0, w0),
                    im.shape[:2],
                )  # im, hw_original, hw_resized
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop('shape') for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image


    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop('shape', None)  # shape is for rect, remove it
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
        label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                              label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
        if self.rect:
            label['rect_shape'] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)
