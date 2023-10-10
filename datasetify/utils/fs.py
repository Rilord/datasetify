import glob
import os
import shutil
from pathlib import Path


img_formats = [
    "bmp",
    "jpg",
    "jpeg",
    "png",
    "tif",
    "tiff",
    "dng",
]  # acceptable image suffixes


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
DEFAULT_CFG_PATH = ROOT / "yolo/cfg/default.yaml"


def scan_txt(filename):
    with open(str(filename), "r", encoding="utf-8") as f:
        data = list(map(lambda x: x.rstrip("\n"), f))
    return data


def extract_basename(filename):
    '''Extract basename from filename'''
    return os.path.splitext(filename)[0]


def make_yolo_dirs(dir="new_dir/", train_sets=("train", "val", "test")):
    '''Create folders'''
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)  # delete dir
    for p in dir, dir / "labels", dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # make dir
    labels_paths = [dir / "labels" / ts for ts in train_sets]
    images_paths = [dir / "images" / ts for ts in train_sets]
    for p in labels_paths:
        p.mkdir(parents=True, exist_ok=True)
    for p in images_paths:
        p.mkdir(parents=True, exist_ok=True)
    return dir

def image_folder2file(folder="images/"):  # from utils import *; image_folder2file()
    '''write a txt file listing all imaged in folder'''
    s = glob.glob(f"{folder}*.*")
    with open(f"{folder[:-1]}.txt", "w") as file:
        for l in s:
            file.write(l + "\n")  # write image list


def file_size(path):
    '''Return file/dir size (MB).'''
    if isinstance(path, (str, Path)):
        mb = 1 << 20  # bytes to MiB (1024 ** 2)
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / mb
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / mb
    return 0.0

def is_dir_writeable(dir_path) -> bool:
    '''
    Check if a directory is writeable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writeable, False otherwise.
    '''
    return os.access(str(dir_path), os.W_OK)

def check_file(file, hard=True):
    file = str(file).strip()

    if not file or ('://' not in file and Path(file).exists()):  # exists ('://' check required in Windows Python<3.10)
        return file

    files = glob.glob(str(ROOT / 'cfg' / '**' / file), recursive=True)  # find file
    if not files and hard:
        raise FileNotFoundError(f"'{file}' does not exist")
    elif len(files) > 1 and hard:
        raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
    return files[0] if len(files) else []  # return file
