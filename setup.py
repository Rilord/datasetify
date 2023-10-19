import re
from pathlib import Path

from setuptools import setup

# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / "README.md").read_text(encoding="utf-8")


def get_version():
    file = PARENT / "datasetify/__init__.py"
    return re.search(
        r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding="utf-8"), re.M
    )[1]


def parse_requirements(file_path: Path):
    """
    Parse a requirements.txt file, ignoring lines that start with '#' and any text after '#'.

    Args:
        file_path (str | Path): Path to the requirements.txt file.

    Returns:
        List[str]: List of parsed requirements.
    """

    requirements = []
    for line in Path(file_path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line.split("#")[0].strip())  # ignore inline comments

    return requirements


setup(
    name="datasetify",  # name of pypi package
    version=get_version(),  # version of pypi package
    python_requires=">=3.8",
    license="Apache License 2.0",
    description=("Tools for managing datasets for image detection"),
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Rilord/datasetify",
    project_urls={"Source": "https://github.com/Rilord/datasetify"},
    author="kodor",
    author_email="devel@kodors.net",
    packages=["datasetify"]
    + [
        str(x)
        for x in Path("datasetify").rglob("*/")
        if x.is_dir() and "__" not in str(x)
    ],
    package_data={
        "": ["*.yaml"],
    },
    include_package_data=True,
    install_requires=parse_requirements(PARENT / "requirements.txt"),
    extras_require={
        "dev": [
            "ipython",
            "check-manifest",
            "pytest",
            "pytest-cov",
            "coverage",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords="machine-learning, deep-learning, vision, ML, DL, AI, YOLO, COCO, Labelme, KITTI",
    entry_points={"console_scripts": ["datasetify = datasetify.cfg:entrypoint"]},
)
