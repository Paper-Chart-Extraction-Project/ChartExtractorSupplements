"""Tiles a yolo dataset."""

# Built-in Imports
from argparse import ArgumentParser
from glob import glob
import os
from pathlib import Path
from PIL import Image
import re
import shutil
from typing import Dict, List, Optional, Tuple, Union

# Internal Imports
from ChartExtractor.utilities.annotations import BoundingBox, Keypoint
from ChartExtractor.utilities.tiling import (
    tile_image,
    tile_annotations,
)

# External Imports
import numpy as np
from tqdm import tqdm


parser: ArgumentParser = ArgumentParser()
parser.add_argument(
    "input_dataset_path",
    help="The filepath to the input YOLO dataset.",
    type=str,
)
parser.add_argument(
    "output_dataset_path",
    help="The filepath to the output tiled YOLO dataset.",
    type=str,
)
parser.add_argument(
    "--horizontal_overlap_ratio",
    help="The amount of overlap the tiles will have left and right.",
    type=float,
)
parser.add_argument(
    "--vertical_overlap_ratio",
    help="The amount of overlap the tiles will have up and down.",
    type=float,
)
tile_size_help_msg: str = (
    "The size of each tile relative to the size of the full image."
)
tile_size_help_msg += (
    "\nIs based off the size of either the width or the height, whichever"
)
tile_size_help_msg += "is smaller."
parser.add_argument("--tile_size_ratio", help=tile_size_help_msg, type=float)


def read_horizontal_overlap_ratio(parser: ArgumentParser) -> float:
    """Reads the 'horizontal_overlap_ratio' parameter.

    Args:
        parser (ArgumentParser):
            The ArgumentParser object that contains the command line arguments.

    Returns:
        A float between 0 and 1 indicating the amount that two adjacent tiles will overlap
        left and right.
    """
    if not parser.horizontal_overlap_ratio:
        return 0.5
    if 0 < parser.horizontal_overlap_ratio > 1:
        err_message: str = f"{parser.horizontal_overlap_ratio} is an invalid value for"
        err_message += "--horizontal_overlap_ratio. Must be between 0 and 1."
        raise ValueError(err_message)
    return parser.horizontal_overlap_ratio


def read_vertical_overlap_ratio(parser: ArgumentParser) -> float:
    """Reads the 'vertical_overlap_ratio' parameter.

    Args:
        parser (ArgumentParser):
            The ArgumentParser object that contains the command line arguments.

    Returns:
        A float between 0 and 1 indicating the amount that two adjacent tiles will overlap
        up and down.
    """
    if not parser.vertical_overlap_ratio:
        return 0.5
    if 0 < parser.vertical_overlap_ratio > 1:
        err_message: str = f"{parser.vertical_overlap_ratio} is an invalid value for"
        err_message += "--vertical_overlap_ratio. Must be between 0 and 1."
        raise ValueError(err_message)
    return parser.vertical_overlap_ratio


def read_tile_size_ratio(parser: ArgumentParser) -> float:
    """Reads the 'tile_size_ratio' parameter.

    Args:
        parser (ArgumentParser):
            The ArgumentParser object that contains the command line arguments.

    Returns:
        A float between 0 and 1 indicating the size of a tile relative to the full image.
    """
    if not parser.tile_size_ratio:
        return 0.5
    if 0 < parser.tile_size_ratio > 1:
        err_message: str = f"{parser.tile_size_ratio} is an invalid value for"
        err_message += "--tile_size_ratio. Must be between 0 and 1."
        raise ValueError(err_message)
    return parser.tile_size_ratio


def read_input_dataset_path(parser: ArgumentParser) -> Path:
    """Reads and validates the input_dataset_path argument.

    Args:
        parser (ArgumentParser):
            The ArgumentParser object that contains the command line arguments.

    Raises:
        FileNotFoundError:
            If the input dataset does not exist.

    Returns:
        A path to the input dataset.
    """
    input_path: Path = Path(parser.input_dataset_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"FileNotFoundError: No such file or directory at {str(input_path.resolve())}."
        )
    elif not Path(parser.input_dataset_path).is_dir():
        raise Exception(
            f"Path exists but is not a directory: {str(input_path.resolve())}."
        )
    return input_path.resolve()


def find_splits(input_dataset_path: Path) -> List[str]:
    """Finds the splits that the dataset uses.

    Args:
        input_dataaset_path (Path):
            The path to the YOLO dataset.

    Returns:
        A list of the splits that the dataset uses.
    """
    potential_splits: List[str] = next(os.walk(str(input_dataset_path / "labels")))
    splits: List[str] = filter(
        lambda s: Path(input_dataset_path / "labels" / s).is_dir(), potential_splits
    )
    return list(splits)


def validate_yolo_dataset(input_dataset_path: Path):
    """Ensures that the yolo dataset is setup correctly.

    Args:
        input_dataaset_path (Path):
            The path to the potential yolo dataset.

    Raises:
        Exception:
            If the YOLO dataset is not valid.
    """
    images_path: Path = input_dataset_path / "images"
    labels_path: Path = input_dataset_path / "labels"

    if not images_path.exists() or not images_path.is_dir():
        raise Exception(
            f"Path to images does not exist, or is not a directory: {str(images_path)}."
        )
    if not labels_path.exists() or not labels_path.is_dir():
        raise Exception(
            f"Path to labels does not exist, or is not a directory: {str(labels_path)}."
        )

    splits: List[str] = find_splits(input_dataset_path)
    for split in splits:
        if not (labels_path / split).is_dir():
            continue
        if not (images_path / split).exists():
            raise Exception(f"Split {split} does not exist in the images folder.")
        for label_name in glob(str(labels_path / split / "*.txt")):
            image_for_label: Path = (images_path / split / label_name).resolve()
            if not image_for_label.exists():
                raise Exception(f"No image for label {image_for_label}.")


def create_output_dataset_directories(
    output_dataset_path: Path,
    splits: List[str],
):
    """Generates the folder structure for the output dataset if it doesn't already exist.

    Args:
        output_dataset_path (Path):
            The path to the output dataset.
        splits (List[str]):
            The list of splits in the dataset.

    Raises:
        Any exception related to folder creation. Typically related to
        filesystem permissions.
    """
    if not output_dataset_path.exists():
        os.mkdir(str(output_dataset_path.resolve()))
    if not (output_dataset_path / "images").exists():
        os.mkdir(str(output_dataset_path / "images"))
    if not (output_dataset_path / "labels").exists():
        os.mkdir(str(output_dataset_path / "labels"))

    for split in splits:
        if not (output_dataset_path / "images" / split).exists():
            os.mkdir(str(output_dataset_path / "images" / split))
        if not (output_dataset_path / "labels" / split).exists():
            os.mkdir(str(output_dataset_path / "labels" / split))


def tile_images(
    input_dataset_path: Path,
    output_dataset_path: Path,
    splits: List[str],
    tile_size: int,
    horizontal_overlap_ratio: float = 0.5,
    vertical_overlap_ratio: float = 0.5,
):
    """Tiles images and saves them to the new dataset.

    Args:
        input_data_path (Path):
            The path to the YOLO dataset to tile.
        output_dataset_path (Path):
            The path to the output dataset.
        splits (List[str]):
            A list of the names of the splits that the dataset uses.
        tile_size (int):
            The size of the tile to use.
        horizontal_overlap_ratio (float):
            The proportion of overlap that two neighboring tiles should have left and right.
        vertical_overlap_ratio (float):
            The proportion of overlap that two neighboring tiles should have up and down.
    """

    def try_open_image(im_path: str) -> Optional[Image]:
        """Trys to open an image, returns None if it cannot. Treat like a Rust result.

        Args:
            im_path (str):
                A path to a file. Possibly an image, but could be anything.

        Returns:
            An optional of a PIL Image.
        """
        try:
            return Image.open(im_path)
        except:
            return None

    for split in splits:
        image_paths: List[str] = glob(str(input_dataset_path / split / "*"))
        for im_path in image_paths:
            image: Optional[Image] = try_open_image(im_path)
            if image is None:
                continue
            image_tiles: List[List[Image]] = tile_image(
                image,
                tile_size,
                tile_size,
                horizontal_overlap_ratio,
                vertical_overlap_ratio,
            )
            for row_ix, row in enumerate(image_tiles):
                for col_ix, tile in enumerate(row):
                    tile.save(
                        output_dataset_path
                        / "images"
                        / split
                        / f"{row_ix}_{col_ix}_{Path(im_path).stem}.jpg"
                    )


def tile_annotations(
    input_dataset_path: Path,
    output_dataset_path: Path,
    splits: List[str],
    tile_size: int,
    horizontal_overlap_ratio: float = 0.5,
    vertical_overlap_ratio: float = 0.5,
):
    """Tiles annotations and saves them to the new dataset.

    Args:
        input_data_path (Path):
            The path to the YOLO dataset to tile.
        output_dataset_path (Path):
            The path to the output dataset.
        splits (List[str]):
            A list of the names of the splits that the dataset uses.
        tile_size (int):
            The size of the tile to use.
        horizontal_overlap_ratio (float):
            The proportion of overlap that two neighboring tiles should have left and right.
        vertical_overlap_ratio (float):
            The proportion of overlap that two neighboring tiles should have up and down.
    """

    def try_open_annotation(
        ann_path: str,
    ) -> Optional[List[Union[BoundingBox, Keypoint]]]:
        """Trys to open an image, returns None if it cannot. Treat like a Rust result.

        Args:
            im_path (str):
                A path to a file. Possibly an image, but could be anything.

        Returns:
            An optional of a PIL Image.
        """
        try:
        except:
    



horizontal_overlap_ratio: float = read_horizontal_overlap_ratio(parser)
vertical_overlap_ratio: float = read_vertical_overlap_ratio(parser)
tile_size_ratio: float = read_tile_size_ratio(parser)
input_dataset_path: Path = read_input_dataset_path(parser)
splits: List[str] = find_splits(input_dataset_path)
output_dataset_path: Path = Path(parser.output_dataset_path)
validate_yolo_dataset(input_dataset_path)
create_output_dataset_directories(output_dataset_path, splits)
tile_images(
    input_dataset_path,
    output_dataset_path,
    splits,
    tile_size_ratio,
    horizontal_overlap_ratio,
    vertical_overlap_ratio,
)

WIDTH, HEIGHT = images[list(images.keys())[0]].size
ID_TO_CATEGORY: Dict[int, str] = {0: "systolic", 1: "diastolic", 2: "heart rate"}
CATEGORY_TO_ID: Dict[str, str] = {v: k for (k, v) in ID_TO_CATEGORY.items()}
CATEGORY_TO_ID = {"systolic": 0, "diastolic": 1, "heart rate": 2}

labels: Dict[str, List[Keypoint]] = {
    Path(lab_name).name: [
        Keypoint.from_yolo(
            line.strip(),
            images[f"{Path(lab_name).stem}.JPG"].size[0],
            images[f"{Path(lab_name).stem}.JPG"].size[1],
            ID_TO_CATEGORY,
        )
        for line in open(lab_name, "r").readlines()
    ]
    for lab_name in glob(str(path_to_data / "bp_yolo_keypoint_labels") + "/*.txt")
}

HORZ_OVERLAP_RATIO, VERT_OVERLAP_RATIO = 0.5, 0.5

tile_ann_dict: Dict[str, Tuple[Image.Image, List[List[Keypoint]]]] = dict()
IMG_DEST: Path = path_to_data / "tiled_bp_yolo_images"
LAB_DEST: Dict[str, Path] = path_to_data / "tiled_bp_yolo_keypoints"

for key in tqdm(images.keys()):
    image: Image.Image = images[key]
    im_name: str = Path(key).name
    im_width, im_height = image.size
    im_labels: List[Keypoint] = labels.get(key.replace(".JPG", ".txt"))
    if im_labels == None:
        continue
    slice_size = min([int(im_width / 5), int(im_height / 5)])
    image_tiles: List[List[Image.Image]] = tile_image(
        image,
        slice_size,  # SLICE_WIDTH,
        slice_size,
        HORZ_OVERLAP_RATIO,
        VERT_OVERLAP_RATIO,
    )
    label_tiles: List[List[Keypoint]] = tile_annotations(
        im_labels,
        im_width,
        im_height,
        slice_size,
        slice_size,
        HORZ_OVERLAP_RATIO,
        VERT_OVERLAP_RATIO,
    )

    for row_ix, row in enumerate(image_tiles):
        for col_ix, image in enumerate(row):
            image.save(str(IMG_DEST) + f"/{row_ix}_{col_ix}_" + im_name)

    for row_ix, row in enumerate(label_tiles):
        for col_ix, tile_labels in enumerate(row):
            yolo_str: str = "\n".join(
                [
                    "0"
                    + l.to_yolo(slice_size, slice_size, CATEGORY_TO_ID, 10, True)[1:]
                    for l in tile_labels
                    if l.category == "heart rate"
                ]
            )
            if yolo_str != "":
                with open(
                    str(LAB_DEST)
                    + f"/{row_ix}_{col_ix}_"
                    + im_name.replace(".JPG", ".txt"),
                    "w",
                ) as f:
                    f.write(yolo_str)


def get_chart_name_from_path(path: str) -> str:
    try:
        return re.findall(r"RC_[0-9][0-9][0-9][0-9]", path)[0]
    except:
        return re.findall(r"RC_IO_[0-9][0-9][0-9][0-9]", path)[0]


val_sheets = [
    "RC_0029",
    "RC_0030",
    "RC_0031",
    "RC_0032",
    "RC_0033",
]

test_sheets = [
    "RC_0034",
    "RC_0035",
    "RC_0036",
    "RC_0037",
    "RC_0038",
    "RC_0039",
    "RC_0043",
    "RC_0044",
    "RC_0045",
]

unlabeled_sheets = []

new_dest_stem = (
    path_to_data / "yolo_datasets" / "new_bp_and_hr_one_vs_rest_hr" / "images"
)
for im_path in tqdm(glob(str(IMG_DEST / "*.JPG"))):
    if get_chart_name_from_path(im_path) in unlabeled_sheets:
        continue
    if get_chart_name_from_path(im_path) in test_sheets:
        dest_branch = "test"
    elif get_chart_name_from_path(im_path) in val_sheets:
        dest_branch = "val"
    else:
        dest_branch = "train"
    new_dest = new_dest_stem / dest_branch

    shutil.copy(im_path, new_dest)

new_dest_stem = (
    path_to_data / "yolo_datasets" / "new_bp_and_hr_one_vs_rest_hr" / "labels"
)
for lab_path in tqdm(glob(str(LAB_DEST / "*.txt"))):
    if get_chart_name_from_path(lab_path) in unlabeled_sheets:
        continue

    if get_chart_name_from_path(lab_path) in unlabeled_sheets:
        continue
    if get_chart_name_from_path(lab_path) in test_sheets:
        dest_branch = "test"
    elif get_chart_name_from_path(lab_path) in val_sheets:
        dest_branch = "val"
    else:
        dest_branch = "train"
    new_dest = new_dest_stem / dest_branch

    shutil.copy(lab_path, new_dest)


def undersample_background(labels_path: Path, images_path: Path, target_pcnt: float):
    """Randomly deletes images with no instances of blood pressure in them to undersample the background.

    Args:
        `labels_path` (Path):
            A pathlib Path to the training set labels.
        `images_path` (Path):
            A pathlib Path to the training set images.
        `target_pcnt` (float):
            A number between 0 and 1 determining the target percentage of backgrounds.
    """
    if not 0 <= target_pcnt <= 1:
        raise ValueError("Target percent must be between 0 and 1 (inclusive).")

    labels: List[str] = [Path(s) for s in glob(str(labels_path / "*.txt"))]
    images: List[str] = [Path(s) for s in glob(str(images_path / "*.JPG"))]
    backgrounds: List[str] = list(
        filter(lambda ti: ti.stem not in [tl.stem for tl in labels], images)
    )
    total_size_of_target_dataset: int = np.ceil(len(labels) / (1 - target_pcnt))
    number_of_backgrounds_to_remove: int = int(
        len(backgrounds) - np.ceil(total_size_of_target_dataset * target_pcnt)
    )
    backgrounds_to_remove: list[int] = sorted(
        np.random.choice(
            a=len(backgrounds), size=number_of_backgrounds_to_remove, replace=False
        )
    )
    backgrounds_to_remove: list[Path] = [
        im_to_remove
        for (ix, im_to_remove) in enumerate(backgrounds)
        if ix in backgrounds_to_remove
    ]
    for path in tqdm(backgrounds_to_remove):
        os.remove(path)


undersample_background(
    path_to_data
    / "yolo_datasets"
    / "new_bp_and_hr_one_vs_rest_hr"
    / "labels"
    / "train",
    path_to_data
    / "yolo_datasets"
    / "new_bp_and_hr_one_vs_rest_hr"
    / "images"
    / "train",
    0.15,
)

undersample_background(
    path_to_data / "yolo_datasets" / "new_bp_and_hr_one_vs_rest_hr" / "labels" / "val",
    path_to_data / "yolo_datasets" / "new_bp_and_hr_one_vs_rest_hr" / "images" / "val",
    0.15,
)
