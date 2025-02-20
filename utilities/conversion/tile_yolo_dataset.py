"""Tiles a yolo dataset."""

# Built-in Imports
from argparse import ArgumentParser, Namespace
from glob import glob
import os
from operator import attrgetter
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
    "--background_proportion",
    help="Proportion of background images in the final dataset (Default: 0.15)",
    type=float,
)
parser.add_argument(
    "--horizontal_overlap_proportion",
    help="The percent amount of horizontal overlap for tiles (Default: 0.5)",
    type=float,
)
parser.add_argument(
    "--vertical_overlap_proportion",
    help="The percent amount vertical overlap for tiles (Default: 0.5)",
    type=float,
)
parser.add_argument(
    "--tile_size_proportion",
    help="The percent of the image's height or width (whichever is smaller) to make each tile (Default: 0.2)",
    type=float,
)
parser.add_argument(
    "--random_seed",
    help="The random seed to use for numpy.",
    type=int,
)


def read_input_dataset_path(args: Namespace) -> Path:
    """Reads and validates the input_dataset_path argument.

    Args:
        args (Namespace):
            The argparse.Namespace object that contains the command line arg data.

    Raises:
        FileNotFoundError:
            If the input dataset does not exist.

    Returns:
        A path to the input dataset.
    """
    input_path: Path = Path(args.input_dataset_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"FileNotFoundError: No such file or directory at {str(input_path.resolve())}."
        )
    elif not Path(args.input_dataset_path).is_dir():
        raise Exception(
            f"Path exists but is not a directory: {str(input_path.resolve())}."
        )
    return input_path.resolve()


def read_float_in_range_0_to_1(args: Namespace, argname: str, default: float) -> float:
    """Reads a float from the command line arguments that must be within the range [0, 1].

    Args:
        args (Namespace):
            The argparse.Namespace object that contains the command line arg data.
        argname (str):
            The name of the argument to get from the command line args.
        default (float):
            The default value to use if the user did not pass in a value.

    Raises:
        ValueError:
            If the supplied value is out of the range [0, 1].

    Returns:
        A float between 0 and 1 encoding the value passed in, or a default.
    """
    if not attrgetter(argname)(args):
        return default
    float_val: float = attrgetter(argname)(args)
    float_is_in_range: bool = 0 <= float_val <= 1
    if not float_is_in_range:
        err_message: str = f"{float_val} is an invalid value for --{argname}."
        err_message += " Value must be between 0 and 1."
        raise ValueError(err_message)
    return float_val


def find_splits(input_dataset_path: Path) -> List[str]:
    """Finds the splits that the dataset uses.

    Args:
        input_dataaset_path (Path):
            The path to the YOLO dataset.

    Returns:
        A list of the splits that the dataset uses.
    """
    potential_splits: List[str] = next(os.walk(str(input_dataset_path / "labels")))[1]
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


def try_open_image(im_path: str) -> Optional[Image.Image]:
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


def tile_images(
    input_dataset_path: Path,
    output_dataset_path: Path,
    splits: List[str],
    tile_size: int,
    horizontal_overlap_proportion: float = 0.5,
    vertical_overlap_proportion: float = 0.5,
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
        horizontal_overlap_proportion (float):
            The proportion of overlap that two neighboring tiles should have left and right.
        vertical_overlap_proportion (float):
            The proportion of overlap that two neighboring tiles should have up and down.
    """
    for split in splits:
        image_paths: List[str] = glob(str(input_dataset_path / split / "*"))
        for im_path in image_paths:
            image: Optional[Image.Image] = try_open_image(im_path)
            if image is None:
                continue
            image_tiles: List[List[Image.Image]] = tile_image(
                image,
                tile_size,
                tile_size,
                horizontal_overlap_proportion,
                vertical_overlap_proportion,
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
    horizontal_overlap_proportion: float = 0.5,
    vertical_overlap_proportion: float = 0.5,
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
        horizontal_overlap_proportion (float):
            The proportion of overlap that two neighboring tiles should have left and right.
        vertical_overlap_proportion (float):
            The proportion of overlap that two neighboring tiles should have up and down.
    """
    def create_image_size_dict() -> Dict[str, Tuple[int, int]]:
        """Creates a dict with image filename stems mapped to image size (width, height).

        Returns:
            A dictionary mapping filename stems to a tuple with an image's (width, height).
        """
        image_size_dict: Dict[str, Tuple[int, int]] = dict()
        for split in splits:
            for file in glob(str(input_dataset_path / split / "*")):
                image: Optional[Image.Image] = try_open_image(file)
                if image is None:
                    continue
                image_size_dict[Path(file).stem] = image.size
        return image_size_dict

    image_size_dict: Dict[str, Tuple[int, int]] = create_image_size_dict()

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
            text: List[str] = open(ann_path, "r").readlines()
            if text[0].replace("\n", "").split(" ") > 5:
                annotation_type = Keypoint
            else:
                annotation_type = BoundingBox
            annotations: List[Union[BoundingBox, Keypoint]] = list()
            for line in text:
                annotations.append(
                    annotation_type.from_yolo(
                        line.strip(),
                        image_size_dict[ann_path.stem][0],
                        image_size_dict[ann_path.stem][1],
                        {line[0]:line[0]}
                    )
                )
            return annotations
        except:
            return None

    for split in splits:
        label_paths: List[str] = glob(str(input_dataset_path / split / "*"))
        for lab_path in label_paths:
            annotations: List[Union[BoundingBox, Keypoint]] = try_open_annotation(
                lab_path
            )
            if annotations is None:
                continue
            annotation_tiles: List[List[List[Union[BoundingBox, Keypoint]]]] = (
                tile_annotations(
                    annotations,
                    image_size_dict[Path(lab_path.stem)][0],
                    image_size_dict[Path(lab_path.stem)][1],
                    tile_size,
                    tile_size,
                    horizontal_overlap_proportion,
                    vertical_overlap_proportion,
                )
            )
            for row in annotation_tiles:
                for tile in row:
                    data_to_save: str = "\n".join(
                        [
                            ann.category
                            + l.to_yolo(slice_size, slice_size, {l.category:l.category}, 10, True)[
                                1:
                            ]
                            for l in tile
                        ]
                    )
                    if yolo_str != "":
                        with open(
                            str(output_dataset_path / split / Path(lab_path).stem)
                            + ".txt",
                            "w",
                        ) as f:
                            f.write(data_to_save)


def undersample_background(
    output_dataset_path: Path, splits: List[str], target_pcnt: float
):
    """Randomly deletes images with no instances of blood pressure in them to undersample the background.

    Args:
        output_dataset_path (Path):
            A pathlib Path to the final dataset.
        splits (List[str]):
            A list of the splits the dataset uses.
        target_pcnt (float):
            A number between 0 and 1 determining the target percentage of backgrounds.
    """
    files_to_delete: List[Path] = list()
    for split in splits:
        label_paths: List[Path] = [
            Path(s) for s in glob(str(output_dataset_path/"labels"/split/"*.txt"))
        ]
        image_paths: List[Path] = [
            Path(s) for s in glob(str(output_dataset_path/"images"/split/"*.jpg"))
        ]
        background_paths: List[Path] = list(
            filter(
                lambda im_path: im_path.stem not in [lab_path.stem for lab_path in labels], 
                image_paths
            )
        )
        total_size_of_target_dataset: int = np.ceil(len(label_paths) / (1 - target_pcnt))
        number_of_backgrounds_to_remove: int = int(
            len(background_paths) - np.ceil(total_size_of_target_dataset * target_pcnt)
        )
        backgrounds_to_remove: list[int] = sorted(
            np.random.choice(
                a=len(background_paths), size=number_of_backgrounds_to_remove, replace=False
            )
        )
        backgrounds_to_remove: list[Path] = [
            im_to_remove
            for (ix, im_to_remove) in enumerate(background_paths)
            if ix in backgrounds_to_remove
        ]
        files_to_delete += backgrounds_to_remove
    for path in tqdm(backgrounds_to_remove):
        os.remove(files_to_delete)


if __name__ == "__main__":
    args: Namespace = parser.parse_args()

    if args.random_seed:
        np.random.seed(args.random_seed)

    horizontal_overlap_proportion: float = read_float_in_range_0_to_1(
        args, "horizontal_overlap_proportion", 0.5
    )
    vertical_overlap_proportion: float = read_float_in_range_0_to_1(
        args, "vertical_overlap_proportion", 0.5
    )
    tile_size_proportion: float = read_float_in_range_0_to_1(
        args, "tile_size_proportion", 0.2
    )
    background_proportion: float = read_float_in_range_0_to_1(
        args, "background_proportion", 0.15
    )
    input_dataset_path: Path = read_input_dataset_path(args)
    splits: List[str] = find_splits(input_dataset_path)
    output_dataset_path: Path = Path(args.output_dataset_path)
    validate_yolo_dataset(input_dataset_path)
    create_output_dataset_directories(output_dataset_path, splits)
    tile_images(
        input_dataset_path,
        output_dataset_path,
        splits,
        tile_size_proportion,
        horizontal_overlap_proportion,
        vertical_overlap_proportion,
    )
    tile_annotations(
        input_dataset_path,
        output_dataset_path,
        splits,
        tile_size_proportion,
        horizontal_overlap_proportion,
        vertical_overlap_proportion,
    )
    undersample_background(output_dataset_path, splits, background_proportion)
