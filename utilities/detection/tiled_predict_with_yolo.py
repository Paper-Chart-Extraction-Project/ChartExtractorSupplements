"""Uses tiling and yolo to identify objects on an image and save them to a yolo formatted label."""


import argparse
from ChartExtractor.object_detection_models.ultralytics_yolov8 import UltralyticsYOLOv8
from ChartExtractor.utilities.annotations import BoundingBox
from ChartExtractor.utilities.detections import Detection
from ChartExtractor.utilities.detection_reassembly import (
    untile_detections,
    non_maximum_suppression,
    intersection_over_minimum,
    intersection_over_union,
)
from ChartExtractor.utilities.image_conversion import pil_to_cv2
from ChartExtractor.utilities.tiling import tile_image
from functools import partial
import json
from pathlib import Path
from PIL import Image
from typing import Dict, List


def read_json(json_filepath: Path) -> Dict:
    """Reads a json file to a dictionary."""
    return json.loads(open(json_filepath, 'r').read())


def validate_model_metadata(metadata: Dict):
    """Validates the metadata file."""
    needed_keys: List[str] = [
        "image_size",
        "horizontal_overlap_ratio",
        "vertical_overlap_ratio",
        "names"
    ]
    for key in needed_keys:
        if metadata.get(key) is None:
            raise KeyError(f"Necessary key {key} does not exist in the model metadata.")


def validate_filepath(path_str: str, must_already_exist: bool) -> Path:
    """Validates and creates a path object from a string."""
    path: Path = Path(path_str)
    if must_already_exist and not path.exists():
        raise FileNotFoundError(f"No such file or directory '{path.resolve()}'")
    if path.is_dir():
        raise FileNotFoundError(
            f"'{path.resolve()}' is a directory, but a file was expected."
        )
    return path


parser = argparse.ArgumentParser()
parser.add_argument(
    "image_path",
    type=partial(validate_filepath, must_already_exist=True),
    help="The path to the image to predict on.",
)
parser.add_argument(
    "yolo_weights_path",
    type=partial(validate_filepath, must_already_exist=True),
    help="The path to the yolo model weights."
)
parser.add_argument(
    "model_metadata_path",
    type=partial(validate_filepath, must_already_exist=True),
    help="The path to the model's metadata."
)
parser.add_argument(
    "output_filepath",
    type=partial(validate_filepath, must_already_exist=False),
    help="The path to where the output should go."
)

args = parser.parse_args()


metadata: Dict = read_json(args.model_metadata_path)
validate_model_metadata(metadata)

model: UltralyticsYOLOv8 = UltralyticsYOLOv8.from_weights_path(args.yolo_weights_path)
image: Image.Image = Image.open(args.image_path)
image_tiles: List[List[Image.Image]] = tile_image(
    image,
    metadata["image_size"],
    metadata["image_size"],
    metadata["horizontal_overlap_ratio"],
    metadata["vertical_overlap_ratio"],
)
detections: List[List[List[Detection]]] = [
    [model(pil_to_cv2(tile), verbose=False)[0] for tile in row]
    for row in image_tiles
]
detections: List[Detection] = untile_detections(
    detections,
    metadata["image_size"],
    metadata["image_size"],
    metadata["horizontal_overlap_ratio"],
    metadata["vertical_overlap_ratio"],
)
detections: List[Detection] = non_maximum_suppression(
    detections,
    threshold=0.8,
    overlap_comparator=intersection_over_minimum,
    sorting_fn=lambda det: det.annotation.area * det.confidence
)
predictions: List[BoundingBox] = [det.annotation for det in detections]
yolo_output: str = "\n".join(
    [
        p.to_yolo(
            image.size[0],
            image.size[1],
            {val:key for (key, val) in metadata["names"].items()}
        )
        for p in predictions
    ]
)

with open(args.output_filepath, 'w') as f:
    f.write(yolo_output)
