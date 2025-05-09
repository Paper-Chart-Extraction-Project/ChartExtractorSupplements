"""Adds annotations from a yolo formatted text file into a label studio json task file."""

import argparse
from functools import partial
from glob import glob
import json
from pathlib import Path
from typing import Dict, List


def read_json(json_filepath: Path) -> Dict:
    """Reads a json file to a dictionary."""
    return json.loads(open(json_filepath, "r").read())


def read_txt_to_lines(txt_filepath: Path) -> List[str]:
    """Reads a text file into a list of its lines."""
    return open(txt_filepath, "r").readlines()


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


def validate_model_metadata(metadata: Dict):
    """Validates the metadata file."""
    needed_keys: List[str] = [
        "image_size",
        "horizontal_overlap_ratio",
        "vertical_overlap_ratio",
        "names",
    ]
    for key in needed_keys:
        if metadata.get(key) is None:
            raise KeyError(f"Necessary key {key} does not exist in the model metadata.")


class YoloLabel:
    def __init__(self, label_str: str, id_to_string: Dict[str, str]):
        cl, x, y, w, h = label_str.replace("\n", "").split(" ")
        self.cl = id_to_string[cl]
        self.x = (float(x)-0.5*float(w))*100
        self.y = (float(y)-0.5*float(h))*100
        self.w = float(w)*100
        self.h = float(h)*100

    def __repr__(self):
        return f"YoloLabel({self.cl}, {self.x:.06f}, {self.y:.06f}, {self.w:.06f}, {self.h:.06f})"
    
    def to_ls_prediction_dict(self, original_width: int, original_height: int) -> Dict:
        """Converts this YoloLabel into a label studio formatted prediction dictionary."""
        return {
            "original_width": original_width,
            "original_height": original_height,
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": self.x,
                "y": self.y,
                "width": self.w,
                "height": self.h,
                "rotation": 0,
                "rectanglelabels": [self.cl]
            }
        }


parser = argparse.ArgumentParser()
parser.add_argument(
    "yolo_labels_path",
    type=partial(validate_filepath, must_already_exist=True),
    help="The path to the yolo labels txt file.",
)
parser.add_argument(
    "label_studio_task_json_path",
    type=partial(validate_filepath, must_already_exist=True),
    help="The path to the label studio task json file.",
)
parser.add_argument(
    "model_metadata_path",
    type=partial(validate_filepath, must_already_exist=True),
    help="The path to the model's metadata.",
)
parser.add_argument(
    "original_image_width",
    type=int,
    help="The original image's width."
)
parser.add_argument(
    "original_image_height",
    type=int,
    help="The original image's height."
)
parser.add_argument(
    "--confidence_path",
    type=partial(validate_filepath, must_already_exist=True),
    help="A txt file where each newline contains a float encoding the confidence of the annotation."
)
args = parser.parse_args()


metadata: Dict = read_json(args.model_metadata_path)
validate_model_metadata(metadata)
task_dict: Dict = read_json(args.label_studio_task_json_path)
yolo_data: List[YoloLabel] = [
    YoloLabel(yolo_str, metadata["names"])
    for yolo_str in read_txt_to_lines(args.yolo_labels_path)
]
task_dict["predictions"].append(
    {
        "result": [
            yl.to_ls_prediction_dict(args.original_image_width, args.original_image_height)
            for yl in yolo_data
        ]
    }
)

with open("test.json", 'w') as f:
    f.write(json.dumps(task_dict))
