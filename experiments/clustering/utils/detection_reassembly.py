"""This module defines functions for reassembling tiled detections."""

from dataclasses import dataclass
from typing import Tuple, Union
from .annotations import BoundingBox, Keypoint

"""This module defines the Detection class representing a single object detection result.

This class is used to store the output of an object detection model, including:

* The predicted location of the object, represented by either a BoundingBox or a Keypoint instance (depending on the model's output format).
* The confidence score assigned by the model to this detection (a float between 0.0 and 1.0).
"""


@dataclass
class Detection:
    """Represents a single detection result from an object detection model.

    Attributes:
        annotation:
            An instance of either BoundingBox or Keypoint class, depending on the
            type of annotation used for localization (bounding box or keypoints).
        confidence:
            A float value between 0.0 and 1.0 representing the confidence score
            assigned by the object detection model to this detection.
    """

    annotation: Union[BoundingBox, Keypoint]
    confidence: float


def compute_area(box: Tuple[float, float, float, float]):
    """Computes the area of a rectangle.

    Args:
        `box` (Tuple[float, float, float, float]):
            A tuple of four floats that define the (left, top, right, bottom) of a rectangle.

    Returns:
        The area of the rectangle.
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def compute_intersection_area(
    box_1: Tuple[float, float, float, float], box_2: Tuple[float, float, float, float]
) -> float:
    """Computes the area of the intersection of two rectangle.

    Args:
        `box_1` (Tuple[float, float, float, float]):
            A tuple of four floats that define the (left, top, right, bottom) of the first rectangle.
        `box_2` (Tuple[float, float, float, float]):
            A tuple of four floats that define the (left, top, right, bottom) of the second rectangle.

    Returns:
        The area of the intersection of the two rectangles box_1 and box_2.
    """
    intersection_left = max(box_1[0], box_2[0])
    intersection_top = max(box_1[1], box_2[1])
    intersection_right = min(box_1[2], box_2[2])
    intersection_bottom = min(box_1[3], box_2[3])
    if intersection_right < intersection_left or intersection_bottom < intersection_top:
        return 0
    intersection_area = compute_area(
        [intersection_left, intersection_top, intersection_right, intersection_bottom]
    )
    return intersection_area


def intersection_over_union(detection_1: Detection, detection_2: Detection) -> float:
    """Calculates the Intersection over Union (IoU) between two detections.

    This function calculates the area of overlap between the bounding boxes of two
    detection objects and divides it by the total area covered by their bounding boxes.

    Args:
        `detection_1` (Detection):
            A Detection object representing the first detection.
        `detection_2` (Detection):
            A Detection object representing the second detection.

    Returns:
        A float value between 0.0 and 1.0 representing the IoU between the two detections.
    """
    box_1, box_2 = detection_1.annotation.box, detection_2.annotation.box
    intersection_area = compute_intersection_area(box_1, box_2)
    union_area = compute_area(box_1) + compute_area(box_2) - intersection_area
    return intersection_area / union_area
