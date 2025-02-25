"""Utilities is a file that holds commonly used functions and classes."""

import random
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw


def draw_bboxes_with_translucence(
    image: Image.Image,
    bounding_boxes: List["BoundingBox"],
    locations_are_relative: bool,
    colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    alpha: int = 40,
) -> Image.Image:
    """Draws bounding boxes on an image with a translucent fill.

    Args:
        `image` (Image.Image):
            A PIL image to draw on top of.
        `bounding_boxes` (List[BoundingBox]):
            A list of bounding boxes to draw on the image.
        `locations_are_relative` (bool):
            If flag is true, then each BoundingBox in bounding_boxes is actually in relative scale
            (Ex: left=0.2, top=0.05, right=0.41, bottom=0.55). Thus, we need to multiply the values
            by the image's width and height to get a correct drawing.
        `alpha` (int):
            A value between 0 and 255 showing the value the alpha layer should be for the box fill.
    """
    im: Image.Image = image.copy()
    mask: Image.Image = Image.new(mode="RGBA", size=im.size, color=(0, 0, 0, 0))
    im_layer_draw: ImageDraw.Draw = ImageDraw.Draw(im)
    mask_layer_draw: ImageDraw.Draw = ImageDraw.Draw(mask)

    if colors is None:
        all_categories: List[str] = list(set(bb.category for bb in bounding_boxes))
        colors: Dict[str, Tuple[int, int, int]] = {
            category: (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            for category in all_categories
        }

    def get_box(bbox: "BoundingBox") -> Tuple[float, float, float, float]:
        if locations_are_relative:
            return [
                bbox.left * im.size[0],
                bbox.top * im.size[1],
                bbox.right * im.size[0],
                bbox.bottom * im.size[1],
            ]
        else:
            return bbox.box

    for box in bounding_boxes:
        im_layer_draw.rectangle(
            get_box(box), outline=tuple(list(colors[box.category]) + [255])
        )
        mask_layer_draw.rectangle(
            get_box(box), fill=tuple(list(colors[box.category]) + [alpha])
        )

    im.paste(mask, (0, 0), mask)
    return im
