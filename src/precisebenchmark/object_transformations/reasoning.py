import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from object_transformations.compose import Compose
from object_transformations.flip import Flip
from object_transformations.move import MoveByPercentage, MoveByPixel, MoveTo
from object_transformations.rotate import Rotate
from object_transformations.scale import (
    ScaleAbsolutelyToPercentage,
    ScaleAbsolutelyToPixels,
    ScaleBy,
)
from object_transformations.shear import Shear
from tqdm import tqdm
from utils.binary_masks import get_object_boundaries
from utils.pascal_voc_parser import parse_voc
from utils.set_seeds import set_seeds


class Reasoning(ScaleAbsolutelyToPixels):
    def __init__(self, obj, all_objects):
        # Get all objects that are not the same as the current object
        other_objects = [o for o in all_objects if o.name != obj.name]

        # Check if there are other objects of the same name
        if len(other_objects) == 0:
            raise ValueError(
                "There should be other objects for reasoning like transformations"
            )

        # Get the object with the maximum area among the other objects
        max_area_obj = max(other_objects, key=lambda o: np.sum(o.mask / 255))
        max_area_obj_bb = get_object_boundaries(max_area_obj.mask)

        if random.choice([True, False]):
            # Height based
            new_height = max_area_obj_bb[3] - max_area_obj_bb[2]
            super().__init__((None, new_height))
            self._get_manually_generated_prompt = lambda: (
                f"Make the {obj.name} as tall as the {max_area_obj.name}."
            )
        else:
            # Width based
            new_width = max_area_obj_bb[1] - max_area_obj_bb[0]
            super().__init__((new_width, None))
            self._get_manually_generated_prompt = lambda: (
                f"Make the {obj.name} as wide as the {max_area_obj.name}."
            )
