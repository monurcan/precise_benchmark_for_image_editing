import random
from typing import Tuple

import numpy as np

from object_transformations.flip import Flip
from object_transformations.move import MoveByPixel, MoveTo, MoveByPercentage
from object_transformations.object_transformation import ObjectTransformation
from object_transformations.rotate import Rotate
from object_transformations.scale import (
    ScaleBy,
    ScaleAbsolutelyToPixels,
    ScaleAbsolutelyToPercentage,
)


class Compose(ObjectTransformation):
    def __init__(self, transformations: Tuple[ObjectTransformation] = None):
        super().__init__()
        self.transform_matrix = None
        if transformations is None:
            # create random sequence of transformations
            transformations = [
                Flip(),
                ScaleBy(),
                ScaleAbsolutelyToPercentage(),
                ScaleAbsolutelyToPixels(),
                MoveByPixel(),
                MoveByPercentage(),
                MoveTo(),
            ]  # Rotate(), Sheer(),

            random.shuffle(transformations)
            subset_size = random.randint(2, len(transformations))
            self.transformations = transformations[:subset_size]
        else:
            self.transformations = transformations
        self.mask_shape = None

    def _process_object(self, mask: np.array) -> np.array:
        self.mask_shape = mask.shape

        self.transform_matrix = np.eye(3)

        x_min, x_max, y_min, y_max = self.get_object_boundaries(mask)
        x_center_init, y_center_init = (x_min + x_max) // 2, (y_min + y_max) // 2
        # this is the origin
        x_center, y_center = x_center_init, y_center_init

        for transformation in self.transformations:
            mask = transformation.process(mask)

            x_min, x_max, y_min, y_max = self.get_object_boundaries(mask)
            x_center, y_center = (x_min + x_max) // 2, (y_min + y_max) // 2

            T_move_initial_center_to_current_center = np.array(
                [
                    [1, 0, x_center - x_center_init],
                    [0, 1, y_center - y_center_init],
                    [0, 0, 1],
                ]
            )
            T_move_current_center_to_initial_center = np.array(
                [
                    [1, 0, x_center_init - x_center],
                    [0, 1, y_center_init - y_center],
                    [0, 0, 1],
                ]
            )
            current_transformation_matrix = (
                T_move_initial_center_to_current_center
                @ transformation.get_matrix()
                @ T_move_current_center_to_initial_center
            )

            self.transform_matrix = (
                current_transformation_matrix @ self.transform_matrix
            )

        return mask

    def _get_base_prompt(self) -> str:
        prompts = [
            transformation._get_base_prompt() for transformation in self.transformations
        ]
        return ".".join(prompts)

    def _get_manually_generated_prompt(self) -> str:
        prompts = [
            transformation._get_manually_generated_prompt()
            for transformation in self.transformations
        ]
        return " then ".join(prompts)

    def _get_transformation_matrix_wrt_object_center(self) -> np.array:
        if self.transform_matrix is None:
            raise ValueError("No transformations have been applied yet")

        return self.transform_matrix

    def _check_overflow(self, mask: np.array, processed_mask: np.array) -> bool:
        for transformation in self.transformations:
            if transformation._check_overflow(mask, processed_mask):
                return True
        return False

    def decompose_transformation_matrix_as_flip_shift_resize_rotation(
        self,
    ) -> dict:
        flip_exists = np.sign(self.transform_matrix[0, 0]) != np.sign(
            self.transform_matrix[1, 1]
        )
        scale_factor = 0.5 * (
            np.sqrt(self.transform_matrix[0, 0] ** 2 + self.transform_matrix[0, 1] ** 2)
            + np.sqrt(
                self.transform_matrix[1, 0] ** 2 + self.transform_matrix[1, 1] ** 2
            )
        )
        rotation_angle = np.degrees(
            np.arctan2(self.transform_matrix[1, 0], self.transform_matrix[1, 1])
        )
        dx = self.transform_matrix[0, 2]
        dy = self.transform_matrix[1, 2]
        output_dict = {
            # "flip": flip_exists,
            "dx": dx / self.mask_shape[1],
            "dy": dy / self.mask_shape[0],
            "sx": scale_factor,
            "sy": scale_factor,
            "rot": rotation_angle,
        }
        return output_dict


# Example usage
if __name__ == "__main__":
    import utils.create_random_simple_shape_mask as mask_generator
    from object_transformations.move import MoveByPercentage, MoveTo
    from object_transformations.scale import (
        ScaleAbsolutelyToPixels,
        ScaleAbsolutelyToPercentage,
    )

    # Generate a random mask
    mask = mask_generator.create_shape_mask("oval", (512, 512))

    # Create individual transformations
    move1 = MoveByPixel(displacement=(50, 100))
    # move2 = Move(displacement=(0, 0))
    move2 = Rotate()
    move3 = ScaleBy(2.4)

    # Create a transformation composition
    composition = Compose((move1, move2, move3))
    processed_mask = composition.process(mask)
    prompt = composition.get_prompt()
    print(prompt)

    import cv2

    cv2.imshow("Original Mask", mask)
    cv2.imshow("Processed Mask", processed_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
