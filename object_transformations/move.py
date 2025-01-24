import math
import random

import cv2
import numpy as np

from object_transformations.object_transformation import ObjectTransformation
from utils.random_variables import (
    generate_bimodal_sample,
    random_sign,
)


class MoveByPixel(ObjectTransformation):
    def __init__(self, displacement: tuple = None):
        super().__init__()
        # Set displacement vector (dx, dy) from tuple, randomize if not provided
        if displacement is None:
            # self.dx, self.dy = random.randint(-150, 150), random.randint(-150, 150)
            self.dx = int(generate_bimodal_sample(80, 40, 200, 40, 0.8)) * random_sign()
            self.dy = int(generate_bimodal_sample(80, 40, 200, 40, 0.8)) * random_sign()
        else:
            self.dx, self.dy = displacement

        (
            self.new_x_start,
            self.new_y_start,
            self.new_x_end_distance_to_right,
            self.new_y_end_distance_to_bottom,
        ) = (None, None, None, None)
        self.direction, self.mask_shape = None, None

    def _process_object(self, mask: np.array) -> np.array:
        # Find object boundaries
        x_min, x_max, y_min, y_max = self.get_object_boundaries(mask)

        # Offset for better transformation - but the transformation matrix wrong
        old_width, old_height = x_max - x_min, y_max - y_min
        x_min -= old_width // 15
        x_max += old_width // 15
        y_min -= old_height // 15
        y_max += old_height // 15
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(mask.shape[1], x_max)
        y_max = min(mask.shape[0], y_max)

        # Extract the object from the original mask
        object = mask[y_min:y_max, x_min:x_max]

        # Create a new mask to move the object
        new_mask = np.zeros(mask.shape, dtype=np.uint8)

        # Compute new starting and ending positions after applying the displacement
        y_start = y_min - self.dy
        x_start = x_min + self.dx
        y_end = y_start + (y_max - y_min)
        x_end = x_start + (x_max - x_min)

        if x_start < 0:
            object = object[:, abs(x_start) :]
            x_start = 0
        if y_start < 0:
            object = object[abs(y_start) :, :]
            y_start = 0
        if x_end > mask.shape[1]:
            object = object[:, : -(x_end - mask.shape[1])]
            x_end = mask.shape[1]
        if y_end > mask.shape[0]:
            object = object[: -(y_end - mask.shape[0]), :]
            y_end = mask.shape[0]

        if object.size == 0:
            return new_mask

        new_mask[y_start:y_end, x_start:x_end] = object

        # For prompt generation
        (
            self.new_x_start,
            self.new_y_start,
            self.new_x_end_distance_to_right,
            self.new_y_end_distance_to_bottom,
        ) = (
            x_start,
            y_start,
            mask.shape[1] - x_end,
            mask.shape[0] - y_end,
        )

        self.mask_shape = mask.shape

        return new_mask

    def _check_overflow(self, mask: np.array, processed_mask: np.array) -> bool:
        before_area = np.sum(mask != 0)
        after_area = np.sum(processed_mask != 0)

        return after_area / before_area < 0.85

    def _get_base_prompt(self) -> str:
        angle_of_displacement = math.degrees(math.atan2(self.dy, self.dx))
        direction_map = {
            (-10, 10): "right",
            (10, 80): "up-right",
            (80, 100): "up",
            (100, 170): "up-left",
            (170, 180): "left",
            (-180, -170): "left",
            (-170, -100): "down-left",
            (-100, -80): "down",
            (-80, -10): "down-right",
        }
        for angle_range, map_direction in direction_map.items():
            if angle_range[0] <= angle_of_displacement <= angle_range[1]:
                self.direction = map_direction

        if not self.direction:
            raise ValueError("Direction not found among the predefined ranges")

        return f"<MOVE> <OBJECT> ({self.dx},{self.dy}) ({100 * self.dx / self.mask_shape[1]:.2f},{100 * self.dy / self.mask_shape[0]:.2f}) {self.direction}"

    def _get_transformation_matrix_wrt_object_center(self) -> np.array:
        return np.array([[1.0, 0, self.dx], [0, 1.0, -self.dy], [0, 0, 1.0]])

    def _get_manually_generated_prompt(self) -> str:
        possible_prompts = [
            f"displace the object by ({self.dx}, {self.dy})",
            f"move the object by ({self.dx}, {self.dy})",
            f"translate the object by a vector ({self.dx}, {self.dy})",
            f"apply a movement to the object by ({self.dx}, {self.dy})",
        ]

        # More verbal approximate explanations
        if not self.direction:
            raise ValueError(
                "First call _get_base_prompt() before _get_manually_generated_prompt()"
            )

        power = ""
        magnitude_of_displacement = math.sqrt(self.dx**2 + self.dy**2)
        if magnitude_of_displacement > 100:
            power = "too much"
        elif magnitude_of_displacement > 50:
            power = "moderately"
        else:
            power = "slightly"

        possible_prompts.extend(
            [
                f"displace the object {power} to {self.direction} direction",
                f"move the object {power} to the {self.direction} direction",
                f"translate the object {power} in the {self.direction} direction",
                f"apply a movement of {power} to the object in the {self.direction} direction",
            ]
        )

        # Move to corner
        if self.new_x_start:
            epsilon_corner_tolerance = 35.0
            if (
                self.new_x_start < epsilon_corner_tolerance
                and self.new_y_start < epsilon_corner_tolerance
            ):
                possible_prompts.append("move the object to the top-left corner")
            elif (
                self.new_x_end_distance_to_right < epsilon_corner_tolerance
                and self.new_y_start < epsilon_corner_tolerance
            ):
                possible_prompts.append("move the object to the top-right corner")
            elif (
                self.new_x_start < epsilon_corner_tolerance
                and self.new_y_end_distance_to_bottom < epsilon_corner_tolerance
            ):
                possible_prompts.append("move the object to the bottom-left corner")
            elif (
                self.new_x_end_distance_to_right < epsilon_corner_tolerance
                and self.new_y_end_distance_to_bottom < epsilon_corner_tolerance
            ):
                possible_prompts.append("move the object to the bottom-right corner")

        return random.choice(possible_prompts)


class MoveByPercentage(MoveByPixel):
    def __init__(self, displacement_percentage: tuple):
        self.dx_percentage, self.dy_percentage = displacement_percentage

    def process(self, mask: np.array) -> np.array:
        x_start, y_start = int(mask.shape[1] * self.dx_percentage / 100), int(
            mask.shape[0] * self.dy_percentage / 100
        )

        super().__init__((x_start, y_start))

        return super().process(mask)


class MoveTo(MoveByPixel):
    def __init__(self, position: str):
        self.position = position

    def process(self, mask: np.array) -> np.array:
        mask_height, mask_width = mask.shape

        # Find object boundaries
        y_min, x_min = np.min(np.where(mask != 0), axis=1)
        y_max, x_max = np.max(np.where(mask != 0), axis=1)

        obj_height, obj_width = y_max - y_min, x_max - x_min

        # map to the new position
        if self.position == "left-up":
            x_start_new, y_start_new = 0, 0
        elif self.position == "left-center":
            x_start_new, y_start_new = 0, (mask_height - obj_height) // 2
        elif self.position == "left-bottom":
            x_start_new, y_start_new = 0, mask_height - obj_height
        elif self.position == "center-up":
            x_start_new, y_start_new = (mask_width - obj_width) // 2, 0
        elif self.position == "center-center":
            x_start_new, y_start_new = (mask_width - obj_width) // 2, (
                mask_height - obj_height
            ) // 2
        elif self.position == "center-bottom":
            x_start_new, y_start_new = (
                mask_width - obj_width
            ) // 2, mask_height - obj_height
        elif self.position == "right-up":
            x_start_new, y_start_new = mask_width - obj_width, 0
        elif self.position == "right-center":
            x_start_new, y_start_new = (
                mask_width - obj_width,
                (mask_height - obj_height) // 2,
            )
        elif self.position == "right-bottom":
            x_start_new, y_start_new = mask_width - obj_width, mask_height - obj_height
        else:
            raise ValueError("Position string is not valid!!!")

        dx = x_start_new - x_min
        dy = -(y_start_new - y_min)

        super().__init__((dx, dy))

        return super().process(mask)


# Test
if __name__ == "__main__":
    import utils.create_random_simple_shape_mask as mask_generator

    # while True:
    mask = mask_generator.create_shape_mask("random", (512, 512))
    cv2.imshow("Original Mask", mask)

    move = MoveByPixel((140, -100))
    processed_mask = move.process(mask)
    cv2.imshow(str(move), processed_mask)

    move = MoveByPercentage((-20, 20))
    processed_mask = move.process(mask)
    cv2.imshow(str(move), processed_mask)

    for stringname in [
        "left-up",
        "left-center",
        "left-bottom",
        "center-up",
        "center-center",
        "center-bottom",
        "right-up",
        "right-center",
        "right-bottom",
    ]:
        move = MoveTo(stringname)

        processed_mask = move.process(mask)

        cv2.imshow(stringname, processed_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
