import random

import cv2
import numpy as np
from numpy._core.multiarray import array as array

from object_transformations.object_transformation import ObjectTransformation
from utils.random_variables import generate_bimodal_sample


class ScaleBy(ObjectTransformation):
    def __init__(self, scale_factor: float = None):
        super().__init__()

        if scale_factor:
            self.scale_factor = scale_factor
        else:
            self.scale_factor = max(
                0, generate_bimodal_sample(0.66, 0.15, 2.35, 0.6, 0.36)
            )

    def _process_object(self, mask: np.array) -> np.array:
        # Find object boundaries
        y_min, x_min = np.min(np.where(mask != 0), axis=1)
        y_max, x_max = np.max(np.where(mask != 0), axis=1)

        # Offset for better transformation
        old_width, old_height = x_max - x_min, y_max - y_min
        x_min -= old_width // 15
        x_max += old_width // 15
        y_min -= old_height // 15
        y_max += old_height // 15
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(mask.shape[1], x_max)
        y_max = min(mask.shape[0], y_max)

        width = x_max - x_min
        height = y_max - y_min
        self.x_center, self.y_center = (x_min + x_max) // 2, (y_min + y_max) // 2

        # Resize the object using the scale factor
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)

        # Extract the object from the original mask
        scaled_object = mask[y_min:y_max, x_min:x_max]
        scaled_object = cv2.resize(scaled_object, (new_width, new_height))

        # Create a new mask image with the resized object
        new_mask = np.zeros(mask.shape, dtype=np.uint8)
        x_start = self.x_center - new_width // 2
        y_start = self.y_center - new_height // 2
        x_end = x_start + new_width
        y_end = y_start + new_height

        # Ensure the resized object does not go out of bounds
        if x_start < 0:
            scaled_object = scaled_object[:, abs(x_start) :]
            x_start = 0
        if y_start < 0:
            scaled_object = scaled_object[abs(y_start) :, :]
            y_start = 0
        if x_end > mask.shape[1]:
            scaled_object = scaled_object[:, : -(x_end - mask.shape[1])]
            x_end = mask.shape[1]
        if y_end > mask.shape[0]:
            scaled_object = scaled_object[: -(y_end - mask.shape[0]), :]
            y_end = mask.shape[0]

        if scaled_object.size == 0:
            return new_mask

        # Place the resized object into the new mask
        new_mask[y_start:y_end, x_start:x_end] = scaled_object

        return new_mask

    def _check_overflow(self, mask: np.array, processed_mask: np.array) -> bool:
        before_area = np.sum(mask != 0)
        after_area = np.sum(processed_mask != 0)

        return abs(after_area / before_area / self.scale_factor**2 - 1) > 0.2

    def _get_base_prompt(self) -> str:
        return f"<SCALE> <OBJECT> {self.scale_factor:.2f}"

    def _get_transformation_matrix_wrt_object_center(self) -> np.array:
        return np.array(
            [
                [self.scale_factor, 0, 0],
                [0, self.scale_factor, 0],
                [0, 0, 1],
            ]
        )

    def _get_manually_generated_prompt(self) -> str:
        possible_prompts = [
            "scale the object by a factor of {self.scale_factor:.2f}",
            "resize the object by a factor of {self.scale_factor:.2f}",
            "change the size of the object by a factor of {self.scale_factor:.2f}",
            "change the scale of the object by a factor of {self.scale_factor:.2f}",
            "adjust the object's size to a factor of {self.scale_factor:.2f}",
            "apply a scale transformation with a factor of {self.scale_factor:.2f}",
            "resize the object to {self.scale_factor:.2f} times its original size",
            "alter the dimensions of the object by a scaling factor of {self.scale_factor:.2f}",
            "modify the object's scale by {self.scale_factor:.2f} for better alignment",
            "transform the object to achieve a scale of {self.scale_factor:.2f}",
            "set the object size to {self.scale_factor:.2f} of its original dimensions",
            "change the object's proportions by a scale factor of {self.scale_factor:.2f}",
        ]

        # Additional prompts based on scale factor
        if self.scale_factor < 1:
            possible_prompts.append(
                "shrink the object by a factor of {self.scale_factor:.2f}"
            )
            possible_prompts.append(
                "reduce the object's size to {self.scale_factor:.2f} times"
            )
            possible_prompts.append(
                "make the object smaller with a factor of {self.scale_factor:.2f}"
            )
        else:
            possible_prompts.append(
                "enlarge the object by a factor of {self.scale_factor:.2f}"
            )
            possible_prompts.append(
                "increase the object's size to {self.scale_factor:.2f} times"
            )
            possible_prompts.append(
                "boost the size of the object to {self.scale_factor:.2f}"
            )

        # Variations based on scale factor
        if 1 < self.scale_factor < 1.3:
            possible_prompts.append("make the object slightly larger")
            possible_prompts.append("make the object a bit bigger")
        elif 0.7 < self.scale_factor < 1:
            possible_prompts.append("make the object slightly smaller")
            possible_prompts.append("make the object a bit smaller")
        elif self.scale_factor > 2:
            possible_prompts.append(
                "significantly increase the size of the object to {self.scale_factor:.2f} times its original"
            )
        elif 1.3 < self.scale_factor <= 2:
            possible_prompts.append(
                "noticeably enlarge the object to {self.scale_factor:.2f} times its initial size"
            )
        elif 0.5 < self.scale_factor <= 0.7:
            possible_prompts.append(
                "reduce the object's size considerably to {self.scale_factor:.2f} times its original dimensions"
            )
        elif self.scale_factor <= 0.5:
            possible_prompts.append(
                "drastically shrink the object to {self.scale_factor:.2f} of its original size"
            )

        if 1.85 < self.scale_factor < 2.15:
            possible_prompts.append("double the object's size")
            possible_prompts.append("make the object twice as big")

        return random.choice(possible_prompts).format(self=self)


class ScaleAbsolutelyToPercentage(ScaleBy):
    def __init__(self, scale_percentage: float = None):
        if scale_percentage is None:
            scale_percentage = random.randint(30, 330)

        self.scale_percentage = scale_percentage / 100

    def _process_object(self, mask: np.array) -> np.array:
        # Scale the mask by the given percentage of the input mask shape
        mask_height, mask_width = mask.shape

        # Find object boundaries
        y_min, x_min = np.min(np.where(mask != 0), axis=1)
        y_max, x_max = np.max(np.where(mask != 0), axis=1)
        obj_width, obj_height = x_max - x_min, y_max - y_min

        # Calculate scaling factor
        relative_scale_factor = (
            mask_width * self.scale_percentage / obj_width
            + mask_height * self.scale_percentage / obj_height
        ) / 2

        # Apply scaling
        super().__init__(relative_scale_factor)

        return super()._process_object(mask)


class ScaleAbsolutelyToPixels(ScaleBy):
    def __init__(self, tuple_new_width_height: tuple = None) -> None:
        if tuple_new_width_height is None and tuple_new_width_height is None:
            self.new_width = random.randint(50, 150)
            self.new_height = None
        else:
            self.new_width = tuple_new_width_height[0]
            self.new_height = tuple_new_width_height[1]

    def _process_object(self, mask: np.array) -> np.array:
        # Find object boundaries
        y_min, x_min = np.min(np.where(mask != 0), axis=1)
        y_max, x_max = np.max(np.where(mask != 0), axis=1)
        obj_width, obj_height = x_max - x_min, y_max - y_min

        # Calculate scaling factor
        if self.new_width is None:
            relative_scale_factor = self.new_height / obj_height
        elif self.new_height is None:
            relative_scale_factor = self.new_width / obj_width
        else:
            relative_scale_factor = max(
                self.new_width / obj_width, self.new_height / obj_height
            )

        # Apply scaling
        super().__init__(relative_scale_factor)

        return super()._process_object(mask)


# Test
if __name__ == "__main__":
    # Realistic test
    # mask = cv2.imread("realistic_mask_test.png", cv2.IMREAD_GRAYSCALE)
    # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Random basic shape test
    import utils.create_random_simple_shape_mask as mask_generator

    mask = mask_generator.create_shape_mask("random", (512, 512))

    # scale = ScaleBy(3.3)
    # scale = ScaleAbsolutelyToPercentage(25.3)
    # scale = ScaleAbsolutelyToPixels(new_width=256)
    scale = ScaleAbsolutelyToPixels(new_height=256)

    processed_mask = scale.process(mask)

    cv2.imshow("Original Mask", mask)
    cv2.imshow("Processed Mask", processed_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
