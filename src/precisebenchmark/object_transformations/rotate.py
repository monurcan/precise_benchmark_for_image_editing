import random

import cv2
import numpy as np

from object_transformations.object_transformation import ObjectTransformation


class Rotate(ObjectTransformation):
    def __init__(self, angle_clockwise: float = None):
        super().__init__()

        self.angle = (
            angle_clockwise
            if angle_clockwise is not None
            else random.uniform(-180, 180)
        )

    def _process_object(self, mask: np.array) -> np.array:
        # Find object boundaries
        x_min, x_max, y_min, y_max = self.get_object_boundaries(mask)

        width = x_max - x_min
        height = y_max - y_min
        x_center, y_center = (x_min + x_max) // 2, (y_min + y_max) // 2

        # Determine new width and height based on the diagonal size
        new_size = int(
            max(width, height) * np.sqrt(2)
        )  # Make sure new_size is an integer

        # Create a rotation matrix with respect to the center of the new square
        self.rotation_matrix = cv2.getRotationMatrix2D(
            (float(new_size) / 2, float(new_size) / 2), -self.angle, 1.0
        )

        # Extract the object from the original mask and place it in a square matrix
        object_ = mask[y_min:y_max, x_min:x_max]
        square_object = np.zeros((new_size, new_size), dtype=np.uint8)
        y_offset = (new_size - height) // 2
        x_offset = (new_size - width) // 2
        square_object[y_offset : y_offset + height, x_offset : x_offset + width] = (
            object_
        )

        # Apply the rotation to the object
        rotated_object = cv2.warpAffine(
            square_object,
            self.rotation_matrix,
            (new_size, new_size),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )

        # Create a new mask and place the rotated object back into it
        new_mask = np.zeros(mask.shape, dtype=np.uint8)

        # Compute the placement of the rotated object in the new mask
        y_start = y_center - new_size // 2
        x_start = x_center - new_size // 2
        y_end = y_start + new_size
        x_end = x_start + new_size

        # Ensure the rotated object does not go out of bounds
        if x_start < 0:
            rotated_object = rotated_object[:, abs(x_start) :]
            x_start = 0
        if y_start < 0:
            rotated_object = rotated_object[abs(y_start) :, :]
            y_start = 0
        if x_end > mask.shape[1]:
            rotated_object = rotated_object[:, : -(x_end - mask.shape[1])]
            x_end = mask.shape[1]
        if y_end > mask.shape[0]:
            rotated_object = rotated_object[: -(y_end - mask.shape[0]), :]
            y_end = mask.shape[0]

        if rotated_object.size == 0:
            return new_mask

        # Place the rotated object back into the new mask
        new_mask[y_start:y_end, x_start:x_end] = rotated_object

        return new_mask

    def _check_overflow(self, mask: np.array, processed_mask: np.array) -> bool:
        before_area = np.sum(mask != 0)
        after_area = np.sum(processed_mask != 0)

        return after_area / before_area < 0.75

    def _get_base_prompt(self) -> str:
        return f"<ROTATE> <OBJECT> {self.angle:.2f}"

    def _get_transformation_matrix_wrt_object_center(self) -> np.array:
        return np.array(
            [
                [np.cos(np.radians(self.angle)), -np.sin(np.radians(self.angle)), 0],
                [np.sin(np.radians(self.angle)), np.cos(np.radians(self.angle)), 0],
                [0, 0, 1],
            ]
        )

    def _get_manually_generated_prompt(self) -> str:
        possible_prompts = [
            f"rotate the object by {-self.angle:.2f} degrees. my convention: counterclockwise is positive",
            f"rotate the object around its center by {-self.angle:.2f} degrees. my convention: counterclockwise is positive",
            f"apply a rotation of {-self.angle:.2f} degrees to the object. my convention: counterclockwise is positive",
            f"turn the object by {-self.angle:.2f} degrees. my convention: counterclockwise is positive",
            f"rotate the object at an angle of {-self.angle:.2f} degrees. my convention: counterclockwise is positive",
        ]

        # Additional prompts based on rotation direction
        if self.angle > 0:
            possible_prompts.append(
                f"rotate the object clockwise by {abs(self.angle):.2f} degrees"
            )
        elif self.angle < 0:
            possible_prompts.append(
                f"rotate the object counterclockwise by {abs(self.angle):.2f} degrees"
            )

        # Variations based on the angle of rotation
        # if self.angle > 90:
        #     possible_prompts.append("rotate the object drastically clockwise")
        # if self.angle < -90:
        #     possible_prompts.append("rotate the object drastically counterclockwise")

        return random.choice(possible_prompts)


# Test
if __name__ == "__main__":
    # Random basic shape test
    import utils.create_random_simple_shape_mask as mask_generator

    mask = mask_generator.create_shape_mask("crescent", (600, 900))

    rotate = Rotate(45)
    processed_mask = rotate.process(mask)
    prompt = rotate.get_prompt()

    cv2.imshow("Original Mask", mask)
    cv2.imshow("Processed Mask", processed_mask)
    print(prompt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
