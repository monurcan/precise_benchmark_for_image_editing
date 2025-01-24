import random

import cv2
import numpy as np

from object_transformations.object_transformation import ObjectTransformation


class Flip(ObjectTransformation):
    def __init__(self):
        super().__init__()

    def _process_object(self, mask: np.array) -> np.array:
        # Find object boundaries
        x_min, x_max, y_min, y_max = self.get_object_boundaries(mask)

        self.object_width = x_max - x_min

        flipped_object = mask[y_min:y_max, x_min:x_max]
        flipped_object = cv2.flip(flipped_object, 1)

        new_mask = np.zeros(mask.shape, dtype=np.uint8)
        new_mask[y_min:y_max, x_min:x_max] = flipped_object

        return new_mask

    def _check_overflow(self, mask: np.array, processed_mask: np.array) -> bool:
        return False

    def _get_base_prompt(self) -> str:
        return "<FLIP> <OBJECT>"

    def _get_transformation_matrix_wrt_object_center(self) -> np.array:
        return np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])

    def _get_manually_generated_prompt(self) -> str:
        possible_prompts = [
            "flip the object horizontally",
            "mirror the object along the vertical axis",
            "reflect the object from left to right",
            "reverse the object along its horizontal axis",
            "apply a horizontal flip to the object",
            "create a mirror image of the object horizontally",
            "invert the object horizontally",
            "swap the object's left and right sides",
            "perform a horizontal reflection on the object",
            "flip the object over the y-axis",
            "turn the object around horizontally",
            "make the object a horizontal mirror of itself",
            "reverse the object's horizontal orientation",
            "mirror the object horizontally for a flipped view",
            "horizontally invert the object's position",
            "create a horizontal reflection of the object",
        ]

        return random.choice(possible_prompts)


# Test
if __name__ == "__main__":
    # Realistic test
    # mask = cv2.imread("realistic_mask_test.png", cv2.IMREAD_GRAYSCALE)
    # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Random basic shape test
    import object_transformations.utils.create_random_simple_shape_mask as mask_generator

    mask = mask_generator.create_shape_mask("crescent", (512, 512))

    flip = Flip()
    processed_mask = flip.process(mask)
    prompt = flip.get_prompt()

    cv2.imshow("Original Mask", mask)
    cv2.imshow("Processed Mask", processed_mask)
    print(prompt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
