from object_transformations.affine import AffineTransform
import numpy as np
import random


class Shear(AffineTransform):
    def __init__(self, shear_coefficients: tuple = None):
        if shear_coefficients is None:
            # TODO: random values between 0.0 and 0.15
            self.shear_x = np.random.rand() * 0.15
            self.shear_y = np.random.rand() * 0.15
        else:
            self.shear_x, self.shear_y = shear_coefficients

    def _process_object(self, mask: np.array) -> np.array:
        affine_mtx_wrt_object_center = np.array(
            [
                [
                    1,
                    self.shear_x,
                    0,
                ],
                [
                    self.shear_y,
                    1,
                    0,
                ],
                [0, 0, 1],
            ]
        )

        super().__init__(affine_mtx_wrt_object_center)

        return super()._process_object(mask)

    def _get_manually_generated_prompt(self) -> str:
        possible_prompts = [
            f"Shear the object by using the following shear coefficients {self.shear_x:.2f} along the x-axis and {self.shear_y:.2f} along the y-axis.",
            f"Apply a shear transformation to the object with shear coefficients {self.shear_x:.2f} in x and {self.shear_y:.2f} in y.",
            f"Transform the object by shearing it with shear coefficients {self.shear_x:.2f} in x and {self.shear_y:.2f} in y.",
            f"Shear the object with shear coefficients {self.shear_x:.2f} in x and {self.shear_y:.2f} in y.",
            f"Use the shear coefficients {self.shear_x:.2f} in x and {self.shear_y:.2f} in y to shear the object.",
        ]

        return random.choice(possible_prompts)


# Test
if __name__ == "__main__":
    # Realistic test
    # mask = cv2.imread("realistic_mask_test.png", cv2.IMREAD_GRAYSCALE)
    # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Random basic shape test
    import utils.create_random_simple_shape_mask as mask_generator
    import cv2

    mask = mask_generator.create_shape_mask("random", (512, 512))

    scale = Shear()

    processed_mask = scale.process(mask)

    cv2.imshow("Original Mask", mask)
    cv2.imshow("Processed Mask", processed_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
