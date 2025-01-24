import random

import cv2
import numpy as np

from object_transformations.object_transformation import ObjectTransformation


class AffineTransform(ObjectTransformation):
    def __init__(self, affine_mtx_wrt_object_center: np.array = None):
        super().__init__()
        if affine_mtx_wrt_object_center is None:
            self.affine_mtx_wrt_object_center = np.array(
                [
                    [
                        random.uniform(0.8, 1.2),
                        random.uniform(-0.2, 0.2),
                        random.uniform(0, 600),
                    ],
                    [
                        random.uniform(-0.2, 0.2),
                        random.uniform(0.8, 1.2),
                        random.uniform(0, 900),
                    ],
                    [0, 0, 1],
                ]
            )
        else:
            self.affine_mtx_wrt_object_center = affine_mtx_wrt_object_center

    def _process_object(self, mask: np.array) -> np.array:
        # Find object boundaries
        x_min, x_max, y_min, y_max = self.get_object_boundaries(mask)
        x_center, y_center = (x_min + x_max) // 2, (y_min + y_max) // 2

        T_move_current_center_to_origin = np.array(
            [
                [1, 0, -x_center],
                [0, 1, -y_center],
                [0, 0, 1],
            ]
        )
        T_move_origin_to_current_center = np.array(
            [
                [1, 0, x_center],
                [0, 1, y_center],
                [0, 0, 1],
            ]
        )
        T_wrt_image_center = (
            T_move_origin_to_current_center
            @ self.affine_mtx_wrt_object_center
            @ T_move_current_center_to_origin
        )

        # Apply the transformation to the object
        transformed_mask = cv2.warpAffine(
            mask,
            T_wrt_image_center[:2],
            (mask.shape[1], mask.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )

        return transformed_mask

    def _check_overflow(self, mask: np.array, processed_mask: np.array) -> bool:
        before_area = np.sum(mask != 0)
        after_area = np.sum(processed_mask != 0)

        return after_area / before_area < 0.6

    def _get_base_prompt(self) -> str:
        return f"<AFFINE> <OBJECT> {self.affine_mtx_wrt_object_center.tolist()}"

    def _get_transformation_matrix_wrt_object_center(self) -> np.array:
        return self.affine_mtx_wrt_object_center

    def _get_manually_generated_prompt(self) -> str:
        return f"apply an affine transformation to the object {self.affine_mtx_wrt_object_center.tolist()}"


# Test
if __name__ == "__main__":
    # Random basic shape test
    import object_transformations.utils.create_random_simple_shape_mask as mask_generator

    mask = mask_generator.create_shape_mask("crescent", (600, 900))

    # affine_mtx = np.array([[1.0, 0, 200], [0, 1.0, 100], [0, 0, 1]])
    # affine_transform = AffineTransform(affine_mtx)
    # processed_mask = affine_transform.process(mask)
    # base_prompt, _ = affine_transform.get_prompt()
    # print(base_prompt)
    # cv2.imshow("Original Mask", mask)
    # cv2.imshow("Processed Mask", processed_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    from object_transformations.compose import Compose
    from object_transformations.flip import Flip
    from object_transformations.move import MoveByPixel, MoveByPercentage, MoveTo
    from object_transformations.rotate import Rotate
    from object_transformations.scale import (
        ScaleBy,
        ScaleAbsolutelyToPercentage,
        ScaleAbsolutelyToPixels,
    )

    composition = Compose()
    processed_mask = composition.process(mask)
    base_prompt, _ = composition.get_prompt()
    composition_mtx = composition.get_matrix()
    print(base_prompt)
    cv2.imshow("Original Mask", mask)
    cv2.imshow("Processed Mask", processed_mask)
    print(composition.decompose_transformation_matrix_as_flip_shift_resize_rotation())

    composition_as_affine = AffineTransform(composition_mtx)
    processed_mask_affine = composition_as_affine.process(mask)
    base_prompt_affine, _ = composition_as_affine.get_prompt()
    print(base_prompt_affine)
    cv2.imshow("Original Mask", mask)
    cv2.imshow("Processed Mask (Affine)", processed_mask_affine)

    # merge into seperate channels
    visualization = cv2.merge([processed_mask, processed_mask, processed_mask_affine])
    cv2.imshow("Visualization", visualization)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
