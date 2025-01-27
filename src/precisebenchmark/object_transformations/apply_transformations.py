from object_transformations.compose import Compose


class ApplyMaskTransformations:
    def __call__(self, transform_dict: dict) -> dict:
        for obj, transforms in transform_dict.items():
            input_mask = transforms["mask"]
            mask_transforms = transforms["mask_transforms"]

            output_mask = input_mask
            for transform in mask_transforms:
                output_mask = transform.process(output_mask)

            transforms["modified_mask"] = output_mask

            compose = Compose(mask_transforms)
            compose.process(input_mask)
            transforms["decomposition"] = (
                compose.decompose_transformation_matrix_as_flip_shift_resize_rotation()
            )
            # del transforms["mask_transforms"]

        return transform_dict
