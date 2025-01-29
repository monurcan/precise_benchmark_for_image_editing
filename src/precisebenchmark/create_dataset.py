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
from object_transformations.reasoning import Reasoning
from object_transformations.shear import Shear
from utils.pascal_voc_parser import parse_voc
from utils.set_seeds import set_seeds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply geometric transformations to the instance masks."
    )

    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder containing the dataset in PASCAL VOC format",
    )
    parser.add_argument(
        "--transform_count",
        type=int,
        default=2,
        help="Number of random transformations per an object in the image",
    )
    parser.add_argument(
        "--composition_probability",
        type=float,
        default=0.05,
        help="Probability of applying a composition of transformations",
    )
    parser.add_argument(
        "--reasoning_probability",
        type=float,
        default=0.02,
        help="Probability of applying a transformations requiring reasoning. Example: make the cat's height as big as the dog's height.",
    )
    parser.add_argument(
        "--save_path", type=str, help="Path to save the transformed images"
    )
    parser.add_argument(
        "--min_percentage_area",
        type=float,
        default=15,
        help="Minimum percentage area of objects to keep.",
    )
    parser.add_argument(
        "--max_percentage_area",
        type=float,
        default=80,
        help="Maximum percentage area of objects to keep.",
    )
    parser.add_argument(
        "--dont_check_truncated",
        action="store_true",
        help="Don't include the truncated objects",
    )
    parser.add_argument(
        "--allow_nonsquare_images",
        action="store_true",
        help="Allow non-square images. If you specify this option, rectangle images will be used with their original dimensions. Otherwise, we will convert it to a square image.",
    )

    return parser.parse_args()


def is_obj_valid(
    obj, check_truncated: bool, min_percentage_area: float, max_percentage_area: float
):
    # Check if the object is truncated
    if check_truncated and obj.truncated:
        return False

    # Check if area percentage is within specified limits
    area = np.sum(obj.mask / 255)
    total_area = obj.mask.size
    percentage_area = (area / total_area) * 100
    if percentage_area < min_percentage_area or percentage_area > max_percentage_area:
        return False

    return True


def get_transformed_masks(
    obj,
    transform_count: int,
    composition_probability: float,
    reasoning_probability: float,
    all_objects,
):
    print(all_objects)

    result = []

    # To guarantee uniqueness of the transformations
    is_flip_applied = False

    for j in range(transform_count):
        random_btw_0_1 = np.random.rand()

        if random_btw_0_1 < composition_probability:
            # Composition of transformations with a certain probability
            transformation = Compose()
        elif len(all_objects) > 1 and (
            random_btw_0_1 < reasoning_probability + composition_probability
        ):
            transformation = Reasoning(obj, all_objects)
        else:
            # Random transformation among 4 different types
            possible_transformations = [
                ScaleBy(),
                ScaleAbsolutelyToPercentage(),
                ScaleAbsolutelyToPixels(),
                MoveByPixel(),
                MoveByPercentage(),
                MoveTo(),
                Rotate(),
                Shear(),
            ]
            if not is_flip_applied:
                possible_transformations.append(Flip())

            transformation = np.random.choice(possible_transformations)

            is_flip_applied = isinstance(transformation, Flip)

        # Apply the transformation to the mask
        processed_mask = transformation.process(obj.mask)
        base_prompt, manually_generated_prompt = transformation.get_prompt()
        transformation_matrix = transformation.get_matrix()

        result.append(
            {
                "obj_name": obj.name,
                "transform_j": j,
                "input_mask": obj.mask,
                "processed_mask": processed_mask,
                "base_prompt": base_prompt,
                "manually_generated_prompt": manually_generated_prompt,
                "transformation_matrix": transformation_matrix,
                "transformation_type": type(transformation).__name__,
                "object_class": obj.name,
            }
        )

    return result


def better_object_class(obj_class):
    if obj_class == "diningtable":
        return "dining table"
    if obj_class == "pottedplant":
        return "potted plant"
    if obj_class == "tvmonitor":
        return "tv monitor"
    return obj_class


def better_manual_prompt(prompt, obj_class):
    prompt = prompt.replace("the object", f"the {better_object_class(obj_class)}")
    return prompt


def save_to_disk(
    filename,
    save_path,
    input_image,
    input_mask,
    obj_name,
    transform_j,
    processed_mask,
    base_prompt,
    manually_generated_prompt,
    transformation_matrix,
    transformation_type,
    object_class,
):
    object_class = better_object_class(object_class)
    manually_generated_prompt = better_manual_prompt(
        manually_generated_prompt, object_class
    )

    if save_path:
        save_folder = Path(save_path) / Path(filename)
        save_folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_folder / f"base_image.png"), input_image)
        cv2.imwrite(str(save_folder / f"base.png"), input_mask)
        cv2.imwrite(
            str(save_folder / f"transformed_{transform_j}.png"),
            processed_mask,
        )

        with open(save_folder / f"prompt_{transform_j}.txt", "w") as f:
            f.write(base_prompt)

        with open(save_folder / f"prompt_human_like_{transform_j}.txt", "w") as f:
            f.write(manually_generated_prompt)

        with open(save_folder / f"transformation_matrix_{transform_j}.txt", "w") as f:
            f.write(str(transformation_matrix))

        with open(save_folder / f"transformation_type_{transform_j}.txt", "w") as f:
            f.write(transformation_type)

        with open(save_folder / f"object_class.txt", "w") as f:
            f.write(object_class)
    else:
        print(
            f"********** Image {filename}, Object {obj_name}, Transform {transform_j} **********"
        )
        print(f"Base Prompt: {base_prompt}")
        print(f"Manually Generated Human-Like Prompt: {manually_generated_prompt}")
        print(f"Matrix: {transformation_matrix}")
        print(f"Transformation Type: {transformation_type}")
        print(f"Object Class: {object_class}")
        cv2.imshow("Original Mask", input_mask)
        cv2.imshow("Transformed Mask", processed_mask)
        cv2.imshow("Original Image", input_image)
        cv2.waitKey(0)


if __name__ == "__main__":
    # Set the seeds for reproducibility
    set_seeds(19)

    args = parse_args()

    # Create the save path if it does not exist
    if args.save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    for voc_object in parse_voc(
        args.input_folder,
        remove_multiple_same_instance_images=True,
        allow_nonsquare_images=args.allow_nonsquare_images,
    ):
        for obj_i, obj in enumerate(voc_object.objects):
            if not is_obj_valid(
                obj,
                not args.dont_check_truncated,
                args.min_percentage_area,
                args.max_percentage_area,
            ):
                continue

            try:
                transformed_masks = get_transformed_masks(
                    obj,
                    args.transform_count,
                    args.composition_probability,
                    args.reasoning_probability,
                    voc_object.objects,
                )
            except Exception as e:
                print(f"Error processing object {obj.name}: {e}")
                continue

            for processed_result in transformed_masks:
                save_to_disk(
                    save_path=args.save_path,
                    filename=voc_object.filename.split(".", 1)[0] + f"_{obj_i}",
                    input_image=voc_object.image,
                    **processed_result,
                )
