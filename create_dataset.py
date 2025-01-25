import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from object_transformations.compose import Compose
from object_transformations.flip import Flip
from object_transformations.move import MoveByPixel, MoveByPercentage, MoveTo
from object_transformations.scale import (
    ScaleBy,
    ScaleAbsolutelyToPercentage,
    ScaleAbsolutelyToPixels,
)

# from object_transformations.rotate import Rotate
from utils.pascal_voc_parser import parse_voc


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
        default=1,
        help="Number of random transformations per an object in the image",
    )
    parser.add_argument(
        "--composition_probability",
        type=float,
        default=0.1,
        help="Probability of applying a composition of transformations",
    )
    parser.add_argument(
        "--save_path", type=str, help="Path to save the transformed images"
    )
    parser.add_argument(
        "--min_percentage_area",
        type=float,
        default=10,
        help="Minimum percentage area of objects to keep.",
    )
    parser.add_argument(
        "--max_percentage_area",
        type=float,
        default=80,
        help="Maximum percentage area of objects to keep.",
    )
    parser.add_argument(
        "--check_truncated",
        action="store_false",
        help="Don't include the truncated objects",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create the save path if it does not exist
    if args.save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    for voc_object in parse_voc(
        args.input_folder, remove_multiple_same_instance_images=True
    ):
        input_image = voc_object.image

        if args.save_path:
            save_folder = Path(voc_object.filename.split(".", 1)[0])
            save_folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_folder / f"base_image.png"), input_image)

        for obj in voc_object.objects:
            print(voc_object.filename)
            print("************************")

            input_instance_mask = obj.mask
            # Check if the object is truncated
            if args.check_truncated and obj.truncated:
                continue

            # Check if area percentage is within specified limits
            area = np.sum(input_instance_mask / 255)
            total_area = input_instance_mask.size
            percentage_area = (area / total_area) * 100
            if (
                percentage_area < args.min_percentage_area
                or percentage_area > args.max_percentage_area
            ):
                continue

            continue

            # To guarantee uniqueness of the transformations
            is_flip_applied = False

            for j in range(args.transform_count):
                if np.random.rand() < args.composition_probability:
                    # Composition of transformations with a certain probability
                    transformation = Compose()
                else:
                    # Random transformation among 4 different types
                    possible_transformations = [
                        ScaleBy(),
                        # ScaleAbsolutelyToPercentage(),
                        # ScaleAbsolutelyToPixels(),
                        MoveByPixel(),
                        # MoveByPercentage(),
                        # MoveTo(),
                    ]  # Rotate(), Sheer(),
                    if not is_flip_applied:
                        possible_transformations.append(Flip())

                    transformation = np.random.choice(possible_transformations)

                    is_flip_applied = isinstance(transformation, Flip)

                    # TODO: add support for ScaleAbsolutelyToPercentage, ScaleAbsolutelyToPixels, MoveByPercentage, MoveTo!!

                # Apply the transformation to the mask
                processed_mask = transformation.process(input_instance_mask)
                base_prompt, manually_generated_prompt = transformation.get_prompt()
                transformation_matrix = transformation.get_matrix()

                if args.save_path:
                    cv2.imwrite(
                        str(save_folder / f"transformed_{j}.png"),
                        processed_mask,
                    )

                    with open(save_folder / f"prompt_{j}.txt", "w") as f:
                        f.write(base_prompt)

                    with open(save_folder / f"prompt_human_like_{j}.txt", "w") as f:
                        f.write(manually_generated_prompt)

                    with open(save_folder / f"transformation_matrix_{j}.txt", "w") as f:
                        f.write(str(transformation_matrix))
                else:
                    print(
                        f"********** Image {voc_object.filename}, Object {obj.name}, Transform {j} **********"
                    )
                    print(f"Base Prompt: {base_prompt}")
                    print(
                        f"Manually Generated Human-Like Prompt: {manually_generated_prompt}"
                    )
                    print(f"Matrix: {transformation_matrix}")
                    cv2.imshow("Original Mask", input_instance_mask)
                    cv2.imshow("Transformed Mask", processed_mask)
                    cv2.imshow("Original Image", input_image)
                    cv2.waitKey(0)
