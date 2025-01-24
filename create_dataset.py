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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply geometric transformations to the binary masks."
    )

    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder containing the dataset in FSS1000 format",
    )
    parser.add_argument(
        "--transform_count",
        type=int,
        default=3,
        help="Number of random transformations per image",
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

    return parser.parse_args()


def load_images_from_folder(folder_path):
    """Load all images from the folder."""
    folder = Path(folder_path)
    mask_paths = list(folder.rglob("*.png"))
    images_paths = [mask_path.with_suffix(".jpg") for mask_path in mask_paths]

    return zip(mask_paths, images_paths)


if __name__ == "__main__":
    args = parse_args()

    # Create the save path if it does not exist
    if args.save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    for mask_path, image_path in load_images_from_folder(args.input_folder):
        input_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        input_image = cv2.imread(str(image_path))

        if args.save_path:
            save_folder = mask_path.parent.name + "_" + mask_path.stem
            save_folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_folder / f"base.png"), input_mask)
            cv2.imwrite(str(save_folder / f"base_image.png"), input_image)

        for j in range(args.transform_count):
            if np.random.rand() < args.composition_probability:
                # Composition of transformations with a certain probability
                transformation = Compose()
            else:
                # Random transformation among 4 different types
                transformation = np.random.choice(
                    [
                        Flip(),
                        ScaleBy(),
                        ScaleAbsolutelyToPercentage(),
                        ScaleAbsolutelyToPixels(),
                        MoveByPixel(),
                        MoveByPercentage(),
                        MoveTo(),
                    ]  # Rotate(), Sheer(),
                )

            # Apply the transformation to the mask
            processed_mask = transformation.process(input_mask)
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
                    f"********** Image {mask_path.parent.name}_{mask_path.stem}, Transform {j} **********"
                )
                print(f"Base Prompt: {base_prompt}")
                print(
                    f"Manually Generated Human-Like Prompt: {manually_generated_prompt}"
                )
                print(f"Matrix: {transformation_matrix}")
                cv2.imshow("Original Mask", input_mask)
                cv2.imshow("Transformed Mask", processed_mask)
                cv2.imshow("Original Image", input_image)
                cv2.waitKey(0)
