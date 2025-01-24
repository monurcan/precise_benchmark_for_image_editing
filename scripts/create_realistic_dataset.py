import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from object_transformations.compose import Compose
from object_transformations.flip import Flip
from object_transformations.move import Move
from object_transformations.rotate import Rotate
from object_transformations.scale import Scale


def load_masks_from_folder(folder_path):
    """Load all binary mask images from the folder."""
    folder = Path(folder_path)
    mask_files = list(folder.glob("*.png"))  # Assuming binary masks are in PNG format
    masks = [
        cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) for mask_file in mask_files
    ]
    return masks, mask_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply transformations to binary masks."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder containing binary mask images",
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
    args = parser.parse_args()

    # Create the save path if it does not exist
    if args.save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    # Load binary masks from the input folder
    masks, mask_files = load_masks_from_folder(args.input_folder)

    for i in tqdm(range(len(masks))):
        # Select a mask from the loaded masks
        mask = masks[i]
        mask_filename = mask_files[i].name

        if args.save_path:
            save_folder = Path(args.save_path) / f"sample_{i}"
            save_folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_folder / f"base.png"), mask)

        for j in range(args.transform_count):
            # Random transformation among 4 different types
            transformations = [Flip(), Rotate(), Scale(), Move()]

            # Apply a composition of transformations with a certain probability
            if np.random.rand() < args.composition_probability:
                transformation = Compose()
            else:
                transformation = np.random.choice(transformations)

            # Apply the transformation to the mask
            processed_mask = transformation.process(mask)
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
                print(f"********** Image {i}, Transform {j} **********")
                print(f"Base Prompt: {base_prompt}")
                print(
                    f"Manually Generated Human-Like Prompt: {manually_generated_prompt}"
                )
                print(f"Matrix: {transformation_matrix}")
                cv2.imshow(f"Original Mask", mask)
                cv2.imshow(f"Transformed Mask", processed_mask)
                cv2.waitKey(0)
