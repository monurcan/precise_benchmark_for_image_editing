import os

from huggingface_hub import HfApi, HfFolder
from PIL import Image
from tqdm import tqdm

import datasets


def load_image(image_path):
    return Image.open(image_path)


def sample_generator(dataset_folder):
    for sample_dir in tqdm(sorted(os.listdir(dataset_folder))):
        sample_path = os.path.join(dataset_folder, sample_dir)

        if not os.path.isdir(sample_path):
            continue

        # Load base image (input_image)
        input_image = load_image(os.path.join(sample_path, "base_image.png"))
        input_mask = load_image(os.path.join(sample_path, "base.png"))

        with open(os.path.join(sample_path, "object_class.txt"), "r") as f:
            object_class = f.read().strip()

        # Iterate over transformations (edit_prompt and edited_mask)
        for i in range(
            len([f for f in os.listdir(sample_path) if f.startswith("transformed_")])
        ):
            edited_mask_path = os.path.join(sample_path, f"transformed_{i}.png")
            edit_prompt_path = os.path.join(sample_path, f"prompt_human_like_{i}.txt")
            edit_gpt_prompt_path = os.path.join(sample_path, f"prompt_gpt_{i}.txt")
            edit_base_prompt_path = os.path.join(sample_path, f"prompt_{i}.txt")
            transformation_type_path = os.path.join(
                sample_path, f"transformation_type_{i}.txt"
            )
            transformation_matrix_path = os.path.join(
                sample_path, f"transformation_matrix_{i}.txt"
            )

            with open(edit_base_prompt_path, "r") as f:
                edit_base_prompt = f.read().strip()

            with open(transformation_matrix_path, "r") as f:
                transformation_matrix = f.read().strip()

            with open(transformation_type_path, "r") as f:
                transformation_type = f.read().strip()

            # Load edited image
            edited_mask = load_image(edited_mask_path)

            # Load prompt
            all_prompts = []
            if os.path.exists(edit_gpt_prompt_path):
                with open(edit_gpt_prompt_path, "r") as f:
                    all_prompts = f.read().strip().splitlines()

            with open(edit_prompt_path, "r") as f:
                all_prompts.append(f.read().strip())

            for example_prompt in all_prompts:
                yield {
                    "id": f"{sample_dir}_transform_{i}",
                    "input_image": input_image,
                    "edit_prompt": example_prompt,
                    "input_mask": input_mask,
                    "edited_mask": edited_mask,
                    "object_class": object_class,
                    "edit_base_prompt": edit_base_prompt,
                    "transformation_matrix": transformation_matrix,
                    "transformation_type": transformation_type,
                }


def create_hf_dataset(dataset_folder, output_hf_dataset_location, upload_to_hf=None):
    # Create a HuggingFace dataset from the generator
    hf_dataset = datasets.Dataset.from_generator(
        lambda: sample_generator(dataset_folder)
    )

    # Define the format and save to disk
    hf_dataset = hf_dataset.cast_column("input_image", datasets.Image())
    hf_dataset = hf_dataset.cast_column("input_mask", datasets.Image())
    hf_dataset = hf_dataset.cast_column("edited_mask", datasets.Image())

    hf_dataset.save_to_disk(output_hf_dataset_location)
    print(f"HuggingFace dataset created at {output_hf_dataset_location}")

    # Upload to Hugging Face hub if flag is set
    if upload_to_hf:
        dataset_name = os.path.basename(output_hf_dataset_location)
        print(f"Uploading dataset to Hugging Face hub under the name: {dataset_name}")

        hf_dataset.push_to_hub(upload_to_hf)
        print(f"Dataset successfully uploaded to {upload_to_hf}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a HuggingFace dataset from a dataset in our format."
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="Path to the dataset folder in our format.",
    )
    parser.add_argument(
        "--output_hf_dataset_location",
        type=str,
        required=True,
        help="Output location for the HuggingFace dataset.",
    )
    parser.add_argument(
        "--upload_to_dataset",
        type=str,
        help="Hugging Face Dataset repo ID to upload the dataset (e.g., 'username/dataset_name').",
    )

    args = parser.parse_args()

    create_hf_dataset(
        args.dataset_folder, args.output_hf_dataset_location, args.upload_to_dataset
    )
