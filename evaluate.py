from datasets import load_dataset
from PIL import Image
import argparse
import time
import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate your outputs on our benchmark."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the edited images or edited masks folder",
    )
    parser.add_argument(
        "--evaluate_reasoning_only",
        action="store_true",
        help="Only evaluate reasoning steps. Input folder should be a folder of edited masks, not images.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to save the evaluation results",
    )
    parser.add_argument(
        "--hf_dataset_path",
        type=str,
        help="Path to the Hugging Face dataset for evaluation",
        default="monurcan/precise_benchmark_for_object_level_image_editing",
    )

    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = (
            "evaluation_results/evaluation_results_" + str(time.time()) + ".json"
        )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    return args


def save_results(results, save_path):
    with open(save_path, "w") as f:
        json.dump(results, f)

    print(results)  # ["categorical_results"] show without image based analysis


def edited_samples(input_folder, sample_id):
    input_folder = Path(input_folder)
    corresponding_sample = next(input_folder.glob(f"{sample_id}.*"), None)

    if not corresponding_sample:
        raise FileNotFoundError(f"No corresponding sample found for {sample_id}")

    corresponding_sample = Image.open(corresponding_sample)

    return corresponding_sample


def compare_two_masks(gt_obj: dict, other_mask: Image):
    gt_mask = gt_obj["edited_mask"]

    # Convert PIL images to NumPy arrays
    gt_mask = np.array(gt_mask, dtype=np.float32)
    other_mask = np.array(other_mask, dtype=np.float32)

    # Check if both masks have the same shape
    if gt_mask.shape != other_mask.shape:
        raise ValueError("The shapes of the original and edited masks do not match.")

    # Ensure binary masks
    threshold_gt_mask = (gt_mask.max() + gt_mask.min()) / 2
    threshold_other_mask = (other_mask.max() + other_mask.min()) / 2
    gt_mask = (gt_mask > threshold_gt_mask).astype(np.float32)
    other_mask = (other_mask > threshold_other_mask).astype(np.float32)

    # Compute Mean Absolute Error (MAE)
    mae = np.mean(np.abs(gt_mask - other_mask))

    # Compute Intersection over Union (IoU)
    intersection = np.logical_and(gt_mask, other_mask).sum()
    union = np.logical_or(gt_mask, other_mask).sum()
    miou = intersection / union if union > 0 else 0.0

    return {
        "miou": miou,
        "mae": mae,
    }


if __name__ == "__main__":
    args = parse_args()

    dataset = load_dataset(
        args.hf_dataset_path,
        split="train",
        streaming=True,
    )

    results = {}

    for sample in tqdm(dataset):
        try:
            corresponding_sample = edited_samples(args.input_folder, sample["id"])

            if not args.evaluate_reasoning_only:
                edited_masks = extract_mask(corresponding_sample)
            else:
                edited_masks = corresponding_sample

            results[sample["id"]] = compare_two_masks(sample, edited_masks)
        except Exception as e:
            print(f"Error processing sample {sample['id']}: {e}")

    save_results(results, args.save_path)
