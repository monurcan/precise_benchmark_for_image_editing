import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam_2 import GroundedSAM2
from datasets import load_dataset
from PIL import Image
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
        help="Path of the .json file to save the evaluation results",
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


def edited_samples(input_folder, sample_id):
    input_folder = Path(input_folder)
    corresponding_sample = next(input_folder.glob(f"{sample_id}.*"), None)

    if not corresponding_sample:
        raise FileNotFoundError(f"No corresponding sample found for {sample_id}")

    corresponding_sample = Image.open(corresponding_sample)

    return corresponding_sample


def extract_mask(image: Image.Image, obj_class: str, base_model) -> Image.Image:

    base_model.ontology = CaptionOntology(
        {
            obj_class: obj_class,
        }
    )

    predictions = base_model.predict(image)

    if predictions.mask.size == 0:
        results_mask = np.zeros_like(image)
    else:
        results_mask = predictions.mask[0].astype(np.uint8) * 255

    return Image.fromarray(results_mask)


def compare_two_masks(gt_obj: dict, other_mask: Image.Image):
    gt_mask = gt_obj["edited_mask"]

    # Convert PIL images to NumPy arrays
    gt_mask = np.array(gt_mask, dtype=np.float32)
    other_mask = np.array(other_mask, dtype=np.float32)

    # Check if both masks have the same shape
    if gt_mask.shape != other_mask.shape:
        raise ValueError(
            f"The shapes of the original and edited masks do not match. gt_mask.shape = {gt_mask.shape}, other_mask.shape = {other_mask.shape}"
        )

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
    iou = intersection / union if union > 0 else 0.0

    return {
        "iou": float(iou),
        "mae": float(mae),
        "object_class": gt_obj["object_class"],
        "transformation_type": gt_obj["transformation_type"],
    }


def find_categorical_results(results):
    categorical_results = {
        "object_class": defaultdict(lambda: defaultdict(list)),
        "transformation_type": defaultdict(lambda: defaultdict(list)),
    }

    for _, sample_result in results.items():
        object_class = sample_result["object_class"]
        transformation_type = sample_result["transformation_type"]

        for metric, value in sample_result.items():
            if metric in ["object_class", "transformation_type"]:
                continue

            categorical_results["object_class"][object_class][metric].append(value)
            categorical_results["transformation_type"][transformation_type][
                metric
            ].append(value)

    # Average
    for category, category_results in categorical_results.items():
        for category_value, category_value_results in category_results.items():
            for metric, values in category_value_results.items():
                categorical_results[category][category_value][metric] = float(
                    np.mean(values)
                )
                number_of_samples = len(values)
            categorical_results[category][category_value][
                "number_of_samples"
            ] = number_of_samples

    return categorical_results


def save_results(results, save_path):
    results["categorical_results"] = find_categorical_results(
        results["individual_results"]
    )
    print("=====================================")
    print("Evaluation finished.")
    print("=====================================")
    print(json.dumps(results["categorical_results"], indent=4))

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print("=====================================")
    print("Saved detailed results to " + save_path)
    print("=====================================")


if __name__ == "__main__":
    args = parse_args()

    # Load the dataset
    dataset = load_dataset(
        args.hf_dataset_path,
        split="train",
    )
    dataset_len = len(dataset)
    dataset = dataset.to_iterable_dataset()

    # TODO: remove!
    # dataset_len = 10
    # dataset = dataset.take(dataset_len)
    ###

    # Initialize the GroundedSAM model
    base_model = GroundedSAM2(
        ontology=CaptionOntology(
            {
                "object": "object",
            }
        ),
        model="Grounding DINO",
    )

    results = {"individual_results": {}}

    for sample in tqdm(dataset, total=dataset_len):
        try:
            corresponding_sample = edited_samples(args.input_folder, sample["id"])

            if not args.evaluate_reasoning_only:
                edited_masks = extract_mask(
                    corresponding_sample, sample["object_class"], base_model
                )
            else:
                edited_masks = corresponding_sample

            results["individual_results"][sample["id"]] = compare_two_masks(
                sample, edited_masks
            )
        except Exception as e:
            print(f"Error processing sample {sample['id']}: {e}")

    save_results(results, args.save_path)
