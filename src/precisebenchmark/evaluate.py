import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
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
        help="Path to your image folder containing one of the following: your edited images, your object mask predictions in the transformed image, your object mask predictions in the input image, or your bounding box predictions in the input image",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path of the .json file to save the evaluation results. Default: evaluation_results/evaluation_results_timeinfo.json",
    )
    parser.add_argument(
        "--hf_dataset_path",
        type=str,
        help="Path to the Hugging Face dataset for evaluation",
        default="monurcan/precise_benchmark_for_object_level_image_editing",
    )
    parser.add_argument(
        "--evaluation_mode",
        type=str,
        choices=[
            "gt_input_masks_vs_my_bounding_boxes",
            "gt_input_masks_vs_my_input_masks",
            "gt_edited_masks_vs_my_edited_masks",
            "gt_edited_masks_vs_my_edited_images",
        ],
        help="""
            There are 4 different evaluation modes.
            - gt_input_masks_vs_my_bounding_boxes: compares ground-truth object mask in the input images with the bounding box images in your input folder. (VLM in our paper.)
            - gt_input_masks_vs_my_input_masks: compares ground-truth object mask in the input image with the binary mask images in your input folder. (SAM in our paper.)
            - gt_edited_masks_vs_my_edited_masks: compares ground-truth object mask in the transformed image with the binary mask images in your input folder. (LLM in our paper.)
            - gt_edited_masks_vs_my_edited_images: compares ground-truth object mask in the transformed image with the binary mask extracted by GroundedSAM from the edited images in your input folder. This is the default mode. (Drawer in our paper.)
        """,
        default="gt_edited_masks_vs_my_edited_images",
    )

    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = (
            "evaluation_results/evaluation_results_" + str(time.time()) + ".json"
        )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    return args


def my_predictions(input_folder, sample_id):
    input_folder = Path(input_folder)
    corresponding_sample = next(input_folder.glob(f"{sample_id}.*"), None)

    if not corresponding_sample:
        raise FileNotFoundError(f"No corresponding sample found for {sample_id}")

    corresponding_sample = Image.open(corresponding_sample)

    return corresponding_sample


def extract_mask(image: Image.Image, obj_class: str, base_model) -> Image.Image:
    from autodistill.detection import CaptionOntology

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


def convert_mask_to_bbox(mask: Image.Image):
    # Convert the mask to a numpy array of type uint8
    mask_np = np.array(mask, dtype=np.uint8)

    # Check if the mask is empty (all pixels are zero)
    if np.all(mask_np == 0):
        return mask.copy()  # Return original mask if empty

    # Find rows and columns with non-zero pixels
    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)

    # Extract bounding box coordinates
    r_indices = np.where(rows)[0]
    c_indices = np.where(cols)[0]

    # Handle edge case where there's only a single row/column of non-zero pixels
    rmin, rmax = r_indices[[0, -1]]
    cmin, cmax = c_indices[[0, -1]]

    # Expand the region to cover the entire bounding box (inclusive)
    # Note: numpy slicing is exclusive on the upper bound, hence +1
    mask_np[rmin : rmax + 1, cmin : cmax + 1] = 255

    # Convert the modified numpy array back to a PIL Image
    return Image.fromarray(mask_np)


def compare_two_masks(gt_mask: Image.Image, other_mask: Image.Image):
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
    }


def meta_type_from_transformation(transformation: str):
    if transformation in {"MoveByPercentage", "MoveByPixel", "MoveTo"}:
        return "Move"
    elif transformation in {
        "ScaleAbsolutelyToPercentage",
        "ScaleAbsolutelyToPixels",
        "ScaleByPercentage",
    }:
        return "Scale"

    return transformation


def find_categorical_results(results):
    categorical_results = {
        "object_class": defaultdict(lambda: defaultdict(list)),
        "meta_transformation_type": defaultdict(lambda: defaultdict(list)),
        "transformation_type": defaultdict(lambda: defaultdict(list)),
        "summary": defaultdict(lambda: defaultdict(list)),
    }

    for _, sample_result in results.items():
        object_class = sample_result["object_class"]
        transformation_type = sample_result["transformation_type"]
        meta_transformation_type = sample_result["meta_transformation_type"]

        for metric, value in sample_result.items():
            if metric in [
                "object_class",
                "transformation_type",
                "meta_transformation_type",
            ]:
                continue

            categorical_results["object_class"][object_class][metric].append(value)
            categorical_results["transformation_type"][transformation_type][
                metric
            ].append(value)
            categorical_results["meta_transformation_type"][meta_transformation_type][
                metric
            ].append(value)
            categorical_results["summary"]["overall"][metric].append(value)

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


def main():
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

    if args.evaluation_mode == "gt_edited_masks_vs_my_edited_images":
        from autodistill.detection import CaptionOntology
        from autodistill_grounded_sam_2 import GroundedSAM2

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

    for gt_sample in tqdm(dataset, total=dataset_len):
        try:
            corresponding_sample = my_predictions(args.input_folder, gt_sample["id"])

            if args.evaluation_mode == "gt_edited_masks_vs_my_edited_images":
                my_prediction = extract_mask(
                    corresponding_sample, gt_sample["object_class"], base_model
                )
            else:
                my_prediction = corresponding_sample

            if args.evaluation_mode in [
                "gt_edited_masks_vs_my_edited_images",
                "gt_edited_masks_vs_my_edited_masks",
            ]:
                target_mask = gt_sample["edited_mask"]
            elif args.evaluation_mode in [
                "gt_input_masks_vs_my_input_masks",
                "gt_input_masks_vs_my_bounding_boxes",
            ]:
                target_mask = gt_sample["input_mask"]
            else:
                raise ValueError("Invalid evaluation mode")

            results["individual_results"][gt_sample["id"]] = {
                **compare_two_masks(target_mask, my_prediction),
                "object_class": gt_sample["object_class"],
                "transformation_type": gt_sample["transformation_type"],
                "meta_transformation_type": meta_type_from_transformation(
                    gt_sample["transformation_type"]
                ),
            }
        except Exception as e:
            print(f"Error processing sample {gt_sample['id']}: {e}")

    save_results(results, args.save_path)


if __name__ == "__main__":
    main()
