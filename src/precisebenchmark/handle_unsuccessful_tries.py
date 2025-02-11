import argparse
import os
import cv2
import numpy as np
from datasets import load_dataset


def get_files(folder):
    """
    Returns a set of file names (not including directories) in the given folder.
    """
    try:
        return {
            f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))
        }
    except FileNotFoundError:
        print(f"Error: Folder '{folder}' not found.")
        exit(1)


def delete_files(folder, files_to_delete):
    """
    Deletes files (by name) from the specified folder.
    """
    for filename in files_to_delete:
        file_path = os.path.join(folder, filename)
        try:
            os.remove(file_path)
            # print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Could not delete {file_path}: {e}")


def use_input_image(folder, files_to_delete, input_images_folder):
    for filename in files_to_delete:
        filename_no_ext = filename.split(".", 1)[0]
        filename_with_img_obj = "_".join(filename_no_ext.split("_")[:3])
        input_image_path = (
            f"{input_images_folder}/{filename_with_img_obj}/base_image.png"
        )
        file_path = os.path.join(folder, filename)
        img = cv2.imread(input_image_path)
        cv2.imwrite(file_path, img)


def use_white_image(folder, files_to_delete):
    for filename in files_to_delete:
        file_path = os.path.join(folder, filename)
        img = np.ones((512, 512, 3), np.uint8) * 255
        cv2.imwrite(file_path, img)


def main():
    parser = argparse.ArgumentParser(
        description="Handle files from input_folder that do not exist in target_folder."
    )
    parser.add_argument("--target_folder", help="Path to the target folder.")
    parser.add_argument(
        "--input_folder", help="Path to the input folder to be cleaned."
    )
    parser.add_argument(
        "--operation_type",
        help="Type of operation to be performed. Options: delete, white or inputimage",
    )
    parser.add_argument(
        "--input_images_folder",
        help="Path to the input images folder to be used in case of inputimage operation.",
    )

    args = parser.parse_args()
    target_folder = args.target_folder
    input_folder = args.input_folder

    # Check if input_images_folder is required for inputimage operation
    if args.operation_type == "inputimage" and args.input_images_folder is None:
        print("Error: input_images_folder is required for inputimage operation.")
        exit(1)

    # Get file sets for both folders
    target_files = get_files(target_folder)
    input_files = get_files(input_folder)

    # Print counts before deletion
    print("Before handling:")
    print(f"  Target folder ({target_folder}) file count: {len(target_files)}")
    print(f"  Input folder ({input_folder}) file count: {len(input_files)}")

    # Identify files in input_folder that do not exist in target_folder
    if args.operation_type == "delete":
        files_to_delete = input_files - target_files
    else:
        files_to_delete = target_files - input_files

    if files_to_delete:
        print(
            "\nDeleting the following files from input folder (not found in target folder):"
        )
        for f in files_to_delete:
            print(f"  {f}")

        if args.operation_type == "delete":
            delete_files(input_folder, files_to_delete)
        elif args.operation_type == "inputimage":
            use_input_image(input_folder, files_to_delete, args.input_images_folder)
        elif args.operation_type == "white":
            use_white_image(input_folder, files_to_delete)
    else:
        print(
            "\nNo files to delete. All files in the input folder exist in the target folder."
        )

    # Get updated file counts
    target_files_after = get_files(target_folder)
    input_files_after = get_files(input_folder)

    # Print counts after deletion
    print("\nAfter handling:")
    print(f"  Target folder ({target_folder}) file count: {len(target_files_after)}")
    print(f"  Input folder ({input_folder}) file count: {len(input_files_after)}")


if __name__ == "__main__":
    main()
