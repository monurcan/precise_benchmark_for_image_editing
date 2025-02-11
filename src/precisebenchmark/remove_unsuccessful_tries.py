#!/usr/bin/env python3
import os
import argparse


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


def main():
    parser = argparse.ArgumentParser(
        description="Remove files from input_folder that do not exist in target_folder."
    )
    parser.add_argument("--target_folder", help="Path to the target folder.")
    parser.add_argument(
        "--input_folder", help="Path to the input folder to be cleaned."
    )

    args = parser.parse_args()
    target_folder = args.target_folder
    input_folder = args.input_folder

    # Get file sets for both folders
    target_files = get_files(target_folder)
    input_files = get_files(input_folder)

    # Print counts before deletion
    print("Before deletion:")
    print(f"  Target folder ({target_folder}) file count: {len(target_files)}")
    print(f"  Input folder ({input_folder}) file count: {len(input_files)}")

    # Identify files in input_folder that do not exist in target_folder
    files_to_delete = input_files - target_files

    if files_to_delete:
        print(
            "\nDeleting the following files from input folder (not found in target folder):"
        )
        for f in files_to_delete:
            print(f"  {f}")
        delete_files(input_folder, files_to_delete)
    else:
        print(
            "\nNo files to delete. All files in the input folder exist in the target folder."
        )

    # Get updated file counts
    target_files_after = get_files(target_folder)
    input_files_after = get_files(input_folder)

    # Print counts after deletion
    print("\nAfter deletion:")
    print(f"  Target folder ({target_folder}) file count: {len(target_files_after)}")
    print(f"  Input folder ({input_folder}) file count: {len(input_files_after)}")


if __name__ == "__main__":
    main()
