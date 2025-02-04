#!/usr/bin/env python3
import os
import json
import glob
import argparse
from tabulate import tabulate


def load_json_files(directory):
    """
    Loads all JSON files from the given directory and returns a dict:
      { filename: json_data["categorical_results"], ... }
    Files that do not have a "categorical_results" field are skipped.
    """
    file_data = {}
    json_files = glob.glob(os.path.join(directory, "*.json"))
    if not json_files:
        print(f"No JSON files found in {directory}")
    for filepath in json_files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if "categorical_results" not in data:
                    print(
                        f"Warning: {filepath} does not contain 'categorical_results'. Skipping."
                    )
                    continue
                filename = os.path.basename(filepath)
                file_data[filename] = data["categorical_results"]
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    return file_data


def format_value(value):
    """
    Format a value with 2 decimal places if it is numeric.
    """
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return value


def generate_comparison_tables(file_data):
    """
    Generates comparison tables automatically for each main key in the "categorical_results".
    For each main key, it determines all subkeys (rows) and all metric names (columns) automatically.
    Then, for each metric it creates a table where rows are subkeys and columns are JSON files.
    """
    output_lines = []

    # Determine union of main keys across all files.
    main_keys = set()
    for cat_results in file_data.values():
        main_keys.update(cat_results.keys())
    main_keys = sorted(main_keys)

    # Prepare a version of file names without the .json extension.
    sorted_files = sorted(file_data.keys())
    short_file_names = {fname: os.path.splitext(fname)[0] for fname in sorted_files}

    for main_key in main_keys:
        output_lines.append("\n" + "=" * 80)
        output_lines.append(f"Tables for main key: '{main_key}'")
        output_lines.append("=" * 80)

        # Collect all subkeys and all metrics for this main key across all files.
        all_subkeys = set()
        all_metrics = set()
        for cat_results in file_data.values():
            if main_key in cat_results:
                sub_dict = cat_results[main_key]
                if isinstance(sub_dict, dict):
                    for subkey, value in sub_dict.items():
                        all_subkeys.add(subkey)
                        if isinstance(value, dict):
                            all_metrics.update(value.keys())
        all_subkeys = sorted(all_subkeys)
        all_metrics = sorted(all_metrics)

        if not all_subkeys:
            output_lines.append(
                f"No subkeys found for main key '{main_key}' in any JSON file.\n"
            )
            continue

        if not all_metrics:
            output_lines.append(f"No metric keys found under main key '{main_key}'.\n")
            continue

        # For each metric, create a table where each row is a subkey and each column a file.
        for metric in all_metrics:
            table = []
            header = [f"{main_key} (subkey)"] + [
                short_file_names[fname] for fname in sorted_files
            ]
            for subkey in all_subkeys:
                row = [subkey]
                for fname in sorted_files:
                    value = ""
                    cat_results = file_data[fname]
                    if main_key in cat_results:
                        sub_dict = cat_results[main_key]
                        if isinstance(sub_dict, dict) and subkey in sub_dict:
                            # Try to get the metric value if it exists.
                            if isinstance(sub_dict[subkey], dict):
                                value = sub_dict[subkey].get(metric, "")
                            else:
                                # In case the subkey value is not a dict, use it directly.
                                value = sub_dict[subkey]
                    row.append(format_value(value))
                table.append(row)
            output_lines.append(f"\nMetric: {metric}")
            output_lines.append(tabulate(table, headers=header, tablefmt="grid"))
        output_lines.append("\n")  # Extra spacing between main keys

    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare JSON files by reading their 'categorical_results' and printing tables."
    )
    parser.add_argument(
        "directory", help="Path to the directory containing JSON files."
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to save the comparison tables as a file. Default: stdout.",
        default=None,
        type=str,
    )
    args = parser.parse_args()

    file_data = load_json_files(args.directory)
    if not file_data:
        print("No valid JSON files were loaded. Exiting.")
        return

    # Generate the comparison tables as a single string.
    output_content = generate_comparison_tables(file_data)

    if args.output:
        try:
            with open(args.output, "w") as f:
                f.write(output_content)
            print(f"Comparison tables saved to {args.output}")
        except Exception as e:
            print(f"Error writing to {args.output}: {e}")
    else:
        print(output_content)


if __name__ == "__main__":
    main()
