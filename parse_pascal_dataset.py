import argparse
import os

import cv2
import numpy as np


def rgb_to_unique_grayscale(image):
    """
    Convert an RGB image (as a NumPy array) to a unique grayscale image
    by encoding the RGB values.
    """
    # Ensure the image is in RGB format (OpenCV reads images in BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split into R, G, B channels as uint32
    R = image[:, :, 0].astype(np.uint32)
    G = image[:, :, 1].astype(np.uint32)
    B = image[:, :, 2].astype(np.uint32)

    # Combine the RGB channels into a unique value
    unique_gray = (R << 16) + (G << 8) + B

    # Return the unique grayscale image
    return unique_gray


white_gray_value = (255 << 16) + (255 << 8) + 255


def process_images(
    input_folder, min_percentage_area, max_percentage_area, output_folder
):
    # List all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Resize image to 512x512, maintaining aspect ratio
            # height, width = image.shape[:2]
            # scale = min(512 / height, 512 / width)
            # new_size = (int(width * scale), int(height * scale))
            # resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            resized_image = image
            new_size = image.shape[1], image.shape[0]

            # Make border around the image
            bordered_image = cv2.copyMakeBorder(
                resized_image,
                top=(512 - new_size[1]) // 2,
                bottom=(512 - new_size[1] + 1) // 2,
                left=(512 - new_size[0]) // 2,
                right=(512 - new_size[0] + 1) // 2,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )

            # Convert to unique grayscale
            unique_gray = rgb_to_unique_grayscale(bordered_image)

            # Find unique values and process masks
            unique_values = np.unique(unique_gray)
            unique_values = unique_values[1:-1]

            if output_folder is None:
                print(f"# of objects: {len(unique_values)}")
                print(f"Unique values: {unique_values}")

                cv2.imshow(f"RGBImage", bordered_image)
                cv2.imshow(f"Image", unique_gray / white_gray_value)

            for value in unique_values:
                # Create a binary mask
                mask = (unique_gray == value).astype(np.uint8)

                # Calculate area of the mask
                area = np.sum(mask)
                total_area = mask.size
                percentage_area = (area / total_area) * 100

                # Check if area percentage is within specified limits
                if min_percentage_area <= percentage_area <= max_percentage_area:
                    if output_folder is None:
                        cv2.imshow(f"Mask", mask * 255)
                        cv2.waitKey(0)  # Wait for a key press to continue
                    else:
                        output_image_path = os.path.join(
                            output_folder, f"{filename[:-4]}_{value}.png"
                        )
                        cv2.imwrite(output_image_path, mask * 255)

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Process Pascal dataset segmentation images."
    )
    parser.add_argument(
        "--input_folder",
        required=True,
        help="Path to the input folder containing images.",
    )
    parser.add_argument(
        "--output_folder",
        help="Path to the output folder.",
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

    args = parser.parse_args()

    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=False)

    process_images(
        args.input_folder,
        args.min_percentage_area,
        args.max_percentage_area,
        args.output_folder,
    )


if __name__ == "__main__":
    main()
