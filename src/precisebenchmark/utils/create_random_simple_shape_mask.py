import random

import cv2
import numpy as np


def create_shape_mask(shape: str, img_size: (int, int), object_coords: (int, int) = None, object_size: int = None) -> np.ndarray:
    """
        No object_size support for some shapes!
    """
    if shape == "random":
        shape = random.choice(
            [
                "circle",
                "star",
                "square",
                "rectangle",
                "triangle",
                "pentagon",
                "oval",
                "crescent",
            ]
        )

    # Create a blank binary mask
    mask = np.zeros(img_size, dtype=np.uint8)

    # Offsets to make the center of the shape within the mask
    offset_y = img_size[0] // 4
    offset_x = img_size[1] // 4

    # Random position for the shape
    height, width = img_size
    if shape == "circle":
        # Random radius
        radius = random.randint(10, 50) if object_size is None else object_size
        # Random center within the mask
        center = (
            random.randint(radius + offset_x, width - radius - offset_x),
            random.randint(radius + offset_y, height - radius - offset_y),
        ) if object_coords is None else object_coords
        cv2.circle(mask, center, radius, 255, -1)  # Draw filled circle

    elif shape == "square":
        # Random size
        size = random.randint(10, 100)  if object_size is None else object_size
        # Random top-left corner
        top_left = (
            random.randint(offset_x, width - size - offset_x),
            random.randint(offset_y, height - size - offset_y),
        ) if object_coords is None else object_coords
        cv2.rectangle(
            mask, top_left, (top_left[0] + size, top_left[1] + size), 255, -1
        )  # Draw filled square

    elif shape == "rectangle":
        # Random width and height
        rect_width = random.randint(10, 100)
        rect_height = random.randint(10, 100)
        # Random top-left corner
        top_left = (
            random.randint(offset_x, width - rect_width - offset_x),
            random.randint(offset_y, height - rect_height - offset_y),
        ) if object_coords is None else object_coords
        cv2.rectangle(
            mask,
            top_left,
            (top_left[0] + rect_width, top_left[1] + rect_height),
            255,
            -1,
        )  # Draw filled rectangle

    elif shape == "star":
        # Define the star points
        points = np.array(
            [
                [0, 10],
                [3, 3],
                [10, 3],
                [4, -1],
                [6, -8],
                [0, -4],
                [-6, -8],
                [-4, -1],
                [-10, 3],
                [-3, 3],
            ],
            np.int32,
        )

        # Random position for the star
        star_size = random.randint(5, 20)  if object_size is None else object_size  # Random size for the star
        points *= star_size  # Scale star points
        # Random center for the star
        center = (
            random.randint(offset_x, width - offset_x),
            random.randint(offset_y, height - offset_y),
        ) if object_coords is None else object_coords
        points += center  # Translate star points to random center

        # Draw filled star
        cv2.fillPoly(mask, [points], 255)

    elif shape == "triangle":
        # Define triangle vertices
        size = random.randint(20, 80)  if object_size is None else object_size
        points = np.array([[0, -size], [-size, size], [size, size]], np.int32)
        points += (
            random.randint(offset_x, width - offset_x),
            random.randint(offset_y, height - offset_y),
        )  if object_coords is None else object_coords
        cv2.fillPoly(mask, [points], 255)  # Draw filled triangle

    elif shape == "pentagon":
        # Define pentagon vertices
        size = random.randint(13, 70)  if object_size is None else object_size
        angle = np.linspace(0, 2 * np.pi, 6)[:-1]  # 5 points
        points = np.column_stack((size * np.cos(angle), size * np.sin(angle))).astype(
            np.int32
        )
        points += (
            random.randint(offset_x, width - offset_x),
            random.randint(offset_y, height - offset_y),
        )  if object_coords is None else object_coords
        cv2.fillPoly(mask, [points], 255)  # Draw filled pentagon

    elif shape == "oval":
        # Random axes lengths
        axes = (random.randint(20, 80), random.randint(10, 40))
        # Random center within the mask
        center = (
            random.randint(axes[0] + offset_x, width - axes[0] - offset_x),
            random.randint(axes[1] + offset_y, height - axes[1] - offset_y),
        ) if object_coords is None else object_coords
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)  # Draw filled ellipse

    elif shape == "crescent":
        # Random sizes for crescent
        outer_radius = random.randint(30, 50)
        inner_radius = outer_radius - random.randint(10, 20)
        center = (
            random.randint(outer_radius + offset_x, width - outer_radius - offset_x),
            random.randint(outer_radius + offset_y, height - outer_radius - offset_y),
        ) if object_coords is None else object_coords

        # Draw outer circle
        cv2.circle(mask, center, outer_radius, 255, -1)
        # Draw inner circle (subtracting)
        cv2.circle(
            mask, (center[0] + inner_radius // 2, center[1]), inner_radius, 0, -1
        )

    else:
        raise ValueError(
            "Unsupported shape: Choose 'circle', 'star', 'square', or 'rectangle'."
        )

    return mask


if __name__ == "__main__":
    shape_mask = create_shape_mask("circle", (200, 200))
    cv2.imshow("Binary Mask", shape_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()
