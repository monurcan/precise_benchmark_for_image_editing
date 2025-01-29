import numpy as np


def get_object_boundaries(mask: np.array) -> tuple[int, int, int, int]:
    """
    Get the object boundaries in the mask image

    Returns:
    x_min (int): The minimum x coordinate of the object boundaries
    x_max (int): The maximum x coordinate of the object boundaries
    y_min (int): The minimum y coordinate of the object boundaries
    y_max (int): The maximum y coordinate of the object boundaries
    """
    where_filter = np.where(mask != 0)
    y_min, x_min = np.min(where_filter, axis=1)
    y_max, x_max = np.max(where_filter, axis=1)

    return x_min, x_max, y_min, y_max
