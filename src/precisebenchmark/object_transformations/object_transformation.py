from abc import ABC, abstractmethod

import numpy as np
from utils.binary_masks import get_object_boundaries

# class ObjectTransformationEasy(ABC):
#     @abstractmethod
#     def process(self, mask: np.array) -> np.array:
#         pass

#     @abstractmethod
#     def _get_transformation_matrix_wrt_object_center(self) -> np.array:
#         pass


class ObjectTransformation(ABC):
    """
    Object transformations class interface
    To create a new object transformation class, inherit this class and implement the abstract methods _process_object and _get_prompt
    These type of classes will have a processing function that will take an binary mask image and return a transformed binary mask image.
    It should also have a function that will return the text prompt for the corresponding transformation.
    The prompt function should be called after the process function is called.
    """

    def __init__(self):
        self.is_process_called = False
        self.check_no_overflow = True
        self.before_area, self.after_area = None, None

    @abstractmethod
    def _process_object(self, mask: np.array) -> np.array:
        pass

    @abstractmethod
    def _check_overflow(self, mask: np.array, processed_mask: np.array) -> bool:
        pass

    def process(self, mask: np.array) -> np.array:
        max_tries = 200
        for _ in range(max_tries):
            processed_mask = self._process_object(mask)
            self.is_process_called = True

            if not self.check_no_overflow or not self._check_overflow(
                mask, processed_mask
            ):
                return processed_mask
            # Else, initialize with new random parameters and try again
            self.__init__()
        raise ValueError(
            f"Failed to find a valid transformation after {max_tries} tries"
        )

    @abstractmethod
    def _get_manually_generated_prompt(self) -> str:
        pass

    @abstractmethod
    def _get_base_prompt(self) -> str:
        pass

    def get_prompt(self) -> tuple[str, str]:
        if not self.is_process_called:
            raise ValueError(
                "The process function should be called before getting the prompt"
            )

        return self._get_base_prompt(), self._get_manually_generated_prompt()

    @abstractmethod
    def _get_transformation_matrix_wrt_object_center(self) -> np.array:
        pass

    def get_matrix(self) -> np.array:
        if not self.is_process_called:
            raise ValueError(
                "The process function should be called before getting the prompt"
            )

        if self._get_transformation_matrix_wrt_object_center().shape != (3, 3):
            raise ValueError("The transformation matrix should be a 3x3 matrix")

        return self._get_transformation_matrix_wrt_object_center()

    def get_object_boundaries(self, mask: np.array) -> tuple[int, int, int, int]:
        return get_object_boundaries(mask)
