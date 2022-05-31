import cv2
from numpy.typing import NDArray


class ImageUtils:
    """
    This class contains static utility methods to process images.
    """

    @staticmethod
    def bgr_to_gray(img: NDArray) -> NDArray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
