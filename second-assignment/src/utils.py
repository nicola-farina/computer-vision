from __future__ import annotations

from typing import List, Tuple

import cv2 as cv

import numpy as np
from numpy.typing import NDArray


class Point:

    _DEFAULT_RADIUS = 15

    def __init__(self, x: float, y: float, radius: int = _DEFAULT_RADIUS, color: Tuple[int, int, int] = None) -> None:
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color if color is not None else Point._random_color()

    @staticmethod
    def from_keypoint(keypoint: cv.KeyPoint, radius: int = _DEFAULT_RADIUS, color: Tuple[int, int, int] = None) -> Point:
        return Point(x=keypoint.pt[0], y=keypoint.pt[1], radius=radius, color=color)

    @staticmethod
    def from_numpy_x_y(np_array: NDArray, radius: int = _DEFAULT_RADIUS, color: Tuple[int, int, int] = None) -> Point:
        return Point(x=np_array[0], y=np_array[1], radius=radius, color=color)

    def get_x_y_int_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)

    def get_x_y_numpy(self) -> NDArray:
        return np.array([self.x, self.y], dtype=np.float32).reshape((1, 2))

    @staticmethod
    def _random_color() -> Tuple[int, int, int]:
        return tuple(np.random.random(size=3) * 256)

    def __repr__(self):
        return f"Point(x: {self.x} y: {self.y})"


class ImageUtils:
    """
    This class contains static utility methods to process images.
    """

    @staticmethod
    def bgr_to_gray(img: NDArray) -> NDArray:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    @staticmethod
    def draw_points(img: NDArray, points: List[Point]) -> NDArray:
        copy = img.copy()
        for point in points:
            copy = cv.circle(img, center=point.get_x_y_int_tuple(), radius=point.radius, color=point.color, thickness=2)
        return copy

    @staticmethod
    def resize_img_by_factor(img: NDArray, factor: float) -> NDArray:
        return cv.resize(img, (0, 0), fx=factor, fy=factor)

