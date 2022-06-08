from itertools import compress
from typing import List

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from utils import Point


class LKOpticalFlow:

    @staticmethod
    def track_keypoints(prev_img: NDArray, curr_img: NDArray, prev_points: List[Point]) -> List[Point]:
        # Convert points to numpy array format for using Lucas Kanade
        np_prev_points = np.array([point.get_x_y_numpy() for point in prev_points])

        new_points, status, _ = cv.calcOpticalFlowPyrLK(
            prevImg=prev_img, nextImg=curr_img,
            prevPts=np_prev_points, nextPts=None
        )

        # Only select those points that had a valid tracking
        good_new_points = new_points[status == 1]
        good_old_points = list(compress(prev_points, status))

        # Reconvert from numpy format to point, keeping size and color of the previous point
        retval = []
        for i, new_point in enumerate(good_new_points):
            retval.append(
                Point.from_numpy_x_y(new_point, radius=good_old_points[i].radius, color=good_old_points[i].color)
            )

        return retval
