import statistics
from itertools import compress
from time import time
from typing import List, Tuple

import cv2 as cv
import numpy
import numpy as np
from numpy.typing import NDArray

from utils import Point


class LKOpticalFlow:

    @staticmethod
    def track_keypoints(prev_img: NDArray, curr_img: NDArray, prev_points: List[Point], compute_avg_error: bool = False) -> Tuple[List[Point], float]:
        # Convert points to numpy array format for using Lucas Kanade
        np_prev_points = np.array([point.get_x_y_numpy() for point in prev_points])

        new_points, status, err = cv.calcOpticalFlowPyrLK(
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

        if compute_avg_error:
            err = err[status == 1]
            avg_error = float(np.mean(err))
            return retval, avg_error
        else:
            return retval, -1

    @staticmethod
    def track_keypoints_timed(prev_img: NDArray, curr_img: NDArray, prev_points: List[Point], compute_avg_error: bool = False) -> Tuple[Tuple[List[Point], float], float]:
        start = time()
        retval = LKOpticalFlow.track_keypoints(prev_img, curr_img, prev_points, compute_avg_error)
        end = time()
        elapsed = end - start
        return retval, elapsed
