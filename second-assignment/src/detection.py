import sys
from enum import Enum
from time import time
from typing import List, Tuple

import cv2 as cv
from numpy.typing import NDArray

from utils import Point


class Algorithm(Enum):
    SIFT = "SIFT"
    ORB = "ORB"
    GFTT = "GFTT"
    FAST = "FAST"


class FeatureDetector:

    def __init__(self, algorithm: Algorithm, nfeatures: int = 1000, fast_thresh: int = 20) -> None:
        if algorithm == Algorithm.SIFT:
            self.detector = cv.SIFT_create(nfeatures=nfeatures)
        elif algorithm == Algorithm.ORB:
            self.detector = cv.ORB_create(nfeatures=nfeatures)
        elif algorithm == Algorithm.GFTT:
            self.detector = cv.GFTTDetector_create(maxCorners=nfeatures)
        elif algorithm == Algorithm.FAST:
            self.detector = cv.FastFeatureDetector_create(threshold=fast_thresh)
        else:
            sys.exit("ERROR: You need to specify a valid algorithm!")

    def detect_keypoints(self, img: NDArray) -> List[Point]:
        return [Point.from_keypoint(kp) for kp in list(self.detector.detect(img, mask=None))]

    def detect_keypoints_timed(self, img: NDArray) -> Tuple[List[Point], float]:
        start = time()
        points = self.detect_keypoints(img)
        end = time()
        elapsed = end - start
        return points, elapsed
