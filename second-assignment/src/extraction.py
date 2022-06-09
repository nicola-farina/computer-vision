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


class FeatureExtractor:

    def __init__(self, algorithm: Algorithm, nfeatures: int = 1000, fast_thresh: int = 20) -> None:
        if algorithm == Algorithm.SIFT:
            self.extractor = cv.SIFT_create(nfeatures=nfeatures)
        elif algorithm == Algorithm.ORB:
            self.extractor = cv.ORB_create(nfeatures=nfeatures)
        elif algorithm == Algorithm.GFTT:
            self.extractor = cv.GFTTDetector_create(maxCorners=nfeatures)
        elif algorithm == Algorithm.FAST:
            self.extractor = cv.FastFeatureDetector_create(threshold=fast_thresh)
        else:
            sys.exit("ERROR: You need to specify a valid algorithm!")

    def extract_keypoints(self, img: NDArray) -> List[Point]:
        return [Point.from_keypoint(kp) for kp in list(self.extractor.detect(img, mask=None))]

    def extract_keypoints_timed(self, img: NDArray) -> Tuple[List[Point], float]:
        start = time()
        points = self.extract_keypoints(img)
        end = time()
        elapsed = end - start
        return points, elapsed
