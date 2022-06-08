import sys
from enum import Enum
from time import time
from typing import List

import cv2 as cv
from numpy.typing import NDArray

from utils import Point


class Algorithm(Enum):
    SIFT = "SIFT"
    ORB = "ORB"
    GFTT = "GFTT"
    FAST = "FAST"


class FeatureExtractor:

    def __init__(self, algorithm: Algorithm) -> None:
        if algorithm == Algorithm.SIFT:
            self.extractor = cv.SIFT_create()
        elif algorithm == Algorithm.ORB:
            self.extractor = cv.ORB_create()
        elif algorithm == Algorithm.GFTT:
            self.extractor = cv.GFTTDetector_create()
        elif algorithm == Algorithm.FAST:
            self.extractor = cv.FastFeatureDetector_create()
        else:
            sys.exit("ERROR: You need to specify a valid algorithm!")

    def extract_keypoints(self, img: NDArray) -> List[Point]:
        return [Point.from_keypoint(kp) for kp in list(self.extractor.detect(img, mask=None))]

    def extract_keypoints_timed(self, img: NDArray, log: dict) -> List[Point]:
        start = time()
        retval = self.extract_keypoints(img)
        end = time()

        log["time"] = end - start
        return retval
