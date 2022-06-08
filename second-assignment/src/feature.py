import sys
from time import time
from typing import List

import cv2 as cv
from cv2 import Feature2D
from numpy.typing import NDArray

from utils import Point


class FeatureExtractor:

    def __init__(self, name: str) -> None:
        self.name = name
        self.extractor = None

    def extract_keypoints(self, img: NDArray) -> List[Point]:
        if self.extractor is not None:
            return [Point.from_keypoint(kp) for kp in list(self.extractor.detect(img, mask=None))]
        else:
            sys.exit("ERROR: you cannot use this class as is, you need to use one of its implementations!")

    def extract_keypoints_timed(self, img: NDArray, log: dict) -> List[Point]:
        start = time()
        retval = self.extract_keypoints(img)
        end = time()

        log["time"] = end - start
        return retval


class SIFT(FeatureExtractor):
    
    def __init__(self) -> None:
        super().__init__("SIFT")
        self.extractor: Feature2D = cv.SIFT_create()


class ORB(FeatureExtractor):

    def __init__(self) -> None:
        super().__init__("ORB")
        self.extractor: Feature2D = cv.ORB_create()


class GFTT(FeatureExtractor):

    def __init__(self) -> None:
        super().__init__("GFTT")
        self.extractor: Feature2D = cv.GFTTDetector_create()


class FAST(FeatureExtractor):

    def __init__(self) -> None:
        super().__init__("FAST")
        self.extractor: Feature2D = cv.FastFeatureDetector_create()
