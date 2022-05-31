from abc import abstractmethod, ABC
from time import time
from typing import Tuple

import cv2 as cv
from cv2 import Feature2D, KeyPoint
from numpy.typing import NDArray


class FeatureExtractor(ABC):
    
    name = None
    _img = None

    def __init__(self, name: str, image: NDArray) -> None:
        self.name = name
        self._img = image

    @abstractmethod
    def extract_features(self):
        pass

    def extract_features_timed(self, log: dict):
        start = time()
        retval = self.extract_features()
        end = time()

        log["time"] = end - start
        return retval


class Sift(FeatureExtractor):
    
    def __init__(self, image: NDArray):
        super().__init__("SIFT", image)

    def extract_features(self) -> Tuple[Tuple[KeyPoint, ...], NDArray]:
        sift: Feature2D = cv.SIFT_create()
        return sift.detectAndCompute(self._img, mask=None)


class Orb(FeatureExtractor):

    def __init__(self, image: NDArray):
        super().__init__("ORB", image)

    def extract_features(self) -> Tuple[Tuple[KeyPoint, ...], NDArray]:
        orb: Feature2D = cv.ORB_create()
        return orb.detectAndCompute(self._img, mask=None)
