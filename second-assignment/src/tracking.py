import cv2 as cv
from cv2 import BFMatcher

from definitions import DATA_DIR
from feature import Orb, Sift
from utils import ImageUtils

if __name__ == "__main__":
    box = cv.imread(str(DATA_DIR / "box.png"))
    box_gray = ImageUtils.bgr_to_gray(box)
    box2 = cv.imread(str(DATA_DIR / "box2.png"))
    box2_gray = ImageUtils.bgr_to_gray(box2)

    extractor = Sift(image=box_gray)
    box_keypoints, box_descriptors = extractor.extract_features()
    extractor2 = Sift(image=box2_gray)
    box2_keypoints, box2_descriptors = extractor2.extract_features()

    matcher: BFMatcher = cv.BFMatcher_create()
    matches = matcher.match(box_descriptors, box2_descriptors)
    matches = list(filter(lambda x: x.distance < 150, matches))

    # Draw the matches
    img_matches = cv.drawMatches(
        img1=box, keypoints1=box_keypoints,
        img2=box2, keypoints2=box2_keypoints,
        matches1to2=matches,
        outImg=None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    )

    # Show matches
    cv.namedWindow("Img", cv.WINDOW_NORMAL)
    cv.imshow("Img", img_matches)
    cv.waitKey(0)
