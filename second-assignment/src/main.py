import cv2 as cv
from cv2 import BFMatcher

from definitions import DATA_DIR
from feature import Orb
from utils import ImageUtils


def main():

    # Define some constants
    sampling = 2

    # Capture video
    video = cv.VideoCapture(str(DATA_DIR / "video.mp4"))

    # Initialize variables
    frame_idx = 0
    prev_frame, prev_keypoints, prev_descriptors = None, None, None
    curr_frame, curr_keypoints, curr_descriptors = None, None, None
    matches = None

    while video.isOpened():
        log = {}

        ret, frame = video.read()

        if not ret:
            break  # End of video

        # Copy frame for processing
        curr_frame = frame.copy()
        curr_frame_gray = ImageUtils.bgr_to_gray(curr_frame)

        # Either sample features every "sampling" frames, or track them in the other frames
        if frame_idx % sampling == 0:
            extractor = Orb(image=curr_frame_gray)
            curr_keypoints, curr_descriptors = extractor.extract_features_timed(log)
        else:
            matcher: BFMatcher = cv.BFMatcher_create()
            matches = matcher.match(prev_descriptors, curr_descriptors)

        if matches:

            # Draw the matches
            img_matches = cv.drawMatches(
                img1=prev_frame, keypoints1=prev_keypoints,
                img2=curr_frame, keypoints2=curr_keypoints,
                matches1to2=matches,
                outImg=None
            )

            # Show matches
            cv.namedWindow("Video", cv.WINDOW_NORMAL)
            cv.imshow("Video", img_matches)

        # Copy values for next iteration
        prev_frame, prev_keypoints, prev_descriptors = frame.copy(), curr_keypoints, curr_descriptors

        # Exit if q is pressed
        if cv.waitKey(1000) == ord('q') or not ret:
            break

        # Increase frame counter
        frame_idx += 1

    video.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
