import cv2 as cv

from definitions import DATA_DIR
from feature import FeatureExtractor, Algorithm
from tracking import LKOpticalFlow
from utils import ImageUtils


def main():
    # Define some constants
    video_path = str(DATA_DIR / "video.mp4")
    window_name = "Feature Extraction and Tracking"
    sampling = 15

    # Initialize variables
    extractor = FeatureExtractor(Algorithm.ORB)
    frame_idx = 0
    prev_frame_gray, prev_keypoints = None, None

    # Capture video
    video = cv.VideoCapture(video_path)
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break  # End of video

        # Copy frame for processing
        curr_frame = frame.copy()
        curr_frame_gray = ImageUtils.bgr_to_gray(curr_frame)

        # Either sample features every "sampling" frames, or track them in the other frames
        if frame_idx % sampling == 0:
            curr_keypoints = extractor.extract_keypoints(curr_frame_gray)
            img_keypoints = ImageUtils.draw_points(curr_frame, curr_keypoints)
            cv.imshow(window_name, img_keypoints)
        else:
            curr_keypoints = LKOpticalFlow.track_keypoints(prev_frame_gray, curr_frame_gray, prev_keypoints)
            img_keypoints = ImageUtils.draw_points(curr_frame, curr_keypoints)
            cv.imshow(window_name, img_keypoints)

        # Copy values for next iteration
        prev_frame_gray, prev_keypoints = curr_frame_gray.copy(), curr_keypoints

        # Exit if q is pressed
        if cv.waitKey(1) == ord('q'):
            break

        # Increase frame counter
        frame_idx += 1

    video.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
