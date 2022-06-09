import cv2 as cv

from definitions import DATA_DIR
from extraction import FeatureExtractor, Algorithm
from tracking import LKOpticalFlow
from utils import ImageUtils


def main():
    # Define some constants
    video_path = str(DATA_DIR / "test.mp4")
    sampling = 30
    resize_factor = 0.5

    # Define algorithm for feature extraction
    algorithm = Algorithm.GFTT
    nfeatures = 1000
    threshold = 20

    # Prepare window for visualization
    window_name = algorithm.value

    # Initialize variables
    extractor = FeatureExtractor(algorithm, nfeatures, threshold)
    frame_idx = 0
    prev_frame_gray, prev_keypoints = None, None

    # Capture video
    video = cv.VideoCapture(video_path)
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    fps = int(video.get(cv.CAP_PROP_FPS))
    fourcc = int(video.get(cv.CAP_PROP_FOURCC))
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH)) // 2
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT)) // 2
    writer = cv.VideoWriter(f"{algorithm.value}.mp4", fourcc, fps, (width, height))

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break  # End of video

        # Copy frame for processing
        curr_frame = frame.copy()
        if 0. < resize_factor < 1.:
            curr_frame = ImageUtils.resize_img_by_factor(curr_frame, resize_factor)
        curr_frame_gray = ImageUtils.bgr_to_gray(curr_frame)

        # Either sample features every "sampling" frames, or track them in the other frames
        if frame_idx % sampling == 0:
            curr_keypoints = extractor.extract_keypoints(curr_frame_gray)
        else:
            curr_keypoints, _ = LKOpticalFlow.track_keypoints(prev_frame_gray, curr_frame_gray, prev_keypoints, compute_avg_error=False)

        img_with_keypoints = ImageUtils.draw_points(curr_frame, curr_keypoints)
        cv.imshow(window_name, img_with_keypoints)

        writer.write(img_with_keypoints)

        # Copy values for next iteration
        prev_frame_gray, prev_keypoints = curr_frame_gray.copy(), curr_keypoints

        # Exit if q is pressed
        if cv.waitKey(1) == ord('q'):
            break

        # Increase frame counter
        frame_idx += 1

    video.release()
    writer.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
