import statistics

import cv2 as cv

from definitions import DATA_DIR
from extraction import FeatureExtractor, Algorithm
from tracking import LKOpticalFlow
from utils import ImageUtils


def extraction_time():
    # Define some constants
    video_path = str(DATA_DIR / "test.mp4")
    sampling = 15
    scaling = 0.5
    nfeatures = ["250", "500", "1000", "2000"]
    thresholds = ["10", "20", "30", "40"]

    # Initialize logs
    logs = {}
    for algorithm in Algorithm:
        logs[algorithm.value] = {}
        if algorithm == Algorithm.FAST:
            for thresh in thresholds:
                logs[algorithm.value][thresh] = {}
                logs[algorithm.value][thresh]["duration"] = []
                logs[algorithm.value][thresh]["nfeatures"] = []
        else:
            for nf in nfeatures:
                logs[algorithm.value][nf] = []

    # Capture video
    video = cv.VideoCapture(video_path)

    frame_idx = 0
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break  # End of video

        # Copy frame for processing, resize
        curr_frame = frame.copy()
        curr_frame = ImageUtils.resize_img_by_factor(curr_frame, scaling)
        curr_frame_gray = ImageUtils.bgr_to_gray(curr_frame)

        # Extract every "sampling" frames
        if frame_idx % sampling == 0:
            for algorithm in Algorithm:
                if algorithm == Algorithm.FAST:
                    for thresh in thresholds:
                        curr_keypoints, elapsed = FeatureExtractor(algorithm, fast_thresh=int(thresh)).extract_keypoints_timed(curr_frame_gray)
                        logs[algorithm.value][thresh]["duration"].append(elapsed)
                        logs[algorithm.value][thresh]["nfeatures"].append(len(curr_keypoints))
                else:
                    for nf in nfeatures:
                        curr_keypoints, elapsed = FeatureExtractor(algorithm, nfeatures=int(nf)).extract_keypoints_timed(curr_frame_gray)
                        logs[algorithm.value][nf].append(elapsed)

        # Increase frame counter
        frame_idx += 1

    # Compute average feature extraction time
    for algorithm in Algorithm:
        print(algorithm.value)
        if algorithm == Algorithm.FAST:
            for thresh in thresholds:
                avg_time = statistics.mean(logs[algorithm.value][thresh]["duration"])
                avg_features = statistics.mean(logs[algorithm.value][thresh]["nfeatures"])
                print(f"\t[{thresh}] => {avg_time} ({avg_features})")
        else:
            for nf in nfeatures:
                avg_time = statistics.mean(logs[algorithm.value][nf])
                print(f"\t[{nf}] => {avg_time}")

    video.release()
    cv.destroyAllWindows()


def features_comparison():
    video_path = str(DATA_DIR / "test.mp4")
    scaling = 0.5
    thresholds = [10, 20, 30, 40]
    algorithm = Algorithm.FAST

    # Capture video
    video = cv.VideoCapture(video_path)

    frame_idx = 0
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break  # End of video

        # Copy frame for processing, resize
        curr_frame = frame.copy()
        curr_frame = ImageUtils.resize_img_by_factor(curr_frame, scaling)
        curr_frame_gray = ImageUtils.bgr_to_gray(curr_frame)

        # Extract every "sampling" frames
        if frame_idx == 90:
            for thresh in thresholds:
                extractor = FeatureExtractor(algorithm, fast_thresh=thresh)
                points = extractor.extract_keypoints(curr_frame_gray)
                img_points = ImageUtils.draw_points(curr_frame, points)
                img_points = ImageUtils.resize_img_by_factor(img_points, 0.5)
                img_points = img_points[240:720, 30:510]
                cv.imwrite(f"{str(thresh)}.png", img_points)
            break
        # Increase frame counter
        frame_idx += 1

    video.release()
    cv.destroyAllWindows()


def tracking_time_and_error():
    # Define some constants
    video_path = str(DATA_DIR / "test.mp4")
    sampling = 30
    scaling = 0.5
    nfeatures = 1000
    threshold = 20

    logs = {}
    for algorithm in Algorithm:
        logs[algorithm.value] = {}
        logs[algorithm.value]["time"] = []
        logs[algorithm.value]["error"] = []

        extractor = FeatureExtractor(algorithm, nfeatures, threshold)

        # Capture video
        video = cv.VideoCapture(video_path)

        prev_frame_gray, prev_keypoints = None, None
        frame_idx = 0
        while video.isOpened():
            ret, frame = video.read()

            if not ret:
                break  # End of video

            # Copy frame for processing, resize
            curr_frame = frame.copy()
            curr_frame = ImageUtils.resize_img_by_factor(curr_frame, scaling)
            curr_frame_gray = ImageUtils.bgr_to_gray(curr_frame)

            # Extract every "sampling" frames, track in other frames
            if frame_idx % sampling == 0:
                curr_keypoints = extractor.extract_keypoints(curr_frame_gray)
            else:
                (curr_keypoints, err), time = LKOpticalFlow.track_keypoints_timed(prev_frame_gray, curr_frame_gray, prev_keypoints, compute_avg_error=True)
                logs[algorithm.value]["time"].append(time)
                logs[algorithm.value]["error"].append(err)
            # Copy values for next iteration
            prev_frame_gray, prev_keypoints = curr_frame_gray.copy(), curr_keypoints
            frame_idx += 1

        video.release()
        cv.destroyAllWindows()

    # Compute average tracking time and error
    for algorithm in Algorithm:
        print(algorithm.value)
        avg_time = statistics.mean(logs[algorithm.value]["time"])
        max_time = max(logs[algorithm.value]["time"])
        std_time = statistics.stdev(logs[algorithm.value]["time"])
        avg_error = statistics.mean(logs[algorithm.value]["error"])
        print(f"\tavg_time: {avg_time} max_time: {max_time} std_time: {std_time} error: {avg_error} ")


if __name__ == "__main__":
    tracking_time_and_error()
