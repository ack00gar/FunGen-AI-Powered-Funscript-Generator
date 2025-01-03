import cv2
from tqdm import tqdm


def compute_histogram(frame):
    """
    Compute the normalized histogram of the H channel in the HSV color space.
    :param frame: Input frame (BGR format).
    :return: Normalized histogram.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def compare_histograms(hist1, hist2):
    """
    Compare two histograms using the correlation method.
    :param hist1: First histogram.
    :param hist2: Second histogram.
    :return: Similarity score (higher values indicate greater similarity).
    """
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def detect_scene_changes(video_path, crop=None, threshold=0.97, frame_start=0, frame_end=None):
    """
    Detect scene changes in a video using histogram comparison.
    :param video_path: Path to the video file.
    :param crop: Will crop the video and keep left/right side.
    :param threshold: Similarity threshold for detecting scene changes.
    :param frame_start: Starting frame for processing.
    :param frame_end: Ending frame for processing.
    :return: List of scene changes as [start_frame, end_frame] pairs.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames_base = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = fps  # Process every second of video

    # Adjust frame range based on input
    if frame_end is None:
        frame_end = total_frames_base
    total_frames = frame_end - frame_start
    total_frames_to_parse = int(total_frames / frame_step)

    # Initialize variables
    scene_changes = []
    prev_hist = None
    prev_cut = frame_start

    # Process frames
    for frame_pos in tqdm(range(total_frames_to_parse), desc="Detecting scene changes"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start + frame_pos * frame_step)
        ret, frame = cap.read()
        if not ret:
            break

        # Crop frame if specified
        if crop == "Left":
            frame = frame[:, :frame.shape[1] // 2]
        elif crop == "Right":
            frame = frame[:, frame.shape[1] // 2:]

        # Compute histogram and compare with previous
        current_hist = compute_histogram(frame)
        if prev_hist is not None:
            similarity = compare_histograms(prev_hist, current_hist)
            if similarity < threshold:
                # Scene change detected
                current_frame = frame_start + frame_pos * frame_step
                tqdm.write(
                    f"Scene change detected at frame {current_frame}, "
                    f"time: {int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 // 60)} min "
                    f"{int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 % 60)} sec"
                )
                scene_changes.append([prev_cut, current_frame])
                prev_cut = current_frame

        prev_hist = current_hist

    # Add the last scene
    if not scene_changes:
        scene_changes.append([frame_start, frame_end])
    elif scene_changes[-1][1] != frame_end:
        scene_changes.append([scene_changes[-1][1], frame_end])

    print(f"Found {len(scene_changes)} raw scenes: {scene_changes}.")
    print(f"Scene changes: {scene_changes}")
    print(f"Merging short scenes...")

    # Merge only short scenes
    merged_scenes = []
    min_scene_length = 1000  # Minimum scene length in frames (adjust as needed)
    for scene in scene_changes:
        scene_length = scene[1] - scene[0]
        if scene_length < min_scene_length:
            # Merge short scenes with the previous or next scene
            if merged_scenes:
                merged_scenes[-1][1] = scene[1]  # Merge with the previous scene
            else:
                merged_scenes.append(scene)  # Add the first scene
        else:
            # Keep scenes longer than the minimum length
            merged_scenes.append(scene)

    print(f"Found {len(merged_scenes)} relevant scenes: {merged_scenes}.")
    cap.release()
    return merged_scenes