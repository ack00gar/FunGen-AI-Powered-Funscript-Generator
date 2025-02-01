from script_generator.state.app_state import AppState
from script_generator.video.ffmpeg.filters import get_video_filters
from script_generator.video.ffmpeg.hwaccel import get_hwaccel_read_args, supports_cuda_scale
from script_generator.video.info.video_info import get_cropped_dimensions, VideoInfo

def get_ffmpeg_read_cmd(state: AppState, frame_start: int | None, output="-", disable_opengl=False):
    video = state.video_info
    width, height = get_cropped_dimensions(video)
    vf = get_video_filters(video, state.video_reader, state.ffmpeg_hwaccel, width, height, disable_opengl)
    start_time = (frame_start / video.fps) * 1000

    # Get supported hardware acceleration backends
    hwaccel_read = get_hwaccel_read_args(video, state.ffmpeg_hwaccel)

    video_filter = ["-vf", vf] if vf else []
    if state.ffmpeg_hwaccel == "vaapi":
        # VAAPI requires specific pixel formats and filters
        video_filter = ["-vf", f"{vf},format=nv12,hwupload"] if vf else ["-vf", "format=nv12,hwupload"]

    if supports_cuda_scale(video, state.ffmpeg_hwaccel):
        video_filter = ["-noautoscale"] + video_filter  # explicitly tell ffmpeg that scaling is done by cuda

    frame_size = width * height * 3  # Size of one frame in bytes

    return [
        state.ffmpeg_path,
        *hwaccel_read,
        '-nostats', '-loglevel', 'warning',
        "-ss", str(start_time / 1000),  # Seek to start time in seconds
        "-i", video.path,
        "-an",  # Disable audio processing
        *video_filter,
        "-f", "rawvideo", "-pix_fmt", "bgr24",  # cv2 requires bgr (over rgb) and Yolo expects bgr images when using numpy frames (converts them internally)
        "-threads", "0", # all threads
        output
    ], frame_size, width, height