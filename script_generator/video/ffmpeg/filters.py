from script_generator.constants import RENDER_RESOLUTION, VR_TO_2D_PITCH
from script_generator.video.ffmpeg.hwaccel import supports_cuda_scale


def get_video_filters(video, video_reader, hwaccel, width, height, disable_opengl=False):
    if video.is_vr:
        return get_vr_video_filters(video, video_reader, hwaccel, disable_opengl)
    else:
        return get_2d_video_filters(video, width, height, hwaccel)

def get_vr_video_filters(video, video_reader, hwaccel, disable_opengl=False):
    fov = int(video.fov * 1)
    if video.is_fisheye:
        projection, iv_fov, ih_fov, v_fov, h_fov, d_fov = "fisheye", fov, fov, 90, 90, fov
    else:
        projection, iv_fov, ih_fov, v_fov, h_fov, d_fov = "he", fov, fov, 90, 90, fov

    cuda = hwaccel == "cuda"

    # hardware accelerated output is not supported with > 8 bit
    scale = f"[0:0]scale_cuda={RENDER_RESOLUTION * 2}:-2,hwdownload" if supports_cuda_scale(video, hwaccel) else f"[0:0]scale={RENDER_RESOLUTION * 2}:-2"
    crop = f"crop={RENDER_RESOLUTION}:{RENDER_RESOLUTION}:0:0"
    out_format = f"format=nv12," if cuda else ""

    if video_reader == "FFmpeg" or disable_opengl:
        filters = [
            scale,
            crop,
            f"{out_format}v360={projection}:in_stereo=2d:output=sg:iv_fov={iv_fov}:ih_fov={ih_fov}:"
            f"d_fov={d_fov}:v_fov={v_fov}:h_fov={h_fov}:pitch={VR_TO_2D_PITCH}:yaw=0:roll=0:"
            f"w={RENDER_RESOLUTION}:h={RENDER_RESOLUTION}:interp=lanczos:reset_rot=1",
            "lutyuv=y=gammaval(0.7)"
        ]
    else:
        filters = [
            scale,
            crop,
            f"{out_format}lutyuv=y=gammaval(0.7)"
        ]

    return f"{','.join(filters)}"


def get_2d_video_filters(video, width, height, hwaccel):
    cuda = hwaccel == "cuda"

    if video.height > RENDER_RESOLUTION:
        scale_width = int(video.width * (height / video.height))
        crop = f",crop={width}:{height}:(iw-{width})/2:0"
        # hardware accelerated output is not supported with > 8 bit
        return f"[0:0]scale_cuda={scale_width}:{height},hwdownload,format=nv12{crop}" if supports_cuda_scale(video, hwaccel) else f"[0:0]scale={scale_width}:{height}{crop}"
    else:
        return "[0:0]hwdownload,format=nv12" if cuda else ""