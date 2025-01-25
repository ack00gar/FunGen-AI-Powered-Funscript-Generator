from config import RENDER_RESOLUTION, PITCH


def get_vr_video_filters(video, video_reader, hwaccel):
    if video.is_fisheye:
        projection, iv_fov, ih_fov, v_fov, h_fov, d_fov = "fisheye", 190, 190, 90, 90, 180
    else:
        projection, iv_fov, ih_fov, v_fov, h_fov, d_fov = "he", 180, 180, 90, 90, 180

    cuda = hwaccel == "cuda"

    scale = f"[0:0]scale_cuda=2160:-2,hwdownload" if cuda else f"[0:0]scale={RENDER_RESOLUTION * 2}:-2"
    crop = f"crop={RENDER_RESOLUTION}:{RENDER_RESOLUTION}:0:0"
    out_format = f"format=nv12," if cuda else ""

    if video_reader == "FFmpeg":
        filters = [
            scale,
            crop,
            f"{out_format}v360={projection}:in_stereo=2d:output=sg:iv_fov={iv_fov}:ih_fov={ih_fov}:"
            f"d_fov={d_fov}:v_fov={v_fov}:h_fov={h_fov}:pitch={PITCH}:yaw=0:roll=0:"
            f"w={RENDER_RESOLUTION}:h={RENDER_RESOLUTION}:interp=lanczos:reset_rot=1",
            "lutyuv=y=gammaval(0.7)"
        ]
    else:
        filters = [
            scale,
            crop,
            f"{out_format}lutyuv=y=gammaval(0.7)"  # TODO Process in open gl and move scale and crop to the gpu
        ]

    return ",".join(filters)

def get_2d_video_filters(video, width, height):
    return f"scale={width}:{height}" if video.height > RENDER_RESOLUTION else None

def get_video_filters(video, video_reader, hwaccel, width, height):
    if video.is_vr:
        return get_vr_video_filters(video, video_reader, hwaccel)
    else:
        return get_2d_video_filters(video, width, height)