import os
import subprocess
from multiprocessing import cpu_count

import imageio
import numpy as np

from config import SUBTRACT_THREADS_FROM_FFMPEG, PITCH, RENDER_RESOLUTION, FFMPEG_PATH, DEBUG_PATH
from script_generator.tasks.abstract_task_processor import AbstractTaskProcessor, TaskProcessorTypes
from script_generator.tasks.tasks import AnalyzeFrameTask
from script_generator.utils.logger import logger
from script_generator.video.video_info import get_cropped_dimensions


class VideoTaskProcessor(AbstractTaskProcessor):
    process_type = TaskProcessorTypes.VIDEO

    def task_logic(self):
        state = self.state
        video = self.state.video_info
        current_frame = state.frame_start

        def get_cmd(vf):
            start_time = (state.frame_start / video.fps) * 1000
            ffmpeg_threads = max(cpu_count() - SUBTRACT_THREADS_FROM_FFMPEG, 1)

            # Get supported hardware acceleration backends
            hwaccel_support = state.ffmpeg_hwaccel_supported

            # Determine the best hardware acceleration backend to use
            hwaccel = []
            if hwaccel_support["cuda"]:
                hwaccel = ["-hwaccel", "cuda"]
            elif hwaccel_support["vaapi"]:
                hwaccel = ["-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128"]
            elif hwaccel_support["amf"]:
                hwaccel = ["-hwaccel", "amf"]
            elif hwaccel_support["videotoolbox"]:
                hwaccel = ["-hwaccel", "videotoolbox"]
            elif hwaccel_support["qsv"]:
                hwaccel = ["-hwaccel", "qsv"]
            elif hwaccel_support["d3d11va"]:
                hwaccel = ["-hwaccel", "d3d11va"]

            # Add platform-specific video filters if required
            video_filter = ["-vf", vf] if vf else []
            if hwaccel_support["vaapi"]:
                # VAAPI requires specific pixel formats and filters
                video_filter = ["-vf", f"{vf},format=nv12,hwupload"] if vf else ["-vf", "format=nv12,hwupload"]

            return [
                FFMPEG_PATH,
                *hwaccel,
                '-nostats', '-loglevel', 'warning',
                "-ss", str(start_time / 1000),  # Seek to start time in seconds
                "-i", video.path,
                "-an",  # Disable audio processing
                "-map", "0:v:0",
                *video_filter,
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-threads", str(ffmpeg_threads),
                "-",  # Output to stdout
            ]

        def vr_video_filters():
            if video.is_fisheye:
                projection, iv_fov, ih_fov, v_fov, h_fov, d_fov = "fisheye", 190, 190, 90, 90, 180
            else:
                projection, iv_fov, ih_fov, v_fov, h_fov, d_fov = "he", 180, 180, 90, 90, 180

            if self.state.video_reader == "FFmpeg":
                filters = [
                    f"scale={RENDER_RESOLUTION * 2}:{RENDER_RESOLUTION}",
                    f"crop={RENDER_RESOLUTION}:{RENDER_RESOLUTION}:0:0",
                    f"v360={projection}:in_stereo=2d:output=sg:iv_fov={iv_fov}:ih_fov={ih_fov}:"
                    f"d_fov={d_fov}:v_fov={v_fov}:h_fov={h_fov}:pitch={PITCH}:yaw=0:roll=0:"
                    f"w={RENDER_RESOLUTION}:h={RENDER_RESOLUTION}:interp=lanczos:reset_rot=1",
                    "lutyuv=y=gammaval(0.7)"
                ]
            else:
                filters = [
                    f"scale={RENDER_RESOLUTION * 2}:{RENDER_RESOLUTION}",
                    f"crop={RENDER_RESOLUTION}:{RENDER_RESOLUTION}:0:0"
                ]

            return ",".join(filters)

        def standard_video_filters():
            return f"scale={width}:{height}" if video.height > RENDER_RESOLUTION else None

        width, height = get_cropped_dimensions(video)

        vf = vr_video_filters() if video.is_vr else standard_video_filters()
        cmd = get_cmd(vf)

        # Debug
        # command = ' '.join(vf)

        # Start FFmpeg process
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frame_size = width * height * 3  # Size of one frame in bytes

        # progress_bar = tqdm(total=203856, unit="frame", desc="Processing frames")
        while True:
            try:
                in_bytes = self.process.stdout.read(frame_size)
                if not in_bytes:
                    break

                task = AnalyzeFrameTask(frame_pos=current_frame)

                frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

                # Debug
                # output_path = os.path.join(DEBUG_PATH, f"frame_{task.id:05d}.png")
                # imageio.imwrite(output_path, frame)

                if self.state.video_reader == "FFmpeg":
                    task.rendered_frame = frame
                else:
                    task.preprocessed_frame = frame

                task.end(str(self.process_type))

                self.finish_task(task)
                current_frame += 1

                # progress_bar.update(1)

            except Exception as e:
                logger.error(f"Error reading frame: {e}")
                return False, None

        self.stop_process()

    # TODO ffmpeg interrupt
    # def release(self):
    #     if self.process:
    #         self.process.stdout.close()
    #         self.process = None
