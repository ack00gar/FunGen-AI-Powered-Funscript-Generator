import subprocess

import numpy as np

from script_generator.debug.errors import FFMpegError
from script_generator.tasks.abstract_task_processor import AbstractTaskProcessor, TaskProcessorTypes
from script_generator.tasks.tasks import AnalyzeFrameTask
from script_generator.utils.logger import logger
from script_generator.video.ffmpeg.commands import get_ffmpeg_read_cmd


class VideoWorker(AbstractTaskProcessor):
    process_type = TaskProcessorTypes.VIDEO
    process = None

    def task_logic(self):
        cmd, frame_size, width, height = get_ffmpeg_read_cmd(
            self.state.video_info,
            self.state.video_reader,
            self.state.ffmpeg_hwaccel,
            self.state.frame_start
        )
        logger.info(f"FFMPEG executing command: {' '.join(cmd)}")

        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        current_frame = self.state.frame_start

        while True:
            try:
                in_bytes = self.process.stdout.read(frame_size)
                if not in_bytes:
                    if current_frame == self.state.frame_start:
                        raise FFMpegError(f"FFMPEG could not read any frames from stdout command: {' '.join(cmd)}")
                    else:
                        logger.info("FFMPEG received last frame")
                    break

                task = AnalyzeFrameTask(frame_pos=current_frame)
                frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

                if self.state.video_reader == "FFmpeg":
                    task.rendered_frame = frame
                else:
                    task.preprocessed_frame = frame

                task.end(str(self.process_type))

                self.finish_task(task)
                current_frame += 1

            except Exception as e:
                logger.error(f"Error reading frame: {e}")
                raise

        self.stop_process()

    # TODO ffmpeg interrupt
    # def release(self):
    #     if self.process:
    #         self.process.stdout.close()
    #         self.process = None
