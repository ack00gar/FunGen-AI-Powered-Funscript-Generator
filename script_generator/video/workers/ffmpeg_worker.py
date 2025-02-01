import subprocess

import numpy as np

from script_generator.debug.errors import FFMpegError
from script_generator.debug.logger import log, log_vid
from script_generator.tasks.abstract_task_processor import AbstractTaskProcessor, TaskProcessorTypes
from script_generator.tasks.tasks import AnalyzeFrameTask
from script_generator.video.ffmpeg.commands import get_ffmpeg_read_cmd


class VideoWorker(AbstractTaskProcessor):
    process_type = TaskProcessorTypes.VIDEO
    process = None

    def task_logic(self):
        cmd, frame_size, width, height = get_ffmpeg_read_cmd(
            self.state,
            self.state.frame_start
        )
        log_vid.info(f"FFMPEG executing command: {' '.join(cmd)}")

        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        current_frame = self.state.frame_start

        try:
            while True:
                in_bytes = self.process.stdout.read(frame_size)
                if not in_bytes:
                    if current_frame == self.state.frame_start:
                        error_output = self.process.stderr.read().decode('utf-8', errors='replace')
                        log_vid.error(f"FFMPEG could not read frames from this video\nFFMPEG command:\n{' '.join(cmd)}\nFFMPEG ERROR:\n{error_output}")
                        raise FFMpegError(f"FFMPEG could not read frames from this video. See the log for details.")
                    else:
                        log_vid.info("FFMPEG received last frame")
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
            log_vid.error(f"Error reading frame: {e}")
            raise

        finally:
            self.stop_process()
            self.release()

    def release(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process.stdout.close()
            self.process.stderr.close()
            self.process = None
