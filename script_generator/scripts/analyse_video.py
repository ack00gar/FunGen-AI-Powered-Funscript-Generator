import queue
import threading
import time
from typing import List

from colorama import Fore, Style
from tqdm import tqdm

from config import QUEUE_MAXSIZE, SEQUENTIAL_MODE, PROGRESS_BAR, UPDATE_PROGRESS_INTERVAL
from script_generator.gui.messages.messages import ProgressMessage
from script_generator.object_detection.post_process_results import YoloAnalysisTaskProcessor
from script_generator.object_detection.yolo import YoloTaskProcessor
from script_generator.state.app_state import AppState
from script_generator.tasks.abstract_task_processor import TaskProcessorTypes
from script_generator.tasks.tasks import AnalyseVideoTask, AnalyseFrameTask
from script_generator.utils.logger import logger
from script_generator.video.video_conversion.vr_to_2d_task_processor import VrTo2DTaskProcessor
from script_generator.video.video_task_processor import VideoTaskProcessor


def analyse_video(state: AppState) -> List[AnalyseFrameTask]:
    logger.info(f"[OBJECT DETECTION] Starting up pipeline with profiling in {'sequential mode' if SEQUENTIAL_MODE else 'parallel mode'}...")

    use_open_gl = state.video_reader == "FFmpeg + OpenGL (Windows)"

    # Initialize batch task
    state.set_video_info()
    state.analyse_task = AnalyseVideoTask()

    # Create queues
    opengl_q = queue.Queue(maxsize=QUEUE_MAXSIZE)
    yolo_q = queue.Queue(maxsize=QUEUE_MAXSIZE)
    analysis_q = queue.Queue(maxsize=QUEUE_MAXSIZE)
    result_q = queue.Queue(maxsize=0)

    # Create threads
    decode_thread = VideoTaskProcessor(state=state, output_queue=opengl_q if use_open_gl else yolo_q)
    opengl_thread = VrTo2DTaskProcessor(state=state, input_queue=opengl_q, output_queue=yolo_q) if use_open_gl else None
    yolo_thread = YoloTaskProcessor(state=state, input_queue=yolo_q, output_queue=analysis_q)
    yolo_analysis_thread = YoloAnalysisTaskProcessor(state=state, input_queue=analysis_q, output_queue=result_q)

    # Start logging thread
    log_thread_stop_event = threading.Event()
    queue_logging_thread = threading.Thread(
        target=log_progress,
        args=(state, opengl_q, yolo_q, analysis_q, result_q, log_thread_stop_event),
        daemon=True,
    )
    queue_logging_thread.start()

    # Sequential mode can be used to determine performance bottlenecks on very short videos
    if SEQUENTIAL_MODE:
        def run_thread(thread, thread_name, out_queue):
            start_time = time.time()
            thread.start()
            thread.join()
            out_queue.put(None)
            logger.info(f"[OBJECT DETECTION] {thread_name} thread done in {time.time() - start_time} s")

        run_thread(decode_thread, TaskProcessorTypes.VIDEO, opengl_q)
        if use_open_gl:
            run_thread(opengl_thread, TaskProcessorTypes.OPENGL, yolo_q)
        run_thread(yolo_thread, TaskProcessorTypes.YOLO, analysis_q)
        run_thread(yolo_analysis_thread, TaskProcessorTypes.YOLO_ANALYSIS, result_q)
    else:
        threads = [decode_thread, opengl_thread, yolo_thread, yolo_analysis_thread] if use_open_gl else [decode_thread, yolo_thread, yolo_analysis_thread]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    state.analyse_task.end_time = time.time()

    log_thread_stop_event.set()

    # wait for update progress thread to close
    time.sleep(UPDATE_PROGRESS_INTERVAL * 1.1)

    log_performance(state=state, results_queue=result_q)

    return result_q.queue


def log_progress(state, opengl_q, yolo_q, analysis_q, results_q, stop_event):
    total_frames = state.video_info.total_frames

    if PROGRESS_BAR:
        with tqdm(
                total=total_frames,
                desc="Analysing video",
                unit="frames",
                position=0,
                unit_scale=False,
                unit_divisor=1,
                ncols=130
        ) as progress_bar:
            while not stop_event.is_set():
                opengl_size = opengl_q.qsize()
                yolo_size = yolo_q.qsize()
                analysis_size = analysis_q.qsize()
                frames_processed = results_q.qsize()

                progress_bar.n = frames_processed
                open_gl = f"OpenGL: {opengl_size:>3}, " if state.video_reader == "FFmpeg + OpenGL (Windows)" else ""
                progress_bar.set_postfix_str(
                    f"Queues: {open_gl}YOLO: {yolo_size:>3}, Analysis: {analysis_size:>3}"
                )
                progress_bar.refresh()

                if frames_processed >= total_frames:
                    stop_event.set()

                if state.update_ui:
                    elapsed_time = time.time() - state.analyse_task.start_time
                    processing_rate = frames_processed / elapsed_time if elapsed_time > 0 else 0
                    remaining_frames = total_frames - frames_processed
                    eta = remaining_frames / processing_rate if processing_rate > 0 else float('inf')
                    try:
                        state.update_ui(ProgressMessage(
                            process="OBJECT_DETECTION",
                            frames_processed=frames_processed,
                            total_frames=total_frames,
                            eta=time.strftime("%H:%M:%S", time.gmtime(eta)) if eta != float('inf') else "Calculating..."
                        ))
                    except Exception as e:
                        logger.error(f"Error in state.update_ui: {e}")

                time.sleep(UPDATE_PROGRESS_INTERVAL)
    else:
        while not stop_event.is_set():
            opengl_size = opengl_q.qsize()
            yolo_size = yolo_q.qsize()
            analysis_size = analysis_q.size()
            frames_processed = results_q.qsize()
            open_gl = f"OpenGL: {opengl_size:>3}, " if state.video_reader == "FFmpeg + OpenGL (Windows)" else ""
            logger.info(f"Queues: {open_gl}YOLO: {yolo_size:>3}, Analysis: {analysis_size:>3}, DONE: {frames_processed:>3}")

            if frames_processed >= total_frames:
                stop_event.set()

            time.sleep(0.5)

def log_performance(state, results_queue):
    analyse_task = state.analyse_task
    # TODO filter out sentinals in task processor
    tasks = [task for task in results_queue.queue if task is not None and hasattr(task, 'profile')]
    total_frames = len(tasks)

    total_pipeline_time = analyse_task.end_time - analyse_task.start_time
    video_duration = total_frames / state.video_info.fps
    avg_processing_fps = total_frames / total_pipeline_time
    realtime_percentage = (avg_processing_fps / 60.0) * 100.0

    log_message = (
        f"\n{'-' * 60}"
        f"\n OBJECT DETECTION COMPLETED {'(sequential mode)' if SEQUENTIAL_MODE else ''}\n"
        f"\n Settings\n"
        f"  - Video reader               : {state.video_reader}\n"     
        f"\n Video stats\n"
        f"  - Total Frames               : {total_frames}\n"
        f"  - Video Duration             : {video_duration:.2f} s\n"
    )

    if SEQUENTIAL_MODE:
        log_message += f"\n Sequential Queue statistics\n"
        for key, total_time in analyse_task.profile.items():
            if key.endswith("_duration"):  # Only include duration metrics
                avg_time = total_time / total_frames if total_frames > 0 else 0.0
                stage_name = key.replace("_duration", "").capitalize()
                log_message += (
                    f"  - {stage_name:<27}: {avg_time * 1000:.0f} ms | "
                    f"{(1 / avg_time if avg_time > 0 else 0):.0f} fps\n"
                )
    else:
        log_message += (
            f"\n Performance stats\n"
            f"  - Average Processing         : {avg_processing_fps:.2f} fps\n"
            f"  - Real-time Processing       : {realtime_percentage:.2f} %\n"
            f"  - Total Pipeline Runtime (s) : {total_pipeline_time:.2f} s\n"
        )
        log_message += f"\n Task Average Times (while running in parallel)\n"

        aggregated_times = {}
        for task in tasks:
            for key, total_time in task.profile.items():
                if key.endswith("_duration"):
                    if key not in aggregated_times:
                        aggregated_times[key] = {"total_time": 0, "task_count": 0}
                    aggregated_times[key]["total_time"] += total_time
                    aggregated_times[key]["task_count"] += 1

        # Calculate and format averages for each key
        for key, data in aggregated_times.items():
            avg_time = data["total_time"] / total_frames if total_frames > 0 else 0.0
            stage_name = key.replace("_duration", "").replace("_", " ").capitalize()
            log_message += (
                f"  - {stage_name:<27}: {avg_time * 1000:.0f} ms | "
                f"{(1 / avg_time if avg_time > 0 else 0):.0f} fps\n"
            )

    log_message += f"{'-' * 60}\n"

    for line in log_message.splitlines():
        logger.info(line)

    print(Fore.LIGHTBLUE_EX + log_message)
    print(Style.RESET_ALL, end="")
