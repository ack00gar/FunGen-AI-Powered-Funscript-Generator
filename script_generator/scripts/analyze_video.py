import queue
import threading
import time
from typing import List

from tqdm import tqdm

from config import SEQUENTIAL_MODE, PROGRESS_BAR, UPDATE_PROGRESS_INTERVAL, NUM_PARALLEL_PIPES
from script_generator.gui.messages.messages import ProgressMessage
from script_generator.object_detection.post_process_results import YoloAnalysisTaskProcessor
from script_generator.object_detection.yolo import YoloTaskProcessor
from script_generator.state.app_state import AppState
from script_generator.tasks.tasks import AnalyzeVideoTask, AnalyzeFrameTask
from script_generator.utils.file import check_create_output_folder
from script_generator.utils.logger import logger
from script_generator.video.video_conversion.vr_to_2d_task_processor import VrTo2DTaskProcessor
from script_generator.video.video_task_processor import VideoTaskProcessor


def analyze_video(state: AppState) -> List[AnalyzeFrameTask]:
    """
    Launch multiple pipelines in parallel.
    Start a single logging thread to monitor all pipelines.
    """
    logger.info(f"[OBJECT DETECTION] Spawning {NUM_PARALLEL_PIPES} pipeline(s).")

    # Create all pipelines, but do not start them yet
    tasks_and_threads = []
    for pipe_id in range(NUM_PARALLEL_PIPES):
        analyze_task, pipeline_threads = analyze_video_pipe(state, pipe_id)
        # Keep track of each pipeline's task object and its associated threads
        tasks_and_threads.append((analyze_task, pipeline_threads))

    # Start all pipeline threads
    for analyze_task, pipeline_threads in tasks_and_threads:
        for t in pipeline_threads:
            # Some pipelines might have a None thread (like opengl_thread = None if not VR)
            if t is not None:
                t.start()

    # Start logging thread that monitors all pipelines
    log_thread_stop_event = threading.Event()
    logging_thread = threading.Thread(
        target=log_progress,
        args=(state, [item[0] for item in tasks_and_threads], log_thread_stop_event),
        daemon=True,
    )
    logging_thread.start()

    # Wait for all pipeline threads to finish
    for analyze_task, pipeline_threads in tasks_and_threads:
        for t in pipeline_threads:
            if t is not None:
                t.join()

    # Check for exceptions from each pipeline thread
    for analyze_task, pipeline_threads in tasks_and_threads:
        for t in pipeline_threads:
            if t is not None:
                t.check_exception()

    # Signal logging thread to stop and wait for it
    log_thread_stop_event.set()
    logging_thread.join()

    # Update end_time for each pipeline and log performance
    for analyze_task, pipeline_threads in tasks_and_threads:
        analyze_task.end_time = time.time()
        # Log performance for that pipeline
        log_performance(state, analyze_task.result_q, analyze_task)

    logger.info("[OBJECT DETECTION] All pipelines complete.")

    # Return tasks
    return [tup[0] for tup in tasks_and_threads]

def analyze_video_pipe(state: AppState, pipe_id) -> AnalyzeVideoTask:
    logger.info(f"[OBJECT DETECTION] Starting up pipeline with id: {pipe_id} and running in {'sequential mode' if SEQUENTIAL_MODE else 'parallel mode'}...")

    log_thread_stop_event = threading.Event()
    threads = []

    # make sure the output folder exists for this video
    check_create_output_folder(state.video_path)

    # Initialize batch task
    state.set_video_info()
    if state.video_reader == "FFmpeg + OpenGL (Windows)":
        if not state.video_info.is_vr:
            logger.warn("Disabled OpenGL in the pipeline as it's not needed for 2D videos")
            state.video_reader = "FFmpeg"

        if state.video_info.is_fisheye:
            logger.warn("Disabled OpenGL as fisheye is not yet supported with the opengl feature")
            state.video_reader = "FFmpeg"

    use_open_gl = state.video_reader == "FFmpeg + OpenGL (Windows)"

    # Create our pipeline task
    analyze_task = AnalyzeVideoTask(pipe_id=pipe_id)

    # Create threads
    decode_thread = VideoTaskProcessor(state=state, task=analyze_task, output_queue=analyze_task.opengl_q if use_open_gl else analyze_task.yolo_q)
    opengl_thread = VrTo2DTaskProcessor(state=state, task=analyze_task, input_queue=analyze_task.opengl_q, output_queue=analyze_task.yolo_q) if use_open_gl else None
    yolo_thread = YoloTaskProcessor(state=state, task=analyze_task, input_queue=analyze_task.yolo_q, output_queue=analyze_task.analysis_q)
    yolo_analysis_thread = YoloAnalysisTaskProcessor(state=state, task=analyze_task, input_queue=analyze_task.analysis_q, output_queue=analyze_task.result_q)

    # If sequential mode, we wrap these in a step-by-step approach
    if SEQUENTIAL_MODE:
        # We'll create "mini-threads" just for the sequential pattern
        threads = [decode_thread]
        if use_open_gl:
            threads.append(opengl_thread)
        threads.append(yolo_thread)
        threads.append(yolo_analysis_thread)
    else:
        # Typical parallel approach
        threads = [decode_thread]
        if use_open_gl:
            threads.append(opengl_thread)
        threads.append(yolo_thread)
        threads.append(yolo_analysis_thread)

    return analyze_task, threads


def log_progress(state: AppState, analyze_tasks: List[AnalyzeVideoTask], stop_event: threading.Event):
    """
    Single logging thread monitoring *all* pipelines in aggregate.
    Displays a combined progress bar and combined queue sizes.
    """
    # Sum total frames across all pipelines
    total_frames = sum([state.video_info.total_frames for _ in analyze_tasks])
    label = f"Analyzing {len(analyze_tasks)} pipeline(s)"

    if PROGRESS_BAR:
        with tqdm(
            total=total_frames,
            desc=label,
            unit="f",
            position=0,
            unit_scale=False,
            unit_divisor=1,
            ncols=130
        ) as progress_bar:
            while not stop_event.is_set():
                # Sum queue sizes across all tasks
                opengl_size = sum(t.opengl_q.qsize() for t in analyze_tasks if t.opengl_q is not None)
                yolo_size = sum(t.yolo_q.qsize() for t in analyze_tasks)
                analysis_size = sum(t.analysis_q.qsize() for t in analyze_tasks)
                frames_processed = sum(t.result_q.qsize() for t in analyze_tasks)

                progress_bar.n = frames_processed
                open_gl = f"OpenGL: {opengl_size:>3}, " if state.video_reader == "FFmpeg + OpenGL (Windows)" else ""
                progress_bar.set_postfix_str(
                    f"Queues: {open_gl}YOLO: {yolo_size:>3}, Analysis: {analysis_size:>3}"
                )
                progress_bar.refresh()

                if frames_processed >= total_frames:
                    stop_event.set()

                if state.update_ui:
                    elapsed_time = time.time() - min(t.start_time for t in analyze_tasks)
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
            opengl_size = sum(t.opengl_q.qsize() for t in analyze_tasks if t.opengl_q is not None)
            yolo_size = sum(t.yolo_q.qsize() for t in analyze_tasks)
            analysis_size = sum(t.analysis_q.qsize() for t in analyze_tasks)
            frames_processed = sum(t.result_q.qsize() for t in analyze_tasks)

            logger.info(
                f"OpenGL: {opengl_size:>3}, YOLO: {yolo_size:>3}, "
                f"Analysis: {analysis_size:>3}, DONE: {frames_processed:>3}"
            )

            if frames_processed >= total_frames:
                stop_event.set()

            time.sleep(UPDATE_PROGRESS_INTERVAL)


def log_performance(state: AppState, results_queue: queue.Queue, analyze_task: AnalyzeVideoTask):
    """
    Logs performance stats for a single pipeline's task.
    If you want to log combined performance for all pipelines,
    you can adapt this function similarly.
    """
    # Filter out any sentinel items
    tasks = [t for t in results_queue.queue if t is not None and hasattr(t, 'profile')]
    total_frames = len(tasks)

    if not analyze_task.end_time:
        analyze_task.end_time = time.time()  # fallback

    total_pipeline_time = analyze_task.end_time - analyze_task.start_time

    # For quick reference
    video_fps = state.video_info.fps if state.video_info else 30.0
    video_duration = total_frames / video_fps if video_fps else 0
    avg_processing_fps = total_frames / total_pipeline_time if total_pipeline_time > 0 else 0
    realtime_percentage = (avg_processing_fps / 60.0) * 100.0

    log_message = (
        f"\n{'-' * 60}"
        f"\n OBJECT DETECTION PIPELINE #{analyze_task.pipe_id} COMPLETED {'(sequential mode)' if SEQUENTIAL_MODE else ''}\n"
        f"\n Settings\n"
        f"  - Video reader               : {state.video_reader}\n"     
        f"\n Video stats\n"
        f"  - Total Frames               : {total_frames}\n"
        f"  - Video Duration             : {video_duration:.2f} s\n"
    )

    if SEQUENTIAL_MODE:
        log_message += f"\n Sequential Queue statistics\n"
        for key, total_time in analyze_task.profile.items():
            if key.endswith("_duration"):  # only duration metrics
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
        for t in tasks:
            for key, ttime in t.profile.items():
                if key.endswith("_duration"):
                    if key not in aggregated_times:
                        aggregated_times[key] = {"total_time": 0, "task_count": 0}
                    aggregated_times[key]["total_time"] += ttime
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