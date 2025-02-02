import cv2
import numpy as np

from script_generator.debug.logger import log_fun
from script_generator.funscript.debug.combined_plot import create_combined_plot
from script_generator.funscript.util.util import load_funscript
from script_generator.utils.file import get_output_file_path


def create_funscript_report(state):
    generated_paths = []
    output_path, _ = get_output_file_path(state.video_path, ".funscript")
    generated_paths.append(output_path)

    if state.reference_script:
        # Load reference funscript
        ref_times, ref_positions, _, _ = load_funscript(state.reference_script)

        # if no 0 at the beginning, add it
        if ref_times and ref_times[0] != 0:
            ref_times.insert(0, 0)
            ref_positions.insert(0, 100)

        # Determine total duration in seconds
        total_duration = ref_times[-1] / 1000 if ref_times else 0
    else:
        ref_times, ref_positions = [], []
        gen_times, gen_positions, _, _ = load_funscript(generated_paths[0])
        total_duration = gen_times[-1] / 1000 if gen_times else 0

    # Select 6 random non-overlapping 20-second sections
    sections = select_random_sections(total_duration, section_duration=10, num_sections=6)

    screenshots_done = False
    screenshots = []

    # Load generated funscripts
    for generated_path in generated_paths:
        gen_times, gen_positions, _, _ = load_funscript(generated_path)
        # Extract data for each section
        ref_sections = []
        gen_sections = []
        for start, end in sections:
            if state.reference_script:
                ref_sec = extract_section(ref_times, ref_positions, start, end)
                ref_sections.append(ref_sec)
            gen_sec = extract_section(gen_times, gen_positions, start, end)
            gen_sections.append(gen_sec)

        if not screenshots_done:
            # Capture screenshots, but only once
            screenshots = capture_screenshots(state.video_path, state.video_info.is_vr, sections)
            screenshots_done = True

        # Plot and combine
        report_path, _ = get_output_file_path(state.video_path, ".png", "report")
        create_combined_plot(
            state, ref_sections, gen_sections, screenshots, sections, report_path,
            ref_times, ref_positions, gen_times, gen_positions
        )

def select_random_sections(total_duration, section_duration=10, num_sections=6):
    sections = []
    segment_duration = total_duration / num_sections  # Duration of each segment

    for i in range(num_sections):
        # Define the start and end of the current segment
        segment_start = i * segment_duration
        segment_end = (i + 1) * segment_duration

        # Ensure the 10-second section stays within the segment
        max_start = segment_end - section_duration
        if max_start < segment_start:
            # raise ValueError(f"Segment {i} is too short to fit a {section_duration}-second section.")
            continue

        # Randomly select a start time within the segment
        start = np.random.uniform(segment_start, max_start)
        end = start + section_duration

        # Add the section to the list
        sections.append((start, end))

    return sections

def extract_section(times, positions, start, end):
    if times is None or not isinstance(times, (list, tuple)):
        log_fun.warning(f"No times for current section {start} - {end}")
        return [], []
    start_ms = start * 1000
    end_ms = end * 1000
    indices = [i for i, t in enumerate(times) if start_ms <= t <= end_ms]
    return [times[i] for i in indices], [positions[i] for i in indices]

def capture_screenshots(video_path, is_vr, sections):
    cap = cv2.VideoCapture(video_path)
    screenshots = []
    for start, _ in sections:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        screenshots.append(np.zeros((100, 160, 3), dtype=np.uint8))
    cap.release()
    return screenshots