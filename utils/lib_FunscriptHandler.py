import logging
import os
import json
from simplification.cutil import simplify_coords
import numpy as np
import cv2
import datetime

from params.config import heatmap_colors, step_size, vw_filter_coeff

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

class FunscriptGenerator:
    def generate(self, global_state):
        output_path = global_state.video_file[:-3] + f"funscript"

        # Backup output file if it exists
        if os.path.exists(output_path):
            print(f"Output path {output_path} already exists, backing up as {output_path}.bak...")
            os.rename(output_path, output_path + ".bak")

        raw_funscript_path = global_state.video_file[:-4] + f"_rawfunscript.json"

        if len(global_state.funscript_data) == 0:
            print("len funscript data is 0, trying to load file")
            # Read the funscript data from the JSON file
            with open(raw_funscript_path, 'r') as f:
                print(f"Loading funscript from {raw_funscript_path}")
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"Error loading funscript from {raw_funscript_path}: {e}")
                    return
        else:
            data = global_state.funscript_data

        try:
            print(f"Generating funscript based on {len(data)} points...")
            # positions = data
            positions = self.filter_positions(data, global_state.fps)
            print(f"Length of filtered positions: {len(positions)}")

            # Extract timestamps and positions
            ats = [p[0] for p in positions]
            positions = [p[1] for p in positions]

            # Remap positions to 0-100 range
            print("Positions adjustment - step 1 (remapping)")
            adjusted_positions = np.interp(positions, (min(positions), max(positions)), (0, 100))

            # Apply thresholding
            if global_state.threshold_enabled:
                print(f"Positions adjustment - step 2 (thresholding)")
                adjusted_positions = adjusted_positions.tolist()  # Convert to list
                adjusted_positions = [
                    0 if p < global_state.threshold_low else 100 if p > global_state.threshold_high else p for p in
                    adjusted_positions]
            else:
                print("Skipping positions adjustment - step 2 (thresholding)")

            # Apply amplitude boosting
            if global_state.boost_enabled:
                print("Positions adjustment - step 3 (amplitude boosting)")
                #self.boost_amplitude(adjusted_positions, boost_factor=1.2, min_value=0, max_value=100)
                adjusted_positions = self.adjust_peaks_and_lows(adjusted_positions,
                                                                peak_boost=global_state.boost_up_percent,
                                                                low_reduction=global_state.boost_down_percent)
            else:
                print("Skipping positions adjustment - step 3 (amplitude boosting)")

            # Round position values to the closest multiple of 5, still between 0 and 100
            print("Positions adjustment - step 4 (rounding to the closest multiple of 5)")
            #adjusted_positions = [round(p, -1) for p in adjusted_positions] # multiple of 10
            adjusted_positions = [round(p / global_state.rounding) * global_state.rounding for p in adjusted_positions]

            # Recombine timestamps and adjusted positions
            print("Re-assembling ats and positions")
            zip_adjusted_positions = list(zip(ats, adjusted_positions))

            # Apply VW simplification if enabled
            if global_state.vw_simplification_enabled:
                print("Positions adjustment - step 5 (VW algorithm simplification)")
                zip_adjusted_positions = simplify_coords(zip_adjusted_positions, global_state.vw_factor)
            else:
                print("Skipping positions adjustment - step 5 (VW algorithm simplification)")

            # Write the final funscript
            self.write_funscript(zip_adjusted_positions, output_path, global_state.fps)
            print(f"Funscript generated and saved to {output_path}")

            # Generate a heatmap
            self.generate_heatmap(output_path,
                                  output_path[:-10] + f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
        except Exception as e:
            print(f"Error generating funscript: {e}")

    def write_funscript(self, distances, output_path, fps):
        with open(output_path, 'w') as f:
            f.write('{"version":"1.0","inverted":false,"range":95,"author":"kAI","actions":[{"at": 0, "pos": 100},')
            i = 0
            for frame, position in distances:
                time_ms = int(frame * 1000 / fps)
                if i > 0:
                    f.write(",")
                f.write(f' {{"at": {time_ms}, "pos": {int(position)}}}')
                i += 1
            f.write("]}\n")

    def generate_heatmap(self, funscript_path, output_image_path):
        # Load funscript data
        times, positions, _, _ = self.load_funscript(funscript_path)
        if not times or not positions:
            print("Failed to load funscript data.")
            return

        # add a timing: 0, position: 100 at the beginning if no value for 0
        if times[0] != 0:
            times.insert(0, 0)
            positions.insert(0, 100)

        # Print loaded data for debugging
        #print(f"Times: {times}")
        #print(f"Positions: {positions}")
        print(f"Total Actions: {len(times)}")
        print(f"Time Range: {times[0]} to {datetime.timedelta(seconds=int(times[-1] / 1000))}")

        # Calculate speed (position change per time interval)
        speeds = np.abs(np.diff(positions) / np.diff(times)) * 1000  # Positions per second
        print(f"Speeds: {speeds}")

        def get_color(intensity):
            if intensity <= 0:
                return heatmap_colors[0]
            if intensity > 5 * step_size:
                return heatmap_colors[6]
            intensity += step_size / 2.0
            index = int(intensity // step_size)
            t = (intensity - index * step_size) / step_size
            return [
                heatmap_colors[index][0] + (heatmap_colors[index + 1][0] - heatmap_colors[index][0]) * t,
                heatmap_colors[index][1] + (heatmap_colors[index + 1][1] - heatmap_colors[index][1]) * t,
                heatmap_colors[index][2] + (heatmap_colors[index + 1][2] - heatmap_colors[index][2]) * t
            ]

        # Create figure and plot
        plt.figure(figsize=(30, 2))
        ax = plt.gca()

        # Draw lines between points with colors based on speed
        for i in range(len(times) - 1):
            x_start = times[i] / 1000  # Convert ms to seconds
            x_end = times[i + 1] / 1000
            y_start = positions[i]
            y_end = positions[i + 1]
            speed = speeds[i]

            # Get color based on speed
            color = get_color(speed)
            line_color = (color[0] / 255, color[1] / 255, color[2] / 255)  # Normalize to [0, 1]

            # Plot the line
            ax.plot([x_start, x_end], [y_start, y_end], color=line_color, linewidth=2)

        # Customize plot
        ax.set_title(
            f'Funscript Heatmap\nDuration: {datetime.timedelta(seconds=int(times[-1] / 1000))} - Avg. Speed {int(np.mean(speeds))} - Actions: {len(times)}')
        ax.set_xlabel('Time (s)')
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xlim(times[0] / 1000, times[-1] / 1000)
        ax.set_ylim(0, 100)

        # Remove borders (spines)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add colorbar
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_heatmap", [
            (heatmap_colors[i][0] / 255, heatmap_colors[i][1] / 255, heatmap_colors[i][2] / 255) for i in
            range(len(heatmap_colors))
        ])
        norm = mcolors.Normalize(vmin=0, vmax=5 * step_size)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.2,
        #                    ticks=np.arange(0, 5 * step_size + 1, step_size))
        # cbar.set_label('Speed (positions/s)')

        # Save the figure
        plt.savefig(output_image_path, bbox_inches='tight', dpi=200)  # Increase resolution
        plt.close()
        print(f"Funscript heatmap saved to {output_image_path}")

    def boost_amplitude(self, signal, boost_factor, min_value=0, max_value=100):
        """
        Boosts the amplitude of a signal and ensures it stays within a specified range.

        Parameters:
            signal (list or np.ndarray): The input signal to boost.
            boost_factor (float): The factor by which to boost the signal's amplitude.
            min_value (int, optional): Minimum value of the range. Default is 0.
            max_value (int, optional): Maximum value of the range. Default is 100.

        Returns:
            np.ndarray: The boosted signal clamped to the specified range.
        """
        # Convert to numpy array for easier manipulation
        signal = np.array(signal)

        # Calculate the midpoint for rescaling
        midpoint = (min_value + max_value) / 2

        # Boost the signal
        boosted_signal = midpoint + (signal - midpoint) * boost_factor

        # Clamp the signal to the specified range
        boosted_signal = np.clip(boosted_signal, min_value, max_value)

        return boosted_signal

    def filter_positions(self, positions, fps):
        """
        Filters positions to remove unnecessary points while preserving key features.

        Args:
            positions (list or np.array): List of [timestamp, value] pairs.
            fps (int): Frames per second of the video.

        Returns:
            list: Filtered list of [timestamp, value] pairs.
        """
        if not positions:
            return []

        # Ensure positions is a list of [timestamp, value] pairs
        positions = np.array(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must be a list of [timestamp, value] pairs")

        filtered_positions = [positions[0]]  # Start with the first position

        min_interval_ms = 50  # Minimum interval between points in milliseconds
        slope_threshold = 0.2  # Adjusted slope threshold for gradual changes

        for i in range(1, len(positions) - 1):
            current_pos = positions[i]
            prev_pos = positions[i - 1]
            next_pos = positions[i + 1]

            # Skip consecutive duplicate positions
            if current_pos[1] == filtered_positions[-1][1] and current_pos[1] == next_pos[1]:
                continue

            # Calculate slopes
            time_diff_prev = current_pos[0] - prev_pos[0]
            time_diff_next = next_pos[0] - current_pos[0]

            slope_prev = (current_pos[1] - prev_pos[1]) / time_diff_prev if time_diff_prev != 0 else 0
            slope_next = (next_pos[1] - current_pos[1]) / time_diff_next if time_diff_next != 0 else 0
            slope_diff = abs(slope_next - slope_prev)

            # Check if the current position is a local extreme
            is_local_extreme = (
                    (current_pos[1] >= prev_pos[1] and current_pos[1] > next_pos[1]) or
                    (current_pos[1] > prev_pos[1] and current_pos[1] >= next_pos[1]) or
                    (current_pos[1] <= prev_pos[1] and current_pos[1] < next_pos[1]) or
                    (current_pos[1] < prev_pos[1] and current_pos[1] <= next_pos[1])
            )

            # Calculate time difference in milliseconds
            time_diff_ms = (current_pos[0] - filtered_positions[-1][0]) * 1000 / fps

            # Add to filtered list based on conditions
            if (is_local_extreme or slope_diff > slope_threshold) and time_diff_ms > min_interval_ms:
                filtered_positions.append(current_pos)

        # Ensure the last point meets the interval requirement
        if len(filtered_positions) > 1 and (
                positions[-1][0] - filtered_positions[-1][0]) * 1000 / fps >= min_interval_ms:
            filtered_positions.append(positions[-1])

        return filtered_positions

    def load_funscript(self, funscript_path):
        # if the funscript path exists
        if not os.path.exists(funscript_path):
            print(f"Funscript not found at {funscript_path}")
            return None, None, None, None

        with open(funscript_path, 'r') as f:
            try:
                print(f"Loading funscript from {funscript_path}")
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}")
                print(f"Error occurred at line {e.lineno}, column {e.colno}")
                print("Dumping the problematic JSON data:")
                f.seek(0)  # Reset file pointer to the beginning
                print(f.read())
                return None, None, None, None

        times = []
        positions = []

        for action in data['actions']:
            times.append(action['at'])
            positions.append(action['pos'])
        print(f"Loaded funscript with {len(times)} actions")

        # Access the chapters
        chapters = data.get("metadata", {}).get("chapters", [])

        relevant_chapters_export = []
        irrelevant_chapters_export = []
        # Print the chapters
        for chapter in chapters:
            if len(chapter['startTime']) > 8:
                chapter['startTime'] = chapter['startTime'][:8]
            if len(chapter['endTime']) > 8:
                chapter['endTime'] = chapter['endTime'][:8]
            print(f"Chapter: {chapter['name']}, Start: {chapter['startTime']}, End: {chapter['endTime']}")
            # convert 00:00:00 to milliseconds
            startTime_ms = int(chapter['startTime'].split(':')[0]) * 60 * 60 * 1000 + int(
                chapter['startTime'].split(':')[1]) * 60 * 1000 + int(chapter['startTime'].split(':')[2]) * 1000
            endTime_ms = int(chapter['endTime'].split(':')[0]) * 60 * 60 * 1000 + int(
                chapter['endTime'].split(':')[1]) * 60 * 1000 + int(chapter['endTime'].split(':')[2]) * 1000
            if chapter['name'] in ['POV Kissing', 'Close Up', 'Asslicking', 'Creampie']:
                irrelevant_chapters_export.append([chapter['name'], startTime_ms, endTime_ms])
            else:  # if chapter['name'] == 'Blow Job':
                relevant_chapters_export.append([chapter['name'], startTime_ms, endTime_ms])

        return times, positions, relevant_chapters_export, irrelevant_chapters_export

    def create_report_funscripts(self, global_state):
        generated_paths = []
        generated_paths.append(global_state.video_file[:-4] + ".funscript")

        if global_state.reference_script:
            # Load reference funscript
            ref_times, ref_positions, _, _ = self.load_funscript(global_state.reference_script)

            # if no 0 at the beginning, add it
            if ref_times and ref_times[0] != 0:
                ref_times.insert(0, 0)
                ref_positions.insert(0, 100)

            # Determine total duration in seconds
            total_duration = ref_times[-1] / 1000 if ref_times else 0
        else:
            ref_times, ref_positions = [], []
            gen_times, gen_positions, _, _ = self.load_funscript(generated_paths[0])
            total_duration = gen_times[-1] / 1000 if gen_times else 0

        # Select 6 random non-overlapping 20-second sections
        sections = self.select_random_sections(total_duration, section_duration=10, num_sections=6)

        screenshots_done = False
        screenshots = []

        # Load generated funscripts
        for generated_path in generated_paths:
            gen_times, gen_positions, _, _ = self.load_funscript(generated_path)
            # Extract data for each section
            ref_sections = []
            gen_sections = []
            for start, end in sections:
                if global_state.reference_script:
                    ref_sec = self.extract_section(ref_times, ref_positions, start, end)
                    ref_sections.append(ref_sec)
                gen_sec = self.extract_section(gen_times, gen_positions, start, end)
                gen_sections.append(gen_sec)

            if not screenshots_done:
                # Capture screenshots, but only once
                screenshots = self.capture_screenshots(global_state.video_file, global_state.isVR, sections)
                screenshots_done = True

            # Plot and combine
            self.create_combined_plot(
                ref_sections, gen_sections, screenshots, sections, global_state.video_file[:-4] + "_report.png",
                ref_times, ref_positions, gen_times, gen_positions
            )


    def select_random_sections(self, total_duration, section_duration=10, num_sections=6):
        sections = []
        segment_duration = total_duration / num_sections  # Duration of each segment

        for i in range(num_sections):
            # Define the start and end of the current segment
            segment_start = i * segment_duration
            segment_end = (i + 1) * segment_duration

            # Ensure the 10-second section stays within the segment
            max_start = segment_end - section_duration
            if max_start < segment_start:
                raise ValueError(f"Segment {i} is too short to fit a {section_duration}-second section.")

            # Randomly select a start time within the segment
            start = np.random.uniform(segment_start, max_start)
            end = start + section_duration

            # Add the section to the list
            sections.append((start, end))

        return sections

    def extract_section(self, times, positions, start, end):
        if times is None or not isinstance(times, (list, tuple)):
            logging.warning(f"No times for current section {start} - {end}")
            return [], []
        start_ms = start * 1000
        end_ms = end * 1000
        indices = [i for i, t in enumerate(times) if start_ms <= t <= end_ms]
        return [times[i] for i in indices], [positions[i] for i in indices]

    def capture_screenshots(self, video_path, isVR, sections):
        cap = cv2.VideoCapture(video_path)
        screenshots = []
        for start, _ in sections:
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
            ret, frame = cap.read()
            if isVR: # left side of the frame only
                frame = frame[:, :frame.shape[1] // 2]
            if ret:
                screenshots.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                screenshots.append(np.zeros((100, 160, 3), dtype=np.uint8))
        cap.release()
        return screenshots

    def create_combined_plot(self, ref_sections, gen_sections, screenshots, sections, output_image_path, ref_times,
                                 ref_positions, gen_times, gen_positions):
        """
        Creates a combined plot with heatmaps as a header, comparative information, and section comparisons below.

        Args:
            ref_sections (list): List of reference sections (times, positions).
            gen_sections (list): List of generated sections (times, positions).
            screenshots (list): List of screenshots for each section.
            sections (list): List of tuples (start, end) for each section.
            output_image_path (str): Path to save the combined plot.
            ref_times (list): Times from the reference funscript.
            ref_positions (list): Positions from the reference funscript.
            gen_times (list): Times from the generated funscript.
            gen_positions (list): Positions from the generated funscript.
        """
        # Create a flexible grid layout
        fig = plt.figure(figsize=(28, 24))
        gs = gridspec.GridSpec(5, 4, height_ratios=[1, .5, 2, 2, 2], width_ratios=[1, 2, 1, 2])

        # Heatmaps (First row: 2 columns spanning the entire width)
        if ref_sections:
            ax_ref_heatmap = fig.add_subplot(gs[0, :2])
            self.generate_heatmap_inline(ax_ref_heatmap, ref_times, ref_positions)
            ax_ref_heatmap.set_title('Reference Funscript Heatmap', fontsize=14)

        ax_gen_heatmap = fig.add_subplot(gs[0, 2:])
        self.generate_heatmap_inline(ax_gen_heatmap, gen_times, gen_positions)
        ax_gen_heatmap.set_title('Generated Funscript Heatmap', fontsize=14)

        if ref_sections:
            # Comparative information (Second row: 2 columns)
            ax_comparative_left = fig.add_subplot(gs[1, :2])
            ref_metrics = self._calculate_metrics(ref_times, ref_positions)
            ref_comparative_text = (
                f"Reference:\n"
                f"Number of Strokes: {ref_metrics['num_strokes']}\n"
                f"Avg Stroke Duration: {ref_metrics['avg_stroke_duration']:.2f}s\n"
                f"Avg Speed: {int(ref_metrics['avg_speed'])} positions/s\n"
                f"Avg Depth of Stroke: {int(ref_metrics['avg_depth'])}\n"
                f"Avg Max: {int(ref_metrics['avg_max'])}\n"
                f"Avg Min: {int(ref_metrics['avg_min'])}"
            )
            ax_comparative_left.text(0.5, 0.5, ref_comparative_text, fontsize=12, va='center', ha='center')
            ax_comparative_left.axis('off')

        ax_comparative_right = fig.add_subplot(gs[1, 2:])
        gen_metrics = self._calculate_metrics(gen_times, gen_positions)
        gen_comparative_text = (
            f"Generated:\n"
            f"Number of Strokes: {gen_metrics['num_strokes']}\n"
            f"Avg Stroke Duration: {gen_metrics['avg_stroke_duration']:.2f}s\n"
            f"Avg Speed: {int(gen_metrics['avg_speed'])} positions/s\n"
            f"Avg Depth of Stroke: {int(gen_metrics['avg_depth'])}\n"
            f"Avg Max: {int(gen_metrics['avg_max'])}\n"
            f"Avg Min: {int(gen_metrics['avg_min'])}"
        )
        ax_comparative_right.text(0.5, 0.5, gen_comparative_text, fontsize=12, va='center', ha='center')
        ax_comparative_right.axis('off')

        # Sections (Rows 3-8: Each row has 2 subplots for screenshot and data plot)
        for i in range(3, 6):  # Section rows
            for j in range(2):  # Two columns per row
                idx = (i - 3) * 2 + j  # Section index
                if idx >= len(sections):
                    break

                # Screenshot (first column)
                ax_screenshot = fig.add_subplot(gs[i-1, j * 2])
                ax_screenshot.imshow(screenshots[idx])
                ax_screenshot.axis('off')

                # Funscript comparison (second column)
                ax_plot = fig.add_subplot(gs[i-1, j * 2 + 1])
                gen_times_sec = [t / 1000 for t in gen_sections[idx][0]]
                ax_plot.plot(gen_times_sec, gen_sections[idx][1], label='Generated', color='blue')

                if ref_sections:
                    ref_times_sec = [t / 1000 for t in ref_sections[idx][0]]
                    ax_plot.plot(ref_times_sec, ref_sections[idx][1], label='Reference', color='red')

                start_time = datetime.timedelta(seconds=int(sections[idx][0]))  # datetime.datetime.fromtimestamp(sections[idx][0]).strftime('%H:%M:%S')
                end_time = datetime.timedelta(seconds=int(sections[idx][1]))  # datetime.datetime.fromtimestamp(sections[idx][1]).strftime('%H:%M:%S')
                ax_plot.set_title(f'Section {idx + 1}: {start_time} - {end_time}', fontsize=10)
                ax_plot.set_xlabel('Time (s)')
                ax_plot.set_ylabel('Position')
                ax_plot.legend()

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        plt.savefig(output_image_path[:-4] + f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png", dpi=100)
        # plt.show()

    def generate_heatmap_inline(self, ax, times, positions):
        """
        Generates a heatmap on the given axes using the existing `generate_heatmap` logic.

        Args:
            ax (matplotlib.axes.Axes): The axes to plot the heatmap on.
            times (list): List of times from the funscript.
            positions (list): List of positions from the funscript.
        """
        if not times or not positions:
            return

        # Calculate speed (position change per time interval)
        speeds = np.abs(np.diff(positions) / np.diff(times)) * 1000  # Positions per second

        def get_color(intensity):
            if intensity <= 0:
                return heatmap_colors[0]
            if intensity > 5 * step_size:
                return heatmap_colors[-1]
            intensity += step_size / 2.0
            index = int(intensity // step_size)
            t = (intensity - index * step_size) / step_size
            return [
                heatmap_colors[index][0] + (heatmap_colors[index + 1][0] - heatmap_colors[index][0]) * t,
                heatmap_colors[index][1] + (heatmap_colors[index + 1][1] - heatmap_colors[index][1]) * t,
                heatmap_colors[index][2] + (heatmap_colors[index + 1][2] - heatmap_colors[index][2]) * t
            ]

        # Draw lines between points with colors based on speed
        for i in range(len(times) - 1):
            x_start = times[i] / 1000  # Convert ms to seconds
            x_end = times[i + 1] / 1000
            y_start = positions[i]
            y_end = positions[i + 1]
            speed = speeds[i]

            # Get color based on speed
            color = get_color(speed)
            line_color = (color[0] / 255, color[1] / 255, color[2] / 255)  # Normalize to [0, 1]

            # Plot the line
            ax.plot([x_start, x_end], [y_start, y_end], color=line_color, linewidth=2)

        # Customize plot
        ax.set_title(
            f'Funscript Heatmap\nDuration: {datetime.timedelta(seconds=int(times[-1] / 1000))} - Avg. Speed {int(np.mean(speeds))} - Actions: {len(times)}')
        ax.set_xlabel('Time (s)')
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xlim(times[0] / 1000, times[-1] / 1000)
        ax.set_ylim(0, 100)

        # Remove borders (spines)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add colorbar
        cmap = LinearSegmentedColormap.from_list("custom_heatmap", [
            (heatmap_colors[i][0] / 255, heatmap_colors[i][1] / 255, heatmap_colors[i][2] / 255) for i in
            range(len(heatmap_colors))
        ])
        norm = Normalize(vmin=0, vmax=5 * step_size)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

    def _calculate_metrics(self, times, positions):
        """
        Calculates metrics for a funscript.

        Args:
            times (list): List of times from the funscript.
            positions (list): List of positions from the funscript.

        Returns:
            dict: Dictionary containing the calculated metrics.
        """
        if not times or not positions:
            return {}

        # Calculate number of strokes
        num_strokes = len(times) - 1

        # Calculate average stroke duration
        stroke_durations = np.diff(times) / 1000  # Convert to seconds
        avg_stroke_duration = np.mean(stroke_durations)

        # Calculate average speed
        speeds = np.abs(np.diff(positions) / np.diff(times)) * 1000  # Positions per second
        #speeds = np.abs(np.diff(positions) / stroke_durations)
        avg_speed = np.mean(speeds)

        # Calculate average depth of stroke
        depths = np.abs(np.diff(positions))
        avg_depth = np.mean(depths)

        # Calculate average max and min
        avg_max = np.mean([max(positions[i], positions[i + 1]) for i in range(len(positions) - 1)])
        avg_min = np.mean([min(positions[i], positions[i + 1]) for i in range(len(positions) - 1)])

        return {
            'num_strokes': num_strokes,
            'avg_stroke_duration': avg_stroke_duration,
            'avg_speed': avg_speed,
            'avg_depth': avg_depth,
            'avg_max': avg_max,
            'avg_min': avg_min
            }

    def adjust_peaks_and_lows(self, positions, peak_boost=10, low_reduction=10, max_flat_length=5):
        """
        Adjusts the peaks and lows of a funscript while avoiding long flat sections at 0 or 100.

        Args:
            positions (list or np.array): List or array of positions (0-100) from the funscript.
            peak_boost (int): Amount to increase peaks by.
            low_reduction (int): Amount to decrease lows by.
            max_flat_length (int): Maximum allowed length of flat sections at 0 or 100.

        Returns:
            list: Adjusted positions.
        """

        # Ensure positions is a list of [timestamp, value] pairs
        if isinstance(positions, np.ndarray):
            positions = positions.tolist()
        elif not isinstance(positions, list):
            raise ValueError("positions must be a list or numpy array of [timestamp, value] pairs")

        if not positions or len(positions) < 3:
            return positions

        # Convert positions to a numpy array for easier manipulation
        positions = np.array(positions, dtype=float)

        # Identify plateaus before boosting
        original_plateaus = self._find_plateaus(positions)
        # Identify peaks and lows
        peaks = self._find_local_maxima(positions)
        lows = self._find_local_minima(positions)
        # Adjust peaks and lows
        positions[peaks] = np.clip(positions[peaks] + peak_boost, 0, 100)
        positions[lows] = np.clip(positions[lows] - low_reduction, 0, 100)
        # Identify plateaus after boosting
        adjusted_plateaus = self._find_plateaus(positions)
        # Compare plateaus and adjust artificially created flats
        positions = self._compare_and_adjust_plateaus(positions, original_plateaus, adjusted_plateaus, max_flat_length)
        return positions.tolist()

    def _find_local_maxima(self, positions):
        """
        Identifies local maxima (peaks) in the positions.

        Args:
            positions (np.array): Array of positions.

        Returns:
            np.array: Boolean array where True indicates a peak.
        """
        peaks = np.zeros_like(positions, dtype=bool)
        for i in range(1, len(positions) - 1):
            if positions[i] > positions[i - 1] and positions[i] > positions[i + 1]:
                peaks[i] = True
        return peaks

    def _find_local_minima(self, positions):
        """
        Identifies local minima (lows) in the positions.

        Args:
            positions (np.array): Array of positions.

        Returns:
            np.array: Boolean array where True indicates a low.
        """
        lows = np.zeros_like(positions, dtype=bool)
        for i in range(1, len(positions) - 1):
            if positions[i] < positions[i - 1] and positions[i] < positions[i + 1]:
                lows[i] = True
        return lows

    def _find_plateaus(self, positions):
        """
        Identifies flat sections (plateaus) in the positions.

        Args:
            positions (np.array): Array of positions.

        Returns:
            list: List of tuples (start, end) representing the indices of plateaus.
        """
        plateaus = []
        start = 0
        for i in range(1, len(positions)):
            if positions[i] != positions[i - 1]:
                if i - start > 1:  # Plateau must have at least 2 points
                    plateaus.append((start, i - 1))
                start = i
        if len(positions) - start > 1:  # Check the last plateau
            plateaus.append((start, len(positions) - 1))
        return plateaus

    def _compare_and_adjust_plateaus(self, positions, original_plateaus, adjusted_plateaus, max_flat_length):
        """
        Compares plateaus before and after adjustments and breaks artificially created flats.

        Args:
            positions (np.array): Array of positions.
            original_plateaus (list): List of plateaus before adjustments.
            adjusted_plateaus (list): List of plateaus after adjustments.
            max_flat_length (int): Maximum allowed length of flat sections at 0 or 100.

        Returns:
            np.array: Adjusted positions.
        """
        positions = positions.copy()  # Work on a copy to avoid modifying the original array
        for plateau in adjusted_plateaus:
            start, end = plateau
            value = positions[start]

            # Check if the plateau is at 0 or 100 and was not present in the original data
            if (value == 0 or value == 100) and not self._is_plateau_in_original(plateau, original_plateaus):
                # Check if the plateau exceeds the maximum allowed length
                if end - start + 1 > max_flat_length:
                    # Break the plateau by adjusting the values
                    for i in range(start, end + 1):
                        positions[i] = positions[i] + 1 if value == 100 else positions[i] - 1

        return positions

    def _is_plateau_in_original(self, plateau, original_plateaus):
        """
        Checks if a plateau was present in the original data.

        Args:
            plateau (tuple): Tuple (start, end) representing the indices of the plateau.
            original_plateaus (list): List of plateaus in the original data.

        Returns:
            bool: True if the plateau was present in the original data, False otherwise.
        """
        start, end = plateau
        for original_start, original_end in original_plateaus:
            if start >= original_start and end <= original_end:
                return True
        return False

if __name__ == "__main__":
    video_path = "/Users/k00gar/Downloads/SLR_SLR Originals_Vote for me_1920p_51071_FISHEYE190_alpha.mp4"

    funscript_handler = FunscriptGenerator()

    # generate heatmap
    funscript_handler.generate_heatmap(video_path[:-4] + ".funscript", video_path[
                                                                        :-4] + f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
