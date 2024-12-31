import os
import json
from simplification.cutil import simplify_coords
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from params.config import heatmap_colors, step_size, vw_filter_coeff
import cv2

class FunscriptGenerator:
    def generate(self, raw_funscript_path, funscript_data, fps, TestMode = False):
        output_path = raw_funscript_path[:-18] + '.funscript'
        if len(funscript_data) == 0:
            # Read the funscript data from the JSON file
            with open(raw_funscript_path, 'r') as f:
                print(f"Loading funscript from {raw_funscript_path}")
                try:
                    data = f.read() #json.load(f)
                    data = eval(data)
                except:
                    print(f"Error loading funscript from {raw_funscript_path}")
        else:
            data = funscript_data

        try:
            print(f"Generating funscript based on {len(data)} points...")
            self.filtered_positions = simplify_coords(data, vw_filter_coeff)  # Use VW algorithm
            print(f"Lenghth of filtered positions: {len(self.filtered_positions) + 1}")
            self.write_funscript(self.filtered_positions, output_path, fps)
            print(f"Funscript generated and saved to {output_path}")
            self.generate_heatmap(output_path, output_path[:-10] + f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

            # Now proceeding with remapping / adjusting
            # Adjust peaks and lows
            ats = [p[0] for p in self.filtered_positions]
            positions = [p[1] for p in self.filtered_positions]
            adjusted_positions = self.adjust_peaks_and_lows(positions, peak_boost=15, low_reduction=25,
                                                            max_flat_length=3)
            # recombine ats and positions
            adjusted_positions = list(zip(ats, adjusted_positions))
            remapped_path = output_path[:-10] + '_adjusted.funscript'
            self.write_funscript(adjusted_positions, remapped_path, fps)
            self.generate_heatmap(remapped_path,
                                  remapped_path[:-10] + f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

        except:
            print(f"Error loading raw funscript from {raw_funscript_path}")

        #if LiveDisplayMode:
        #    # plot a comparative graph
        #    self.plot_comparison(self.points, self.filtered_positions, points_v2)


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

        # Print loaded data for debugging
        print(f"Times: {times}")
        print(f"Positions: {positions}")
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


    def filter_positions(self, positions, fps):
        if not positions:
            return []

        filtered_positions = [positions[0]]  # Start with the first position

        min_interval_ms = 50  # Minimum interval between points in milliseconds
        slope_threshold = 0.2  # Adjusted slope threshold for gradual changes

        def calculate_slope(pos1, time1, pos2, time2):
            return (pos2 - pos1) / (time2 - time1) if (time2 - time1) != 0 else 0

        for i in range(1, len(positions) - 1):
            current_pos = positions[i]

            # Skip None values
            if current_pos is None:
                continue

            prev_pos = positions[i - 1]
            next_pos = positions[i + 1]

            # Skip consecutive duplicate positions
            if current_pos[1] == filtered_positions[-1][1] and current_pos[1] == next_pos[1]:
                continue

            # Calculate slopes
            slope_prev = calculate_slope(prev_pos[1], prev_pos[0], current_pos[1], current_pos[0])
            slope_next = calculate_slope(current_pos[1], current_pos[0], next_pos[1], next_pos[0])
            slope_diff = abs(slope_next - slope_prev)

            is_local_extreme = ((current_pos[1] >= prev_pos[1] and current_pos[1] > next_pos[1])
                                or (current_pos[1] > prev_pos[1] and current_pos[1] >= next_pos[1])
                                or (current_pos[1] <= prev_pos[1] and current_pos[1] < next_pos[1])
                                or (current_pos[1] < prev_pos[1] and current_pos[1] <= next_pos[1]))

            # Add to filtered lists based on conditions
            if (is_local_extreme or slope_diff > slope_threshold) and (abs(
                    current_pos[0] - filtered_positions[-1][0]) * 1000 / fps) > min_interval_ms:
                filtered_positions.append(current_pos)

        # Ensure the last point meets the interval requirement
        if len(filtered_positions) > 1 and positions[-1][0] - filtered_positions[-1][0] >= min_interval_ms:
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

    def compare_funscripts(self, reference_path, script1, video_path, isVR, output_image_path, script2=None):
        generated_paths = []
        generated_paths.append(script1)
        if script2 is not None:
            generated_paths.append(script2)

        # Load reference funscript
        ref_times, ref_positions, _, _ = self.load_funscript(reference_path)

        # Determine total duration in seconds
        total_duration = ref_times[-1] / 1000 if ref_times else 0

        # Select 6 random non-overlapping 20-second sections
        sections = self.select_random_sections(total_duration, section_duration=10, num_sections=6)

        screenshots_done = False

        # Load generated funscripts
        for generated_path in generated_paths:
            gen_times, gen_positions, _, _ = self.load_funscript(generated_path)
            # Extract data for each section
            ref_sections = []
            gen_sections = []
            for start, end in sections:
                ref_sec = self.extract_section(ref_times, ref_positions, start, end)
                gen_sec = self.extract_section(gen_times, gen_positions, start, end)
                ref_sections.append(ref_sec)
                gen_sections.append(gen_sec)

            if not screenshots_done:
                # Capture screenshots, but only once
                screenshots = self.capture_screenshots(video_path, isVR, sections)
                screenshots_done = True

            # Plot and combine
            #self.create_combined_plot(ref_sections, gen_sections, screenshots, sections, output_image_path)
            # Create the combined plot
            self.create_combined_plot(
                ref_sections, gen_sections, screenshots, sections, output_image_path,
                ref_times, ref_positions, gen_times, gen_positions
            )

    def analyze_funscripts(self, script_paths, video_path, isVR, output_image_path):
        """
        Analyzes one, two, or three funscripts and generates a report.

        Args:
            script_paths (list): List of paths to the funscripts (1, 2, or 3).
            video_path (str): Path to the video file.
            isVR (bool): Whether the video is VR (to crop the frame).
            output_image_path (str): Path to save the output image.
        """
        if not script_paths or len(script_paths) > 3:
            raise ValueError("Please provide 1, 2, or 3 funscript paths.")

        # Load funscripts
        funscripts = []
        for path in script_paths:
            times, positions, _, _ = self.load_funscript(path)
            funscripts.append((times, positions))

        # Determine total duration in seconds
        total_duration = funscripts[0][0][-1] / 1000 if funscripts[0][0] else 0

        # Select 6 random non-overlapping 20-second sections
        sections = self.select_random_sections(total_duration, section_duration=10, num_sections=6)

        # Capture screenshots
        screenshots = self.capture_screenshots(video_path, isVR, sections)

        # Extract data for each section
        section_data = []
        for times, positions in funscripts:
            sections_data = []
            for start, end in sections:
                sec = self.extract_section(times, positions, start, end)
                sections_data.append(sec)
            section_data.append(sections_data)

        # Create the combined plot
        self.create_flexible_plot(section_data, screenshots, sections, output_image_path, funscripts)

    def create_flexible_plot(self, section_data, screenshots, sections, output_image_path, funscripts):
        """
        Creates a flexible plot for 1, 2, or 3 funscripts.

        Args:
            section_data (list): List of section data for each funscript.
            screenshots (list): List of screenshots for each section.
            sections (list): List of tuples (start, end) for each section.
            output_image_path (str): Path to save the plot.
            funscripts (list): List of tuples (times, positions) for each funscript.
        """
        num_funscripts = len(funscripts)
        rows_per_funscript = 2  # Heatmap + stats
        total_rows = rows_per_funscript + 6  # 6 sections

        # Create a grid: total_rows rows, num_funscripts columns
        #   fig, axes = plt.subplots(nrows=total_rows, ncols=2, figsize=(10 * num_funscripts, 28))
        fig, axes = plt.subplots(nrows=total_rows, ncols=2, figsize=(10, 28))
        if num_funscripts == 1:
            axes = axes.reshape(-1, 1)  # Ensure axes is 2D even for a single funscript

        # Plot heatmaps and stats for each funscript
        for i, (times, positions) in enumerate(funscripts):
            # Heatmap
            ax_heatmap = axes[0, i]
            self.generate_heatmap_inline(ax_heatmap, times, positions)
            ax_heatmap.set_title(f'Funscript {i + 1} Heatmap')

            # Stats
            ax_stats = axes[1, i]
            ax_stats.axis('off')
            metrics = self._calculate_metrics(times, positions)
            stats_text = (
                f"Stats:\n"
                f"Number of Strokes: {metrics['num_strokes']}\n"
                f"Avg Stroke Duration: {metrics['avg_stroke_duration']:.2f}s\n"
                f"Avg Speed: {int(metrics['avg_speed'])} positions/s\n"
                f"Avg Depth of Stroke: {int(metrics['avg_depth'])}\n"
                f"Avg Max: {int(metrics['avg_max'])}\n"
                f"Avg Min: {int(metrics['avg_min'])}"
            )
            ax_stats.text(0.1, 0.5, stats_text, fontsize=10, va='center')

        # original version - Plot sections in the remaining rows
        """for i in range(6):  # 6 sections
            # Plot screenshot in the first column
            ax_screenshot = axes[(i + 2) * 2]  # First column of each row
            ax_screenshot.imshow(screenshots[i])
            ax_screenshot.axis('off')  # Hide axes for the screenshot

            # Plot funscript data in the second column
            ax_plot = axes[(i + 2) * 2 + 1]  # Second column of each row
            gen_times_sec = [t / 1000 for t in gen_sections[i][0]]
            ax_plot.plot(gen_times_sec, gen_sections[i][1], label='Generated', color='blue')

            ref_times_sec = [t / 1000 for t in ref_sections[i][0]]
            ax_plot.plot(ref_times_sec, ref_sections[i][1], label='Reference', color='red')

            # Set labels and title
            start_time = datetime.datetime.fromtimestamp(sections[i][0]).strftime('%H:%M:%S')
            end_time = datetime.datetime.fromtimestamp(sections[i][1]).strftime('%H:%M:%S')
            ax_plot.set_title(f'Section {i + 1}: {start_time} - {end_time}')
            ax_plot.set_xlabel('Time (s)')
            ax_plot.set_ylabel('Position')
            ax_plot.legend()
            ax_plot.set_xlim(sections[i][0], sections[i][1])
        """

        # Adjust layout
        plt.tight_layout()

        # Plot sections
        for i in range(6):  # 6 sections
            #ax_screenshot = axes[rows_per_funscript + i]  # axes[rows_per_funscript + i, j]
            ax_screenshot = axes[(i + 2) * 2]  # First column of each row
            ax_screenshot.imshow(screenshots[i])
            ax_screenshot.axis('off')

            # Plot funscript data in the second column
            ax_plot = axes[(i + 2) * 2 + 1]  # Second column of each row
            for j in range(num_funscripts):

                #ax_plot = axes[rows_per_funscript + i, j]
                times_sec = [t / 1000 for t in section_data[j][i][0]]
                ax_plot.plot(times_sec, section_data[j][i][1], label=f'Funscript {j + 1}',
                             color=['blue', 'red', 'green'][j])

            # Set labels and title
            start_time = datetime.datetime.fromtimestamp(sections[i][0]).strftime('%H:%M:%S')
            end_time = datetime.datetime.fromtimestamp(sections[i][1]).strftime('%H:%M:%S')
            ax_plot.set_title(f'Section {i + 1}: {start_time} - {end_time}')
            ax_plot.set_xlabel('Time (s)')
            ax_plot.set_ylabel('Position')
            ax_plot.legend()
            ax_plot.set_xlim(sections[i][0], sections[i][1])

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        plt.savefig(output_image_path[:-4] + f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png", dpi=200)
        plt.close()

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
        # Create a 8x2 grid: 8 rows, 2 columns (1st row for heatmaps, 2nd row for comparative info, 3rd-8th for sections)
        fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(10, 28))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Generate and add reference heatmap to the first row, first column
        ax_ref_heatmap = axes[0]
        self.generate_heatmap_inline(ax_ref_heatmap, ref_times, ref_positions)
        ax_ref_heatmap.set_title('Reference Funscript Heatmap')

        # Generate and add generated heatmap to the first row, second column
        ax_gen_heatmap = axes[1]
        self.generate_heatmap_inline(ax_gen_heatmap, gen_times, gen_positions)
        ax_gen_heatmap.set_title('Generated Funscript Heatmap')

        # Calculate comparative information
        ref_metrics = self._calculate_metrics(ref_times, ref_positions)
        gen_metrics = self._calculate_metrics(gen_times, gen_positions)

        # Add comparative information to the second row
        ax_comparative_left = axes[2]  # First column of the second row
        ax_comparative_left.axis('off')  # Hide axes for the comparative info
        ref_comparative_text = (
            f"Reference:\n"
            f"Number of Strokes: {ref_metrics['num_strokes']}\n"
            f"Avg Stroke Duration: {ref_metrics['avg_stroke_duration']:.2f}s\n"
            f"Avg Speed: {int(ref_metrics['avg_speed'])} positions/s\n"
            f"Avg Depth of Stroke: {int(ref_metrics['avg_depth'])}\n"
            f"Avg Max: {int(ref_metrics['avg_max'])}\n"
            f"Avg Min: {int(ref_metrics['avg_min'])}"
        )
        ax_comparative_left.text(0.1, 0.5, ref_comparative_text, fontsize=10, va='center')

        ax_comparative_right = axes[3]  # Second column of the second row
        ax_comparative_right.axis('off')  # Hide axes for the comparative info
        gen_comparative_text = (
            f"Generated:\n"
            f"Number of Strokes: {gen_metrics['num_strokes']}\n"
            f"Avg Stroke Duration: {gen_metrics['avg_stroke_duration']:.2f}s\n"
            f"Avg Speed: {int(gen_metrics['avg_speed'])} positions/s\n"
            f"Avg Depth of Stroke: {int(gen_metrics['avg_depth'])}\n"
            f"Avg Max: {int(gen_metrics['avg_max'])}\n"
            f"Avg Min: {int(gen_metrics['avg_min'])}"
        )
        ax_comparative_right.text(0.1, 0.5, gen_comparative_text, fontsize=10, va='center')

        # Plot sections in the remaining rows
        for i in range(6):  # 6 sections
            # Plot screenshot in the first column
            ax_screenshot = axes[(i + 2) * 2]  # First column of each row
            ax_screenshot.imshow(screenshots[i])
            ax_screenshot.axis('off')  # Hide axes for the screenshot

            # Plot funscript data in the second column
            ax_plot = axes[(i + 2) * 2 + 1]  # Second column of each row
            gen_times_sec = [t / 1000 for t in gen_sections[i][0]]
            ax_plot.plot(gen_times_sec, gen_sections[i][1], label='Generated', color='blue')

            ref_times_sec = [t / 1000 for t in ref_sections[i][0]]
            ax_plot.plot(ref_times_sec, ref_sections[i][1], label='Reference', color='red')

            # Set labels and title
            start_time = datetime.datetime.fromtimestamp(sections[i][0]).strftime('%H:%M:%S')
            end_time = datetime.datetime.fromtimestamp(sections[i][1]).strftime('%H:%M:%S')
            ax_plot.set_title(f'Section {i + 1}: {start_time} - {end_time}')
            ax_plot.set_xlabel('Time (s)')
            ax_plot.set_ylabel('Position')
            ax_plot.legend()
            ax_plot.set_xlim(sections[i][0], sections[i][1])

        # Adjust layout
        plt.tight_layout()

        # Save the combined figure
        plt.savefig(output_image_path[:-4] + f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png", dpi=200)
        #plt.show()

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
            positions (list): List of positions (0-100) from the funscript.
            peak_boost (int): Amount to increase peaks by.
            low_reduction (int): Amount to decrease lows by.
            max_flat_length (int): Maximum allowed length of flat sections at 0 or 100.

        Returns:
            list: Adjusted positions.
        """
        if not positions or len(positions) < 3:
            return positions

        # Convert positions to a numpy array for easier manipulation
        positions = np.array(positions)

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
