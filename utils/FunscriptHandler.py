import os
import json
from simplification.cutil import simplify_coords
import matplotlib.pyplot as plt

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

            filter_coeff = 12.0

            self.filtered_positions = simplify_coords(data, filter_coeff)  # Use VW algorithm

            print(f"Lenghth of filtered positions: {len(self.filtered_positions)}")

            #output_path = raw_funscript_path[:-18] + '_vw_' + str(filter_coeff) + '.funscript'
            #output_path = raw_funscript_path[:-18] + '_vw_' + str(filter_coeff) + '.funscript'

            self.write_funscript(self.filtered_positions, output_path, fps)

            print(f"Funscript generated and saved to {output_path}")

        except:
            print(f"Error loading raw funscript from {raw_funscript_path}")
        """
        # for alternative version, make every point 0 if distance is under 20
        points_v2 = []
        multiplier = 1.2
        for i in range(len(self.filtered_positions)):
            if self.filtered_positions[i][1] < 10:
                distance = 0
            else:
                if self.filtered_positions[i][1] < 50:
                    distance = int(self.filtered_positions[i][1] / multiplier)
                else:
                    distance = min(int(self.filtered_positions[i][1] * multiplier), 100)
            # print(f"Point {i}: was {self.filtered_positions[i][1]}, now is {distance}")
            points_v2.append((self.filtered_positions[i][0], distance))

        write_path = output_path[:-10] + '_remapped.funscript'
        self.write_funscript(points_v2, write_path, fps)
        """

        #if TestMode:
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

    def dump_section(self, section_id, frames, distances, output_path, fps):
        print(f"Generating json file for section {section_id} content based on {len(distances)} distances...")
        points = []
        # build points based on the list of frames, distances
        for i in range(len(frames)):
            points.append((int(frames[i] * 1000 / fps), int(distances[i])))
        filtered_positions = simplify_coords(points, 5.0)
        print(f"Filtered positions: {len(filtered_positions)}")

        # Convert filtered positions to a list of dictionaries
        filtered_data = [{"at": timing_ms, "pos": distance} for timing_ms, distance in filtered_positions]

        # Create the output file path
        output_file = f"{output_path}_section_{section_id}.json"

        # Dump filtered positions to json file
        with open(output_file, 'w') as f:
            json.dump(filtered_data, f, indent=4)

        print(f"Section {section_id} dumped to {output_file}")

    def assemble_sections(self, number_of_sections, output_path):
        # assemble json section files to a final funscript file
        final_data = []

        # Load each section's JSON file and append its data to the final data
        for section_id in range(1, number_of_sections + 1):
            section_file = f"{output_path}_section_{section_id}.json"
            with open(section_file, 'r') as f:
                section_data = json.load(f)
                final_data.extend(section_data)

        # Sort the final data by frame number
        final_data.sort(key=lambda x: x['at'])

        # Create the final funscript file
        final_file = f"{output_path}"
        #with open(final_file, 'w') as f:
        #    json.dump(final_data, f, indent=4)
        with open(output_path, 'w') as f:
            #f.write("[\n")
            f.write('{"version":"1.0","inverted":false,"range":95,"author":"kAI","actions":[')
            i = 0
            for timing_ms, position in self.filtered_positions:
                if i > 0:
                    f.write(",")
                f.write(f' {{"at": {timing_ms}, "pos": {int(position)}}}')
                i += 1
            f.write("]}\n")

        print(f"Final funscript file assembled and saved to {final_file}")

    def plot_comparison(self, original_points, rdp_filtered_points, vw_filtered_points):
        plt.figure(figsize=(12, 6))

        # Plot original points
        plt.plot([p[0] for p in original_points], [p[1] for p in original_points], 'b-', label='Original Points')

        # Plot RDP filtered points
        plt.plot([p[0] for p in rdp_filtered_points], [p[1] for p in rdp_filtered_points], 'r-', label='RDP Filtered Points')

        # Plot VW filtered points
        plt.plot([p[0] for p in vw_filtered_points], [p[1] for p in vw_filtered_points], 'g-', label='VW Filtered Points')

        plt.xlabel('Frames')
        plt.ylabel('Distances')
        plt.title('Comparison of RDP and VW Filtered Positions')
        plt.legend()
        plt.savefig('comparison_plot.png')
        plt.show()
        #plt.savefig('comparison_plot.png')

    def smooth_distance_ema(self, distances, alpha=0.3):
        smoothed = [distances[0]]
        for i in range(1, len(distances)):
            smoothed.append(alpha * distances[i] + (1 - alpha) * smoothed[-1])
        return [[i, smoothed[i]] for i in range(len(smoothed))]

    def keep_significant_points(self, positions):
        significant = [positions[0]]
        for i in range(1, len(positions) - 1):
            if positions[i][1] != positions[i - 1][1] and positions[i][1] != positions[i + 1][1]:
                significant.append(positions[i])
        significant.append(positions[-1])
        return significant

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