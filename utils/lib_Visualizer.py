import cv2
import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    def draw_bounding_box(self, image, box, label, color, offset_x=0):
        x1, y1, x2, y2 = map(int, box)
        x1 += offset_x
        x2 += offset_x
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - label_height - baseline), (x1 + label_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return image

    def draw_gauge(self, image, distance):
        gauge_width = 20
        gauge_height = 200
        gauge_x = int(3 * image.shape[1] / 4)
        gauge_y = int(3 * image.shape[0] / 5)
        cv2.rectangle(image, (gauge_x, gauge_y), (gauge_x + gauge_width, gauge_y + gauge_height), (0, 0, 0), -1)
        fill_height = int((distance / 100) * gauge_height)
        cv2.rectangle(image, (gauge_x, gauge_y + gauge_height - fill_height), (gauge_x + gauge_width, gauge_y + gauge_height), (0, 255, 0), -1)
        # Draw the distance text on top of the filled part of the gauge
        #cv2.putText(image, str(int(distance)), (gauge_x + gauge_width // 2, gauge_y + gauge_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, str(int(distance)), (gauge_x, gauge_y + gauge_height - fill_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image
"""
    def draw_limited_graph(self, image, distances, frames, nb_frames):
        # Plot the frame indices vs positions for the last nb frames.
        # Determine the number of frames to plot
        num_frames_to_plot = min(nb_frames, len(distances))

        # Get the last 'num_frames_to_plot' frames
        last_frame_indices = frames[-num_frames_to_plot:]
        last_positions = distances[-num_frames_to_plot:]

        graph_height = image.shape[0] // 4
        graph_width = image.shape[1] - 20 - 10
        graph_x = 10
        graph_y = image.shape[0] // 10

        fig = plt.figure(figsize=(graph_width / 100, graph_height / 100), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(range(len(last_frame_indices)), last_positions, color='green', linewidth=1)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, nb_frames)  # Limit the x-axis to nb_frames
        fig.canvas.draw()

        # Get the width and height of the figure
        width, height = fig.canvas.get_width_height()

        # Convert the figure to an image
        graph_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape((height, width, 3))
        graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGBA2RGB)  # Convert from RGBA to RGB
        graph_image = cv2.resize(graph_image, (graph_width, graph_height))

        #graph_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(
        #    fig.canvas.get_width_height()[::-1] + (3,))
        #graph_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(
        #    fig.canvas.get_width_height()[::-1] + (4,))
        #graph_image = cv2.resize(graph_image, (graph_width, graph_height))

        image[graph_y:graph_y + graph_height, graph_x:graph_x + graph_width] = cv2.addWeighted(
            image[graph_y:graph_y + graph_height, graph_x:graph_x + graph_width],
            0.5,
            graph_image,
            0.5,
            0
        )
        plt.close(fig)
        return image

    def draw_graph(self, image, distances, frames, reference_positions, reference_frames):
        graph_height = image.shape[0] // 4
        graph_width = image.shape[1] - 20 - 10
        graph_x = 10
        graph_y = image.shape[0] // 10
        fig = plt.figure(figsize=(graph_width / 100, graph_height / 100), dpi=100)
        ax = fig.add_subplot(111)
        if reference_positions:
            ax.plot(reference_frames, reference_positions, color='red', linewidth=1)
        ax.plot(range(len(distances)), distances, color='green', linewidth=1)
        # Draw the last 600 frames

        ax.set_ylim(0, 100)
        ax.set_xlim(0, len(distances))
        fig.canvas.draw()
        graph_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        graph_image = cv2.resize(graph_image, (graph_width, graph_height))
        image[graph_y:graph_y + graph_height, graph_x:graph_x + graph_width] = cv2.addWeighted(
            image[graph_y:graph_y + graph_height, graph_x:graph_x + graph_width],
            0.5,
            graph_image,
            0.5,
            0
        )
        plt.close(fig)
        return image

"""
