# VR-Funscript-AI-Generator

---

# Video Processing with YOLO and Funscript Generation

This project is a Python-based tool for generating Funscript files from VR videos using Computer Vision (CV) and AI techniques. It leverages YOLO (You Only Look Once) object detection and custom tracking algorithms to automate the process of creating Funscript files for interactive devices.

## Features

- **YOLO Object Detection**: Uses a pre-trained YOLO model to detect and track objects in video frames.
- **Funscript Generation**: Generates Funscript data based on the tracked objects' movements.
- **Scene Change Detection**: Automatically detects scene changes in the video to improve tracking accuracy.
- **Visualization**: Provides real-time visualization of object tracking and Funscript data (in test mode).
- **VR Support**: Optimized for VR videos, with options to process specific regions of the frame.

## Project Genesis and Evolution

This project started as a dream to automate Funscript generation for VR videos. Here’s a brief history of its development:

### Initial Approach (OpenCV Trackers)

The first version relied on OpenCV trackers to detect and track objects in the video.
While functional, the approach was slow (8–20 FPS) and struggled with occlusions and complex scenes.

### Transition to YOLO

To improve accuracy and speed, the project shifted to using YOLO object detection.
A custom YOLO model was trained on a dataset of VR video frames, significantly improving detection quality.
The new approach runs at 90 FPS on a Mac mini M4 pro, making it much more efficient.

### Original Post

For more details and discussions, check out the original post on EroScripts:  
[VR Funscript Generation Helper (Python + CV/AI)](https://discuss.eroscripts.com/t/vr-funscript-generation-helper-python-now-cv-ai/202554)

## YOLO Model

The YOLO model used in this project is based on YOLOv11n, which was fine-tuned with 9 new classes and 4,500+ frames randomly extracted from a VR video library. Here’s how the model was developed:

### Initial Training:
A few hundred frames were manually tagged and boxed to create an initial dataset.
The model was trained on this dataset to generate preliminary detection results.
### Iterative Improvement
The trained model was used to suggest bounding boxes in additional frames.
The suggested boxes were manually adjusted, and the dataset was expanded.
This process was repeated iteratively to improve the model’s accuracy.
### Final Training
After gathering 4,500+ images and 30,149 annotations, the model was trained for 200 epochs.
YOLOv11s and YOLOv11m were also tested, but YOLOv11n was chosen for its balance of accuracy and inference speed.
### Hardware:
The model runs on a Mac using MPS (Metal Performance Shaders) for accelerated inference on ARM chips.
Other versions of the model (ONNX and PT) are also available for use on other platforms.

## Pipeline Overview

The pipeline for generating Funscript files is as follows:

- **YOLO Object Detection**:
A YOLO model detects relevant objects (e.g., penis, hands, mouth, etc.) in each frame of the video.
The detection results are saved to a .json file.
- **Tracking Algorithm**:
A custom tracking algorithm processes the YOLO detection results to track the positions of objects over time.
The algorithm calculates distances and interactions between objects to determine the Funscript position.
- **Funscript Generation**:
The tracked data is used to generate a raw Funscript file.
- **Simplifier**:
The raw Funscript data is simplified to remove noise and smooth out the motion.
The final .funscript file is saved.
- **Heatmap Generation**:
A heatmap is generated to visualize the Funscript data.


## Prerequisites

Before using this project, ensure you have the following installed:

- **Python 3.8 or higher**
- **FFmpeg** (for video processing)
- **CUDA** (optional, for GPU acceleration)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/video-processing-yolo-funscript.git
   cd video-processing-yolo-funscript
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy opencv-python tqdm ultralytics scipy matplotlib simplification
   ```
### Libraries Used

The project relies on the following Python libraries:

- **numpy**: For numerical computations and array manipulations.
- **opencv-python**: For computer vision tasks like video processing and image manipulation.
- **tqdm**: For displaying progress bars during long-running tasks.
- **ultralytics**: For YOLO object detection and tracking.
- **scipy**: For scientific computing, including interpolation (interp1d).
- **matplotlib**: For plotting and visualization.
- **simplification**: For simplifying Funscript data.
- **logging**: For logging debug and runtime information.
- **argparse**: For parsing command-line arguments.
- **subprocess**: For running external commands (e.g., FFmpeg).
- **collections**: For specialized container datatypes like deque and defaultdict.
- **datetime**: For handling timestamps and date-related operations.
- **json**: For reading and writing JSON files.

3. **Download the YOLO model**:
   - Place your YOLO model file (e.g., `k00gar-11n-200ep-best.mlpackage`) in the `models/` directory.
   - Alternatively, you can specify a custom path to the model using the `--yolo_model` argument.

## Usage

### Basic Command

To process a video, run the following command:

```bash
python main.py /path/to/video.mp4
```

### Optional Arguments

- `--yolo_model`: Path to the YOLO model file (default: `models/k00gar-11n-200ep-best.mlpackage`).
- `--test_mode`: Enable test mode for real-time visualization of object tracking.
- `--debug_mode`: Enable debug mode to save detailed logs.
- `--is_vr`: Enable VR mode for processing VR videos.

Example:

```bash
python main.py /path/to/video.mp4 --yolo_model /path/to/custom_model.mlpackage --test_mode --is_vr
```

### Output Files

The script generates the following files in the same directory as the input video:

1. `_rawyolo.json`: Raw YOLO detection data.
2. `_cuts.json`: Detected scene changes.
3. `_rawfunscript.json`: Raw Funscript data.
4. `.funscript`: Final Funscript file.
5. `_heatmap.png`: Heatmap visualization of the Funscript data.

## How It Works

1. **YOLO Detection**:
   - The script uses a YOLO model to detect and track objects in each video frame.
   - For VR videos, it processes only the center third of the left half of the frame.

2. **Scene Change Detection**:
   - Detects scene changes to reset tracking and ensure accuracy.

3. **Tracking and Funscript Generation**:
   - Tracks specific objects (e.g., body parts) and generates Funscript data based on their movements.

4. **Visualization (Test Mode)**:
   - Displays bounding boxes and Funscript data in real-time for debugging and verification.

5. **Debugging (Debug Mode)**:
   - Saves detailed logs for debugging purposes.

## Example

1. **Generate funscript**:
```bash
python FSGenerator.py /path/to/vr_video.mp4 --yolo_model models/k00gar-11n-200ep-best.mlpackage --test_mode --is_vr
```
This command processes a VR video, detects objects using the specified YOLO model, and generates Funscript data. The `--test_mode` flag enables real-time visualization.

2. **Debugging example**:

Display a Specific Frame with debug information:
```bash
debugger.display_frame(frame_id)
```
Play the Video with debug information:
```bash
debugger.play_video(frame_id)
```
Record the Debugged Video:
```bash
debugger.play_video(frame, record=True, downsize_ratio=2, duration=10)
```

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Submit a pull request.

## License

This project is licensed under the **Non-Commercial License**. You are free to use the software for personal, non-commercial purposes only. Commercial use, redistribution, or modification for commercial purposes is strictly prohibited without explicit permission from the copyright holder.

See the [LICENSE](LICENSE) file for full details.

## Acknowledgments

- **YOLO**: Thanks to the Ultralytics team for the YOLO implementation.
- **FFmpeg**: For video processing capabilities.
- **Eroscripts Community**: For the inspiration and use cases.

## Support

If you encounter any issues or have questions, please open an issue on the [GitHub repository](https://github.com/yourusername/video-processing-yolo-funscript/issues).

---

This README provides a comprehensive overview of the project, making it easy for users to understand, set up, and use the tool. It also encourages contributions and provides support information.