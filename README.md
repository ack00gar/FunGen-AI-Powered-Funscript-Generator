# VR-Funscript-AI-Generator

Here’s a detailed and professional `README.md` for your GitHub repository. It explains the purpose of the project, how to set it up, and how to use it effectively. This README is designed to be clear and accessible for both technical and non-technical users.

---

# Video Processing with YOLO and Funscript Generation

This project is designed to process video files using YOLO (You Only Look Once) object detection and generate **Funscript** data for interactive applications. It is particularly useful for VR videos, where it can track specific objects (e.g., body parts) and create corresponding motion data.

## Features

- **YOLO Object Detection**: Uses a pre-trained YOLO model to detect and track objects in video frames.
- **Funscript Generation**: Generates Funscript data based on the tracked objects' movements.
- **Scene Change Detection**: Automatically detects scene changes in the video to improve tracking accuracy.
- **Visualization**: Provides real-time visualization of object tracking and Funscript data (in test mode).
- **VR Support**: Optimized for VR videos, with options to process specific regions of the frame.

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
   pip install -r requirements.txt
   ```

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

## Example

```bash
python main.py /path/to/vr_video.mp4 --yolo_model models/k00gar-11n-200ep-best.mlpackage --test_mode --is_vr
```

This command processes a VR video, detects objects using the specified YOLO model, and generates Funscript data. The `--test_mode` flag enables real-time visualization.

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