# VR-Funscript-AI-Generator

This project is a Python-based tool for generating Funscript files from VR videos using Computer Vision (CV) and AI techniques. It leverages YOLO (You Only Look Once) object detection and custom tracking algorithms to automate the process of creating Funscript files for interactive devices.

If you find this project useful, consider supporting me on:

- **Ko-fi**: [![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/H2H818EIJV) 
- **Patreon**: [https://www.patreon.com/c/k00gar](https://www.patreon.com/c/k00gar)

Your support helps me continue developing and improving this project!

Join the **Discord community** for discussions and support: [Discord Community](https://discord.gg/WYkjMbtCZA)

The necessary YOLO models will also be available via the Discord.

---

## DISCLAIMER

This project is at its very early stages of development, still faulty and broken, and is for research and educational purposes only. It is not intended for commercial use.
Please, do not use this project for any commercial purposes without prior consent from the author. It is for individual use only.

---

## Features

- **YOLO Object Detection**: Uses a pre-trained YOLO model to detect and track objects in video frames.
- **Funscript Generation**: Generates Funscript data based on the tracked objects' movements.
- **Scene Change Detection**: Automatically detects scene changes in the video to improve tracking accuracy.
- **Visualization**: Provides real-time visualization of object tracking and Funscript data (in test mode).
- **VR Support**: Optimized for VR videos, with options to process specific regions of the frame.

---

## Project Genesis and Evolution

This project started as a dream to automate Funscript generation for VR videos. Here’s a brief history of its development:

- **Initial Approach (OpenCV Trackers)**: The first version relied on OpenCV trackers to detect and track objects in the video. While functional, the approach was slow (8–20 FPS) and struggled with occlusions and complex scenes.

- **Transition to YOLO**: To improve accuracy and speed, the project shifted to using YOLO object detection. A custom YOLO model was trained on a dataset of VR video frames, significantly improving detection quality. The new approach runs at 90 FPS on a Mac mini M4 pro, making it much more efficient.

- **Original Post**: For more details and discussions, check out the original post on EroScripts:  
[VR Funscript Generation Helper (Python + CV/AI)](https://discuss.eroscripts.com/t/vr-funscript-generation-helper-python-now-cv-ai/202554)

---

## YOLO Model

The YOLO model used in this project is based on YOLOv11n, which was fine-tuned with 10 new classes and 4,500+ frames randomly extracted from a VR video library. Here’s how the model was developed:

- **Initial Training**: A few hundred frames were manually tagged and boxed to create an initial dataset. The model was trained on this dataset to generate preliminary detection results.
- **Iterative Improvement**: The trained model was used to suggest bounding boxes in additional frames. The suggested boxes were manually adjusted, and the dataset was expanded. This process was repeated iteratively to improve the model’s accuracy.
- **Final Training**: After gathering 4,500+ images and 30,149 annotations, the model was trained for 200 epochs. YOLOv11s and YOLOv11m were also tested, but YOLOv11n was chosen for its balance of accuracy and inference speed.
- **Hardware**: The model runs on a Mac using MPS (Metal Performance Shaders) for accelerated inference on ARM chips. Other versions of the model (ONNX and PT) are also available for use on other platforms.

---

## Pipeline Overview

The pipeline for generating Funscript files is as follows:

1. **YOLO Object Detection**: A YOLO model detects relevant objects (e.g., penis, hands, mouth, etc.) in each frame of the video. The detection results are saved to a `.json` file.
2. **Tracking Algorithm**: A custom tracking algorithm processes the YOLO detection results to track the positions of objects over time. The algorithm calculates distances and interactions between objects to determine the Funscript position.
3. **Funscript Generation**: The tracked data is used to generate a raw Funscript file.
4. **Simplifier**: The raw Funscript data is simplified to remove noise and smooth out the motion. The final `.funscript` file is saved.
5. **Heatmap Generation**: A heatmap is generated to visualize the Funscript data.

---

## Prerequisites
 
Before using this project, ensure you have the following installed:

- **Python 3.8 or higher**
- **FFmpeg** added to your path (for video processing)
- **CUDA** (optional, for GPU acceleration)

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ack00gar/VR-Funscript-AI-Generator.git
   cd VR-Funscript-AI-Generator
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy opencv-python tqdm ultralytics scipy matplotlib simplification
   ```

3. **Use a venv as suggested by Zalunda**
   * Install miniconda 
   * Start a miniconda command prompt 
   * Execute (assuming you already cloned VR-Funscript-AI-Generator and copied the model into models folder)
   ```bash
   conda create -n VRFunAIGen python=3.11
   conda activate VRFunAIGen
   pip install numpy opencv-python tqdm ultralytics scipy matplotlib simplification
   pip uninstall torch torchvision torchaudio
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   cd <VR-Funscript-AI-Generator folder>
   python FSGenerator.py
   ```
   
   While executing, you’ll need to say “yes” a few times. The lines “pip uninstall / pip3 install” is to replace the “CPU” version of torch with a “cuda enabled / GPU” version (you might need to install nvidia CUDA stuff for it to works, I’m not sure).
   
   Zalunda also suggests the creation of a batch file, after the setup, to start the application in the right conda environment:

   ```bash
   @echo off
   call <PATH_TO_MINICONDA>\miniconda3\condabin\conda activate VRFunAIGen
   cd /d "<PATH_TO_SOURCES>\VR-Funscript-AI-Generator"
   python FSGenerator.py
   pause
   ```

4. **Download the YOLO model**:
   - Place your YOLO model file (e.g., `k00gar-11n-200ep-best.mlpackage`) in the `models/` sub-directory.
   - Alternatively, you can specify a custom path to the model using the `--yolo_model` argument.

5. **Update the params/config.py**:
   - If ffmpeg and ffprobe paths are not in your system path, the program will default to the following values.
   - You can update the params/config.py file, which contains:

   ```bash
   # ffmpeg and ffprobe paths - replace with your own if not in your system path   
    win_ffmpeg_path = "C:/ffmpeg/bin/ffmpeg.exe"
    mac_ffmpeg_path = "/usr/local/bin/ffmpeg"
    lin_ffmpeg_path = "/usr/bin/ffmpeg"

    win_ffprobe_path = "C:/ffmpeg/bin/ffprobe.exe"
    mac_ffprobe_path = "/usr/local/bin/ffprobe"
    lin_ffprobe_path = "/usr/bin/ffprobe"
   ```

---

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
- **tkinter**: For creating a basic GUI for file selection and parameter configuration.

---

## Howto & Video Input and Preprocessing

### Input File Requirements

The input video file should be a standard video file. For VR videos, ensure the file is in **Side-by-Side (SBS)** format. The algorithm will process the **left panel** by default.

**Note**: While the algorithm can handle up to 8K videos, it is strongly recommended to process videos at **1920p resolution** for optimal performance. Videos should not be lower than **1080p** to maintain detection accuracy. If your video exceeds 1920p in height, the script will automatically suggest resizing options (1920p, 1440p, or 1080p) while preserving the aspect ratio.

### Video Resizing

If the input video height exceeds 1920 pixels, the script will prompt you to resize the video to a lower resolution. The resizing process uses **FFmpeg** and excludes audio by default. The following options are available:
- **1920p**: Resize to 1920 pixels in height (recommended for most cases).
- **1440p**: Resize to 1440 pixels in height.
- **1080p**: Resize to 1080 pixels in height.

### VR Video Projection and Undistortion

For VR videos, the **projection type** and **undistortion settings** are critical for accurate object detection and tracking. The script supports two main projection types:
1. **Fisheye**: Used for videos with a fisheye lens projection. The script automatically detects fisheye videos based on the filename or metadata.
2. **Equirectangular**: Used for standard 360° VR videos. This is the default projection if fisheye is not detected.

#### Key Parameters for VR Video Processing

The undistortion process is handled using FFmpeg's `v360` filter, which corrects the video frames based on the specified projection and field-of-view (FOV) parameters. Key parameters include:
- **Input Vertical FOV (`iv_fov`)**: The vertical field of view of the input video.
- **Input Horizontal FOV (`ih_fov`)**: The horizontal field of view of the input video.
- **Output Vertical FOV (`v_fov`)**: The desired vertical field of view after undistortion.
- **Output Horizontal FOV (`h_fov`)**: The desired horizontal field of view after undistortion.
- **Diagonal FOV (`d_fov`)**: The diagonal field of view used for undistortion.

#### Example FFmpeg Command for VR Video Processing

The following FFmpeg command is used for undistorting VR videos:

```bash
ffmpeg -ss <start_time> -i <input_video> -vf "crop=w=iw/2:h=ih:x=0:y=0,v360=<type>:output=sg:iv_fov=<iv_fov>:ih_fov=<ih_fov>:d_fov=<d_fov>:v_fov=<v_fov>:h_fov=<h_fov>:pitch=-25:yaw=0:roll=0:w=<width>:h=<height>:interp=lanczos:reset_rot=1,lutyuv=y=gammaval(0.7)" -f rawvideo -pix_fmt bgr24 -vsync 0 -threads 0 -
```

#### Adjusting Projection Settings

To ensure accurate detection and tracking, you may need to adjust the projection settings based on the specific characteristics of your VR video. These settings can be found and modified in the `utils/lib_VideoReaderFFmpeg.py` file. Use the `utils/test_detect_compare_unwarped.py` script to test different projection settings before processing the video.

---

### Key Points to Remember

1. **Video Resolution**: Resize videos to 1920p for optimal performance.
2. **VR Video Projection**: Ensure the correct projection type (Fisheye or Equirectangular) is selected for undistortion.
3. **Undistortion Settings**: Adjust FOV and other parameters in `utils/lib_VideoReaderFFmpeg.py` for accurate results.
4. **Testing**: Use `utils/test_detect_compare_unwarped.py` to test projection settings before full processing.

---

### Basic Command

To process a video, run the following command:

```bash
python FSGenerator.py /path/to/video.mp4
```
or Run the script directly from your IDE.

---

### Output Files

The script generates the following files in the same directory as the input video:

1. `_rawyolo.json`: Raw YOLO detection data.
2. `_cuts.json`: Detected scene changes.
3. `_rawfunscript.json`: Raw Funscript data.
4. `.funscript`: Final Funscript file.
5. `_heatmap.png`: Heatmap visualization of the Funscript data.
6. `_comparefunscripts.png`: Comparison visualization between the generated Funscript and the reference Funscript (if provided).
7. `_adjusted.funscript`: Funscript file with adjusted amplitude.

---

## How It Works

1. **YOLO Detection**: The script uses a YOLO model to detect and track objects in each video frame. For VR videos, it processes only the center third of the left half of the frame.
2. **Scene Change Detection**: Detects scene changes to reset tracking and ensure accuracy.
3. **Tracking and Funscript Generation**: Tracks specific objects (e.g., body parts) and generates Funscript data based on their movements.
4. **Visualization (Test Mode)**: Displays bounding boxes and Funscript data in real-time for debugging and verification.
5. **Debugging (Debug Mode)**: Saves detailed logs for debugging purposes.

---

## Example

1. **Generate Funscript**:
   ```bash
   python FSGenerator.py /path/to/vr_video.mp4
   ```
   This command starts the UI.

   You can also simply run it from your IDE, giving it a `video_path` to process.

2. **Debugging Example**:

   The debugger is accessible from the GUI.

   If you want to call it from the code, you can do the following: 

   - Display a Specific Frame with debug information:
     ```bash
     debugger.display_frame(frame_id)
     ```
   - Play the Video with debug information:
     ```bash
     debugger.play_video(frame_id)
     ```
   - Record the Debugged Video:
     ```bash
     debugger.play_video(frame, record=True, downsize_ratio=2, duration=10)
     ```

   Or run `Display_debug_results.py` from your IDE with the desired parameters.

---

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Submit a pull request.

---

## License

This project is licensed under the **Non-Commercial License**. You are free to use the software for personal, non-commercial purposes only. Commercial use, redistribution, or modification for commercial purposes is strictly prohibited without explicit permission from the copyright holder.

This project is not intended for commercial use, nor for generating and distributing in a commercial environment.

For commercial use, please contact me.

See the [LICENSE](LICENSE) file for full details.

---

## Acknowledgments

- **YOLO**: Thanks to the Ultralytics team for the YOLO implementation.
- **FFmpeg**: For video processing capabilities.
- **Eroscripts Community**: For the inspiration and use cases.

---

## Support

If you encounter any issues or have questions, please open an issue on GitHub.

Join the **Discord community** for discussions and support:  
[Discord Community](https://discord.gg/WYkjMbtCZA)

---
