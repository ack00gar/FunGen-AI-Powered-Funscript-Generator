import platform
import shutil

import torch


def find_ffmpeg_path(win_ffmpeg_path, mac_ffmpeg_path, lin_ffmpeg_path):
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    else:
        print("ffmpeg not found in PATH, defaulting to platform values in params/config.py")

        if platform.system() == "Windows":
            return win_ffmpeg_path
        elif platform.system() == "Darwin":
            return mac_ffmpeg_path
        elif platform.system() == "Linux":
            return lin_ffmpeg_path
        else:
            raise FileNotFoundError("ffmpeg not found in PATH and no default path for this OS")


def find_ffprobe_path(win_ffprobe_path, mac_ffprobe_path, lin_ffprobe_path):
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        return ffprobe_path
    else:
        print("ffprobe not found in PATH, defaulting to platform values in params/config.py")

        if platform.system() == "Windows":
            return win_ffprobe_path
        elif platform.system() == "Darwin":
            return mac_ffprobe_path
        elif platform.system() == "Linux":
            return lin_ffprobe_path
        else:
            raise FileNotFoundError("ffprobe not found in PATH and no default path for this OS")

def get_yolo_model_path(YOLO_MODELS):
    # Check if the device is an Apple device
    if platform.system() == 'Darwin':
        print(f"Apple device detected, loading {YOLO_MODELS[0]} for MPS inference.")
        return YOLO_MODELS[0]

    # Check if CUDA is available (for GPU support)
    elif torch.cuda.is_available():
        print(f"CUDA is available, loading {YOLO_MODELS[1]} for GPU inference.")
        return YOLO_MODELS[1]

    # Fallback to ONNX model for other platforms without CUDA
    else:
        print("CUDA not available, if this is unexpected, please install CUDA and check your version of torch.")
        print("You might need to install a dependency with the following command (example):")
        print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print(f"Falling back to CPU inference, loading {YOLO_MODELS[2]}.")
        print("WARNING: CPU inference may be slow on some devices.")

        return YOLO_MODELS[2]
