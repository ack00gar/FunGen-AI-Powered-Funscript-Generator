import os
import threading
from typing import Optional, Callable

class TensorRTCompilerError(Exception):
    pass

def is_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def is_tensorrt_installed() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False

def compile_yolo_to_tensorrt(
    pt_model_path: str,
    output_dir: str,
    progress_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Compiles a YOLO .pt model to a TensorRT .engine file using ultralytics.
    Returns the path to the generated .engine file on success.
    Raises TensorRTCompilerError on failure.
    """
    if not os.path.isfile(pt_model_path):
        raise TensorRTCompilerError(f"Model file not found: {pt_model_path}")
    if not pt_model_path.lower().endswith('.pt'):
        raise TensorRTCompilerError("Selected file is not a .pt model file.")
    if not os.path.isdir(output_dir):
        raise TensorRTCompilerError(f"Output directory does not exist: {output_dir}")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise TensorRTCompilerError("ultralytics package is not installed.")
    try:
        import torch
    except ImportError:
        raise TensorRTCompilerError("torch package is not installed.")
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        raise TensorRTCompilerError("TensorRT Python package is not installed.")
    if not torch.cuda.is_available():
        raise TensorRTCompilerError("No CUDA-capable device detected.")

    model_basename = os.path.splitext(os.path.basename(pt_model_path))[0]
    output_path = os.path.join(output_dir, model_basename + ".engine")

    try:
        if progress_callback:
            progress_callback("Loading YOLO model...")
        model = YOLO(pt_model_path)
        if progress_callback:
            progress_callback("Exporting to TensorRT .engine (this may take a while)...")
        # Export with fixed batch size 1, 640x640, half precision, no dynamic shapes
        model.export(
            format="engine",
            half=True,
            batch=1,
            simplify=True,
            imgsz=640,
            device='cuda',
            dynamic=False,
            optimize=False,
            workspace=4,
            save_dir=output_dir,
            name=model_basename + ".engine"
        )
        # Find the most recently created .engine file in the output dir
        import glob
        engine_files = glob.glob(os.path.join(output_dir, "*.engine"))
        if engine_files:
            latest_engine = max(engine_files, key=os.path.getctime)
            if os.path.abspath(latest_engine) != os.path.abspath(output_path):
                import shutil
                shutil.move(latest_engine, output_path)
        if not os.path.exists(output_path):
            raise TensorRTCompilerError(".engine file was not created.")
        if progress_callback:
            progress_callback(f"Model compiled successfully! Output: {output_path}")
        return output_path
    except Exception as e:
        raise TensorRTCompilerError(f"TensorRT compilation failed: {e}")

def unload_yolo_model(model: Optional[object]):
    """
    Unloads a YOLO model instance if needed (placeholder for future resource cleanup).
    """
    # In ultralytics, models are just Python objects; explicit unload is not required.
    # This function is a placeholder in case future resource management is needed.
    del model 