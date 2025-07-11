import os
import glob
import shutil

from ultralytics import YOLO

# SETTINGS (CHANGE THIS IF NEEDED):
model_to_convert_list = [
    "yolo11s-pose.pt",
    "FunGen-12s-pov-1.1.0.pt",
    # Add more model filenames here
]
batch_size = 2  # Set your desired batch size here

# Conversion Code
for model_to_convert in model_to_convert_list:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", model_to_convert)
    if not os.path.exists(model_path):
        print(f"Model not found in path: {model_path}")
        print(f"Did you download the latest model and put it in the models directory?")
        continue

    model = YOLO(model_path)
    output_name = os.path.splitext(model_to_convert)[0] + f"_{batch_size}frameBatch.engine"
    output_path = os.path.join(os.path.dirname(model_path), output_name)
    model.export(format="engine", half=True, batch=batch_size, simplify=True, imgsz=640, device='cuda',
                 dynamic=False, optimize=False, workspace=4, save_dir=os.path.dirname(model_path),
                 name=output_name)

    # Find the most recently created .engine file and rename it to the desired output_path
    engine_files = glob.glob(os.path.join(os.path.dirname(model_path), "*.engine"))
    if engine_files:
        latest_engine = max(engine_files, key=os.path.getctime)
        if os.path.abspath(latest_engine) != os.path.abspath(output_path):
            shutil.move(latest_engine, output_path)
        print(f"Renamed exported engine to: {output_path}")
    else:
        print("No .engine file found after export.")