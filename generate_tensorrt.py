import os

from ultralytics import YOLO

# SETTINGS (CHANGE THIS IF NEEDED):
model_to_convert = "FunGen-12s-pov-1.1.0.pt"

# Conversion Code
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", model_to_convert)
if not os.path.exists(model_path):
    print(f"Model not found in path: {model_path}")
    print(f"Did you download the latest model and put it in the models directory?")

model = YOLO(model_path)
model.export(format="engine", half=True, batch=30, simplify=True)
print("Successfully generated .engine TensorRT model in models directory")