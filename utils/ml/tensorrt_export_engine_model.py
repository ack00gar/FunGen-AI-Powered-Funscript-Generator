#!/usr/bin/env python3
"""
Standalone script for exporting YOLO models to TensorRT.
This script is designed to be run as a subprocess to allow interruption.
"""

import sys
import os
import json
import traceback
import logging

# Configure logging for TensorRT export utility
logger = logging.getLogger(__name__)

def _setup_standalone_logger():
    """Setup minimal logging for standalone script execution."""
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(levelname)-8s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

def export_model(model_path, output_dir):
    """Export a YOLO model to TensorRT format."""
    _setup_standalone_logger()
    try:
        logger.info("Starting TensorRT export process...")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Output directory: {output_dir}")
        
        logger.info("Importing ultralytics...")
        from ultralytics import YOLO
        
        logger.info("Loading YOLO model...")
        model = YOLO(model_path)
        
        logger.info("Starting TensorRT export (this may take several minutes)...")
        logger.info("Export settings: half=True, batch=1, simplify=True")
        model.export(format="engine", half=True, batch=1, simplify=True)
        logger.info("TensorRT export completed!")
        
        logger.info("Searching for generated engine files...")
        # Find the specific engine file that should have been created
        model_basename = os.path.splitext(os.path.basename(model_path))[0]
        expected_engine_path = os.path.join(output_dir, model_basename + ".engine")
        
        if os.path.exists(expected_engine_path):
            logger.info(f"Found engine file: {expected_engine_path}")
            result = {
                'success': True,
                'engine_file': expected_engine_path
            }
        else:
            logger.error("No engine file was created!")
            result = {
                'success': False,
                'error': 'No engine file was created'
            }
    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    
    # Output result as JSON
    print(json.dumps(result))
    sys.exit(0 if result['success'] else 1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({
            'success': False,
            'error': 'Usage: python export_model.py <model_path> <output_dir>'
        }))
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    export_model(model_path, output_dir) 