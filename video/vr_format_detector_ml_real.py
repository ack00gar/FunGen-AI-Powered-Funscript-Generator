"""
Real ML-Based VR Format Detector

Uses actual machine learning (Random Forest) to learn format detection from training data.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import logging
import json
import pickle
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class VRFormatFeatureExtractor:
    """Extract discriminative features from video frames for ML classification."""

    @staticmethod
    def extract_features(frames: List[np.ndarray], video_info: Dict) -> np.ndarray:
        """
        Extract feature vector from video frames.

        Returns:
            1D numpy array with features
        """
        if not frames:
            return np.zeros(20)

        height, width = frames[0].shape[:2]
        aspect_ratio = width / height

        # Initialize feature accumulators
        sbs_correlations = []
        tb_correlations = []
        corner_darkness_scores = []
        edge_gradients = []
        content_coverage_scores = []
        brightness_ratios = []
        center_brightness_values = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # === SBS/TB Correlation Features ===
            mid_w = w // 2
            left_half = gray[:, :mid_w]
            right_half = gray[:, mid_w:2*mid_w]

            hist_left = cv2.calcHist([left_half], [0], None, [256], [0, 256])
            hist_right = cv2.calcHist([right_half], [0], None, [256], [0, 256])
            sbs_corr = cv2.compareHist(hist_left, hist_right, cv2.HISTCMP_CORREL)
            sbs_correlations.append(sbs_corr)

            mid_h = h // 2
            top_half = gray[:mid_h, :]
            bottom_half = gray[mid_h:2*mid_h, :]

            hist_top = cv2.calcHist([top_half], [0], None, [256], [0, 256])
            hist_bottom = cv2.calcHist([bottom_half], [0], None, [256], [0, 256])
            tb_corr = cv2.compareHist(hist_top, hist_bottom, cv2.HISTCMP_CORREL)
            tb_correlations.append(tb_corr)

            # === Projection Features ===
            single_eye = gray[:, :mid_w]
            h_eye, w_eye = single_eye.shape

            # Corner darkness
            corner_size = min(h_eye, w_eye) // 10
            tl = single_eye[:corner_size, :corner_size]
            tr = single_eye[:corner_size, -corner_size:]
            bl = single_eye[-corner_size:, :corner_size]
            br = single_eye[-corner_size:, -corner_size:]

            corner_brightness = np.mean([np.mean(tl), np.mean(tr), np.mean(bl), np.mean(br)])
            center_brightness = np.mean(single_eye[
                h_eye//2-corner_size:h_eye//2+corner_size,
                w_eye//2-corner_size:w_eye//2+corner_size
            ])

            corner_darkness = 1.0 - (corner_brightness / 255.0)
            corner_darkness_scores.append(corner_darkness)

            brightness_ratio = center_brightness / (corner_brightness + 1e-6)
            brightness_ratios.append(brightness_ratio)
            center_brightness_values.append(center_brightness / 255.0)

            # Edge gradients
            left_edge = single_eye[:, :corner_size]
            right_edge = single_eye[:, -corner_size:]
            left_grad = np.mean(np.abs(cv2.Sobel(left_edge, cv2.CV_64F, 1, 0, ksize=3)))
            right_grad = np.mean(np.abs(cv2.Sobel(right_edge, cv2.CV_64F, 1, 0, ksize=3)))
            edge_gradients.append((left_grad + right_grad) / 2)

            # === FOV Features ===
            edge_width = w_eye // 20
            left_content = np.mean(single_eye[:, :edge_width])
            right_content = np.mean(single_eye[:, -edge_width:])
            center_content = np.mean(single_eye[:, w_eye//2-edge_width:w_eye//2+edge_width])

            edge_content = (left_content + right_content) / 2
            content_ratio = edge_content / (center_content + 1e-6)

            edges = cv2.Canny(single_eye, 50, 150)
            edge_density = (np.sum(edges[:, :w_eye//4]) + np.sum(edges[:, -w_eye//4:])) / (h_eye * w_eye // 2)

            content_coverage = content_ratio * 0.6 + edge_density * 0.4
            content_coverage_scores.append(content_coverage)

        # Build feature vector (20 features)
        features = np.array([
            # Resolution features (4)
            aspect_ratio,
            np.log10(width * height),
            float(1.8 <= aspect_ratio <= 2.2),  # is_sbs_aspect
            float(0.45 <= aspect_ratio <= 0.55),  # is_tb_aspect

            # Stereo correlation features (6)
            np.mean(sbs_correlations),
            np.std(sbs_correlations),
            np.max(sbs_correlations),
            np.mean(tb_correlations),
            np.std(tb_correlations),
            np.mean(sbs_correlations) - np.mean(tb_correlations),  # corr_diff

            # Projection features (6)
            np.mean(corner_darkness_scores),
            np.std(corner_darkness_scores),
            np.mean(brightness_ratios),
            np.mean(center_brightness_values),
            np.mean(edge_gradients),
            np.std(edge_gradients),

            # FOV features (2)
            np.mean(content_coverage_scores),
            np.std(content_coverage_scores),

            # Combined features (2)
            np.mean(corner_darkness_scores) * np.mean(brightness_ratios),  # fisheye_indicator
            np.max(sbs_correlations) * float(1.8 <= aspect_ratio <= 2.2),  # sbs_confidence
        ])

        return features


class RealMLVRFormatDetector:
    """
    Real ML-based VR format detector using Random Forest.

    Trains actual classifiers to predict:
    - Video type (2D/VR)
    - Layout (sbs/tb)
    - Projection (he/fisheye)
    - FOV (180/190/200)
    """

    def __init__(self, model_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """Initialize detector."""
        self.logger = logger or logging.getLogger(__name__)
        self.feature_extractor = VRFormatFeatureExtractor()

        # Separate classifiers for each prediction task
        self.video_type_clf = None  # 2D vs VR
        self.layout_clf = None  # sbs/tb (for VR only)
        self.projection_clf = None  # he/fisheye (for VR only)
        self.fov_clf = None  # 180/190/200 (for VR only)

        self.feature_names = None
        self.training_stats = None

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def train(self, training_data_path: str) -> Dict:
        """Train all classifiers on the training dataset."""
        self.logger.info(f"Training on dataset: {training_data_path}")

        with open(training_data_path, 'r') as f:
            training_data = json.load(f)

        self.logger.info(f"Loaded {len(training_data)} training samples")

        # Prepare training data
        X_list = []
        y_video_type = []
        y_layout = []
        y_projection = []
        y_fov = []

        valid_samples = 0

        for sample in training_data:
            if 'error' in sample or 'video_path' not in sample:
                continue

            video_type = sample.get('video_type', 'UNKNOWN')
            if video_type == 'UNKNOWN':
                continue

            # Load REAL extracted features from the dataset
            # This is the key fix - use actual frame features, not simulated ones!
            if 'features' in sample:
                # New format: real features extracted from frames
                features = np.array(sample['features'])
            else:
                # Old format: simulate from metadata (fallback for compatibility)
                self.logger.warning(f"Sample missing 'features' field, using metadata simulation (this is not ideal!)")
                width, height = map(int, sample.get('resolution', '1920x1080').split('x'))
                aspect_ratio = width / height

                features = np.array([
                    aspect_ratio,
                    np.log10(width * height),
                    float(1.8 <= aspect_ratio <= 2.2),
                    float(0.45 <= aspect_ratio <= 0.55),
                    sample.get('layout_confidence', 0.5),
                    0.05,  # std placeholder
                    sample.get('layout_confidence', 0.5),
                    0.5 if sample.get('layout') == 'tb' else 0.3,
                    0.05,
                    sample.get('layout_confidence', 0.5) - 0.4,
                    sample.get('projection_confidence', 0.5) if sample.get('projection') == 'fisheye' else 0.3,
                    0.05,
                    1.2 if sample.get('projection') == 'fisheye' else 1.0,
                    0.7,
                    50.0 if sample.get('projection') == 'fisheye' else 30.0,
                    5.0,
                    0.4 if sample.get('fov') == 200 else 0.25,
                    0.05,
                    0.6 if sample.get('projection') == 'fisheye' else 0.3,
                    0.8 if sample.get('layout') == 'sbs' else 0.4,
                ])

            X_list.append(features)

            # Labels
            y_video_type.append(0 if video_type == '2D' else 1)  # 0=2D, 1=VR

            if video_type == 'VR':
                layout = sample.get('layout', 'sbs')
                y_layout.append(0 if layout == 'sbs' else 1)  # 0=sbs, 1=tb

                projection = sample.get('projection', 'he')
                y_projection.append(0 if projection == 'he' else 1)  # 0=he, 1=fisheye

                fov = sample.get('fov', 200)
                if fov == 180:
                    y_fov.append(0)
                elif fov == 190:
                    y_fov.append(1)
                else:  # 200
                    y_fov.append(2)
            else:
                # For 2D videos, use dummy labels
                y_layout.append(-1)
                y_projection.append(-1)
                y_fov.append(-1)

            valid_samples += 1

        X = np.array(X_list)

        self.logger.info(f"Training on {valid_samples} valid samples")
        self.logger.info(f"Feature matrix shape: {X.shape}")

        # Train video type classifier (2D vs VR)
        self.logger.info("Training video type classifier (2D vs VR)...")
        self.video_type_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.video_type_clf.fit(X, y_video_type)

        # Train layout classifier (SBS vs TB) - only on VR samples
        vr_mask = np.array(y_video_type) == 1
        X_vr = X[vr_mask]
        y_layout_vr = np.array(y_layout)[vr_mask]

        self.logger.info(f"Training layout classifier on {len(X_vr)} VR samples...")
        self.layout_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight='balanced'
        )
        self.layout_clf.fit(X_vr, y_layout_vr)

        # Train projection classifier (HE vs Fisheye)
        y_projection_vr = np.array(y_projection)[vr_mask]
        self.logger.info("Training projection classifier (HE vs Fisheye)...")
        self.projection_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight='balanced'
        )
        self.projection_clf.fit(X_vr, y_projection_vr)

        # Train FOV classifier
        y_fov_vr = np.array(y_fov)[vr_mask]
        self.logger.info("Training FOV classifier (180/190/200)...")
        self.fov_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            class_weight='balanced'
        )
        self.fov_clf.fit(X_vr, y_fov_vr)

        # Calculate training stats
        stats = self._calculate_training_stats(X, y_video_type, y_layout_vr, y_projection_vr, y_fov_vr)
        self.training_stats = stats

        return stats

    def _calculate_training_stats(self, X, y_video_type, y_layout, y_projection, y_fov) -> Dict:
        """Calculate training statistics and accuracy."""
        stats = {
            'total_samples': len(X),
            'video_type_accuracy': accuracy_score(y_video_type, self.video_type_clf.predict(X)),
            'layout_accuracy': accuracy_score(y_layout, self.layout_clf.predict(X[np.array(y_video_type) == 1])),
            'projection_accuracy': accuracy_score(y_projection, self.projection_clf.predict(X[np.array(y_video_type) == 1])),
            'fov_accuracy': accuracy_score(y_fov, self.fov_clf.predict(X[np.array(y_video_type) == 1])),
        }

        self.logger.info(f"Training accuracies:")
        self.logger.info(f"  Video Type (2D vs VR): {stats['video_type_accuracy']:.1%}")
        self.logger.info(f"  Layout (SBS vs TB): {stats['layout_accuracy']:.1%}")
        self.logger.info(f"  Projection (HE vs Fisheye): {stats['projection_accuracy']:.1%}")
        self.logger.info(f"  FOV (180/190/200): {stats['fov_accuracy']:.1%}")

        return stats

    def detect(self, video_path: str, video_info: Dict, num_frames: int = 3) -> Dict:
        """Detect VR format using trained ML classifiers."""
        try:
            # Extract frames
            frames = self._extract_middle_frames(video_path, video_info, num_frames)

            if not frames:
                return self._create_error_result("Failed to extract frames")

            # Extract features
            features = self.feature_extractor.extract_features(frames, video_info)
            features = features.reshape(1, -1)  # Reshape for prediction

            # Predict video type
            video_type_pred = self.video_type_clf.predict(features)[0]
            video_type_prob = self.video_type_clf.predict_proba(features)[0]
            video_type = '2D' if video_type_pred == 0 else 'VR'
            video_type_conf = np.max(video_type_prob)

            if video_type == '2D':
                return {
                    'video_type': '2D',
                    'layout': '2d',
                    'projection': None,
                    'fov': None,
                    'format_string': '2D',
                    'confidence': video_type_conf,
                    'detection_method': 'ml_random_forest',
                    'details': {
                        'probabilities': {'2D': video_type_prob[0], 'VR': video_type_prob[1]}
                    }
                }

            # Predict VR attributes
            layout_pred = self.layout_clf.predict(features)[0]
            layout_prob = self.layout_clf.predict_proba(features)[0]
            layout = 'sbs' if layout_pred == 0 else 'tb'
            layout_conf = np.max(layout_prob)

            projection_pred = self.projection_clf.predict(features)[0]
            projection_prob = self.projection_clf.predict_proba(features)[0]
            projection = 'he' if projection_pred == 0 else 'fisheye'
            projection_conf = np.max(projection_prob)

            fov_pred = self.fov_clf.predict(features)[0]
            fov_prob = self.fov_clf.predict_proba(features)[0]
            fov_map = {0: 180, 1: 190, 2: 200}
            fov = fov_map[fov_pred]
            fov_conf = np.max(fov_prob)

            format_string = f"{projection}_{layout}"
            overall_confidence = (video_type_conf + layout_conf + projection_conf + fov_conf) / 4.0

            # Build probabilities dict with safe indexing (handle missing classes)
            layout_probs = {'sbs': 0.0, 'tb': 0.0}
            if len(layout_prob) > 0:
                layout_probs['sbs'] = float(layout_prob[0])
            if len(layout_prob) > 1:
                layout_probs['tb'] = float(layout_prob[1])

            projection_probs = {'he': 0.0, 'fisheye': 0.0}
            if len(projection_prob) > 0:
                projection_probs['he'] = float(projection_prob[0])
            if len(projection_prob) > 1:
                projection_probs['fisheye'] = float(projection_prob[1])

            fov_probs = {180: 0.0, 190: 0.0, 200: 0.0}
            if len(fov_prob) > 0:
                fov_probs[180] = float(fov_prob[0])
            if len(fov_prob) > 1:
                fov_probs[190] = float(fov_prob[1])
            if len(fov_prob) > 2:
                fov_probs[200] = float(fov_prob[2])

            return {
                'video_type': 'VR',
                'layout': layout,
                'projection': projection,
                'fov': fov,
                'format_string': format_string,
                'confidence': overall_confidence,
                'detection_method': 'ml_random_forest',
                'details': {
                    'video_type_confidence': video_type_conf,
                    'layout_confidence': layout_conf,
                    'projection_confidence': projection_conf,
                    'fov_confidence': fov_conf,
                    'probabilities': {
                        'layout': layout_probs,
                        'projection': projection_probs,
                        'fov': fov_probs
                    }
                }
            }

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return self._create_error_result(str(e))

    def _extract_middle_frames(self, video_path: str, video_info: Dict, num_frames: int) -> List[np.ndarray]:
        """Extract sample frames from middle of video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = video_info.get('nb_frames', 1000)

        start_frame = int(total_frames * 0.25)
        end_frame = int(total_frames * 0.75)
        sample_interval = max(1, (end_frame - start_frame) // num_frames)

        frames = []
        for i in range(num_frames):
            frame_num = start_frame + (i * sample_interval)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        cap.release()
        return frames

    def _create_error_result(self, error_msg: str) -> Dict:
        """Create error result."""
        return {
            'video_type': 'UNKNOWN',
            'layout': None,
            'projection': None,
            'fov': None,
            'format_string': 'UNKNOWN',
            'confidence': 0.0,
            'detection_method': 'ml_random_forest',
            'error': error_msg
        }

    def save_model(self, model_path: str):
        """Save trained model to file."""
        with open(model_path, 'wb') as f:
            pickle.dump({
                'video_type_clf': self.video_type_clf,
                'layout_clf': self.layout_clf,
                'projection_clf': self.projection_clf,
                'fov_clf': self.fov_clf,
                'training_stats': self.training_stats,
                'version': '1.0_rf'
            }, f)
        self.logger.info(f"Model saved to: {model_path}")

    def load_model(self, model_path: str):
        """Load trained model from file."""
        # Suppress sklearn version warnings during model loading
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.video_type_clf = data['video_type_clf']
                self.layout_clf = data['layout_clf']
                self.projection_clf = data['projection_clf']
                self.fov_clf = data['fov_clf']
                self.training_stats = data.get('training_stats')
        self.logger.info(f"Model loaded from: {model_path}")
