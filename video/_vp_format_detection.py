"""VideoProcessor FormatDetectionMixin — extracted from video_processor.py."""

import json
import os
import subprocess
import sys
from video.vr_format_detector_ml_real import RealMLVRFormatDetector


class FormatDetectionMixin:
    """Mixin fragment for VideoProcessor."""

    @staticmethod
    def _detect_format_from_filename(filename: str) -> dict:
        """
        Detects video format information from filename suffixes.

        Returns:
            dict with keys:
            - 'type': 'VR', '2D', or None (if cannot determine)
            - 'projection': projection type if VR (e.g., 'fisheye', 'he', 'eac')
            - 'layout': stereoscopic layout if VR (e.g., '_sbs', '_tb', '_lr')
            - 'fov': FOV value if specific lens detected (e.g., 200 for MKX200)
        """
        upper_filename = filename.upper()
        result = {
            'type': None,
            'projection': None,
            'layout': None,
            'fov': None
        }

        # Check for 2D markers first
        if '_2D' in upper_filename or '_FLAT' in upper_filename:
            result['type'] = '2D'
            return result

        # Check for custom fisheye lenses
        if '_MKX200' in upper_filename or 'MKX200' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'fisheye'
            result['layout'] = '_sbs'
            result['fov'] = 200
            return result
        elif '_MKX220' in upper_filename or 'MKX220' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'fisheye'
            result['layout'] = '_sbs'
            result['fov'] = 220
            return result
        elif '_RF52' in upper_filename or 'RF52' in upper_filename or '_VRCA220' in upper_filename or 'VRCA220' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'fisheye'
            result['layout'] = '_sbs'
            return result

        # Check for standard fisheye (flexible matching - with or without underscore)
        if '_F180' in upper_filename or 'F180_' in upper_filename or '_180F' in upper_filename or '180F_' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'fisheye'
            result['layout'] = '_sbs'
            result['fov'] = 180
            return result
        if 'FISHEYE190' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'fisheye'
            result['layout'] = '_sbs'
            result['fov'] = 190
            return result
        if 'FISHEYE200' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'fisheye'
            result['layout'] = '_sbs'
            result['fov'] = 200
            return result
        if 'FISHEYE220' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'fisheye'
            result['layout'] = '_sbs'
            result['fov'] = 220
            return result
        if 'FISHEYE' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'fisheye'
            result['layout'] = '_sbs'
            result['fov'] = 190
            return result

        # Check for equiangular cubemap
        if '_EAC360' in upper_filename or '_360EAC' in upper_filename or 'EAC360' in upper_filename or '360EAC' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'eac'
            if '_LR' in upper_filename:
                result['layout'] = '_lr'
            elif '_RL' in upper_filename:
                result['layout'] = '_rl'
            elif '_TB' in upper_filename or '_BT' in upper_filename:
                result['layout'] = '_tb'
            elif '_3DH' in upper_filename:
                result['layout'] = '_sbs'
            elif '_3DV' in upper_filename:
                result['layout'] = '_tb'
            else:
                result['layout'] = '_sbs'
            return result

        # Check for equirectangular 360
        if '_360' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'he'
            if '_LR' in upper_filename:
                result['layout'] = '_lr'
            elif '_RL' in upper_filename:
                result['layout'] = '_rl'
            elif '_TB' in upper_filename or '_BT' in upper_filename:
                result['layout'] = '_tb'
            elif '_3DH' in upper_filename:
                result['layout'] = '_sbs'
            elif '_3DV' in upper_filename:
                result['layout'] = '_tb'
            else:
                result['layout'] = '_sbs'
            return result

        # Check for equirectangular 180
        if '_180' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'he'
            if '_LR' in upper_filename:
                result['layout'] = '_lr'
            elif '_RL' in upper_filename:
                result['layout'] = '_rl'
            elif '_TB' in upper_filename or '_BT' in upper_filename:
                result['layout'] = '_tb'
            elif '_3DH' in upper_filename:
                result['layout'] = '_sbs'
            elif '_3DV' in upper_filename:
                result['layout'] = '_tb'
            else:
                result['layout'] = '_sbs'
            return result

        return result

    @staticmethod
    def _classify_by_resolution(width: int, height: int) -> str:
        """
        Classifies video as '2D', 'most_likely_VR', or 'uncertain' based on resolution.

        Returns:
            '2D': Definitely 2D based on resolution
            'most_likely_VR': Resolution suggests VR (should trigger ML)
            'uncertain': Cannot determine (should check other heuristics)
        """
        # < 1080p -> 2D
        if height < 1080 and width < 1920:
            return '2D'

        # Exactly 1920x1080p or 3840x2160p (and portrait 2160x3840) -> 2D
        if (width == 1920 and height == 1080) or (width == 3840 and height == 2160) or (width == 2160 and height == 3840):
            return '2D'

        # Check if width = 2x height or height = 2x width (VR aspect ratios)
        is_sbs_aspect = width > 1000 and 1.8 <= (width / height) <= 2.2
        is_tb_aspect = height > 1000 and 1.8 <= (height / width) <= 2.2

        if is_sbs_aspect or is_tb_aspect:
            return 'most_likely_VR'

        # Bigger than 2160p -> most likely VR
        if height > 2160 or width > 3840:
            return 'most_likely_VR'

        return 'uncertain'

    def _update_video_parameters(self):
        """
        Consolidates logic for determining video type and building the FFmpeg filter string.
        Called from open_video and reapply_video_settings.

        Detection priority:
        1. Filename-based detection (most specific)
        2. Resolution-based classification
        3. ML detection (only if resolution suggests VR)
        """
        if not self.video_info:
            return

        width = self.video_info.get('width', 0)
        height = self.video_info.get('height', 0)

        # Skip detection if user has manually set the video type
        if self.video_type_setting != 'auto':
            self.determined_video_type = self.video_type_setting
            # Clear VR metadata if manually set to 2D
            if self.video_type_setting == '2D':
                self.vr_input_format = ""
                self.vr_fov = 0
            self.logger.info(f"Using configured video type: {self.determined_video_type}")
            self.ffmpeg_filter_string = self._build_ffmpeg_filter_string()
            self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
            return

        # STEP 1: Try filename-based detection first
        filename_result = self._detect_format_from_filename(self.video_path)

        if filename_result['type'] == '2D':
            self.logger.info(f"Filename indicates 2D video (contains _2D or _FLAT)")
            self.determined_video_type = '2D'
            # Clear VR metadata for 2D videos
            self.vr_input_format = ""
            self.vr_fov = 0
            self.ffmpeg_filter_string = self._build_ffmpeg_filter_string()
            self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
            return

        if filename_result['type'] == 'VR':
            self.logger.info(f"Filename indicates VR video: projection={filename_result['projection']}, layout={filename_result['layout']}, fov={filename_result['fov']}")
            self.determined_video_type = 'VR'

            # Apply detected format
            if filename_result['projection'] and filename_result['layout']:
                self.vr_input_format = f"{filename_result['projection']}{filename_result['layout']}"
                self.logger.info(f"Set VR format to: {self.vr_input_format}")

            # Apply detected FOV if available
            if filename_result['fov']:
                self.vr_fov = filename_result['fov']
                self.logger.info(f"Set VR FOV to: {self.vr_fov}")

            self.ffmpeg_filter_string = self._build_ffmpeg_filter_string()
            self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
            return

        # STEP 2: Filename inconclusive - check resolution
        resolution_classification = self._classify_by_resolution(width, height)

        if resolution_classification == '2D':
            self.logger.info(f"Resolution {width}x{height} classified as 2D (< 1080p or standard 2D resolution)")
            self.determined_video_type = '2D'
            # Clear VR metadata for 2D videos
            self.vr_input_format = ""
            self.vr_fov = 0
            self.ffmpeg_filter_string = self._build_ffmpeg_filter_string()
            self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
            return

        # STEP 3: Resolution suggests VR - run ML detection
        if resolution_classification == 'most_likely_VR':
            self.logger.debug(f"Resolution {width}x{height} suggests VR - running ML detection")

            # Try ML detection if model available
            # Cache ML detection to avoid re-running expensive inference on every settings change
            if os.path.exists(self.ml_model_path) and not hasattr(self, '_ml_detection_cached'):
                try:
                    # Lazy load detector
                    if self.ml_detector is None:
                        self.logger.debug("Loading ML format detector...")
                        self.ml_detector = RealMLVRFormatDetector(logger=self.logger)
                        self.ml_detector.load_model(self.ml_model_path)
                        self.logger.debug("ML format detector loaded successfully")

                    # Detect format
                    ml_result = self.ml_detector.detect(self.video_path, self.video_info, num_frames=3)

                    if ml_result and ml_result.get('confidence', 0) > 0.5:
                        self.logger.debug(f"ML detected format: {ml_result.get('format_string')} "
                                        f"(confidence: {ml_result.get('confidence'):.2f})")

                        # Apply ML results
                        self.determined_video_type = ml_result['video_type']

                        if ml_result['video_type'] == 'VR':
                            self.vr_input_format = ml_result['format_string']
                            if ml_result.get('fov'):
                                self.vr_fov = ml_result['fov']
                        else:
                            # ML detected 2D - clear VR metadata
                            self.vr_input_format = ""
                            self.vr_fov = 0

                        self._ml_detection_cached = True  # Cache result to avoid re-running on settings changes
                        self.ffmpeg_filter_string = self._build_ffmpeg_filter_string()
                        self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
                        self.logger.debug(f"Frame size bytes updated to: {self.frame_size_bytes} for YOLO size {self.yolo_input_size}")
                        return
                    else:
                        self.logger.debug("ML detection confidence low, falling back to resolution heuristics")

                except Exception as e:
                    self.logger.warning(f"ML detection failed: {e}, falling back to resolution heuristics")

        # STEP 4: Fallback - use resolution-based heuristics
        # Check for VR-like aspect ratios
        is_sbs_resolution = width > 1000 and 1.8 <= (width / height) <= 2.2
        is_tb_resolution = height > 1000 and 1.8 <= (height / width) <= 2.2

        if is_sbs_resolution or is_tb_resolution:
            self.logger.info(f"Resolution aspect ratio suggests VR (SBS: {is_sbs_resolution}, TB: {is_tb_resolution})")
            self.determined_video_type = 'VR'

            # Determine format based on aspect ratio
            suggested_base = 'he'
            suggested_layout = '_tb' if is_tb_resolution else '_sbs'

            self.vr_input_format = f"{suggested_base}{suggested_layout}"
            self.logger.info(f"Auto-detected VR format: {self.vr_input_format}")
        else:
            self.logger.info(f"Resolution {width}x{height} does not suggest VR - defaulting to 2D")
            self.determined_video_type = '2D'
            # Clear VR metadata for 2D videos
            self.vr_input_format = ""
            self.vr_fov = 0

        self.ffmpeg_filter_string = self._build_ffmpeg_filter_string()
        self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
        self.logger.info(f"Frame size bytes updated to: {self.frame_size_bytes} for YOLO size {self.yolo_input_size}")

    @staticmethod
    def get_video_type_heuristic(video_path: str, use_ml: bool = False) -> str:
        """
        A lightweight heuristic to guess the video type (2D/VR) and format (SBS/TB)
        without fully opening the video. Uses ffprobe for metadata.

        Args:
            video_path: Path to video file
            use_ml: If True, attempt ML detection first (requires model in /models)

        Returns:
            String like "2D", "VR (he_sbs)", "VR (fisheye_tb)", or "Unknown"
        """
        if not os.path.exists(video_path):
            return "Unknown"

        try:
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                   '-show_entries', 'stream=width,height,pix_fmt', '-of', 'json', video_path]
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True, timeout=5, creationflags=creation_flags)
            data = json.loads(result.stdout)
            stream_info = data.get('streams', [{}])[0]
            width = int(stream_info.get('width', 0))
            height = int(stream_info.get('height', 0))
            pix_fmt = stream_info.get('pix_fmt', '')
        except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, IndexError, ValueError, OSError):
            return "Unknown"

        if width == 0 or height == 0:
            return "Unknown"

        # Try ML detection if requested
        if use_ml:
            try:
                model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'vr_detector_model_rf.pkl')
                if os.path.exists(model_path):
                    detector = RealMLVRFormatDetector(logger=None)
                    detector.load_model(model_path)

                    video_info = {'width': width, 'height': height, 'pix_fmt': pix_fmt}
                    ml_result = detector.detect(video_path, video_info, num_frames=3)

                    if ml_result and ml_result.get('confidence', 0) > 0.5:
                        if ml_result['video_type'] == '2D':
                            return "2D"
                        else:
                            return f"VR ({ml_result['format_string']})"
            except (ImportError, OSError, ValueError, RuntimeError):
                pass  # Fall back to filename heuristics

        # Fallback to filename heuristics
        is_sbs_resolution = width > 1000 and 1.8 * height <= width <= 2.2 * height
        is_tb_resolution = height > 1000 and 1.8 * width <= height <= 2.2 * width
        upper_video_path = video_path.upper()
        vr_keywords = ['VR', '_180', '_360', 'SBS', '_TB', 'FISHEYE', 'EQUIRECTANGULAR', 'LR_', 'Oculus', '_3DH', 'MKX200']
        has_vr_keyword = any(kw in upper_video_path for kw in vr_keywords)

        if not (is_sbs_resolution or is_tb_resolution or has_vr_keyword):
            return "2D"

        # If VR, guess the specific format
        suggested_base = 'he'
        suggested_layout = '_sbs'
        if is_tb_resolution or any(kw in upper_video_path for kw in ['_TB', 'TB_', 'TOPBOTTOM', 'OVERUNDER', '_OU', 'OU_']):
            suggested_layout = '_tb'
        if any(kw in upper_video_path for kw in ['FISHEYE', 'MKX', 'RF52']):
            suggested_base = 'fisheye'

        return f"VR ({suggested_base}{suggested_layout})"
