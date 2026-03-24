"""Video information display mixin for InfoGraphsUI."""
import imgui
import os
from application.utils import _format_time
from application.utils.imgui_layout_helpers import (
    begin_settings_columns, end_settings_columns, row_label, row_end, row_separator,
)


class VideoInfoMixin:

    def _get_k_resolution_label(self, width, height):
        if width <= 0 or height <= 0:
            return ""
        if (1280, 720) == (width, height):
            return " (HD)"
        if (1920, 1080) == (width, height):
            return " (Full HD)"
        if (2560, 1440) == (width, height):
            return " (QHD/2.5K)"
        if (3840, 2160) == (width, height):
            return " (4K UHD)"
        if width >= 7600:
            return " (8K)"
        if width >= 6600:
            return " (7K)"
        if width >= 5600:
            return " (6K)"
        if width >= 5000:
            return " (5K)"
        if width >= 3800:
            return " (4K)"
        return ""

    def _render_content_video_info(self):
        self.video_info_perf.start_timing()

        if not self.app.processor or not self.app.processor.video_info:
            imgui.text_disabled("No video loaded.")
            self.video_info_perf.end_timing()
            return

        file_mgr = self.app.file_manager
        info = self.app.processor.video_info
        processor = self.app.processor
        width, height = info.get("width", 0), info.get("height", 0)

        # --- File ---
        begin_settings_columns("vi_file_cols")

        row_label("Path")
        path = os.path.dirname(file_mgr.video_path) if file_mgr.video_path else "N/A"
        imgui.text_wrapped(path)
        row_end()

        row_label("File")
        imgui.text_wrapped(info.get("filename", "N/A"))
        row_end()

        row_label("Size")
        size_bytes = info.get("file_size", 0)
        if size_bytes > 1024 * 1024 * 1024:
            imgui.text(f"{size_bytes / (1024**3):.2f} GB")
        elif size_bytes > 0:
            imgui.text(f"{size_bytes / (1024**2):.2f} MB")
        else:
            imgui.text("N/A")
        row_end()

        row_separator()

        # --- Video ---
        begin_settings_columns("vi_video_cols")

        row_label("Resolution")
        imgui.text(f"{width}x{height}{self._get_k_resolution_label(width, height)}")
        row_end()

        row_label("Duration")
        imgui.text(_format_time(self.app, info.get('duration', 0.0)))
        row_end()

        row_label("Total Frames")
        imgui.text(f"{info.get('total_frames', 0):,}")
        row_end()

        row_label("Frame Rate")
        fps_mode = "VFR" if info.get("is_vfr", False) else "CFR"
        imgui.text(f"{info.get('fps', 0):.3f} ({fps_mode})")
        row_end()

        row_label("Bitrate")
        bitrate_bps = info.get("bitrate", 0)
        if bitrate_bps > 0:
            imgui.text(f"{bitrate_bps / 1_000_000:.2f} Mbit/s")
        else:
            imgui.text("N/A")
        row_end()

        row_label("Bit Depth")
        imgui.text(f"{info.get('bit_depth', 'N/A')} bit")
        row_end()

        row_label("Codec")
        codec_name = info.get('codec_name', 'N/A')
        imgui.text(codec_name.upper() if codec_name != 'N/A' else codec_name)
        codec_long = info.get('codec_long_name', '')
        if codec_long and imgui.is_item_hovered():
            imgui.set_tooltip(codec_long)
        row_end()

        row_label("Detected Type")
        imgui.text(processor.determined_video_type or "N/A")
        row_end()

        # VR-specific rows
        if processor.determined_video_type == 'VR':
            row_label("VR Format")
            imgui.text((processor.vr_input_format or "N/A").upper())
            row_end()

            row_label("VR FOV")
            vr_fov = processor.vr_fov if hasattr(processor, 'vr_fov') else 0
            imgui.text(f"{vr_fov} deg" if vr_fov > 0 else "N/A")
            row_end()

        row_label("Active Source")
        if (hasattr(processor, "_active_video_source_path")
                and processor._active_video_source_path != processor.video_path):
            imgui.text("Preprocessed")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    f"Using: {os.path.basename(processor._active_video_source_path)}\n"
                    "All filtering/de-warping is pre-applied.")
        else:
            imgui.text("Original")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    f"Using: {os.path.basename(processor.video_path)}\n"
                    "Filters are applied on-the-fly.")
        row_end()

        end_settings_columns()

        # --- Audio ---
        if info.get("has_audio"):
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            begin_settings_columns("vi_audio_cols")

            row_label("Audio Codec")
            a_codec = info.get("audio_codec_name", "")
            imgui.text(a_codec.upper() if a_codec else "N/A")
            a_long = info.get("audio_codec_long_name", "")
            if a_long and imgui.is_item_hovered():
                imgui.set_tooltip(a_long)
            row_end()

            row_label("Audio Bitrate")
            a_bps = info.get("audio_bitrate", 0)
            imgui.text(f"{a_bps / 1000:.0f} kbps" if a_bps > 0 else "N/A")
            row_end()

            row_label("Sample Rate")
            a_sr = info.get("audio_sample_rate", 0)
            imgui.text(f"{a_sr:,} Hz" if a_sr > 0 else "N/A")
            row_end()

            row_label("Channels")
            a_ch = info.get("audio_channels", 0)
            if a_ch == 1:
                imgui.text("Mono")
            elif a_ch == 2:
                imgui.text("Stereo")
            elif a_ch > 0:
                imgui.text(f"{a_ch}ch")
            else:
                imgui.text("N/A")
            row_end()

            end_settings_columns()
        else:
            imgui.spacing()
            imgui.text_colored("No audio stream", 0.5, 0.5, 0.5, 1.0)

        imgui.spacing()
        self.video_info_perf.end_timing()
