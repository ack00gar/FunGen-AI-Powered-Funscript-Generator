"""Video information display mixin for InfoGraphsUI."""
import imgui
import os
from application.utils import _format_time


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
        file_mgr = self.app.file_manager

        imgui.columns(2, "video_info_stats", border=False)
        imgui.set_column_width(0, 120 * imgui.get_io().font_global_scale)

        if self.app.processor and self.app.processor.video_info:
            path = (
                os.path.dirname(file_mgr.video_path)
                if file_mgr.video_path
                else "N/A (Drag & Drop Video)"
            )
            filename = self.app.processor.video_info.get("filename", "N/A")

            info = self.app.processor.video_info
            width, height = info.get("width", 0), info.get("height", 0)

            imgui.text("Path:")
            imgui.next_column()
            imgui.text_wrapped(path)
            imgui.next_column()

            imgui.text("File:")
            imgui.next_column()
            imgui.text_wrapped(filename)
            imgui.next_column()

            imgui.text("Resolution:")
            imgui.next_column()
            imgui.text(
                f"{width}x{height}{self._get_k_resolution_label(width, height)}"
            )
            imgui.next_column()

            imgui.text("Duration:")
            imgui.next_column()
            imgui.text(f"{_format_time(self.app, info.get('duration', 0.0))}")
            imgui.next_column()

            imgui.text("Total Frames:")
            imgui.next_column()
            imgui.text(f"{info.get('total_frames', 0)}")
            imgui.next_column()

            imgui.text("Frame Rate:")
            imgui.next_column()
            fps_text = f"{info.get('fps', 0):.3f}"
            fps_mode = " (VFR)" if info.get("is_vfr", False) else " (CFR)"
            imgui.text(fps_text + fps_mode)
            imgui.next_column()

            imgui.text("Size:")
            imgui.next_column()
            size_bytes = info.get("file_size", 0)
            if size_bytes > 0:
                if size_bytes > 1024 * 1024 * 1024:
                    size_str = f"{size_bytes / (1024**3):.2f} GB"
                else:
                    size_str = f"{size_bytes / (1024**2):.2f} MB"
            else:
                size_str = "N/A"
            imgui.text(size_str)
            imgui.next_column()

            imgui.text("Bitrate:")
            imgui.next_column()
            bitrate_bps = info.get("bitrate", 0)
            if bitrate_bps > 0:
                bitrate_mbps = bitrate_bps / 1_000_000
                bitrate_str = f"{bitrate_mbps:.2f} Mbit/s"
            else:
                bitrate_str = "N/A"
            imgui.text(bitrate_str)
            imgui.next_column()

            imgui.text("Bit Depth:")
            imgui.next_column()
            imgui.text(f"{info.get('bit_depth', 'N/A')} bit")
            imgui.next_column()

            imgui.text("Codec:")
            imgui.next_column()
            codec_name = info.get('codec_name', 'N/A')
            imgui.text(codec_name.upper() if codec_name != 'N/A' else codec_name)
            if imgui.is_item_hovered():
                codec_long = info.get('codec_long_name', 'N/A')
                if codec_long != 'N/A':
                    imgui.set_tooltip(codec_long)
            imgui.next_column()

            imgui.text("Detected Type:")
            imgui.next_column()
            imgui.text(f"{self.app.processor.determined_video_type or 'N/A'}")
            imgui.next_column()

            # Show VR format and FOV if VR video
            if self.app.processor.determined_video_type == 'VR':
                imgui.text("VR Format:")
                imgui.next_column()
                vr_format = self.app.processor.vr_input_format or 'N/A'
                imgui.text(vr_format.upper())
                imgui.next_column()

                imgui.text("VR FOV:")
                imgui.next_column()
                vr_fov = self.app.processor.vr_fov if hasattr(self.app.processor, 'vr_fov') else 0
                if vr_fov > 0:
                    imgui.text(f"{vr_fov}deg")
                else:
                    imgui.text("N/A")
                imgui.next_column()

            imgui.text("Active Source:")
            imgui.next_column()
            processor = self.app.processor
            if (
                hasattr(processor, "_active_video_source_path")
                and processor._active_video_source_path != processor.video_path
            ):
                imgui.text("Preprocessed File")
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        f"Using: {os.path.basename(processor._active_video_source_path)}\n"
                        "All filtering/de-warping is pre-applied."
                    )
            else:
                imgui.text("Original File")
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        f"Using: {os.path.basename(processor.video_path)}\n"
                        "Filters are applied on-the-fly."
                    )
            imgui.next_column()

            # --- Audio Stream Info ---
            imgui.separator()
            if info.get("has_audio"):
                imgui.text("Audio Codec:")
                imgui.next_column()
                a_codec = info.get("audio_codec_name", "")
                imgui.text(a_codec.upper() if a_codec else "N/A")
                if imgui.is_item_hovered():
                    a_long = info.get("audio_codec_long_name", "")
                    if a_long:
                        imgui.set_tooltip(a_long)
                imgui.next_column()

                imgui.text("Audio Bitrate:")
                imgui.next_column()
                a_bps = info.get("audio_bitrate", 0)
                if a_bps > 0:
                    imgui.text(f"{a_bps / 1000:.0f} kbps")
                else:
                    imgui.text("N/A")
                imgui.next_column()

                imgui.text("Sample Rate:")
                imgui.next_column()
                a_sr = info.get("audio_sample_rate", 0)
                if a_sr > 0:
                    imgui.text(f"{a_sr} Hz")
                else:
                    imgui.text("N/A")
                imgui.next_column()

                imgui.text("Channels:")
                imgui.next_column()
                a_ch = info.get("audio_channels", 0)
                if a_ch == 1:
                    imgui.text("Mono")
                elif a_ch == 2:
                    imgui.text("Stereo")
                elif a_ch > 0:
                    imgui.text(f"{a_ch}ch")
                else:
                    imgui.text("N/A")
                imgui.next_column()
            else:
                imgui.text("Audio:")
                imgui.next_column()
                imgui.text_colored("No audio stream", 0.5, 0.5, 0.5, 1.0)
                imgui.next_column()
        else:
            imgui.text("Status:")
            imgui.next_column()
            imgui.text("Video details not loaded.")
            imgui.next_column()
        imgui.columns(1)
        imgui.spacing()

        self.video_info_perf.end_timing()
