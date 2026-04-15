"""Proxy suggestion + progress dialogs.

Two modals, driven by a small state machine on the app instance:

- ProxyController.open_suggest(source_path, video_info, vr_input_format)
    Shows the suggestion popup. User can Create, Skip, or check "don't ask again".
- On Create -> a worker thread runs the encode, the suggestion popup closes
  and the progress modal opens. Cancel kills the encode.
- On completion -> swap the active source to the proxy (if enabled) and
  close the progress modal.

The encoder itself lives in video.proxy_builder. This file is pure UI glue.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import imgui
import tempfile

from application.utils.imgui_helpers import center_next_window_pivot
from video.proxy_builder import (
    ProxyBuilder, ProxyJob, PROXY_SUFFIX, SIDECAR_SUFFIX,
    proxy_path_for, resolve_proxy_target_path, is_valid_proxy,
    is_proxy_filename, should_suggest_proxy,
)


class ProxyController:
    """One per application. Owns the modal open-state, the worker thread,
    and progress values."""

    def __init__(self, app):
        self.app = app
        # Suggestion modal.
        self._suggest_open = False
        self._suggest_needs_open = False  # one-shot flag: call imgui.open_popup in render()
        self._suggest_source: Optional[str] = None
        self._suggest_video_info: Optional[dict] = None
        self._suggest_vr_format: str = "he_sbs"
        self._suggest_dont_ask = False
        self._suggest_output_mode: str = "next_to_source"
        self._suggest_custom_folder: str = ""
        # Progress modal.
        self._progress_open = False
        self._progress_needs_open = False  # True for one frame to trigger open_popup
        self._progress_fraction = 0.0
        self._progress_time_s = 0.0
        self._progress_speed = 0.0
        self._progress_eta_s = 0.0
        self._progress_source: str = ""
        self._progress_target: str = ""
        self._cancel_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._worker_result: Optional[bool] = None
        # Wall-clock + progress-reception timing (so UI feels alive even
        # before ffmpeg emits its first -progress line).
        self._encode_started_at: float = 0.0
        self._last_progress_ts: float = 0.0
        self._source_fps: float = 30.0
        # Live preview (ffmpeg writes a JPEG once every ~7s; we poll it).
        self._preview_path: Optional[str] = None
        self._preview_lock = threading.Lock()
        self._preview_bytes: Optional[bytes] = None  # raw JPEG
        self._preview_pending = False                # new bytes since last upload
        self._preview_mtime: float = 0.0
        self._preview_tex_id: int = 0
        self._preview_w: int = 0
        self._preview_h: int = 0
        self._preview_stop = threading.Event()
        self._preview_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------- public API

    def open_suggest_if_needed(self, source_path: str, video_info: dict,
                               determined_video_type: str,
                               vr_input_format: str) -> None:
        """Call after a successful open_video; may pop the suggestion modal.
        Silent no-op if already-valid proxy exists, if user dismissed, etc.
        """
        settings = self.app.app_settings
        if not settings.get("video_proxy_suggest_on_open", True):
            return
        if settings.get("video_proxy_ask_dismissed", False):
            return
        if is_proxy_filename(source_path):
            return  # already a proxy
        if is_valid_proxy(source_path):
            return  # will be auto-loaded elsewhere; no need to suggest
        min_gb = float(settings.get("video_proxy_min_size_gb", 1.5))
        if not should_suggest_proxy(video_info, determined_video_type, min_gb):
            return
        self._suggest_source = source_path
        self._suggest_video_info = dict(video_info)
        self._suggest_vr_format = vr_input_format or "he_sbs"
        self._suggest_dont_ask = False
        self._suggest_output_mode = settings.get(
            "video_proxy_output_mode", "next_to_source")
        self._suggest_custom_folder = settings.get(
            "video_proxy_custom_folder", "") or ""
        self._suggest_open = True
        self._suggest_needs_open = True

    def open_suggest_manual(self) -> None:
        """Menu entry target: force the suggestion popup for the current
        video, even if 'don't ask again' is set or a valid proxy exists."""
        proc = self.app.processor
        if not proc or not proc.video_info or not proc.video_path:
            self.app.logger.info("No video loaded; open one first.",
                                 extra={"status_message": True})
            return
        if proc.determined_video_type not in ("VR", "2D"):
            self.app.logger.info("Proxy not available for this source.",
                                 extra={"status_message": True})
            return
        self._suggest_source = proc.video_path
        self._suggest_video_info = dict(proc.video_info)
        self._suggest_vr_format = (proc.vr_input_format or "he_sbs"
                                   if proc.determined_video_type == "VR" else "2d")
        self._suggest_dont_ask = False
        settings = self.app.app_settings
        self._suggest_output_mode = settings.get(
            "video_proxy_output_mode", "next_to_source")
        self._suggest_custom_folder = settings.get(
            "video_proxy_custom_folder", "") or ""
        self._suggest_open = True
        self._suggest_needs_open = True

    # ---------------------------------------------------------------- render

    def render(self) -> None:
        """Call once per frame from the main menu/UI render loop."""
        self._render_suggest()
        self._render_progress()

    # ------------------------------------------------------------- internals

    def _render_suggest(self) -> None:
        if not self._suggest_open:
            return
        if self._suggest_needs_open:
            imgui.open_popup("Proxy suggestion##fungen_proxy")
            self._suggest_needs_open = False
        center_next_window_pivot()
        imgui.set_next_window_size_constraints((560, 0), (900, 600))
        imgui.set_next_window_size(640, 0, condition=imgui.APPEARING)
        opened = imgui.begin_popup_modal("Proxy suggestion##fungen_proxy",
                                         flags=imgui.WINDOW_NO_COLLAPSE)[0]
        if not opened:
            # Popup not visible yet (frame after open_popup queued). Bail;
            # we'll render it next tick. Do NOT flip _suggest_open off here.
            return

        src = self._suggest_source or ""
        info = self._suggest_video_info or {}
        w = info.get("width", 0)
        h = info.get("height", 0)
        size_bytes = int(info.get("file_size") or info.get("file_size_bytes") or 0)
        size_gb = size_bytes / (1024 ** 3)
        dur_s = float(info.get("duration", 0) or 0)

        is_vr = self._suggest_vr_format != "2d"
        kind = "VR" if is_vr else "2D"
        imgui.text(f"Large {kind} file: {os.path.basename(src)}")
        imgui.text_disabled(f"{w}x{h}, {size_gb:.1f} GB, "
                            f"{int(dur_s//60):02d}:{int(dur_s%60):02d}")
        imgui.spacing()
        if is_vr:
            imgui.text("Create an iframe 1080p flat proxy for fast scripting?")
        else:
            imgui.text("Create an iframe 1080p proxy for fast scripting?")
        imgui.text_disabled(
            "Every frame is a keyframe (scripter 'iframe' convention): instant\n"
            "forward, backward and random seek. Original stays untouched."
        )
        imgui.spacing()

        # Rough size estimate: all-I 1080p HEVC at 15 Mbps.
        est_size_gb = (15 * 1_000_000 * dur_s / 8) / (1024 ** 3) if dur_s > 0 else 0.0
        imgui.text_disabled(f"Estimated proxy size: ~{est_size_gb:.1f} GB")
        imgui.spacing()

        # Output-location picker (mirrors Settings -> Proxies -> Output location)
        settings = self.app.app_settings
        imgui.text("Save to:")
        modes = [
            ("next_to_source", "Next to original"),
            ("output_folder",  "FunGen output folder"),
            ("custom",         "Custom folder"),
        ]
        for value, label in modes:
            if imgui.radio_button(f"{label}##SugOut_{value}",
                                   self._suggest_output_mode == value):
                self._suggest_output_mode = value
            imgui.same_line()
        imgui.dummy(1, 1)  # end the radio row
        resolved_target = resolve_proxy_target_path(
            src, self._suggest_output_mode,
            output_folder=os.path.abspath(
                settings.get("output_folder_path", "output") or "output"),
            custom_folder=self._suggest_custom_folder,
        )
        if self._suggest_output_mode == "custom":
            imgui.push_item_width(-130)
            ch_cp, np_ = imgui.input_text("##SugCustomFolder",
                                           self._suggest_custom_folder, 1024)
            imgui.pop_item_width()
            if ch_cp:
                self._suggest_custom_folder = np_
            imgui.same_line()
            if imgui.small_button("Browse...##SugCustomBrowse"):
                try:
                    from tkinter import Tk, filedialog
                    root = Tk(); root.withdraw(); root.attributes("-topmost", True)
                    picked = filedialog.askdirectory(
                        title="Proxy output folder",
                        initialdir=self._suggest_custom_folder
                                    or os.path.expanduser("~"))
                    root.destroy()
                    if picked:
                        self._suggest_custom_folder = picked
                except Exception as e:
                    self.app.logger.warning(f"Folder picker failed: {e}")
        imgui.push_style_color(imgui.COLOR_TEXT, 0.7, 0.7, 0.7, 1.0)
        imgui.text_wrapped(f"  -> {resolved_target}")
        imgui.pop_style_color()
        imgui.spacing()

        changed, self._suggest_dont_ask = imgui.checkbox(
            "Don't ask me again", self._suggest_dont_ask)
        imgui.spacing()

        create_clicked = imgui.button("Create Proxy", width=140)
        imgui.same_line()
        skip_clicked = imgui.button("Skip", width=100)

        if create_clicked:
            self._apply_dont_ask()
            # Persist the chosen mode/folder so next launch remembers the
            # user's preference without requiring Settings.
            settings.set("video_proxy_output_mode", self._suggest_output_mode)
            if self._suggest_output_mode == "custom":
                settings.set("video_proxy_custom_folder",
                             self._suggest_custom_folder)
            self._start_encode(src, self._suggest_vr_format,
                               dur_s, info)
            self._suggest_open = False
            imgui.close_current_popup()
        elif skip_clicked:
            self._apply_dont_ask()
            self._suggest_open = False
            imgui.close_current_popup()

        imgui.end_popup()

    def _render_progress(self) -> None:
        if not self._progress_open:
            # If a worker finished since last frame, react to the result here.
            if self._worker is not None and self._worker_result is not None:
                self._finalize_worker()
            return

        # One-shot open_popup on the frame we started the encode (open_popup
        # must be called from the main UI thread, never from the worker).
        if self._progress_needs_open:
            imgui.open_popup("Building proxy...##fungen_proxy_prog")
            self._progress_needs_open = False

        center_next_window_pivot()
        imgui.set_next_window_size_constraints((600, 0), (900, 800))
        opened = imgui.begin_popup_modal(
            "Building proxy...##fungen_proxy_prog",
            flags=(imgui.WINDOW_NO_COLLAPSE
                   | imgui.WINDOW_ALWAYS_AUTO_RESIZE))[0]
        if not opened:
            return

        # --- Header: source + target ---------------------------------------
        imgui.text_colored("Building proxy",
                           0.85, 0.85, 0.95, 1.0)
        imgui.spacing()
        imgui.text(os.path.basename(self._progress_source))
        imgui.push_style_color(imgui.COLOR_TEXT, 0.6, 0.6, 0.65, 1.0)
        imgui.text_wrapped(f"-> {self._progress_target}")
        imgui.pop_style_color()
        imgui.separator()

        # --- Preview thumbnail (centered) ----------------------------------
        preview = self._upload_preview_if_pending()
        if preview is not None:
            tex_id, fw, fh = preview
            if fw > 0 and fh > 0:
                disp_w = 540.0
                disp_h = disp_w * (fh / fw)
                avail_w = imgui.get_content_region_available_width()
                pad = max(0.0, (avail_w - disp_w) * 0.5)
                if pad > 0:
                    imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + pad)
                imgui.image(tex_id, disp_w, disp_h)
                imgui.spacing()
        else:
            # Placeholder while ffmpeg warms up so the modal doesn't
            # suddenly grow by 300px on the first frame.
            placeholder_h = 200.0
            imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.10, 0.10, 0.12, 1.0)
            imgui.begin_child("##proxy_prev_placeholder",
                              width=0, height=placeholder_h, border=True)
            msg = "Waiting for first preview frame..."
            tsz = imgui.calc_text_size(msg)
            avail_w = imgui.get_content_region_available_width()
            imgui.set_cursor_pos_x(max(0.0, (avail_w - tsz[0]) * 0.5))
            imgui.set_cursor_pos_y((placeholder_h - tsz[1]) * 0.5)
            imgui.text_disabled(msg)
            imgui.end_child()
            imgui.pop_style_color()
            imgui.spacing()

        # --- Progress bar + stats ------------------------------------------
        elapsed = (time.time() - self._encode_started_at
                   if self._encode_started_at else 0.0)
        got_first_progress = (self._progress_fraction > 0.0
                              or self._last_progress_ts > 0.0)
        if got_first_progress:
            imgui.progress_bar(
                self._progress_fraction, (-1, 0),
                f"{self._progress_fraction*100:.1f}%")
            m1 = int(self._progress_time_s // 60); s1 = int(self._progress_time_s % 60)
            m2 = int(self._progress_eta_s // 60);  s2 = int(self._progress_eta_s % 60)
            imgui.text_disabled(
                f"{m1:02d}:{s1:02d} encoded   "
                f"{self._progress_speed:.1f}x   "
                f"ETA {m2:02d}:{s2:02d}")
        else:
            pulse = (elapsed % 2.0) / 2.0
            imgui.progress_bar(pulse, (-1, 0), "Starting encoder...")
            me = int(elapsed // 60); se = int(elapsed % 60)
            imgui.text_disabled(f"Elapsed {me:02d}:{se:02d}   (waiting for ffmpeg)")

        imgui.spacing(); imgui.separator(); imgui.spacing()

        # --- Cancel (right-aligned) ----------------------------------------
        worker_done = (self._worker is not None
                       and self._worker_result is not None)
        if worker_done:
            imgui.close_current_popup()

        btn_w = 120.0
        avail_w = imgui.get_content_region_available_width()
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, avail_w - btn_w))
        if imgui.button("Cancel", width=btn_w) and not worker_done:
            self._cancel_event.set()
            imgui.close_current_popup()

        imgui.end_popup()

        if worker_done:
            self._finalize_worker()

    def _apply_dont_ask(self) -> None:
        if self._suggest_dont_ask:
            self.app.app_settings.set("video_proxy_ask_dismissed", True)

    def _start_encode(self, source: str, vr_format: str,
                      duration_s: float, info: dict) -> None:
        settings = self.app.app_settings
        mode = settings.get("video_proxy_output_mode", "next_to_source")
        output_folder = os.path.abspath(
            settings.get("output_folder_path", "output") or "output")
        custom_folder = settings.get("video_proxy_custom_folder", "") or ""
        target = resolve_proxy_target_path(source, mode,
                                           output_folder=output_folder,
                                           custom_folder=custom_folder)
        nb_frames = int(info.get("total_frames", 0) or 0)
        fps = float(info.get("fps", 30.0) or 30.0)

        self._progress_fraction = 0.0
        self._progress_time_s = 0.0
        self._progress_speed = 0.0
        self._progress_eta_s = 0.0
        self._progress_source = source
        self._progress_target = target
        self._cancel_event = threading.Event()
        self._worker_result = None
        self._progress_open = True
        self._progress_needs_open = True
        self._source_fps = fps
        self._encode_started_at = time.time()
        self._last_progress_ts = 0.0

        # Preview: stable path in the system tempdir, overwritten by ffmpeg.
        self._preview_path = os.path.join(
            tempfile.gettempdir(), "fungen_proxy_preview.jpg")
        try:
            if os.path.exists(self._preview_path):
                os.remove(self._preview_path)
        except OSError:
            pass
        self._preview_mtime = 0.0
        self._preview_bytes = None
        self._preview_pending = False
        self._start_preview_poller()

        def _on_progress(frac: float, out_time_s: float, speed: float, eta_s: float):
            self._progress_fraction = frac
            self._progress_time_s = out_time_s
            self._progress_speed = speed
            self._progress_eta_s = eta_s
            self._last_progress_ts = time.time()

        job = ProxyJob(
            source_path=source,
            vr_input_format=vr_format,
            duration_s=duration_s,
            source_nb_frames=nb_frames,
            source_fps=fps,
            progress_cb=_on_progress,
            cancel_event=self._cancel_event,
            target_path=target,
            preview_path=self._preview_path,
        )

        def _run():
            builder = ProxyBuilder(logger=self.app.logger)
            ok = builder.encode(job)
            self._worker_result = ok

        self._worker = threading.Thread(target=_run, daemon=True,
                                         name="ProxyEncode")
        self._worker.start()

    def _start_preview_poller(self) -> None:
        self._preview_stop.clear()
        path = self._preview_path
        def _run():
            while not self._preview_stop.is_set():
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    self._preview_stop.wait(0.5)
                    continue
                if mtime != self._preview_mtime:
                    try:
                        with open(path, "rb") as f:
                            data = f.read()
                    except OSError:
                        self._preview_stop.wait(0.5)
                        continue
                    if data:
                        with self._preview_lock:
                            self._preview_bytes = data
                            self._preview_pending = True
                            self._preview_mtime = mtime
                self._preview_stop.wait(0.5)
        self._preview_thread = threading.Thread(
            target=_run, daemon=True, name="ProxyPreviewPoller")
        self._preview_thread.start()

    def _stop_preview_poller(self) -> None:
        self._preview_stop.set()
        if self._preview_thread and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=1.0)
        self._preview_thread = None
        # Best-effort cleanup of the temp preview file.
        try:
            if self._preview_path and os.path.exists(self._preview_path):
                os.remove(self._preview_path)
        except OSError:
            pass

    def _upload_preview_if_pending(self):
        """Decode the latest JPEG (if any) and upload to the GL texture.
        Returns (tex_id, w, h) or None if no frame available yet."""
        with self._preview_lock:
            pending = self._preview_pending
            data = self._preview_bytes
            self._preview_pending = False
        if pending and data:
            try:
                import numpy as np
                import cv2
                arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    return None
                import OpenGL.GL as gl
                gui = getattr(self.app, "gui_instance", None)
                if gui is None or not hasattr(gui, "update_texture"):
                    return None
                if self._preview_tex_id == 0 or not gl.glIsTexture(self._preview_tex_id):
                    self._preview_tex_id = int(gl.glGenTextures(1))
                    gl.glBindTexture(gl.GL_TEXTURE_2D, self._preview_tex_id)
                    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                gui.update_texture(self._preview_tex_id, img)
                self._preview_h, self._preview_w = img.shape[:2]
            except Exception as e:
                self.app.logger.debug(f"Preview upload failed: {e}")
                return None
        if self._preview_tex_id and self._preview_w > 0 and self._preview_h > 0:
            return (self._preview_tex_id, self._preview_w, self._preview_h)
        return None

    def _finalize_worker(self) -> None:
        ok = self._worker_result
        self._worker = None
        self._worker_result = None
        self._progress_open = False
        self._stop_preview_poller()
        if not ok:
            if self._cancel_event.is_set():
                self.app.logger.info("Proxy encode canceled.",
                                     extra={"status_message": True})
            else:
                self.app.logger.error("Proxy encode failed. See log for details.",
                                      extra={"status_message": True})
            return
        # Success: offer to swap source.
        if self.app.app_settings.get("video_proxy_autoswitch_on_complete", True):
            self._swap_to_proxy()
        self.app.logger.info("Proxy ready.",
                             extra={"status_message": True})

    def _swap_to_proxy(self) -> None:
        """Swap the active source to the proxy by reopening the video file."""
        proxy = self._progress_target
        if not proxy or not os.path.exists(proxy):
            return
        fm = getattr(self.app, "file_manager", None)
        if fm and hasattr(fm, "open_video_from_path"):
            try:
                # Preserve the current frame by deferring to normal reopen; the
                # app state flow already seeks to 0 on video-load, which is
                # acceptable for a just-finished encode from idle.
                fm.open_video_from_path(proxy)
            except Exception as e:
                self.app.logger.warning(f"Could not swap to proxy: {e}")
