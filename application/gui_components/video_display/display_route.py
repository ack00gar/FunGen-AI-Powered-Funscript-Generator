"""Single-tick display routing decision.

Everything that selects what pixels reach ``imgui.image`` (which texture,
which UV, which aspect, which overlays, what status text) is computed by
:func:`compute_display_route` and consumed by the draw loop as a single
struct. Keeps the draw site free of scattered branching.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class DisplayRoute:
    source: str  # 'mpv_shader' | 'mpv_direct' | 'cpu_tracker' | 'blank'
    texture_id: int = 0
    uv: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    content_aspect: float = 16 / 9
    fill_panel: bool = False
    show_overlays: bool = True
    overlay_status: Optional[str] = None
    status_busy: bool = False
    shader_locked: bool = False


_BLANK_UV = (0.0, 0.0, 1.0, 1.0)


def _compose_uv(outer: Tuple[float, float, float, float],
                inner: Tuple[float, float, float, float]
                ) -> Tuple[float, float, float, float]:
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    ow = ox1 - ox0
    oh = oy1 - oy0
    return (ox0 + ix0 * ow, oy0 + iy0 * oh,
            ox0 + ix1 * ow, oy0 + iy1 * oh)


def _status_and_busy(proc, mpv_display) -> Tuple[Optional[str], bool]:
    if proc is None:
        return None, False
    if getattr(proc, '_video_open_in_progress', False):
        return "Opening video...", True
    err = getattr(mpv_display, 'last_load_error', None) if mpv_display else None
    if err:
        return f"Video load error: {err}", False
    import time
    seek_since = float(getattr(proc, '_seek_in_progress_since', 0.0) or 0.0)
    if seek_since > 0.0 and (time.monotonic() - seek_since) > 0.2:
        return "Seeking...", True
    return None, False


def compute_display_route(app) -> DisplayRoute:
    proc = getattr(app, 'processor', None)
    gui = getattr(app, 'gui_instance', None)
    app_state = getattr(app, 'app_state_ui', None)
    settings = getattr(app, 'app_settings', None)
    mpv_display = getattr(gui, 'mpv_display', None) if gui else None
    tracker = getattr(app, 'tracker', None)

    if gui is None or proc is None or app_state is None:
        return DisplayRoute(source='blank',
                            overlay_status="No video",
                            show_overlays=False)

    status_text, status_busy = _status_and_busy(proc, mpv_display)

    video_loaded = bool(getattr(proc, 'video_path', '') and getattr(proc, 'video_info', None))
    if not video_loaded:
        return DisplayRoute(source='blank',
                            overlay_status=status_text or "No video loaded",
                            status_busy=status_busy,
                            show_overlays=False)

    tracker_active = bool(tracker and getattr(tracker, 'tracking_active', False))

    determined_type = getattr(proc, 'determined_video_type', '') or ''
    is_vr = (determined_type == 'VR'
             or getattr(proc, 'video_type_setting', '') == 'VR')

    vr_mode = 'shader_dewarp'
    shader_locked = False
    if settings:
        try:
            vr_mode = settings.get('vr_display_mode', 'shader_dewarp')
        except Exception:
            pass
        try:
            shader_locked = bool(settings.get('vr_shader_lock_to_tracker', False))
        except Exception:
            pass
    if vr_mode == 'v360_baked':
        vr_mode = 'shader_dewarp'
        shader_locked = True
        if settings is not None:
            try:
                settings.set('vr_display_mode', 'shader_dewarp')
                settings.set('vr_shader_lock_to_tracker', True)
            except Exception:
                pass

    shader_obj = getattr(gui, 'vr_dewarp_shader', None)
    shader_ready = bool(shader_obj is not None and shader_obj.is_ready)

    mpv_ok = (mpv_display is not None
              and getattr(mpv_display, 'is_loaded', False))

    # Use mpv whenever loaded so 2D paused/stepping matches playback (no 640 flip).
    use_mpv = mpv_ok and not tracker_active

    zoom_uv = _BLANK_UV
    try:
        zoom_uv = app_state.get_video_uv_coords()
    except Exception:
        pass

    eye_uv = _BLANK_UV
    if is_vr:
        try:
            from video import vr_panel
            eye = vr_panel.read_setting(settings, default=vr_panel.EYE_LEFT)
            fmt = getattr(proc, 'vr_input_format', '') or ''
            region = vr_panel.resolve_eye(fmt, eye)
            eye_uv = (region.x, region.y,
                      region.x + region.w, region.y + region.h)
        except Exception:
            eye_uv = _BLANK_UV

    if tracker_active:
        return DisplayRoute(
            source='cpu_tracker',
            texture_id=int(getattr(gui, 'frame_texture_id', 0)),
            uv=zoom_uv,
            content_aspect=_content_aspect_from_proc(proc),
            fill_panel=False,
            show_overlays=True,
            overlay_status=status_text,
            status_busy=status_busy,
        )

    if not use_mpv:
        return DisplayRoute(
            source='cpu_tracker',
            texture_id=int(getattr(gui, 'frame_texture_id', 0)),
            uv=zoom_uv,
            content_aspect=_content_aspect_from_proc(proc),
            fill_panel=False,
            show_overlays=True,
            overlay_status=status_text,
            status_busy=status_busy,
        )

    mpv_w = int(getattr(gui, 'mpv_display_w', 1) or 1)
    mpv_h = int(getattr(gui, 'mpv_display_h', 1) or 1)
    mpv_aspect = mpv_w / mpv_h if mpv_h > 0 else 1.0

    if is_vr and vr_mode == 'shader_dewarp' and shader_ready:
        dewarp_tex = int(getattr(gui, 'vr_dewarp_texture_id', 0))
        return DisplayRoute(
            source='mpv_shader',
            texture_id=dewarp_tex,
            uv=zoom_uv,
            content_aspect=1.0,
            fill_panel=not shader_locked,
            show_overlays=shader_locked,
            overlay_status=status_text,
            status_busy=status_busy,
            shader_locked=shader_locked,
        )

    if is_vr:  # passthrough
        final_uv = _compose_uv(eye_uv, zoom_uv)
        eye_w = max(1e-6, eye_uv[2] - eye_uv[0])
        eye_h = max(1e-6, eye_uv[3] - eye_uv[1])
        content_aspect = mpv_aspect * (eye_w / eye_h)
        return DisplayRoute(
            source='mpv_direct',
            texture_id=int(getattr(gui, 'mpv_display_texture_id', 0)),
            uv=final_uv,
            content_aspect=content_aspect,
            fill_panel=False,
            show_overlays=True,
            overlay_status=status_text,
            status_busy=status_busy,
        )

    return DisplayRoute(
        source='mpv_direct',
        texture_id=int(getattr(gui, 'mpv_display_texture_id', 0)),
        uv=zoom_uv,
        content_aspect=mpv_aspect,
        fill_panel=False,
        show_overlays=True,
        overlay_status=status_text,
        status_busy=status_busy,
    )


def _content_aspect_from_proc(proc) -> float:
    cf = getattr(proc, 'current_frame', None)
    if cf is not None and cf.shape[0] > 0 and cf.shape[1] > 0:
        return cf.shape[1] / cf.shape[0]
    dfw = int(getattr(proc, '_display_frame_w', 0) or 0)
    dfh = int(getattr(proc, '_display_frame_h', 0) or 0)
    if dfw > 0 and dfh > 0:
        return dfw / dfh
    return 16 / 9


def fit_rect_to_panel(content_aspect: float,
                      avail_w: float, avail_h: float,
                      zoom: float = 1.0
                      ) -> Tuple[float, float, float, float]:
    if avail_w <= 0 or avail_h <= 0 or content_aspect <= 0:
        return 0.0, 0.0, 0.0, 0.0
    fit_w = avail_w
    fit_h = fit_w / content_aspect
    if fit_h > avail_h:
        fit_h = avail_h
        fit_w = fit_h * content_aspect
    fit_w *= max(0.1, float(zoom))
    fit_h *= max(0.1, float(zoom))
    fit_w = min(fit_w, avail_w)
    fit_h = min(fit_h, avail_h)
    off_x = (avail_w - fit_w) * 0.5
    off_y = (avail_h - fit_h) * 0.5
    return fit_w, fit_h, off_x, off_y
