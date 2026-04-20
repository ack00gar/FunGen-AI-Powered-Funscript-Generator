"""Typed, auto-completing view over AppSettings.

Goal: replace `app.app_settings.get("audio_volume", 0.8)` with
`app.app_settings.config.audio.volume`. Readers/writers stay live —
sections read/write through the underlying AppSettings store, so debounced
saves and the on-disk JSON format are unchanged.

Defaults declared here are the source of truth; the runtime still merges
with settings_manager.get_default_settings() at load time, so both must
agree. The per-setting default on the property is the fallback if the key
is missing from settings.json AND from get_default_settings().
"""
from typing import Any, Protocol

from config import constants, ui_metrics


class _Store(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...
    def set(self, key: str, value: Any) -> None: ...


class _Section:
    def __init__(self, store: _Store) -> None:
        self._store = store


class AudioConfig(_Section):
    @property
    def enabled(self) -> bool:
        return bool(self._store.get("audio_enabled", True))

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._store.set("audio_enabled", bool(value))

    @property
    def volume(self) -> float:
        return float(self._store.get("audio_volume", 0.8))

    @volume.setter
    def volume(self, value: float) -> None:
        self._store.set("audio_volume", float(value))

    @property
    def muted(self) -> bool:
        return bool(self._store.get("audio_muted", False))

    @muted.setter
    def muted(self, value: bool) -> None:
        self._store.set("audio_muted", bool(value))


class UIConfig(_Section):
    # Window + scale
    @property
    def window_width(self) -> int:
        return int(self._store.get("window_width", constants.DEFAULT_WINDOW_WIDTH))

    @window_width.setter
    def window_width(self, value: int) -> None:
        self._store.set("window_width", int(value))

    @property
    def window_height(self) -> int:
        return int(self._store.get("window_height", constants.DEFAULT_WINDOW_HEIGHT))

    @window_height.setter
    def window_height(self, value: int) -> None:
        self._store.set("window_height", int(value))

    @property
    def layout_mode(self) -> str:
        return str(self._store.get("ui_layout_mode", constants.DEFAULT_UI_LAYOUT))

    @layout_mode.setter
    def layout_mode(self, value: str) -> None:
        self._store.set("ui_layout_mode", str(value))

    @property
    def full_width_nav(self) -> bool:
        return bool(self._store.get("full_width_nav", True))

    @full_width_nav.setter
    def full_width_nav(self, value: bool) -> None:
        self._store.set("full_width_nav", bool(value))

    @property
    def global_font_scale(self) -> float:
        return float(self._store.get("global_font_scale", 1.0))

    @global_font_scale.setter
    def global_font_scale(self, value: float) -> None:
        self._store.set("global_font_scale", float(value))

    @property
    def auto_system_scaling(self) -> bool:
        return bool(self._store.get("auto_system_scaling_enabled", True))

    @auto_system_scaling.setter
    def auto_system_scaling(self, value: bool) -> None:
        self._store.set("auto_system_scaling_enabled", bool(value))

    @property
    def theme(self) -> str:
        return str(self._store.get("theme", "dark"))

    @theme.setter
    def theme(self, value: str) -> None:
        self._store.set("theme", str(value))

    # Timeline interaction
    @property
    def timeline_pan_speed_multiplier(self) -> int:
        return int(self._store.get("timeline_pan_speed_multiplier", 20))

    @timeline_pan_speed_multiplier.setter
    def timeline_pan_speed_multiplier(self, value: int) -> None:
        self._store.set("timeline_pan_speed_multiplier", int(value))

    @property
    def timeline_pan_offset_ms(self) -> float:
        return float(self._store.get("timeline_pan_offset_ms", 0.0))

    @timeline_pan_offset_ms.setter
    def timeline_pan_offset_ms(self, value: float) -> None:
        self._store.set("timeline_pan_offset_ms", float(value))

    @property
    def timeline_base_height(self) -> int:
        return int(self._store.get("timeline_base_height", ui_metrics.TIMELINE_BASE_DEFAULT_PX))

    @timeline_base_height.setter
    def timeline_base_height(self, value: int) -> None:
        self._store.set("timeline_base_height", int(value))

    # Layout — quad-block fractions
    @property
    def left_bottom_block_height_frac(self) -> float:
        return float(self._store.get("left_bottom_block_height_frac", 0.45))

    @left_bottom_block_height_frac.setter
    def left_bottom_block_height_frac(self, value: float) -> None:
        self._store.set("left_bottom_block_height_frac", float(value))

    @property
    def right_bottom_block_height_frac(self) -> float:
        return float(self._store.get("right_bottom_block_height_frac", 0.45))

    @right_bottom_block_height_frac.setter
    def right_bottom_block_height_frac(self, value: float) -> None:
        self._store.set("right_bottom_block_height_frac", float(value))

    # Window / panel visibility
    @property
    def show_control_panel_window(self) -> bool:
        return bool(self._store.get("show_control_panel_window", True))

    @show_control_panel_window.setter
    def show_control_panel_window(self, value: bool) -> None:
        self._store.set("show_control_panel_window", bool(value))

    @property
    def show_video_display_window(self) -> bool:
        return bool(self._store.get("show_video_display_window", True))

    @show_video_display_window.setter
    def show_video_display_window(self, value: bool) -> None:
        self._store.set("show_video_display_window", bool(value))

    @property
    def show_video_navigation_window(self) -> bool:
        return bool(self._store.get("show_video_navigation_window", True))

    @show_video_navigation_window.setter
    def show_video_navigation_window(self, value: bool) -> None:
        self._store.set("show_video_navigation_window", bool(value))

    @property
    def show_info_graphs_window(self) -> bool:
        return bool(self._store.get("show_info_graphs_window", True))

    @show_info_graphs_window.setter
    def show_info_graphs_window(self, value: bool) -> None:
        self._store.set("show_info_graphs_window", bool(value))

    @property
    def show_left_top_block(self) -> bool:
        return bool(self._store.get("show_left_top_block", True))

    @show_left_top_block.setter
    def show_left_top_block(self, value: bool) -> None:
        self._store.set("show_left_top_block", bool(value))

    @property
    def show_left_bottom_block(self) -> bool:
        return bool(self._store.get("show_left_bottom_block", True))

    @show_left_bottom_block.setter
    def show_left_bottom_block(self, value: bool) -> None:
        self._store.set("show_left_bottom_block", bool(value))

    @property
    def show_right_top_block(self) -> bool:
        return bool(self._store.get("show_right_top_block", True))

    @show_right_top_block.setter
    def show_right_top_block(self, value: bool) -> None:
        self._store.set("show_right_top_block", bool(value))

    @property
    def show_right_bottom_block(self) -> bool:
        return bool(self._store.get("show_right_bottom_block", True))

    @show_right_bottom_block.setter
    def show_right_bottom_block(self, value: bool) -> None:
        self._store.set("show_right_bottom_block", bool(value))

    # Timeline visibility
    @property
    def show_funscript_interactive_timeline(self) -> bool:
        return bool(self._store.get("show_funscript_interactive_timeline", True))

    @show_funscript_interactive_timeline.setter
    def show_funscript_interactive_timeline(self, value: bool) -> None:
        self._store.set("show_funscript_interactive_timeline", bool(value))

    @property
    def show_funscript_interactive_timeline2(self) -> bool:
        return bool(self._store.get("show_funscript_interactive_timeline2", False))

    @show_funscript_interactive_timeline2.setter
    def show_funscript_interactive_timeline2(self, value: bool) -> None:
        self._store.set("show_funscript_interactive_timeline2", bool(value))

    @property
    def show_funscript_timeline(self) -> bool:
        return bool(self._store.get("show_funscript_timeline", True))

    @show_funscript_timeline.setter
    def show_funscript_timeline(self, value: bool) -> None:
        self._store.set("show_funscript_timeline", bool(value))

    # Overlay / feature visibility
    @property
    def show_heatmap(self) -> bool:
        return bool(self._store.get("show_heatmap", True))

    @show_heatmap.setter
    def show_heatmap(self, value: bool) -> None:
        self._store.set("show_heatmap", bool(value))

    @property
    def show_stage2_overlay(self) -> bool:
        return bool(self._store.get("show_stage2_overlay", True))

    @show_stage2_overlay.setter
    def show_stage2_overlay(self, value: bool) -> None:
        self._store.set("show_stage2_overlay", bool(value))

    @property
    def show_simulator_3d(self) -> bool:
        return bool(self._store.get("show_simulator_3d", True))

    @show_simulator_3d.setter
    def show_simulator_3d(self, value: bool) -> None:
        self._store.set("show_simulator_3d", bool(value))

    @property
    def simulator_3d_overlay_mode(self) -> bool:
        return bool(self._store.get("simulator_3d_overlay_mode", True))

    @simulator_3d_overlay_mode.setter
    def simulator_3d_overlay_mode(self, value: bool) -> None:
        self._store.set("simulator_3d_overlay_mode", bool(value))

    @property
    def show_timeline_editor_buttons(self) -> bool:
        return bool(self._store.get("show_timeline_editor_buttons", False))

    @show_timeline_editor_buttons.setter
    def show_timeline_editor_buttons(self, value: bool) -> None:
        self._store.set("show_timeline_editor_buttons", bool(value))

    @property
    def show_advanced_options(self) -> bool:
        return bool(self._store.get("show_advanced_options", False))

    @show_advanced_options.setter
    def show_advanced_options(self, value: bool) -> None:
        self._store.set("show_advanced_options", bool(value))

    @property
    def show_toast_notifications(self) -> bool:
        return bool(self._store.get("show_toast_notifications", True))

    @show_toast_notifications.setter
    def show_toast_notifications(self, value: bool) -> None:
        self._store.set("show_toast_notifications", bool(value))

    @property
    def show_video_feed(self) -> bool:
        return bool(self._store.get("show_video_feed", True))

    @show_video_feed.setter
    def show_video_feed(self, value: bool) -> None:
        self._store.set("show_video_feed", bool(value))

    @property
    def hd_video_display(self) -> bool:
        return bool(self._store.get("hd_video_display", True))

    @hd_video_display.setter
    def hd_video_display(self, value: bool) -> None:
        self._store.set("hd_video_display", bool(value))

    @property
    def show_audio_waveform(self) -> bool:
        return bool(self._store.get("show_audio_waveform", True))

    @show_audio_waveform.setter
    def show_audio_waveform(self, value: bool) -> None:
        self._store.set("show_audio_waveform", bool(value))

    @property
    def show_bpm_overlay(self) -> bool:
        return bool(self._store.get("show_bpm_overlay", False))

    @show_bpm_overlay.setter
    def show_bpm_overlay(self, value: bool) -> None:
        self._store.set("show_bpm_overlay", bool(value))

    @property
    def show_script_gauge(self) -> bool:
        return bool(self._store.get("show_script_gauge", False))

    @show_script_gauge.setter
    def show_script_gauge(self, value: bool) -> None:
        self._store.set("show_script_gauge", bool(value))

    @property
    def show_video_controls_overlay(self) -> bool:
        return bool(self._store.get("show_video_controls_overlay", True))

    @show_video_controls_overlay.setter
    def show_video_controls_overlay(self, value: bool) -> None:
        self._store.set("show_video_controls_overlay", bool(value))

    @property
    def use_simplified_funscript_preview(self) -> bool:
        return bool(self._store.get("use_simplified_funscript_preview", False))

    @use_simplified_funscript_preview.setter
    def use_simplified_funscript_preview(self, value: bool) -> None:
        self._store.set("use_simplified_funscript_preview", bool(value))

    @property
    def timeline_zoom_factor_ms_per_px(self) -> float:
        return float(self._store.get("timeline_zoom_factor_ms_per_px", 20.0))

    @timeline_zoom_factor_ms_per_px.setter
    def timeline_zoom_factor_ms_per_px(self, value: float) -> None:
        self._store.set("timeline_zoom_factor_ms_per_px", float(value))

    @property
    def timeline_show_video_sync_line(self) -> bool:
        return bool(self._store.get("timeline_show_video_sync_line", False))

    @timeline_show_video_sync_line.setter
    def timeline_show_video_sync_line(self, value: bool) -> None:
        self._store.set("timeline_show_video_sync_line", bool(value))

    @property
    def show_3d_simulator_logo(self) -> bool:
        return bool(self._store.get("show_3d_simulator_logo", True))

    @show_3d_simulator_logo.setter
    def show_3d_simulator_logo(self, value: bool) -> None:
        self._store.set("show_3d_simulator_logo", bool(value))


class TrackingConfig(_Section):
    # Session selection
    @property
    def selected_tracker_name(self) -> str:
        return str(self._store.get("selected_tracker_name", constants.DEFAULT_TRACKER_NAME))

    @selected_tracker_name.setter
    def selected_tracker_name(self, value: str) -> None:
        self._store.set("selected_tracker_name", str(value))

    @property
    def selected_processing_speed_mode(self) -> str:
        return str(self._store.get("selected_processing_speed_mode", "REALTIME"))

    @selected_processing_speed_mode.setter
    def selected_processing_speed_mode(self, value: str) -> None:
        self._store.set("selected_processing_speed_mode", str(value))

    @property
    def slow_motion_fps(self) -> float:
        return float(self._store.get("slow_motion_fps", 10.0))

    @slow_motion_fps.setter
    def slow_motion_fps(self, value: float) -> None:
        self._store.set("slow_motion_fps", float(value))

    # Axis configuration
    @property
    def axis_mode(self) -> str:
        return str(self._store.get("tracking_axis_mode", "both"))

    @axis_mode.setter
    def axis_mode(self, value: str) -> None:
        self._store.set("tracking_axis_mode", str(value))

    @property
    def single_axis_output_target(self) -> str:
        return str(self._store.get("single_axis_output_target", "primary"))

    @single_axis_output_target.setter
    def single_axis_output_target(self, value: str) -> None:
        self._store.set("single_axis_output_target", str(value))

    @property
    def funscript_output_delay_frames(self) -> int:
        return int(self._store.get("funscript_output_delay_frames", 0))

    @funscript_output_delay_frames.setter
    def funscript_output_delay_frames(self, value: int) -> None:
        self._store.set("funscript_output_delay_frames", int(value))

    # Tracker dropdown visibility
    @property
    def show_legacy_trackers(self) -> bool:
        return bool(self._store.get("tracker_show_legacy", False))

    @show_legacy_trackers.setter
    def show_legacy_trackers(self, value: bool) -> None:
        self._store.set("tracker_show_legacy", bool(value))

    @property
    def show_experimental_trackers(self) -> bool:
        return bool(self._store.get("tracker_show_experimental", True))

    @show_experimental_trackers.setter
    def show_experimental_trackers(self, value: bool) -> None:
        self._store.set("tracker_show_experimental", bool(value))

    @property
    def show_community_trackers(self) -> bool:
        return bool(self._store.get("tracker_show_community", True))

    @show_community_trackers.setter
    def show_community_trackers(self, value: bool) -> None:
        self._store.set("tracker_show_community", bool(value))

    @property
    def show_tool_trackers(self) -> bool:
        return bool(self._store.get("tracker_show_tool", False))

    @show_tool_trackers.setter
    def show_tool_trackers(self, value: bool) -> None:
        self._store.set("tracker_show_tool", bool(value))

    # Class filtering
    @property
    def discarded_classes(self) -> list:
        val = self._store.get("discarded_tracking_classes",
                              constants.CLASSES_TO_DISCARD_BY_DEFAULT)
        return list(val) if val is not None else []

    @discarded_classes.setter
    def discarded_classes(self, value: list) -> None:
        self._store.set("discarded_tracking_classes", list(value))

    # Live tracker tuning
    @property
    def live_confidence_threshold(self) -> float:
        return float(self._store.get("live_tracker_confidence_threshold",
                                     constants.DEFAULT_TRACKER_CONFIDENCE_THRESHOLD))

    @live_confidence_threshold.setter
    def live_confidence_threshold(self, value: float) -> None:
        self._store.set("live_tracker_confidence_threshold", float(value))

    @property
    def live_sensitivity(self) -> float:
        return float(self._store.get("live_tracker_sensitivity",
                                     constants.DEFAULT_LIVE_TRACKER_SENSITIVITY))

    @live_sensitivity.setter
    def live_sensitivity(self, value: float) -> None:
        self._store.set("live_tracker_sensitivity", float(value))

    @property
    def live_base_amplification(self) -> float:
        return float(self._store.get("live_tracker_base_amplification",
                                     constants.DEFAULT_LIVE_TRACKER_BASE_AMPLIFICATION))

    @live_base_amplification.setter
    def live_base_amplification(self, value: float) -> None:
        self._store.set("live_tracker_base_amplification", float(value))

    @property
    def live_class_amp_multipliers(self) -> dict:
        val = self._store.get("live_tracker_class_amp_multipliers",
                              constants.DEFAULT_CLASS_AMP_MULTIPLIERS)
        return dict(val) if val is not None else {}

    @live_class_amp_multipliers.setter
    def live_class_amp_multipliers(self, value: dict) -> None:
        self._store.set("live_tracker_class_amp_multipliers", dict(value))

    @property
    def live_roi_padding(self) -> int:
        return int(self._store.get("live_tracker_roi_padding",
                                   constants.DEFAULT_TRACKER_ROI_PADDING))

    @live_roi_padding.setter
    def live_roi_padding(self, value: int) -> None:
        self._store.set("live_tracker_roi_padding", int(value))

    @property
    def live_roi_update_interval(self) -> int:
        return int(self._store.get("live_tracker_roi_update_interval",
                                   constants.DEFAULT_ROI_UPDATE_INTERVAL))

    @live_roi_update_interval.setter
    def live_roi_update_interval(self, value: int) -> None:
        self._store.set("live_tracker_roi_update_interval", int(value))

    @property
    def live_roi_smoothing_factor(self) -> float:
        return float(self._store.get("live_tracker_roi_smoothing_factor",
                                     constants.DEFAULT_ROI_SMOOTHING_FACTOR))

    @live_roi_smoothing_factor.setter
    def live_roi_smoothing_factor(self, value: float) -> None:
        self._store.set("live_tracker_roi_smoothing_factor", float(value))

    @property
    def live_roi_persistence_frames(self) -> int:
        return int(self._store.get("live_tracker_roi_persistence_frames",
                                   constants.DEFAULT_ROI_PERSISTENCE_FRAMES))

    @live_roi_persistence_frames.setter
    def live_roi_persistence_frames(self, value: int) -> None:
        self._store.set("live_tracker_roi_persistence_frames", int(value))

    @property
    def live_use_sparse_flow(self) -> bool:
        return bool(self._store.get("live_tracker_use_sparse_flow", False))

    @live_use_sparse_flow.setter
    def live_use_sparse_flow(self, value: bool) -> None:
        self._store.set("live_tracker_use_sparse_flow", bool(value))

    @property
    def live_dis_flow_preset(self) -> str:
        return str(self._store.get("live_tracker_dis_flow_preset",
                                   constants.DEFAULT_DIS_FLOW_PRESET))

    @live_dis_flow_preset.setter
    def live_dis_flow_preset(self, value: str) -> None:
        self._store.set("live_tracker_dis_flow_preset", str(value))

    @property
    def live_dis_finest_scale(self) -> int:
        return int(self._store.get("live_tracker_dis_finest_scale",
                                   constants.DEFAULT_DIS_FINEST_SCALE))

    @live_dis_finest_scale.setter
    def live_dis_finest_scale(self, value: int) -> None:
        self._store.set("live_tracker_dis_finest_scale", int(value))

    @property
    def live_flow_smoothing_window(self) -> int:
        return int(self._store.get("live_tracker_flow_smoothing_window",
                                   constants.DEFAULT_FLOW_HISTORY_SMOOTHING_WINDOW))

    @live_flow_smoothing_window.setter
    def live_flow_smoothing_window(self, value: int) -> None:
        self._store.set("live_tracker_flow_smoothing_window", int(value))

    # Rolling autotune (streamer integration)
    @property
    def live_rolling_autotune_enabled(self) -> bool:
        return bool(self._store.get("live_tracker_rolling_autotune_enabled", False))

    @live_rolling_autotune_enabled.setter
    def live_rolling_autotune_enabled(self, value: bool) -> None:
        self._store.set("live_tracker_rolling_autotune_enabled", bool(value))

    @property
    def live_rolling_autotune_interval_ms(self) -> int:
        return int(self._store.get("live_tracker_rolling_autotune_interval_ms", 5000))

    @live_rolling_autotune_interval_ms.setter
    def live_rolling_autotune_interval_ms(self, value: int) -> None:
        self._store.set("live_tracker_rolling_autotune_interval_ms", int(value))

    @property
    def live_rolling_autotune_window_ms(self) -> int:
        return int(self._store.get("live_tracker_rolling_autotune_window_ms", 5000))

    @live_rolling_autotune_window_ms.setter
    def live_rolling_autotune_window_ms(self, value: int) -> None:
        self._store.set("live_tracker_rolling_autotune_window_ms", int(value))

    # Oscillation detector
    @property
    def oscillation_grid_size(self) -> int:
        return int(self._store.get("oscillation_detector_grid_size", 20))

    @oscillation_grid_size.setter
    def oscillation_grid_size(self, value: int) -> None:
        self._store.set("oscillation_detector_grid_size", int(value))

    @property
    def oscillation_sensitivity(self) -> float:
        return float(self._store.get("oscillation_detector_sensitivity", 2.5))

    @oscillation_sensitivity.setter
    def oscillation_sensitivity(self, value: float) -> None:
        self._store.set("oscillation_detector_sensitivity", float(value))

    @property
    def oscillation_mode(self) -> str:
        return str(self._store.get("stage3_oscillation_detector_mode", "current"))

    @oscillation_mode.setter
    def oscillation_mode(self, value: str) -> None:
        self._store.set("stage3_oscillation_detector_mode", str(value))

    @property
    def oscillation_dynamic_amp_enabled(self) -> bool:
        return bool(self._store.get("live_oscillation_dynamic_amp_enabled", True))

    @oscillation_dynamic_amp_enabled.setter
    def oscillation_dynamic_amp_enabled(self, value: bool) -> None:
        self._store.set("live_oscillation_dynamic_amp_enabled", bool(value))

    @property
    def oscillation_amp_window_ms(self) -> int:
        return int(self._store.get("live_oscillation_amp_window_ms", 4000))

    @oscillation_amp_window_ms.setter
    def oscillation_amp_window_ms(self, value: int) -> None:
        self._store.set("live_oscillation_amp_window_ms", int(value))

    @property
    def oscillation_enable_decay(self) -> bool:
        return bool(self._store.get("oscillation_enable_decay", True))

    @oscillation_enable_decay.setter
    def oscillation_enable_decay(self, value: bool) -> None:
        self._store.set("oscillation_enable_decay", bool(value))

    @property
    def oscillation_hold_duration_ms(self) -> int:
        return int(self._store.get("oscillation_hold_duration_ms", 250))

    @oscillation_hold_duration_ms.setter
    def oscillation_hold_duration_ms(self, value: int) -> None:
        self._store.set("oscillation_hold_duration_ms", int(value))

    @property
    def oscillation_decay_factor(self) -> float:
        return float(self._store.get("oscillation_decay_factor", 0.95))

    @oscillation_decay_factor.setter
    def oscillation_decay_factor(self, value: float) -> None:
        self._store.set("oscillation_decay_factor", float(value))

    @property
    def oscillation_use_simple_amplification(self) -> bool:
        return bool(self._store.get("oscillation_use_simple_amplification", False))

    @oscillation_use_simple_amplification.setter
    def oscillation_use_simple_amplification(self, value: bool) -> None:
        self._store.set("oscillation_use_simple_amplification", bool(value))


class PerformanceConfig(_Section):
    @property
    def stage1_producers(self) -> int:
        return int(self._store.get("num_producers_stage1", constants.DEFAULT_S1_NUM_PRODUCERS))

    @stage1_producers.setter
    def stage1_producers(self, value: int) -> None:
        self._store.set("num_producers_stage1", int(value))

    @property
    def stage1_consumers(self) -> int:
        return int(self._store.get("num_consumers_stage1", constants.DEFAULT_S1_NUM_CONSUMERS))

    @stage1_consumers.setter
    def stage1_consumers(self, value: int) -> None:
        self._store.set("num_consumers_stage1", int(value))

    @property
    def stage2_of_workers(self) -> int:
        return int(self._store.get("num_workers_stage2_of", constants.DEFAULT_S2_OF_WORKERS))

    @stage2_of_workers.setter
    def stage2_of_workers(self, value: int) -> None:
        self._store.set("num_workers_stage2_of", int(value))

    @property
    def adaptive_batch_tuning_enabled(self) -> bool:
        return bool(self._store.get("adaptive_batch_tuning_enabled", True))

    @adaptive_batch_tuning_enabled.setter
    def adaptive_batch_tuning_enabled(self, value: bool) -> None:
        self._store.set("adaptive_batch_tuning_enabled", bool(value))

    @property
    def hardware_acceleration_method(self) -> str:
        return str(self._store.get("hardware_acceleration_method", "none"))

    @hardware_acceleration_method.setter
    def hardware_acceleration_method(self, value: str) -> None:
        self._store.set("hardware_acceleration_method", str(value))

    @property
    def default_secondary_axis(self) -> str:
        return str(self._store.get("default_secondary_axis", "roll"))

    @default_secondary_axis.setter
    def default_secondary_axis(self, value: str) -> None:
        self._store.set("default_secondary_axis", str(value))

    @property
    def ffmpeg_path(self) -> str:
        return str(self._store.get("ffmpeg_path", "ffmpeg"))

    @ffmpeg_path.setter
    def ffmpeg_path(self, value: str) -> None:
        self._store.set("ffmpeg_path", str(value))

    @property
    def vr_unwarp_method(self) -> str:
        return str(self._store.get("vr_unwarp_method", "v360"))

    @vr_unwarp_method.setter
    def vr_unwarp_method(self, value: str) -> None:
        self._store.set("vr_unwarp_method", str(value))

    @property
    def main_loop_normal_fps_target(self) -> int:
        return int(self._store.get("main_loop_normal_fps_target", 60))

    @main_loop_normal_fps_target.setter
    def main_loop_normal_fps_target(self, value: int) -> None:
        self._store.set("main_loop_normal_fps_target", int(value))

    @property
    def available_ffmpeg_hwaccels(self) -> list:
        val = self._store.get("available_ffmpeg_hwaccels", None)
        return list(val) if val else []

    @available_ffmpeg_hwaccels.setter
    def available_ffmpeg_hwaccels(self, value: list) -> None:
        self._store.set("available_ffmpeg_hwaccels", list(value) if value else [])


class EnergySaverConfig(_Section):
    @property
    def enabled(self) -> bool:
        return bool(self._store.get("energy_saver_enabled", True))

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._store.set("energy_saver_enabled", bool(value))

    @property
    def threshold_seconds(self) -> float:
        return float(self._store.get("energy_saver_threshold_seconds", 30.0))

    @threshold_seconds.setter
    def threshold_seconds(self, value: float) -> None:
        self._store.set("energy_saver_threshold_seconds", float(value))

    @property
    def fps(self) -> int:
        return int(self._store.get("energy_saver_fps", 1))

    @fps.setter
    def fps(self, value: int) -> None:
        self._store.set("energy_saver_fps", int(value))


class OutputConfig(_Section):
    @property
    def folder_path(self) -> str:
        return str(self._store.get("output_folder_path", constants.DEFAULT_OUTPUT_FOLDER))

    @folder_path.setter
    def folder_path(self, value: str) -> None:
        self._store.set("output_folder_path", str(value))

    @property
    def autosave_final_next_to_video(self) -> bool:
        return bool(self._store.get("autosave_final_funscript_to_video_location", True))

    @autosave_final_next_to_video.setter
    def autosave_final_next_to_video(self, value: bool) -> None:
        self._store.set("autosave_final_funscript_to_video_location", bool(value))

    @property
    def generate_roll_file(self) -> bool:
        return bool(self._store.get("generate_roll_file", True))

    @generate_roll_file.setter
    def generate_roll_file(self, value: bool) -> None:
        self._store.set("generate_roll_file", bool(value))

    @property
    def export_raw_as_funscript(self) -> bool:
        return bool(self._store.get("export_raw_as_funscript", False))

    @export_raw_as_funscript.setter
    def export_raw_as_funscript(self, value: bool) -> None:
        self._store.set("export_raw_as_funscript", bool(value))

    @property
    def batch_mode_overwrite_strategy(self) -> int:
        return int(self._store.get("batch_mode_overwrite_strategy", 0))

    @batch_mode_overwrite_strategy.setter
    def batch_mode_overwrite_strategy(self, value: int) -> None:
        self._store.set("batch_mode_overwrite_strategy", int(value))

    @property
    def metadata_verbose(self) -> bool:
        return bool(self._store.get("metadata_verbose", True))

    @metadata_verbose.setter
    def metadata_verbose(self, value: bool) -> None:
        self._store.set("metadata_verbose", bool(value))

    @property
    def performance_metadata(self) -> bool:
        return bool(self._store.get("performance_metadata", False))

    @performance_metadata.setter
    def performance_metadata(self, value: bool) -> None:
        self._store.set("performance_metadata", bool(value))

    @property
    def metadata_creator_identity(self) -> str:
        return str(self._store.get("metadata_creator_identity", ""))

    @metadata_creator_identity.setter
    def metadata_creator_identity(self, value: str) -> None:
        self._store.set("metadata_creator_identity", str(value))

    @property
    def save_preprocessed_video(self) -> bool:
        return bool(self._store.get("save_preprocessed_video", True))

    @save_preprocessed_video.setter
    def save_preprocessed_video(self, value: bool) -> None:
        self._store.set("save_preprocessed_video", bool(value))

    @property
    def retain_stage2_database(self) -> bool:
        return bool(self._store.get("retain_stage2_database", True))

    @retain_stage2_database.setter
    def retain_stage2_database(self, value: bool) -> None:
        self._store.set("retain_stage2_database", bool(value))


class ModelConfig(_Section):
    @property
    def yolo_det_path(self) -> str:
        return str(self._store.get("yolo_det_model_path", "") or "")

    @yolo_det_path.setter
    def yolo_det_path(self, value: str) -> None:
        self._store.set("yolo_det_model_path", str(value) if value else "")

    @property
    def yolo_pose_path(self) -> str:
        return str(self._store.get("yolo_pose_model_path", "") or "")

    @yolo_pose_path.setter
    def yolo_pose_path(self, value: str) -> None:
        self._store.set("yolo_pose_model_path", str(value) if value else "")


class LoggingConfig(_Section):
    @property
    def level(self) -> str:
        return str(self._store.get("logging_level", "INFO") or "INFO")

    @level.setter
    def level(self, value: str) -> None:
        self._store.set("logging_level", str(value))


class UpdaterConfig(_Section):
    @property
    def check_on_startup(self) -> bool:
        return bool(self._store.get("updater_check_on_startup", True))

    @check_on_startup.setter
    def check_on_startup(self, value: bool) -> None:
        self._store.set("updater_check_on_startup", bool(value))

    @property
    def check_periodically(self) -> bool:
        return bool(self._store.get("updater_check_periodically", True))

    @check_periodically.setter
    def check_periodically(self, value: bool) -> None:
        self._store.set("updater_check_periodically", bool(value))

    @property
    def suppress_popup(self) -> bool:
        return bool(self._store.get("updater_suppress_popup", False))

    @suppress_popup.setter
    def suppress_popup(self, value: bool) -> None:
        self._store.set("updater_suppress_popup", bool(value))


class AutosaveConfig(_Section):
    @property
    def enabled(self) -> bool:
        return bool(self._store.get("autosave_enabled", True))

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._store.set("autosave_enabled", bool(value))

    @property
    def interval_seconds(self) -> int:
        return int(self._store.get("autosave_interval_seconds", 120))

    @interval_seconds.setter
    def interval_seconds(self, value: int) -> None:
        self._store.set("autosave_interval_seconds", int(value))

    @property
    def on_exit(self) -> bool:
        return bool(self._store.get("autosave_on_exit", True))

    @on_exit.setter
    def on_exit(self, value: bool) -> None:
        self._store.set("autosave_on_exit", bool(value))


class ProxyConfig(_Section):
    @property
    def suggest_on_open(self) -> bool:
        return bool(self._store.get("video_proxy_suggest_on_open", True))

    @suggest_on_open.setter
    def suggest_on_open(self, value: bool) -> None:
        self._store.set("video_proxy_suggest_on_open", bool(value))

    @property
    def autoswitch_on_complete(self) -> bool:
        return bool(self._store.get("video_proxy_autoswitch_on_complete", True))

    @autoswitch_on_complete.setter
    def autoswitch_on_complete(self, value: bool) -> None:
        self._store.set("video_proxy_autoswitch_on_complete", bool(value))

    @property
    def min_size_gb(self) -> float:
        return float(self._store.get("video_proxy_min_size_gb", 1.5))

    @min_size_gb.setter
    def min_size_gb(self, value: float) -> None:
        self._store.set("video_proxy_min_size_gb", float(value))

    @property
    def ask_dismissed(self) -> bool:
        return bool(self._store.get("video_proxy_ask_dismissed", False))

    @ask_dismissed.setter
    def ask_dismissed(self, value: bool) -> None:
        self._store.set("video_proxy_ask_dismissed", bool(value))

    @property
    def output_mode(self) -> str:
        return str(self._store.get("video_proxy_output_mode", "next_to_source"))

    @output_mode.setter
    def output_mode(self, value: str) -> None:
        self._store.set("video_proxy_output_mode", str(value))

    @property
    def custom_folder(self) -> str:
        return str(self._store.get("video_proxy_custom_folder", "") or "")

    @custom_folder.setter
    def custom_folder(self, value: str) -> None:
        self._store.set("video_proxy_custom_folder", str(value) if value else "")


class ChapterConfig(_Section):
    @property
    def auto_save_standalone(self) -> bool:
        return bool(self._store.get("chapter_auto_save_standalone", False))

    @auto_save_standalone.setter
    def auto_save_standalone(self, value: bool) -> None:
        self._store.set("chapter_auto_save_standalone", bool(value))

    @property
    def backup_on_regenerate(self) -> bool:
        return bool(self._store.get("chapter_backup_on_regenerate", True))

    @backup_on_regenerate.setter
    def backup_on_regenerate(self, value: bool) -> None:
        self._store.set("chapter_backup_on_regenerate", bool(value))

    @property
    def skip_if_exists(self) -> bool:
        return bool(self._store.get("chapter_skip_if_exists", False))

    @skip_if_exists.setter
    def skip_if_exists(self, value: bool) -> None:
        self._store.set("chapter_skip_if_exists", bool(value))

    @property
    def overwrite_on_analysis(self) -> bool:
        return bool(self._store.get("overwrite_chapters_on_analysis", False))

    @overwrite_on_analysis.setter
    def overwrite_on_analysis(self, value: bool) -> None:
        self._store.set("overwrite_chapters_on_analysis", bool(value))

    @property
    def auto_processing_use_profiles(self) -> bool:
        return bool(self._store.get("auto_processing_use_chapter_profiles", True))

    @auto_processing_use_profiles.setter
    def auto_processing_use_profiles(self, value: bool) -> None:
        self._store.set("auto_processing_use_chapter_profiles", bool(value))

    @property
    def type_recent_max(self) -> int:
        return int(self._store.get("chapter_type_recent_max", 5))

    @type_recent_max.setter
    def type_recent_max(self, value: int) -> None:
        self._store.set("chapter_type_recent_max", int(value))


class FunscriptConfig(_Section):
    @property
    def point_simplification_enabled(self) -> bool:
        return bool(self._store.get("funscript_point_simplification_enabled", True))

    @point_simplification_enabled.setter
    def point_simplification_enabled(self, value: bool) -> None:
        self._store.set("funscript_point_simplification_enabled", bool(value))

    @property
    def point_simplification_tolerance(self) -> int:
        return int(self._store.get("funscript_point_simplification_tolerance", 2))

    @point_simplification_tolerance.setter
    def point_simplification_tolerance(self, value: int) -> None:
        self._store.set("funscript_point_simplification_tolerance", int(value))

    @property
    def enable_signal_enhancement(self) -> bool:
        return bool(self._store.get("enable_signal_enhancement", True))

    @enable_signal_enhancement.setter
    def enable_signal_enhancement(self, value: bool) -> None:
        self._store.set("enable_signal_enhancement", bool(value))

    @property
    def signal_enhancement_motion_low(self) -> float:
        return float(self._store.get("signal_enhancement_motion_threshold_low", 12.0))

    @signal_enhancement_motion_low.setter
    def signal_enhancement_motion_low(self, value: float) -> None:
        self._store.set("signal_enhancement_motion_threshold_low", float(value))

    @property
    def signal_enhancement_motion_high(self) -> float:
        return float(self._store.get("signal_enhancement_motion_threshold_high", 30.0))

    @signal_enhancement_motion_high.setter
    def signal_enhancement_motion_high(self, value: float) -> None:
        self._store.set("signal_enhancement_motion_threshold_high", float(value))

    @property
    def signal_enhancement_change_threshold(self) -> int:
        return int(self._store.get("signal_enhancement_signal_change_threshold", 6))

    @signal_enhancement_change_threshold.setter
    def signal_enhancement_change_threshold(self, value: int) -> None:
        self._store.set("signal_enhancement_signal_change_threshold", int(value))

    @property
    def signal_enhancement_strength(self) -> float:
        return float(self._store.get("signal_enhancement_strength", 0.25))

    @signal_enhancement_strength.setter
    def signal_enhancement_strength(self, value: float) -> None:
        self._store.set("signal_enhancement_strength", float(value))

    @property
    def enable_auto_post_processing(self) -> bool:
        return bool(self._store.get("enable_auto_post_processing", False))

    @enable_auto_post_processing.setter
    def enable_auto_post_processing(self, value: bool) -> None:
        self._store.set("enable_auto_post_processing", bool(value))

    @property
    def auto_post_proc_final_rdp_enabled(self) -> bool:
        return bool(self._store.get("auto_post_proc_final_rdp_enabled", False))

    @auto_post_proc_final_rdp_enabled.setter
    def auto_post_proc_final_rdp_enabled(self, value: bool) -> None:
        self._store.set("auto_post_proc_final_rdp_enabled", bool(value))

    @property
    def auto_post_proc_final_rdp_epsilon(self) -> float:
        return float(self._store.get("auto_post_proc_final_rdp_epsilon", 10.0))

    @auto_post_proc_final_rdp_epsilon.setter
    def auto_post_proc_final_rdp_epsilon(self, value: float) -> None:
        self._store.set("auto_post_proc_final_rdp_epsilon", float(value))

    # Timeline 1 Ultimate Autotune stages (funscript cleanup pipeline)
    @property
    def t1_presmoothing_enabled(self) -> bool:
        return bool(self._store.get("timeline1_ultimate_presmoothing_enabled", True))

    @t1_presmoothing_enabled.setter
    def t1_presmoothing_enabled(self, value: bool) -> None:
        self._store.set("timeline1_ultimate_presmoothing_enabled", bool(value))

    @property
    def t1_presmoothing_max_window(self) -> int:
        return int(self._store.get("timeline1_ultimate_presmoothing_max_window", 15))

    @t1_presmoothing_max_window.setter
    def t1_presmoothing_max_window(self, value: int) -> None:
        self._store.set("timeline1_ultimate_presmoothing_max_window", int(value))

    @property
    def t1_peaks_enabled(self) -> bool:
        return bool(self._store.get("timeline1_ultimate_peaks_enabled", True))

    @t1_peaks_enabled.setter
    def t1_peaks_enabled(self, value: bool) -> None:
        self._store.set("timeline1_ultimate_peaks_enabled", bool(value))

    @property
    def t1_peaks_prominence(self) -> int:
        return int(self._store.get("timeline1_ultimate_peaks_prominence", 10))

    @t1_peaks_prominence.setter
    def t1_peaks_prominence(self, value: int) -> None:
        self._store.set("timeline1_ultimate_peaks_prominence", int(value))

    @property
    def t1_recovery_enabled(self) -> bool:
        return bool(self._store.get("timeline1_ultimate_recovery_enabled", True))

    @t1_recovery_enabled.setter
    def t1_recovery_enabled(self, value: bool) -> None:
        self._store.set("timeline1_ultimate_recovery_enabled", bool(value))

    @property
    def t1_recovery_threshold(self) -> float:
        return float(self._store.get("timeline1_ultimate_recovery_threshold", 1.8))

    @t1_recovery_threshold.setter
    def t1_recovery_threshold(self, value: float) -> None:
        self._store.set("timeline1_ultimate_recovery_threshold", float(value))

    @property
    def t1_normalization_enabled(self) -> bool:
        return bool(self._store.get("timeline1_ultimate_normalization_enabled", True))

    @t1_normalization_enabled.setter
    def t1_normalization_enabled(self, value: bool) -> None:
        self._store.set("timeline1_ultimate_normalization_enabled", bool(value))

    @property
    def t1_speed_limit_enabled(self) -> bool:
        return bool(self._store.get("timeline1_ultimate_speed_limit_enabled", True))

    @t1_speed_limit_enabled.setter
    def t1_speed_limit_enabled(self, value: bool) -> None:
        self._store.set("timeline1_ultimate_speed_limit_enabled", bool(value))

    @property
    def t1_speed_threshold(self) -> float:
        return float(self._store.get("timeline1_ultimate_speed_threshold", 500.0))

    @t1_speed_threshold.setter
    def t1_speed_threshold(self, value: float) -> None:
        self._store.set("timeline1_ultimate_speed_threshold", float(value))

    @property
    def export_format(self) -> str:
        return str(self._store.get("funscript_export_format", "separate"))

    @export_format.setter
    def export_format(self, value: str) -> None:
        self._store.set("funscript_export_format", str(value))

    @property
    def enable_enhanced_preview(self) -> bool:
        return bool(self._store.get("enable_enhanced_funscript_preview", False))

    @enable_enhanced_preview.setter
    def enable_enhanced_preview(self, value: bool) -> None:
        self._store.set("enable_enhanced_funscript_preview", bool(value))

    @property
    def speed_limit_threshold(self) -> float:
        return float(self._store.get("speed_limit_threshold", 400.0))

    @speed_limit_threshold.setter
    def speed_limit_threshold(self, value: float) -> None:
        self._store.set("speed_limit_threshold", float(value))

    @property
    def auto_apply_post_processing(self) -> bool:
        return bool(self._store.get("auto_apply_post_processing", False))

    @auto_apply_post_processing.setter
    def auto_apply_post_processing(self, value: bool) -> None:
        self._store.set("auto_apply_post_processing", bool(value))

    @property
    def interactive_refinement_mode_enabled(self) -> bool:
        return bool(self._store.get("interactive_refinement_mode_enabled", False))

    @interactive_refinement_mode_enabled.setter
    def interactive_refinement_mode_enabled(self, value: bool) -> None:
        self._store.set("interactive_refinement_mode_enabled", bool(value))


class HeatmapConfig(_Section):
    @property
    def max_speed(self) -> float:
        return float(self._store.get("heatmap_max_speed", 1000.0))

    @max_speed.setter
    def max_speed(self, value: float) -> None:
        self._store.set("heatmap_max_speed", float(value))

    @property
    def highlight_overspeed(self) -> bool:
        return bool(self._store.get("heatmap_highlight_overspeed", False))

    @highlight_overspeed.setter
    def highlight_overspeed(self, value: bool) -> None:
        self._store.set("heatmap_highlight_overspeed", bool(value))


class NavigationConfig(_Section):
    @property
    def seek_n_frames(self) -> int:
        return int(self._store.get("seek_n_frames", 10))

    @seek_n_frames.setter
    def seek_n_frames(self, value: int) -> None:
        self._store.set("seek_n_frames", int(value))

    @property
    def nav_cache_bytes(self) -> int:
        return int(self._store.get("nav_cache_bytes", 0))

    @nav_cache_bytes.setter
    def nav_cache_bytes(self, value: int) -> None:
        self._store.set("nav_cache_bytes", int(value))

    @property
    def debug_logging(self) -> bool:
        return bool(self._store.get("debug_nav_logging", False))

    @debug_logging.setter
    def debug_logging(self, value: bool) -> None:
        self._store.set("debug_nav_logging", bool(value))


class MPVConfig(_Section):
    @property
    def render_backend(self) -> str:
        return str(self._store.get("mpv_render_backend", "gl"))

    @render_backend.setter
    def render_backend(self, value: str) -> None:
        self._store.set("mpv_render_backend", str(value))

    @property
    def hwdec_override(self) -> Any:
        """Returns the user-configured hwdec value, or None if not set.

        Callers should pick a platform-appropriate default when None.
        """
        return self._store.get("mpv_hwdec", None)

    @hwdec_override.setter
    def hwdec_override(self, value: Any) -> None:
        self._store.set("mpv_hwdec", value)


class WSAPIConfig(_Section):
    @property
    def enabled(self) -> bool:
        return bool(self._store.get("ws_api_enabled", False))

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._store.set("ws_api_enabled", bool(value))

    @property
    def port(self) -> int:
        return int(self._store.get("ws_api_port", 8760))

    @port.setter
    def port(self, value: int) -> None:
        self._store.set("ws_api_port", int(value))


class KeyboardLayoutConfig(_Section):
    @property
    def saved_layout(self) -> Any:
        return self._store.get("keyboard_layout", None)

    @saved_layout.setter
    def saved_layout(self, value: Any) -> None:
        self._store.set("keyboard_layout", value)


class FirstRunConfig(_Section):
    @property
    def complete(self) -> bool:
        return bool(self._store.get("is_first_run_complete", False))

    @complete.setter
    def complete(self, value: bool) -> None:
        self._store.set("is_first_run_complete", bool(value))


class Stage3Config(_Section):
    """Stage 3 (offline analysis) tracker parameters.

    These are duplicates of the live-tracker tuning with a `tracker_` prefix
    — they exist because Stage 3 can be re-run independently of live
    tuning. Defaults fall through to constants when keys are missing.
    """
    @property
    def confidence_threshold(self) -> float:
        return float(self._store.get("tracker_confidence_threshold",
                                     constants.DEFAULT_TRACKER_CONFIDENCE_THRESHOLD))

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        self._store.set("tracker_confidence_threshold", float(value))

    @property
    def roi_padding(self) -> int:
        return int(self._store.get("tracker_roi_padding", constants.DEFAULT_TRACKER_ROI_PADDING))

    @roi_padding.setter
    def roi_padding(self, value: int) -> None:
        self._store.set("tracker_roi_padding", int(value))

    @property
    def roi_update_interval(self) -> int:
        return int(self._store.get("s3_roi_update_interval", constants.DEFAULT_ROI_UPDATE_INTERVAL))

    @roi_update_interval.setter
    def roi_update_interval(self, value: int) -> None:
        self._store.set("s3_roi_update_interval", int(value))

    @property
    def roi_smoothing_factor(self) -> float:
        return float(self._store.get("tracker_roi_smoothing_factor",
                                     constants.DEFAULT_ROI_SMOOTHING_FACTOR))

    @roi_smoothing_factor.setter
    def roi_smoothing_factor(self, value: float) -> None:
        self._store.set("tracker_roi_smoothing_factor", float(value))

    @property
    def dis_flow_preset(self) -> str:
        return str(self._store.get("tracker_dis_flow_preset", "ULTRAFAST"))

    @dis_flow_preset.setter
    def dis_flow_preset(self, value: str) -> None:
        self._store.set("tracker_dis_flow_preset", str(value))

    @property
    def flow_history_window_smooth(self) -> int:
        return int(self._store.get("tracker_flow_history_window_smooth", 3))

    @flow_history_window_smooth.setter
    def flow_history_window_smooth(self, value: int) -> None:
        self._store.set("tracker_flow_history_window_smooth", int(value))

    @property
    def adaptive_flow_scale(self) -> bool:
        return bool(self._store.get("tracker_adaptive_flow_scale", True))

    @adaptive_flow_scale.setter
    def adaptive_flow_scale(self, value: bool) -> None:
        self._store.set("tracker_adaptive_flow_scale", bool(value))

    @property
    def use_sparse_flow(self) -> bool:
        return bool(self._store.get("tracker_use_sparse_flow", False))

    @use_sparse_flow.setter
    def use_sparse_flow(self, value: bool) -> None:
        self._store.set("tracker_use_sparse_flow", bool(value))

    @property
    def base_amplification(self) -> float:
        return float(self._store.get("tracker_base_amplification",
                                     constants.DEFAULT_LIVE_TRACKER_BASE_AMPLIFICATION))

    @base_amplification.setter
    def base_amplification(self, value: float) -> None:
        self._store.set("tracker_base_amplification", float(value))

    @property
    def class_specific_multipliers(self) -> dict:
        val = self._store.get("tracker_class_specific_multipliers",
                              constants.DEFAULT_CLASS_AMP_MULTIPLIERS)
        return dict(val) if val is not None else {}

    @class_specific_multipliers.setter
    def class_specific_multipliers(self, value: dict) -> None:
        self._store.set("tracker_class_specific_multipliers", dict(value))

    @property
    def y_offset(self) -> int:
        return int(self._store.get("tracker_y_offset", constants.DEFAULT_LIVE_TRACKER_Y_OFFSET))

    @y_offset.setter
    def y_offset(self, value: int) -> None:
        self._store.set("tracker_y_offset", int(value))

    @property
    def x_offset(self) -> int:
        return int(self._store.get("tracker_x_offset", constants.DEFAULT_LIVE_TRACKER_X_OFFSET))

    @x_offset.setter
    def x_offset(self, value: int) -> None:
        self._store.set("tracker_x_offset", int(value))

    @property
    def sensitivity(self) -> float:
        return float(self._store.get("tracker_sensitivity",
                                     constants.DEFAULT_LIVE_TRACKER_SENSITIVITY))

    @sensitivity.setter
    def sensitivity(self, value: float) -> None:
        self._store.set("tracker_sensitivity", float(value))

    @property
    def num_warmup_frames(self) -> int:
        return int(self._store.get("s3_num_warmup_frames", 10))

    @num_warmup_frames.setter
    def num_warmup_frames(self, value: int) -> None:
        self._store.set("s3_num_warmup_frames", int(value))

    @property
    def roi_narrow_factor_hjbj(self) -> float:
        return float(self._store.get("roi_narrow_factor_hjbj",
                                     constants.DEFAULT_ROI_NARROW_FACTOR_HJBJ))

    @roi_narrow_factor_hjbj.setter
    def roi_narrow_factor_hjbj(self, value: float) -> None:
        self._store.set("roi_narrow_factor_hjbj", float(value))

    @property
    def min_roi_dim_hjbj(self) -> int:
        return int(self._store.get("min_roi_dim_hjbj", constants.DEFAULT_MIN_ROI_DIM_HJBJ))

    @min_roi_dim_hjbj.setter
    def min_roi_dim_hjbj(self, value: int) -> None:
        self._store.set("min_roi_dim_hjbj", int(value))

    @property
    def show_roi_debug(self) -> bool:
        return bool(self._store.get("s3_show_roi_debug", False))

    @show_roi_debug.setter
    def show_roi_debug(self, value: bool) -> None:
        self._store.set("s3_show_roi_debug", bool(value))

    @property
    def chunk_size(self) -> int:
        return int(self._store.get("s3_chunk_size", 1000))

    @chunk_size.setter
    def chunk_size(self, value: int) -> None:
        self._store.set("s3_chunk_size", int(value))

    @property
    def overlap_size(self) -> int:
        return int(self._store.get("s3_overlap_size", 30))

    @overlap_size.setter
    def overlap_size(self, value: int) -> None:
        self._store.set("s3_overlap_size", int(value))

    @property
    def debug_prints_stage2(self) -> bool:
        return bool(self._store.get("debug_prints_stage2", False))

    @debug_prints_stage2.setter
    def debug_prints_stage2(self, value: bool) -> None:
        self._store.set("debug_prints_stage2", bool(value))


class ProjectConfig(_Section):
    @property
    def recent_projects(self) -> list:
        val = self._store.get("recent_projects", [])
        return list(val) if val is not None else []

    @recent_projects.setter
    def recent_projects(self, value: list) -> None:
        self._store.set("recent_projects", list(value))

    @property
    def last_opened_path(self) -> Any:
        return self._store.get("last_opened_project_path", "")

    @last_opened_path.setter
    def last_opened_path(self, value: Any) -> None:
        self._store.set("last_opened_project_path", value)


class BatchConfig(_Section):
    @property
    def watch_path(self) -> str:
        return str(self._store.get("batch_watch_path", "") or "")

    @watch_path.setter
    def watch_path(self, value: str) -> None:
        self._store.set("batch_watch_path", str(value) if value else "")

    @property
    def watch_recursive(self) -> bool:
        return bool(self._store.get("batch_watch_recursive", False))

    @watch_recursive.setter
    def watch_recursive(self, value: bool) -> None:
        self._store.set("batch_watch_recursive", bool(value))


class PluginPipelineConfig(_Section):
    @property
    def presets(self) -> dict:
        val = self._store.get("plugin_pipeline_presets", {})
        return dict(val) if val is not None else {}

    @presets.setter
    def presets(self, value: dict) -> None:
        self._store.set("plugin_pipeline_presets", dict(value))

    @property
    def auto_assignments(self) -> dict:
        val = self._store.get("auto_pipeline_assignments", {})
        return dict(val) if val is not None else {}

    @auto_assignments.setter
    def auto_assignments(self, value: dict) -> None:
        self._store.set("auto_pipeline_assignments", dict(value))


class VRDisplayConfig(_Section):
    @property
    def mode(self) -> str:
        return str(self._store.get("vr_display_mode", "shader_dewarp"))

    @mode.setter
    def mode(self, value: str) -> None:
        self._store.set("vr_display_mode", str(value))

    @property
    def panel_selection(self) -> str:
        return str(self._store.get("vr_panel_selection", "left"))

    @panel_selection.setter
    def panel_selection(self, value: str) -> None:
        self._store.set("vr_panel_selection", str(value))

    @property
    def filter_stage2(self) -> bool:
        return bool(self._store.get("vr_filter_stage2", True))

    @filter_stage2.setter
    def filter_stage2(self, value: bool) -> None:
        self._store.set("vr_filter_stage2", bool(value))

    @property
    def mode_enabled(self) -> bool:
        return bool(self._store.get("vr_mode_enabled", False))

    @mode_enabled.setter
    def mode_enabled(self, value: bool) -> None:
        self._store.set("vr_mode_enabled", bool(value))

    @property
    def crop_panel(self) -> bool:
        return bool(self._store.get("vr_crop_panel", False))

    @crop_panel.setter
    def crop_panel(self, value: bool) -> None:
        self._store.set("vr_crop_panel", bool(value))

    @property
    def display_aspect(self) -> float:
        return float(self._store.get("vr_display_aspect", 1.0))

    @display_aspect.setter
    def display_aspect(self, value: float) -> None:
        self._store.set("vr_display_aspect", float(value))

    @property
    def quality_mode(self) -> str:
        return str(self._store.get("vr_quality_mode", "auto"))

    @quality_mode.setter
    def quality_mode(self, value: str) -> None:
        self._store.set("vr_quality_mode", str(value))

    @property
    def shader_sg_scale(self) -> float:
        return float(self._store.get("vr_shader_sg_scale", 1.840))

    @shader_sg_scale.setter
    def shader_sg_scale(self, value: float) -> None:
        self._store.set("vr_shader_sg_scale", float(value))

    @property
    def shader_lock_to_tracker(self) -> bool:
        return bool(self._store.get("vr_shader_lock_to_tracker", False))

    @shader_lock_to_tracker.setter
    def shader_lock_to_tracker(self, value: bool) -> None:
        self._store.set("vr_shader_lock_to_tracker", bool(value))

    @property
    def shader_supersample(self) -> bool:
        return bool(self._store.get("vr_shader_supersample", True))

    @shader_supersample.setter
    def shader_supersample(self, value: bool) -> None:
        self._store.set("vr_shader_supersample", bool(value))


class DeviceControlConfig(_Section):
    @property
    def enabled(self) -> bool:
        return bool(self._store.get("device_control_enabled", True))

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._store.set("device_control_enabled", bool(value))

    @property
    def buttplug_server_address(self) -> str:
        return str(self._store.get("buttplug_server_address", "localhost"))

    @buttplug_server_address.setter
    def buttplug_server_address(self, value: str) -> None:
        self._store.set("buttplug_server_address", str(value))

    @property
    def buttplug_server_port(self) -> int:
        return int(self._store.get("buttplug_server_port", 12345))

    @buttplug_server_port.setter
    def buttplug_server_port(self, value: int) -> None:
        self._store.set("buttplug_server_port", int(value))

    @property
    def buttplug_auto_connect(self) -> bool:
        return bool(self._store.get("buttplug_auto_connect", False))

    @buttplug_auto_connect.setter
    def buttplug_auto_connect(self, value: bool) -> None:
        self._store.set("buttplug_auto_connect", bool(value))

    @property
    def preferred_backend(self) -> str:
        return str(self._store.get("device_control_preferred_backend", "buttplug"))

    @preferred_backend.setter
    def preferred_backend(self, value: str) -> None:
        self._store.set("device_control_preferred_backend", str(value))

    @property
    def last_connected_device_type(self) -> str:
        return str(self._store.get("device_control_last_connected_device_type", "") or "")

    @last_connected_device_type.setter
    def last_connected_device_type(self, value: str) -> None:
        self._store.set("device_control_last_connected_device_type", str(value) if value else "")

    @property
    def max_rate_hz(self) -> float:
        return float(self._store.get("device_control_max_rate_hz", 20.0))

    @max_rate_hz.setter
    def max_rate_hz(self, value: float) -> None:
        self._store.set("device_control_max_rate_hz", float(value))

    @property
    def selected_devices(self) -> list:
        val = self._store.get("device_control_selected_devices", [])
        return list(val) if val is not None else []

    @selected_devices.setter
    def selected_devices(self, value: list) -> None:
        self._store.set("device_control_selected_devices", list(value))

    @property
    def log_commands(self) -> bool:
        return bool(self._store.get("device_control_log_commands", False))

    @log_commands.setter
    def log_commands(self, value: bool) -> None:
        self._store.set("device_control_log_commands", bool(value))


class RecordingConfig(_Section):
    @property
    def gamepad_center_mode(self) -> bool:
        """Stick mapping mode. True: rest=50, full travel -> 0..100 (sign-preserving).
        False: rest=0, deflection magnitude -> 0..100 (direction-agnostic)."""
        return bool(self._store.get("recording_gamepad_center_mode", True))

    @gamepad_center_mode.setter
    def gamepad_center_mode(self, value: bool) -> None:
        self._store.set("recording_gamepad_center_mode", bool(value))


class XBVRConfig(_Section):
    @property
    def host(self) -> str:
        return str(self._store.get("xbvr_host", "localhost"))

    @host.setter
    def host(self, value: str) -> None:
        self._store.set("xbvr_host", str(value))

    @property
    def port(self) -> int:
        return int(self._store.get("xbvr_port", 9999))

    @port.setter
    def port(self, value: int) -> None:
        self._store.set("xbvr_port", int(value))

    @property
    def enabled(self) -> bool:
        return bool(self._store.get("xbvr_enabled", True))

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._store.set("xbvr_enabled", bool(value))


class AppConfig:
    def __init__(self, store: _Store) -> None:
        self.audio = AudioConfig(store)
        self.ui = UIConfig(store)
        self.tracking = TrackingConfig(store)
        self.performance = PerformanceConfig(store)
        self.energy_saver = EnergySaverConfig(store)
        self.output = OutputConfig(store)
        self.models = ModelConfig(store)
        self.logging = LoggingConfig(store)
        self.updater = UpdaterConfig(store)
        self.autosave = AutosaveConfig(store)
        self.proxy = ProxyConfig(store)
        self.chapter = ChapterConfig(store)
        self.funscript = FunscriptConfig(store)
        self.device_control = DeviceControlConfig(store)
        self.recording = RecordingConfig(store)
        self.xbvr = XBVRConfig(store)
        self.project = ProjectConfig(store)
        self.batch = BatchConfig(store)
        self.plugin_pipeline = PluginPipelineConfig(store)
        self.vr_display = VRDisplayConfig(store)
        self.stage3 = Stage3Config(store)
        self.heatmap = HeatmapConfig(store)
        self.navigation = NavigationConfig(store)
        self.mpv = MPVConfig(store)
        self.ws_api = WSAPIConfig(store)
        self.first_run = FirstRunConfig(store)
        self.keyboard_layout = KeyboardLayoutConfig(store)
