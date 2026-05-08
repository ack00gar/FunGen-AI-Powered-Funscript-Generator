"""Developer performance panels mixin for InfoGraphsUI (behind Advanced Options)."""
import imgui
import time
from application.gui_components.ig_panels._performance import render_graph
from config.constants_colors import CurrentTheme


class DeveloperPerfMixin:

    def _render_content_pipeline_timing(self):
        """Render video processing pipeline performance metrics."""
        processor = self.app.processor
        if not processor or not processor.is_video_open():
            imgui.text_disabled("No video loaded")
            return

        imgui.text_colored("Video Processing Pipeline", *CurrentTheme.BLUE_LIGHT)
        imgui.separator()
        imgui.spacing()

        # Active backends
        imgui.columns(2, "pipeline_info", border=False)
        imgui.set_column_width(0, 180 * imgui.get_io().font_global_scale)

        imgui.text("Frame Source:")
        imgui.next_column()
        fs = getattr(processor, 'frame_source', None)
        fs_label = type(fs).__name__ if fs is not None else "(none)"
        imgui.text_colored(fs_label, *CurrentTheme.REFERENCE_OVERLAY)
        imgui.next_column()

        imgui.text("Display Backend:")
        imgui.next_column()
        gui = getattr(self.app, 'gui_instance', None)
        mpv_disp = getattr(gui, 'mpv_display', None) if gui else None
        if mpv_disp is not None and getattr(mpv_disp, 'is_loaded', False):
            disp_label, disp_color = "libmpv (render API)", CurrentTheme.GREEN
        else:
            disp_label, disp_color = "CPU upload (fallback)", CurrentTheme.YELLOW
        imgui.text_colored(disp_label, *disp_color)
        imgui.next_column()

        imgui.text("Unwarp Method:")
        imgui.next_column()
        # The frame_source applies v360 inside ffmpeg's -vf chain for VR
        # (so the tracker's numpy frames are already rectilinear). The
        # libmpv display path receives the same flat frames the ffmpeg
        # subprocess produced; mpv itself does no v360.
        if processor.is_vr_active_or_potential():
            method_text = "CPU (ffmpeg -vf v360)"
            method_color = CurrentTheme.ORANGE
        else:
            method_text = "N/A (2D video)"
            method_color = CurrentTheme.DESCRIPTION_TEXT
        imgui.text_colored(method_text, *method_color)
        imgui.next_column()

        # YOLO Model
        imgui.text("YOLO Model:")
        imgui.next_column()
        if hasattr(processor, 'yolo_processor') and processor.yolo_processor:
            model_name = getattr(processor.yolo_processor, 'model_name', 'unknown')
            imgui.text(model_name)
        else:
            imgui.text_disabled("Not loaded")
        imgui.next_column()

        # Live decode FPS (most useful in MAX_SPEED mode where wall-clock
        # pacing is off and the user has no other indicator of throughput).
        imgui.text("Decode FPS:")
        imgui.next_column()
        actual_fps = float(getattr(processor, 'actual_fps', 0.0) or 0.0)
        if actual_fps > 0:
            target = float(getattr(processor, 'fps', 0.0) or 0.0)
            ratio = (actual_fps / target) if target > 0 else 0.0
            if ratio >= 1.0:
                fps_color = CurrentTheme.GREEN
            elif ratio >= 0.5:
                fps_color = CurrentTheme.YELLOW
            else:
                fps_color = CurrentTheme.RED_LIGHT
            ratio_str = f" ({ratio:.1f}x source)" if target > 0 else ""
            imgui.text_colored(f"{actual_fps:.1f}{ratio_str}", *fps_color)
        else:
            imgui.text_disabled("idle")
        imgui.next_column()

        imgui.columns(1)
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Pipeline timing metrics
        imgui.text_colored("Component Timings:", *CurrentTheme.BLUE_LIGHT)
        imgui.spacing()

        available_width = imgui.get_content_region_available_width()

        # Get timing data from processor (if available)
        decode_time = getattr(processor, '_last_decode_time_ms', 0.0)
        unwarp_time = getattr(processor, '_last_unwarp_time_ms', 0.0)
        yolo_time = getattr(processor, '_last_yolo_time_ms', 0.0)
        flow_time = getattr(processor, '_last_flow_time_ms', 0.0)
        total_time = decode_time + unwarp_time + yolo_time + flow_time

        def render_timing_bar(label, time_ms, color):
            """Render a timing bar with label and value."""
            imgui.text(f"{label}:")
            imgui.same_line(position=180 * imgui.get_io().font_global_scale)

            if time_ms > 0:
                imgui.text_colored(f"{time_ms:.2f}ms", *color)
                imgui.same_line()
                # Progress bar showing relative contribution
                if total_time > 0:
                    percentage = (time_ms / total_time)
                    imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *color)
                    imgui.progress_bar(percentage, size=(150, 0), overlay=f"{percentage*100:.1f}%")
                    imgui.pop_style_color()
            else:
                imgui.text_disabled("N/A")

        # FFmpeg subprocess decode (v360 filter baked in when VR is active)
        render_timing_bar("FFmpeg Decode", decode_time, CurrentTheme.REFERENCE_OVERLAY)

        # v360 unwarp runs as part of the ffmpeg -vf chain and is rolled
        # into decode time. Show an explanatory label instead of N/A.
        if processor.is_vr_active_or_potential():
            imgui.text("CPU v360 Unwarp:")
            imgui.same_line(position=180 * imgui.get_io().font_global_scale)
            if unwarp_time > 0:
                imgui.text_colored(f"{unwarp_time:.2f}ms", *CurrentTheme.ORANGE)
            else:
                imgui.text_disabled("(baked into decode)")

        # YOLO Inference
        render_timing_bar("YOLO Inference", yolo_time, CurrentTheme.PURPLE)

        # Optical flow (DIS) - populated by offline trackers via flow_ms
        render_timing_bar("DIS Flow", flow_time, CurrentTheme.GREEN_LIGHT)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Total and FPS
        imgui.text_colored("Pipeline Summary:", *CurrentTheme.BLUE_LIGHT)
        imgui.spacing()

        imgui.columns(2, "pipeline_summary", border=False)
        imgui.set_column_width(0, 180 * imgui.get_io().font_global_scale)

        imgui.text("Total Time:")
        imgui.next_column()
        if total_time > 0:
            if total_time < 16.67:
                total_color = CurrentTheme.GREEN  # Green (>60 FPS)
            elif total_time < 33.33:
                total_color = CurrentTheme.YELLOW  # Yellow (30-60 FPS)
            else:
                total_color = CurrentTheme.RED  # Red (<30 FPS)
            imgui.text_colored(f"{total_time:.2f}ms", *total_color)
        else:
            imgui.text_disabled("N/A")
        imgui.next_column()

        imgui.text("Est. Pipeline FPS:")
        imgui.next_column()
        if total_time > 0:
            fps = 1000.0 / total_time
            if fps >= 60:
                fps_color = CurrentTheme.GREEN  # Green
            elif fps >= 30:
                fps_color = CurrentTheme.YELLOW  # Yellow
            else:
                fps_color = CurrentTheme.RED  # Red
            imgui.text_colored(f"{fps:.1f} FPS", *fps_color)
        else:
            imgui.text_disabled("N/A")
        imgui.next_column()

        imgui.columns(1)
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Nav debug logging toggle.
        current = self.app.app_settings.config.navigation.debug_logging
        changed, new_val = imgui.checkbox("Debug nav logging", current)
        if changed:
            self.app.app_settings.config.navigation.debug_logging = bool(new_val)
            try:
                self.app.save_app_settings()
            except Exception:
                pass
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Emit a NAV log line on every arrow/scrub tick with\n"
                "action, from/to frame, path (cache/source/seek), and\n"
                "duration. Use when diagnosing arrow-nav glitches.")

        imgui.spacing()

        # Performance recommendations
        if total_time > 0:
            imgui.text_disabled("Tip: Check individual component times to identify bottlenecks")

        # Live pipeline metrics: seek timings, prefetcher work, loop state.
        # Populated in real time by VideoProcessor / FramePrefetcher hooks.
        self._render_pipeline_live_metrics(processor)

    def _render_pipeline_live_metrics(self, processor):
        metrics = getattr(processor, "metrics", None)
        if metrics is None:
            return
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        imgui.text_colored("Live Diagnostics:", *CurrentTheme.BLUE_LIGHT)
        imgui.spacing()

        # Trace recorder controls.
        trace = getattr(processor, "trace", None)
        if trace is not None:
            recording = trace.is_enabled()
            label = "[REC] Stop" if recording else "Start recording"
            if imgui.button(label):
                if recording:
                    trace.stop_recording()
                else:
                    trace.start_recording()
            imgui.same_line()
            out = trace.out_path() or "(unset)"
            imgui.text_disabled(out)
            imgui.spacing()
            self._render_trace_summary(trace)
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

        # Loop state (where the system is right now).
        age_s = metrics.loop_state_age_s()
        age_str = f"{age_s:.1f}s ago" if age_s < 60 else "stale"
        state = "tracker" if metrics.loop_tracker_needs else (
            "mpv" if metrics.loop_mpv_drives else "idle")
        src_state = "paused" if metrics.loop_src_paused else "running"
        imgui.text(f"Loop: {state}  |  ffmpeg src: {src_state}  ({age_str})")

        # Recent seeks: total / ffmpeg / mpv breakdown.
        seeks = metrics.recent_seeks()
        imgui.spacing()
        imgui.text("Recent seeks (newest first):")
        if not seeks:
            imgui.text_disabled("  none yet")
        else:
            now_t = time.monotonic()
            for ev in reversed(seeks[-6:]):
                age = now_t - ev.timestamp
                if ev.total_ms < 30:
                    color = CurrentTheme.GREEN
                elif ev.total_ms < 100:
                    color = CurrentTheme.YELLOW
                else:
                    color = CurrentTheme.RED
                line = (
                    f"  -{age:5.1f}s  f={ev.target_frame:<7}  "
                    f"total={ev.total_ms:5.1f}ms  "
                    f"ff={ev.ffmpeg_ms:5.1f}ms  "
                    f"mpv={ev.mpv_ms:5.1f}ms  "
                    f"{'tracker' if ev.tracker_needs else 'mpv-only'}")
                imgui.text_colored(line, *color)

        # Pause / resume single-slot timings.
        imgui.spacing()
        if metrics.last_pause_ms > 0 or metrics.last_resume_ms > 0:
            imgui.text(
                f"Last pause: {metrics.last_pause_ms:.1f}ms  |  "
                f"last resume: {metrics.last_resume_ms:.1f}ms")


    def _render_trace_summary(self, trace):
        """Show the most recent root traces with their top child legs."""
        if not trace.is_enabled():
            imgui.text_disabled("Tracing off; press Start recording to capture.")
            return
        roots = trace.recent_roots(8)
        if not roots:
            imgui.text_disabled("No traces yet. Click around or press play.")
            return
        imgui.text("Recent traces (newest last):")
        for r in roots:
            color = (CurrentTheme.GREEN if r.ms < 30 else
                     CurrentTheme.YELLOW if r.ms < 100 else CurrentTheme.RED)
            line = f"  {r.name:<22} {r.ms:7.2f}ms"
            if r.attrs:
                # Render the most useful attrs compactly.
                kv = "  ".join(f"{k}={v}" for k, v in list(r.attrs.items())[:3])
                line = f"{line}  {kv}"
            imgui.text_colored(line, *color)
            for c in r.children[:2]:
                imgui.text_disabled(f"    {c.name:<20} {c.ms:7.2f}ms")

    def _render_disk_io_section(self):
        """Render the independent Disk I/O section."""
        stats = self._get_system_stats()

        available_width = imgui.get_content_region_available_width()

        disk_read_mb_s = stats.get("disk_read_mb_s", [])
        disk_write_mb_s = stats.get("disk_write_mb_s", [])
        current_read_rate = disk_read_mb_s[-1] if disk_read_mb_s else 0.0
        current_write_rate = disk_write_mb_s[-1] if disk_write_mb_s else 0.0
        current_time = time.time()

        # Handle 3-second delay for zero values
        # Read rate logic
        if current_read_rate > 0:
            self._last_non_zero_read_rate = current_read_rate
            self._read_zero_start_time = None
            read_rate = current_read_rate
        else:
            if self._read_zero_start_time is None:
                self._read_zero_start_time = current_time
                read_rate = self._last_non_zero_read_rate
            elif current_time - self._read_zero_start_time >= self._zero_delay_duration:
                read_rate = 0.0
            else:
                read_rate = self._last_non_zero_read_rate

        # Write rate logic
        if current_write_rate > 0:
            self._last_non_zero_write_rate = current_write_rate
            self._write_zero_start_time = None
            write_rate = current_write_rate
        else:
            if self._write_zero_start_time is None:
                self._write_zero_start_time = current_time
                write_rate = self._last_non_zero_write_rate
            elif current_time - self._write_zero_start_time >= self._zero_delay_duration:
                write_rate = 0.0
            else:
                write_rate = self._last_non_zero_write_rate

        # Header status colors based on individual read/write activity
        def get_io_color_and_status(rate):
            if rate < 10:
                return CurrentTheme.GREEN
            elif rate < 100:
                return CurrentTheme.YELLOW
            else:
                return (1.0, 0.4, 0.4, 1.0)  # Lighter red for better readability

        read_color = get_io_color_and_status(read_rate)
        write_color = get_io_color_and_status(write_rate)

        imgui.text_colored(f"           Read {read_rate:.2f} MB/s", *read_color)
        imgui.same_line()
        imgui.text_colored(" | ", *CurrentTheme.WHITE)
        imgui.same_line()
        imgui.text_colored(f"Write {write_rate:.2f} MB/s", *write_color)

        # Read MB/s graph
        render_graph(
            "disk_read_mb_s",
            disk_read_mb_s,
            f"Read {read_rate:.2f} MB/s",
            available_width,
            scale_min=0,
            scale_max=max(1.0, max(disk_read_mb_s) if disk_read_mb_s else 1.0),
            height=45,
            color=(0.4, 0.8, 1.0, 0.9),
        )

        # Write MB/s graph
        render_graph(
            "disk_write_mb_s",
            disk_write_mb_s,
            f"Write {write_rate:.2f} MB/s",
            available_width,
            scale_min=0,
            scale_max=max(1.0, max(disk_write_mb_s) if disk_write_mb_s else 1.0),
            height=45,
            color=(1.0, 0.6, 0.2, 0.9),
        )

        imgui.spacing()

    def _render_content_ui_performance(self):
        """Render UI performance information with clean, organized layout."""
        self.ui_performance_perf.start_timing()
        app = self.app
        gui = app.gui_instance if hasattr(app, "gui_instance") else None
        if not gui:
            imgui.text_disabled("Performance data not available.")
            self.ui_performance_perf.end_timing()
            return

        imgui.text("UI Component Performance")
        imgui.separator()
        imgui.spacing()

        current_stats = (
            list(gui.component_render_times.items())
            if hasattr(gui, "component_render_times")
            else []
        )
        current_total = 0

        if current_stats:
            current_total = sum(t for _, t in current_stats)
            imgui.text_colored("Current Frame:", *CurrentTheme.BLUE_LIGHT)

            if current_total < 16.67:
                status_color = CurrentTheme.GREEN
                status_text = "[SMOOTH]"
                fps_target = "60+ FPS"
            elif current_total < 33.33:
                status_color = CurrentTheme.YELLOW
                status_text = "[GOOD]"
                fps_target = "30-60 FPS"
            else:
                status_color = CurrentTheme.RED
                status_text = "[SLOW]"
                fps_target = "<30 FPS"

            imgui.same_line()
            imgui.text_colored(
                f" {current_total:.1f}ms {status_text} ({fps_target})", *status_color
            )
        else:
            imgui.text_disabled("No current frame data available")

        # Enhanced breakdown with visual indicators
        if current_stats:
            imgui.spacing()
            imgui.text("Component Breakdown - All Components:")
            imgui.spacing()

            # Show ALL components with columns for better organization
            imgui.columns(3, "AllComponentColumns", True)
            imgui.text("Component")
            imgui.next_column()
            imgui.text("Time (ms)")
            imgui.next_column()
            imgui.text("% of Total")
            imgui.next_column()
            imgui.separator()

            # Show all components sorted by time (most expensive first)
            for component, time_ms in current_stats:
                percentage = (time_ms / current_total) * 100 if current_total > 0 else 0

                # Color code the component name based on impact
                if time_ms > 5.0:
                    imgui.text_colored(component, *CurrentTheme.RED)  # Red - high impact
                elif time_ms > 1.0:
                    imgui.text_colored(component, *CurrentTheme.YELLOW)  # Yellow - moderate impact
                else:
                    imgui.text_colored(component, *CurrentTheme.GREEN)  # Green - low impact

                imgui.next_column()
                imgui.text(f"{time_ms:.3f}")
                imgui.next_column()
                imgui.text(f"{percentage:.1f}%")
                imgui.next_column()

            imgui.columns(1)

            # Optional: Visual bars for top components
            imgui.spacing()
            if len(current_stats) > 5 and imgui.collapsing_header("Visual Performance Bars##PerfBars")[0]:
                imgui.text("Top Performance Impact:")
                # Show top 5 with visual bars
                for component, time_ms in current_stats[:5]:
                    percentage = (time_ms / current_total) * 100 if current_total > 0 else 0

                    # Color code based on performance impact
                    if time_ms > 5.0:
                        color = CurrentTheme.RED  # Red
                    elif time_ms > 1.0:
                        color = CurrentTheme.YELLOW  # Yellow
                    else:
                        color = CurrentTheme.GREEN  # Green

                    imgui.text(f"{component}: {time_ms:.2f}ms ({percentage:.1f}%)")

                    # Progress bar visualization
                    bar_width = min(percentage / 100.0, 1.0)
                    imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *color)
                    imgui.progress_bar(bar_width, size=(300, 0))
                    imgui.pop_style_color()

        # Extended Performance Monitoring Section
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if imgui.collapsing_header("Extended Performance Monitoring##ExtendedPerf")[0]:
            gui = app.gui_instance if hasattr(app, "gui_instance") else None
            if gui:
                imgui.columns(2, "ExtendedPerfColumns", True)

                # Video Decoding Performance
                imgui.text_colored("Video Decoding:", *CurrentTheme.BLUE_LIGHT)
                imgui.next_column()
                if hasattr(gui, 'video_decode_times') and gui.video_decode_times:
                    avg_decode = sum(gui.video_decode_times) / len(gui.video_decode_times)
                    recent_decode = gui.video_decode_times[-1] if gui.video_decode_times else 0
                    imgui.text(f"Recent: {recent_decode:.2f}ms | Avg: {avg_decode:.2f}ms")
                else:
                    imgui.text_disabled("No decode data available")
                imgui.next_column()

                # GPU Memory Usage
                imgui.text_colored("GPU Memory:", *CurrentTheme.BLUE_LIGHT)
                imgui.next_column()
                if hasattr(gui, 'gpu_memory_usage') and gui.gpu_memory_usage > 0:
                    if gui.gpu_memory_usage < 50:
                        color = CurrentTheme.GREEN  # Green
                    elif gui.gpu_memory_usage < 80:
                        color = CurrentTheme.YELLOW  # Yellow
                    else:
                        color = CurrentTheme.RED  # Red
                    imgui.text_colored(f"{gui.gpu_memory_usage:.1f}% used", *color)
                else:
                    imgui.text_disabled("GPU monitoring unavailable")
                imgui.next_column()

                # Disk I/O Performance
                imgui.text_colored("Disk I/O:", *CurrentTheme.BLUE_LIGHT)
                imgui.next_column()
                if hasattr(gui, 'disk_io_times') and gui.disk_io_times:
                    recent_io = gui.disk_io_times[-1] if gui.disk_io_times else 0
                    avg_io = sum(gui.disk_io_times) / len(gui.disk_io_times)
                    if avg_io < 5.0:
                        color = CurrentTheme.GREEN
                    elif avg_io < 20.0:
                        color = CurrentTheme.YELLOW
                    else:
                        color = CurrentTheme.RED
                    imgui.text_colored(f"Recent: {recent_io:.2f}ms | Avg: {avg_io:.2f}ms", *color)
                else:
                    imgui.text_disabled("No I/O data available")
                imgui.next_column()

                # Network Operations
                imgui.text_colored("Network Ops:", *CurrentTheme.BLUE_LIGHT)
                imgui.next_column()
                if hasattr(gui, 'network_operation_times') and gui.network_operation_times:
                    recent_net = gui.network_operation_times[-1] if gui.network_operation_times else 0
                    avg_net = sum(gui.network_operation_times) / len(gui.network_operation_times)
                    if avg_net < 100.0:
                        color = CurrentTheme.GREEN
                    elif avg_net < 500.0:
                        color = CurrentTheme.YELLOW
                    else:
                        color = CurrentTheme.RED
                    imgui.text_colored(f"Recent: {recent_net:.1f}ms | Avg: {avg_net:.1f}ms", *color)
                else:
                    imgui.text_disabled("No network data available")
                imgui.next_column()

                imgui.columns(1)

                # Performance Budget Analysis
                imgui.spacing()
                imgui.text_colored("Frame Budget Analysis (60fps = 16.67ms):", *CurrentTheme.YELLOW_LIGHT)
                if current_total > 0:
                    budget_used = (current_total / 16.67) * 100
                    remaining = max(0, 16.67 - current_total)

                    if budget_used < 60:
                        budget_color = CurrentTheme.GREEN
                        status = "PLENTY OF HEADROOM"
                    elif budget_used < 90:
                        budget_color = CurrentTheme.YELLOW
                        status = "GOOD PERFORMANCE"
                    else:
                        budget_color = CurrentTheme.RED
                        status = "FRAME BUDGET EXCEEDED"

                    imgui.text(f"Budget used: ")
                    imgui.same_line()
                    imgui.text_colored(f"{budget_used:.1f}% ({status})", *budget_color)
                    imgui.text(f"Remaining budget: {remaining:.2f}ms")

                    # Visual budget bar
                    imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *budget_color)
                    imgui.progress_bar(min(budget_used / 100.0, 1.0), size=(300, 0), overlay=f"{budget_used:.1f}%")
                    imgui.pop_style_color()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Frontend: Always read from queue for continuous data
        current_accumulated_times = {}
        current_frame_count = 0

        # Get the most recent data from the queue
        if hasattr(gui, "_frontend_perf_queue") and gui._frontend_perf_queue:
            latest_data = gui._frontend_perf_queue[-1]  # Get most recent entry
            current_accumulated_times = latest_data.get('accumulated_times', {})
            current_frame_count = latest_data.get('frame_count', 0)

        if current_accumulated_times and current_frame_count > 0:
            imgui.text_colored("Average Performance:", *CurrentTheme.BLUE_LIGHT)
            imgui.same_line()
            imgui.text(f"({current_frame_count} frames tracked)")

            avg_stats = list(current_accumulated_times.items())
            avg_total = (
                sum(total_time / current_frame_count for _, total_time in avg_stats)
                if current_frame_count > 0
                else 0
            )

            avg_stats_for_expensive = list(current_accumulated_times.items())
            if current_frame_count > 0:
                avg_stats_for_expensive.sort(
                    key=lambda x: x[1] / current_frame_count, reverse=True
                )
            else:
                avg_stats_for_expensive.sort(key=lambda x: x[0].lower())

            if avg_total < 16.67:
                avg_status_color = CurrentTheme.GREEN
                avg_status_text = "[EXCELLENT]"
            elif avg_total < 33.33:
                avg_status_color = CurrentTheme.YELLOW
                avg_status_text = "[GOOD]"
            else:
                avg_status_color = CurrentTheme.RED
                avg_status_text = "[NEEDS OPTIMIZATION]"

            imgui.text_colored(
                f"Overall: {avg_total:.1f}ms {avg_status_text}", *avg_status_color
            )

            imgui.spacing()

            imgui.text("Most Expensive Components (avg):")
            for i, (component, total_time) in enumerate(
                avg_stats_for_expensive[:3]
            ):
                avg_time = (
                    total_time / current_frame_count if current_frame_count > 0 else 0.0
                )
                time_color = (
                    (0.0, 1.0, 0.0, 1.0)
                    if avg_time < 5.0
                    else (1.0, 0.8, 0.0, 1.0)
                    if avg_time < 16.67
                    else CurrentTheme.RED
                )
                imgui.text_colored(
                    f"  {i+1}. {component}: {avg_time:.2f}ms", *time_color
                )

            if len(avg_stats_for_expensive) > 3:
                imgui.text_disabled(
                    f"  ... and {len(avg_stats_for_expensive) - 3} more components"
                )
        else:
            imgui.text_disabled("No historical data available")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if (
            current_stats
            and len(current_stats) > 0
            and current_accumulated_times
            and current_frame_count > 0
        ):
            imgui.text_colored("All Components:", *CurrentTheme.BLUE_LIGHT)

            if not hasattr(self, "_perf_sort_mode"):
                self._perf_sort_mode = 0

            sort_modes = ["by Current", "by Average", "A-Z"]
            sort_label = sort_modes[self._perf_sort_mode]
            if imgui.small_button(f"Sort: {sort_label}"):
                self._perf_sort_mode = (self._perf_sort_mode + 1) % 3

            all_component_names = set()
            current_dict = dict(current_stats)
            avg_dict = {}

            for component, _ in current_stats:
                all_component_names.add(component)

            inv_frame_count = 1.0 / current_frame_count if current_frame_count > 0 else 0.0
            for component, total_time in current_accumulated_times.items():
                all_component_names.add(component)
                avg_dict[component] = total_time * inv_frame_count

            complete_stats = []
            for component in all_component_names:
                ct = current_dict.get(component, 0.0)
                avg_time = avg_dict.get(component, 0.0)
                complete_stats.append((component, ct, avg_time))

            if self._perf_sort_mode == 0:
                complete_stats.sort(key=lambda x: x[1], reverse=True)
            elif self._perf_sort_mode == 1:
                complete_stats.sort(key=lambda x: x[2], reverse=True)
            else:
                complete_stats.sort(key=lambda x: x[0].lower())

            imgui.spacing()
            imgui.columns(4, "complete_perf_table", border=False)
            imgui.set_column_width(0, 180)
            imgui.set_column_width(1, 70)
            imgui.set_column_width(2, 70)
            imgui.set_column_width(3, 70)
            imgui.text("Component")
            imgui.next_column()
            imgui.text("Time (ms)")
            imgui.next_column()
            imgui.text("Avg (ms)")
            imgui.next_column()
            imgui.text("% of Total")
            imgui.next_column()
            imgui.separator()

            for component, ct, avg_time in complete_stats:
                percentage = (
                    (ct / current_total) * 100
                    if current_total > 0 and ct > 0
                    else 0.0
                )
                imgui.text(component)
                imgui.next_column()

                if ct > 0:
                    time_color = (
                        (0.0, 1.0, 0.0, 1.0)
                        if ct < 16.67
                        else (1.0, 0.8, 0.0, 1.0)
                        if ct < 33.33
                        else CurrentTheme.RED
                    )
                    imgui.text_colored(f"{ct:.2f}", *time_color)
                else:
                    imgui.text_disabled("0.00")
                imgui.next_column()

                if avg_time > 0:
                    avg_color = (
                        (0.0, 1.0, 0.0, 1.0)
                        if avg_time < 5.0
                        else (1.0, 0.8, 0.0, 1.0)
                        if avg_time < 16.67
                        else CurrentTheme.RED
                    )
                    imgui.text_colored(f"{avg_time:.2f}", *avg_color)
                else:
                    imgui.text_disabled("0.00")
                imgui.next_column()

                if percentage > 0:
                    imgui.text(f"{percentage:.1f}%")
                else:
                    imgui.text_disabled("-")
                imgui.next_column()

            imgui.columns(1)

        elif current_stats and len(current_stats) > 0:
            imgui.text_colored(
                "All Components (current frame only):", *CurrentTheme.BLUE_LIGHT
            )
            imgui.text_disabled("Historical data not yet available")

            imgui.spacing()

            imgui.columns(3, "current_only_table", border=False)
            imgui.set_column_width(0, 200)
            imgui.set_column_width(1, 80)
            imgui.text("Component")
            imgui.next_column()
            imgui.text("Time (ms)")
            imgui.next_column()
            imgui.text("% of Total")
            imgui.next_column()
            imgui.separator()

            for component, render_time in current_stats:
                percentage = (render_time / current_total) * 100 if current_total > 0 else 0
                imgui.text(component)
                imgui.next_column()
                time_color = (
                    (0.0, 1.0, 0.0, 1.0)
                    if render_time < 16.67
                    else (1.0, 0.8, 0.0, 1.0)
                    if render_time < 33.33
                    else CurrentTheme.RED
                )
                imgui.text_colored(f"{render_time:.2f}", *time_color)
                imgui.next_column()
                imgui.text(f"{percentage:.1f}%")
                imgui.next_column()

            imgui.columns(1)

        imgui.spacing()
        imgui.separator()

        if hasattr(gui, "last_perf_log_time") and hasattr(gui, "perf_log_interval"):
            time_since_log = time.time() - gui.last_perf_log_time
            next_log_in = gui.perf_log_interval - time_since_log

            imgui.text_disabled(f"Next debug log in: {next_log_in:.1f}s")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Performance data is logged to console every few seconds.\n\n"
                    "Color coding:\n"
                    "Green: Excellent performance\n"
                    "Yellow: Acceptable performance\n"
                    "Red: Needs optimization"
                )

        self.ui_performance_perf.end_timing()
