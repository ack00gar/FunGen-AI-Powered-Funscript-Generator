"""System performance monitoring mixin for InfoGraphsUI."""
import imgui
import numpy as np
import time


def render_graph(label, data, overlay_text, available_width,
                 scale_min=0, scale_max=100, height=60, color=None):
    """Shared graph rendering helper used by performance and disk I/O sections."""
    np_data = np.array(data, dtype=np.float32) if data else np.array([], dtype=np.float32)
    current_value = data[-1] if data else 0.0
    if color is None:
        if current_value < 50:
            color = (0.2, 0.8, 0.2, 0.8)
        elif current_value < 80:
            color = (1.0, 0.8, 0.2, 0.8)
        else:
            color = (1.0, 0.2, 0.2, 0.8)

    imgui.push_style_color(imgui.COLOR_PLOT_LINES, *color)
    imgui.plot_lines(
        f"##{label}",
        np_data,
        overlay_text=overlay_text,
        scale_min=scale_min,
        scale_max=scale_max,
        graph_size=(available_width, height),
    )
    imgui.pop_style_color()


class PerformanceMixin:

    def _render_content_performance(self):
        self.perf_monitor.start_timing()

        stats = self._get_system_stats()

        available_width = imgui.get_content_region_available_width()

        def render_core_bars(per_core_usage):
            if not per_core_usage:
                return

            bar_width = max(8, (available_width - 20) / len(per_core_usage))
            bar_height = 30
            spacing = 2

            total_width = (bar_width + spacing) * len(per_core_usage) - spacing

            # Pre-compute outside loop
            dl = imgui.get_window_draw_list()
            bg_color = imgui.get_color_u32_rgba(0.2, 0.2, 0.2, 0.8)
            text_color = imgui.get_color_u32_rgba(0.8, 0.8, 0.8, 1.0)
            color_green = imgui.get_color_u32_rgba(0.2, 0.8, 0.2, 1.0)
            color_yellow = imgui.get_color_u32_rgba(1.0, 0.8, 0.2, 1.0)
            color_red = imgui.get_color_u32_rgba(1.0, 0.2, 0.2, 1.0)

            for i, core_load in enumerate(per_core_usage):
                bar_x = imgui.get_cursor_screen_pos()[0] + i * (bar_width + spacing)
                bar_y = imgui.get_cursor_screen_pos()[1]

                color = color_green if core_load < 50 else (color_yellow if core_load < 80 else color_red)

                dl.add_rect_filled(
                    bar_x, bar_y, bar_x + bar_width, bar_y + bar_height, bg_color
                )

                usage_height = (core_load / 100.0) * bar_height
                dl.add_rect_filled(
                    bar_x,
                    bar_y + bar_height - usage_height,
                    bar_x + bar_width,
                    bar_y + bar_height,
                    color,
                )

                text = f"C{i}"
                text_size = imgui.calc_text_size(text)
                text_x = bar_x + (bar_width - text_size[0]) / 2
                text_y = bar_y + bar_height + 2
                dl.add_text(text_x, text_y, text_color, text)

            imgui.dummy(total_width, bar_height + 20)

        # CPU
        physical_cores = stats.get("cpu_physical_cores", stats["cpu_core_count"])
        logical_cores = stats["cpu_core_count"]
        core_info = (
            f"({physical_cores}P/{logical_cores}L cores)"
            if physical_cores != logical_cores
            else f"({logical_cores} cores)"
        )

        cpu_load = stats["cpu_load"]
        current_cpu = cpu_load[-1] if cpu_load else 0

        if current_cpu < 50:
            header_color = (0.2, 0.8, 0.2, 1.0)
            cpu_status = "[OK]"
        elif current_cpu < 80:
            header_color = (1.0, 0.8, 0.2, 1.0)
            cpu_status = "[HIGH]"
        else:
            header_color = (1.0, 0.2, 0.2, 1.0)
            cpu_status = "[CRIT]"

        imgui.text_colored(f"CPU {core_info} {cpu_status}", *header_color)
        render_graph("cpu_load", cpu_load, f"{current_cpu:.1f}%", available_width, height=50)

        # CPU freq (MHz to GHz) if available
        cpu_freq = stats.get("cpu_freq", 0)
        if cpu_freq > 0:
            freq_ghz = cpu_freq / 1000.0
            imgui.same_line()
            imgui.text_colored(f" @ {freq_ghz:.1f}GHz", 0.7, 0.7, 0.7, 1.0)

        # CPU temp if available
        cpu_temp = stats.get("cpu_temp", None)
        if cpu_temp is not None:
            imgui.same_line()
            imgui.text_colored(f" | {cpu_temp:.0f}C", 0.7, 0.7, 0.7, 1.0)

        imgui.spacing()

        per_core_usage = stats.get("cpu_per_core", [])
        if per_core_usage:
            imgui.text("Per-Core Usage:")
            render_core_bars(per_core_usage)
        else:
            imgui.text_disabled("Per-core data not available")

        imgui.separator()
        imgui.spacing()

        # RAM
        ram_percent = stats["ram_usage_percent"]
        ram_gb = stats["ram_usage_gb"]
        ram_total = stats.get("ram_total_gb", 0)
        current_ram = ram_percent[-1] if ram_percent else 0

        if current_ram < 60:
            ram_color = (0.2, 0.8, 0.2, 1.0)
        elif current_ram < 85:
            ram_color = (1.0, 0.8, 0.2, 1.0)
        else:
            ram_color = (1.0, 0.2, 0.2, 1.0)

        ram_status = (
            "[OK]" if current_ram < 60 else "[HIGH]" if current_ram < 85 else "[CRIT]"
        )
        imgui.text_colored(f"Memory (RAM) {ram_status}", *ram_color)
        last_ram_gb = ram_gb[-1] if ram_gb else 0.0
        render_graph(
            "ram_usage",
            ram_percent,
            f"{current_ram:.1f}% ({last_ram_gb:.1f}/{ram_total:.1f}GB)",
            available_width,
            height=55,
        )

        swap_percent = stats.get("swap_usage_percent", 0.0)
        swap_gb = stats.get("swap_usage_gb", 0.0)
        if swap_percent > 0:
            swap_color = (
                (0.2, 0.8, 0.2, 1.0)
                if swap_percent < 50
                else (1.0, 0.8, 0.2, 1.0)
                if swap_percent < 80
                else (1.0, 0.2, 0.2, 1.0)
            )
            imgui.text_colored(
                f"Swap: {swap_percent:.1f}% ({swap_gb:.1f}GB)", *swap_color
            )
        else:
            imgui.text_colored("Swap: Not in use", 0.2, 0.8, 0.2, 1.0)

        imgui.separator()

        # GPU
        if stats.get("gpu_available", False):
            gpu_name = stats.get("gpu_name", "Unknown GPU")
            if len(gpu_name) > 35:
                gpu_name = gpu_name[:32] + "..."

            gpu_load = stats.get("gpu_load", [])
            current_gpu = gpu_load[-1] if gpu_load else 0.0

            if current_gpu < 50:
                gpu_color = (0.2, 0.8, 0.2, 1.0)
            elif current_gpu < 80:
                gpu_color = (1.0, 0.8, 0.2, 1.0)
            else:
                gpu_color = (1.0, 0.2, 0.2, 1.0)

            gpu_status = (
                "[OK]" if current_gpu < 50 else "[HIGH]" if current_gpu < 80 else "[CRIT]"
            )
            header = f"GPU - {gpu_name} {gpu_status}"

            # Append temp if available
            gpu_temp = stats.get("gpu_temp", None)
            if gpu_temp is not None:
                header += f" | {gpu_temp:.0f}C"

            imgui.text_colored(header, *gpu_color)

            if any(load > 0 for load in gpu_load):
                render_graph("gpu_load", gpu_load, f"{current_gpu:.1f}%", available_width, height=55)

                gpu_mem = stats.get("gpu_mem_usage_percent", [])
                current_gpu_mem = gpu_mem[-1] if gpu_mem else 0.0

                # Best effort: get memory used/total from gpu_info if present
                mem_overlay = f"{current_gpu_mem:.1f}%"
                gpu_info = stats.get("gpu_info", None)
                try:
                    if (
                        isinstance(gpu_info, dict)
                        and "gpu" in gpu_info
                        and gpu_info["gpu"]
                    ):
                        fb = gpu_info["gpu"][0].get("fb_memory_usage", {})
                        used = fb.get("used", 0.0)
                        total = fb.get("total", 0.0)

                        # pynvml.smi may provide MiB values or strings; try to coerce
                        def _to_mib(x):
                            if isinstance(x, (int, float)):
                                return float(x)
                            s = str(x)
                            for ch in ["MiB", "MB", "GiB", "GB"]:
                                s = s.replace(ch, "")
                            try:
                                return float(s.strip())
                            except Exception:
                                return None

                        used_mib = _to_mib(used)
                        total_mib = _to_mib(total)
                        if used_mib is not None and total_mib is not None and total_mib > 0:
                            used_gb = used_mib / 1024.0
                            total_gb = total_mib / 1024.0
                            mem_overlay = f"{current_gpu_mem:.1f}% ({used_gb:.1f}/{total_gb:.1f}GB)"
                except Exception:
                    pass

                render_graph("gpu_mem", gpu_mem, mem_overlay, available_width, height=45)
            else:
                if stats.get("os") == "Darwin" and "Apple" in gpu_name:
                    imgui.text_colored(
                        "[INFO] GPU detected but metrics may require admin access",
                        0.6, 0.8, 1.0, 1.0,
                    )
                    imgui.text_colored(
                        "      Run with 'sudo' to enable GPU monitoring",
                        0.5, 0.6, 0.7, 1.0,
                    )
                else:
                    imgui.text_colored(
                        "[INFO] GPU monitoring not available", 0.7, 0.7, 0.7, 1.0
                    )
                imgui.spacing()

            imgui.spacing()

        self.perf_monitor.end_timing()

    def _render_system_report_section(self):
        """Render system report with copy-to-clipboard button."""
        # Lazy-generate report (cached until manually refreshed)
        if not hasattr(self, '_system_report_text'):
            self._system_report_text = None

        if imgui.button("Generate Report##sysReport"):
            from application.utils.system_report import generate_report
            self._system_report_text = generate_report()

        if self._system_report_text:
            imgui.same_line()
            if imgui.button("Copy to Clipboard##sysReportCopy"):
                try:
                    import pyperclip
                    pyperclip.copy(self._system_report_text)
                    self.app.logger.info("System report copied to clipboard.",
                                         extra={'status_message': True, 'duration': 3.0})
                except ImportError:
                    # Fallback for macOS
                    try:
                        process = __import__('subprocess').Popen(
                            ['pbcopy'], stdin=__import__('subprocess').PIPE, text=True)
                        process.communicate(self._system_report_text)
                        self.app.logger.info("System report copied to clipboard.",
                                             extra={'status_message': True, 'duration': 3.0})
                    except Exception as e:
                        self.app.logger.warning(f"Could not copy to clipboard: {e}",
                                                extra={'status_message': True})

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Scrollable text region
            avail_h = max(200, imgui.get_content_region_available()[1] - 10)
            imgui.begin_child("##sysReportScroll", width=-1, height=avail_h, border=True)
            imgui.text_unformatted(self._system_report_text)
            imgui.end_child()
        else:
            imgui.spacing()
            imgui.text_disabled("Click 'Generate Report' to collect system information.")

    def _check_memory_alerts(self, stats):
        """Check memory usage and trigger alerts if thresholds are exceeded."""
        if not hasattr(self, "_last_alert_time"):
            self._last_alert_time = {"ram": 0}

        current_time = time.time()
        alert_cooldown = 300  # 5 minutes

        ram_percent = stats.get("ram_usage_percent", [])
        if ram_percent:
            current_ram = ram_percent[-1]
            if current_ram >= 90 and (current_time - self._last_alert_time["ram"]) > alert_cooldown:
                ram_gb = stats.get("ram_usage_gb", [])
                ram_gb_val = ram_gb[-1] if ram_gb else 0.0
                ram_total = stats.get("ram_total_gb", 0.0)
                self.app.logger.warning(
                    f"[CRITICAL] HIGH MEMORY USAGE: {current_ram:.1f}% "
                    f"({ram_gb_val:.1f}/{ram_total:.1f} GB) - "
                    "Consider closing unnecessary applications or upgrading RAM.",
                    extra={"status_message": True},
                )
                self._last_alert_time["ram"] = current_time
            elif current_ram >= 85 and (current_time - self._last_alert_time["ram"]) > alert_cooldown:
                self.app.logger.warning(
                    f"[WARNING] Memory usage is high: {current_ram:.1f}% - Monitor for potential issues.",
                    extra={"status_message": True},
                )
                self._last_alert_time["ram"] = current_time
