"""Efficient performance monitoring for UI components with memory management."""
import imgui
import time
from collections import deque


class ComponentPerformanceMonitor:

    def __init__(self, component_name: str, history_size: int = 60):
        self.component_name = component_name
        self.history = deque(maxlen=history_size)  # Efficient O(1) append/pop
        self.start_time = None

    def start_timing(self):
        """Start timing a render cycle."""
        self.start_time = time.perf_counter()

    def end_timing(self):
        """End timing and record the measurement."""
        if self.start_time is not None:
            render_time_ms = (time.perf_counter() - self.start_time) * 1000
            self.history.append(render_time_ms)
            self.start_time = None
            return render_time_ms
        return 0.0

    def get_stats(self):
        """Get performance statistics."""
        if not self.history:
            return {"current": 0.0, "avg": 0.0, "max": 0.0, "min": 0.0, "count": 0}

        history_list = list(self.history)
        return {
            "current": history_list[-1],
            "avg": sum(history_list) / len(history_list),
            "max": max(history_list),
            "min": min(history_list),
            "count": len(history_list),
        }

    def get_status_info(self):
        """Get color-coded status information."""
        stats = self.get_stats()
        current = stats["current"]

        if current < 1.0:
            return "[EXCELLENT]", (0.0, 1.0, 0.0, 1.0)  # Bright green
        elif current < 5.0:
            return "[VERY GOOD]", (0.2, 0.8, 0.2, 1.0)  # Green
        elif current < 16.67:
            return "[GOOD]", (0.4, 0.8, 0.4, 1.0)  # Light green
        elif current < 33.33:
            return "[OK]", (1.0, 0.8, 0.2, 1.0)  # Yellow
        elif current < 50.0:
            return "[SLOW]", (1.0, 0.5, 0.0, 1.0)  # Orange
        else:
            return "[VERY SLOW]", (1.0, 0.2, 0.2, 1.0)  # Red

    def render_info(self, show_detailed=True):
        """Render performance information in imgui."""
        stats = self.get_stats()
        status_text, status_color = self.get_status_info()

        if show_detailed:
            imgui.text_colored(
                f"{self.component_name} Performance {status_text}", *status_color
            )
            imgui.text(
                f"Current: {stats['current']:.2f}ms | "
                f"Avg: {stats['avg']:.2f}ms | "
                f"Max: {stats['max']:.2f}ms | "
                f"Min: {stats['min']:.2f}ms"
            )

            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    f"{self.component_name} Render Performance:\n"
                    f"Current frame: {stats['current']:.2f}ms\n"
                    f"Average ({stats['count']} frames): {stats['avg']:.2f}ms\n"
                    f"Maximum: {stats['max']:.2f}ms\n"
                    f"Minimum: {stats['min']:.2f}ms\n\n"
                    "Performance Targets:\n"
                    "< 1ms: Excellent\n"
                    "< 5ms: Very Good\n"
                    "< 16.67ms: Good (60 FPS)\n"
                    "< 33.33ms: OK (30 FPS)\n"
                    "> 33.33ms: Needs optimization"
                )
        else:
            imgui.text_colored(
                f"{self.component_name}: {stats['current']:.1f}ms {status_text}",
                *status_color,
            )
