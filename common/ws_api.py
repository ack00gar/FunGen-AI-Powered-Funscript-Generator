"""
WebSocket API server for external tool integration.

Lightweight async WS server that exposes FunGen state and controls to
external tools (custom trackers, automation scripts, companion apps).

Usage:
    api = FunGenWSAPI(app_logic_instance)
    api.start()  # starts in background thread
    # ... later
    api.stop()

Protocol:
    Send:    {"type": "command", "name": "get_state", "data": {}}
    Receive: {"type": "response", "name": "get_state", "success": true, "data": {...}}
"""

import asyncio
import json
import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_WS_API_PORT = 8769


class FunGenWSAPI:
    """WebSocket API server for FunGen."""

    def __init__(self, app, port: int = DEFAULT_WS_API_PORT):
        self.app = app
        self.port = port
        self._server = None
        self._loop = None
        self._thread = None
        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="WSAPIServer")
        self._thread.start()
        logger.info(f"WebSocket API server starting on port {self.port}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("WebSocket API server stopped")

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            import websockets
            self._loop.run_until_complete(self._serve(websockets))
        except ImportError:
            logger.warning("websockets library not available, WS API disabled")
        except OSError as e:
            logger.warning(f"WS API server failed to start: {e}")
        except Exception as e:
            if self._running:
                logger.error(f"WS API server error: {e}")

    async def _serve(self, websockets):
        async with websockets.serve(self._handle_client, "127.0.0.1", self.port):
            logger.info(f"WebSocket API listening on ws://127.0.0.1:{self.port}")
            while self._running:
                await asyncio.sleep(0.5)

    async def _handle_client(self, websocket, path=None):
        """Handle a connected client."""
        logger.info(f"WS API client connected")
        try:
            async for message in websocket:
                try:
                    msg = json.loads(message)
                    response = self._dispatch(msg)
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error", "message": "Invalid JSON"
                    }))
                except Exception as e:
                    await websocket.send(json.dumps({
                        "type": "error", "message": str(e)
                    }))
        except Exception:
            pass
        finally:
            logger.info("WS API client disconnected")

    def _dispatch(self, msg: dict) -> dict:
        """Route a command to its handler."""
        name = msg.get("name", "")
        data = msg.get("data", {})

        handlers = {
            "get_state": self._cmd_get_state,
            "seek": self._cmd_seek,
            "play": self._cmd_play,
            "pause": self._cmd_pause,
            "stop": self._cmd_stop,
            "get_funscript": self._cmd_get_funscript,
            "add_action": self._cmd_add_action,
            "get_chapters": self._cmd_get_chapters,
            "get_tracker_info": self._cmd_get_tracker_info,
        }

        handler = handlers.get(name)
        if not handler:
            return {"type": "response", "name": name, "success": False,
                    "error": f"Unknown command: {name}"}

        try:
            result = handler(data)
            return {"type": "response", "name": name, "success": True, "data": result}
        except Exception as e:
            return {"type": "response", "name": name, "success": False, "error": str(e)}

    # ---- Command handlers ----

    def _cmd_get_state(self, data: dict) -> dict:
        proc = self.app.processor
        if not proc:
            return {"video_loaded": False}
        return {
            "video_loaded": proc.is_video_open(),
            "current_frame": proc.current_frame_index,
            "total_frames": getattr(proc, 'total_frames', 0),
            "fps": proc.fps,
            "is_playing": proc.is_processing and not proc.pause_event.is_set(),
            "video_path": getattr(self.app.file_manager, 'video_path', ''),
        }

    def _cmd_seek(self, data: dict) -> dict:
        proc = self.app.processor
        if not proc or not proc.is_video_open():
            raise ValueError("No video loaded")
        frame = data.get("frame")
        ms = data.get("ms")
        if frame is not None:
            proc.seek_video(int(frame))
        elif ms is not None:
            from common.frame_utils import ms_to_frame
            proc.seek_video(ms_to_frame(ms, proc.fps))
        else:
            raise ValueError("Provide 'frame' or 'ms'")
        return {"frame": proc.current_frame_index}

    def _cmd_play(self, data: dict) -> dict:
        proc = self.app.processor
        if proc and proc.is_video_open():
            if not proc.is_processing:
                proc.start_processing()
            elif proc.pause_event.is_set():
                proc.pause_event.clear()
        return {"playing": True}

    def _cmd_pause(self, data: dict) -> dict:
        proc = self.app.processor
        if proc and proc.is_processing:
            proc.pause_event.set()
        return {"playing": False}

    def _cmd_stop(self, data: dict) -> dict:
        proc = self.app.processor
        if proc and proc.is_processing:
            proc.stop_processing()
        return {"stopped": True}

    def _cmd_get_funscript(self, data: dict) -> dict:
        fs = None
        if self.app.processor and self.app.processor.tracker:
            fs = self.app.processor.tracker.funscript
        if not fs:
            return {"actions": [], "count": 0}
        axis = data.get("axis", "primary")
        start_ms = data.get("start_ms")
        end_ms = data.get("end_ms")
        if start_ms is not None and end_ms is not None:
            actions = fs.get_actions_in_range(int(start_ms), int(end_ms), axis)
        else:
            actions = fs.get_axis_actions(axis) or []
        return {"actions": actions, "count": len(actions)}

    def _cmd_add_action(self, data: dict) -> dict:
        fs = None
        if self.app.processor and self.app.processor.tracker:
            fs = self.app.processor.tracker.funscript
        if not fs:
            raise ValueError("No funscript available")
        at = data.get("at")
        pos = data.get("pos")
        if at is None or pos is None:
            raise ValueError("Provide 'at' (ms) and 'pos' (0-100)")
        fs.add_action(int(at), int(pos), is_from_live_tracker=False)
        return {"at": int(at), "pos": int(pos)}

    def _cmd_get_chapters(self, data: dict) -> dict:
        fs_proc = getattr(self.app, 'funscript_processor', None)
        if not fs_proc:
            return {"chapters": []}
        chapters = []
        for ch in (fs_proc.video_chapters or []):
            chapters.append({
                "name": ch.position_short_name,
                "start_frame": ch.start_frame_id,
                "end_frame": ch.end_frame_id,
                "source": ch.source,
            })
        return {"chapters": chapters, "count": len(chapters)}

    def _cmd_get_tracker_info(self, data: dict) -> dict:
        tracker = getattr(self.app, 'tracker', None)
        if not tracker:
            return {"tracker": None}
        info = tracker.get_tracker_info() if hasattr(tracker, 'get_tracker_info') else None
        return {
            "name": info.name if info else None,
            "display_name": info.display_name if info else None,
            "category": info.category if info else None,
            "tracking_active": tracker.tracking_active if hasattr(tracker, 'tracking_active') else False,
        }
