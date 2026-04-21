"""GLFW gamepad polling for live-scripting in recording mode.

Polls GLFW gamepad state (axes + buttons), applies deadzone and axis mapping,
and returns a normalised 0-100 position suitable for RecordingCapture.

All poll() calls MUST happen on the main/GUI thread (GLFW requirement).
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import glfw


# ---------------------------------------------------------------------------
# GLFW gamepad axis/button constants (duplicated here for clarity)
# ---------------------------------------------------------------------------
_AXIS_LEFT_X = 0
_AXIS_LEFT_Y = 1
_AXIS_RIGHT_X = 2
_AXIS_RIGHT_Y = 3

_BUTTON_A = 0       # Cross on PS
_BUTTON_B = 1       # Circle on PS
_BUTTON_START = 6

_AXIS_NAME_TO_INDEX = {
    "left_x":  _AXIS_LEFT_X,
    "left_y":  _AXIS_LEFT_Y,
    "right_x": _AXIS_RIGHT_X,
    "right_y": _AXIS_RIGHT_Y,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GamepadInfo:
    joystick_id: int       # GLFW slot 0-15
    name: str              # e.g. "Xbox Wireless Controller"
    is_gamepad: bool       # True if GLFW has a standard mapping


@dataclass
class GamepadState:
    primary: float         # 0-100 mapped position (selected axis)
    secondary: float       # 0-100 mapped position (secondary axis)
    button_a: bool = False
    button_b: bool = False
    button_start: bool = False
    raw_axes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# GamepadInput — main poller
# ---------------------------------------------------------------------------
class GamepadInput:
    """Polls GLFW gamepad state.  MUST be called from the main/GUI thread."""

    def __init__(self):
        self.deadzone: float = 0.15
        self.axis_mapping: str = "left_y"
        self.secondary_mapping: str = "right_x"
        self.invert_primary: bool = False
        self.invert_secondary: bool = False
        self.center_mode: bool = True
        self.active_joystick_id: int = 0
        self._detected_name: str = ""

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect_gamepads(self) -> List[GamepadInfo]:
        """Scan GLFW joystick slots 0-15, return those that are present."""
        found: List[GamepadInfo] = []
        for jid in range(16):
            if glfw.joystick_present(jid):
                name = glfw.get_joystick_name(jid)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                is_gp = bool(glfw.joystick_is_gamepad(jid))
                found.append(GamepadInfo(joystick_id=jid, name=name or f"Joystick {jid}", is_gamepad=is_gp))
        if found:
            self._detected_name = found[0].name
        return found

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------
    def poll(self) -> Optional[GamepadState]:
        """Read current gamepad state.  Returns *None* if pad is absent."""
        jid = self.active_joystick_id
        if not glfw.joystick_present(jid):
            return None

        # Prefer gamepad API (standard mapping) over raw joystick.
        if glfw.joystick_is_gamepad(jid):
            state = glfw.get_gamepad_state(jid)
            if state is None:
                return None
            axes = [float(x) for x in state.axes]
            buttons = [int(b) for b in state.buttons]
        else:
            axes_ptr, a_count = glfw.get_joystick_axes(jid)
            btn_ptr, b_count = glfw.get_joystick_buttons(jid)
            if not axes_ptr or a_count <= 0:
                return None
            axes = [float(axes_ptr[i]) for i in range(a_count)]
            axes += [0.0] * max(0, 4 - len(axes))
            buttons = [int(btn_ptr[i]) for i in range(b_count)] if btn_ptr else []
            buttons += [0] * max(0, 7 - len(buttons))

        pri_idx = _AXIS_NAME_TO_INDEX.get(self.axis_mapping, _AXIS_LEFT_Y)
        sec_idx = _AXIS_NAME_TO_INDEX.get(self.secondary_mapping, _AXIS_RIGHT_X)

        raw_pri = axes[pri_idx] if pri_idx < len(axes) else 0.0
        raw_sec = axes[sec_idx] if sec_idx < len(axes) else 0.0

        pri = self._apply_deadzone(raw_pri)
        sec = self._apply_deadzone(raw_sec)

        if self.center_mode:
            # rest=50, full stick travel maps to 0..100 (sign-preserving)
            primary_100 = (1.0 - pri) * 50.0
            secondary_100 = (1.0 - sec) * 50.0
        else:
            # rest=0, deflection magnitude maps to 0..100 (direction-agnostic)
            primary_100 = 100.0 * abs(pri)
            secondary_100 = 100.0 * abs(sec)

        # Invert after mode transform; applied before, abs() erased it.
        if self.invert_primary:
            primary_100 = 100.0 - primary_100
        if self.invert_secondary:
            secondary_100 = 100.0 - secondary_100

        return GamepadState(
            primary=max(0.0, min(100.0, primary_100)),
            secondary=max(0.0, min(100.0, secondary_100)),
            button_a=bool(buttons[_BUTTON_A]) if _BUTTON_A < len(buttons) else False,
            button_b=bool(buttons[_BUTTON_B]) if _BUTTON_B < len(buttons) else False,
            button_start=bool(buttons[_BUTTON_START]) if _BUTTON_START < len(buttons) else False,
            raw_axes=axes[:4],
        )

    # ------------------------------------------------------------------
    # Deadzone
    # ------------------------------------------------------------------
    def _apply_deadzone(self, value: float) -> float:
        """Scaled deadzone: |v|<dz → 0, else remap [dz,1] → [0,1]."""
        dz = self.deadzone
        av = abs(value)
        if av < dz:
            return 0.0
        sign = 1.0 if value >= 0 else -1.0
        return sign * min(1.0, (av - dz) / (1.0 - dz))


# ---------------------------------------------------------------------------
# CalibrationRoutine — measures controller→screen input lag
# ---------------------------------------------------------------------------
class CalibrationRoutine:
    """10-beat visual calibration to measure controller input lag.

    Usage::

        cal = CalibrationRoutine(gamepad)
        cal.start()
        # Each frame:
        done = cal.update()
        show_flash = cal.get_current_beat_active()
        if done:
            print(cal.result_ms)
    """

    NUM_BEATS = 10
    BEAT_INTERVAL_S = 1.0       # 1 second between beats
    BEAT_WINDOW_S = 0.45        # beat flash lasts this long

    def __init__(self, gamepad: GamepadInput):
        self._gamepad = gamepad
        self._start_time: float = 0.0
        self._beat_times: List[float] = []   # expected beat times (monotonic)
        self._press_deltas: List[float] = [] # measured delta per beat
        self.current_beat: int = 0
        self.is_running: bool = False
        self.result_ms: Optional[float] = None
        self._waiting_for_release: bool = False

    def start(self):
        """Begin (or restart) calibration."""
        self._start_time = time.monotonic()
        self._beat_times = [
            self._start_time + (i + 1) * self.BEAT_INTERVAL_S
            for i in range(self.NUM_BEATS)
        ]
        self._press_deltas.clear()
        self.current_beat = 0
        self.is_running = True
        self.result_ms = None
        self._waiting_for_release = False

    def get_current_beat_active(self) -> bool:
        """Return True when the current beat's visual cue should be displayed."""
        if not self.is_running or self.current_beat >= self.NUM_BEATS:
            return False
        now = time.monotonic()
        beat_start = self._beat_times[self.current_beat]
        return beat_start <= now < beat_start + self.BEAT_WINDOW_S

    def update(self) -> bool:
        """Call each frame.  Returns True when calibration is complete."""
        if not self.is_running:
            return self.result_ms is not None

        if self.current_beat >= self.NUM_BEATS:
            self._finish()
            return True

        state = self._gamepad.poll()
        if state is None:
            return False

        now = time.monotonic()
        beat_time = self._beat_times[self.current_beat]

        # Wait for the beat window to start before accepting presses
        if now < beat_time:
            return False

        if self._waiting_for_release:
            if not state.button_a:
                self._waiting_for_release = False
                self.current_beat += 1
            return False

        if state.button_a:
            delta = now - beat_time
            self._press_deltas.append(delta)
            self._waiting_for_release = True

        # Auto-advance if user missed this beat (2× interval past)
        if now > beat_time + self.BEAT_INTERVAL_S * 2:
            self.current_beat += 1

        return False

    def _finish(self):
        """Compute result, dropping best and worst."""
        self.is_running = False
        deltas = sorted(self._press_deltas)
        if len(deltas) >= 4:
            trimmed = deltas[1:-1]  # drop best and worst
        else:
            trimmed = deltas
        if trimmed:
            self.result_ms = (sum(trimmed) / len(trimmed)) * 1000.0
        else:
            self.result_ms = 0.0
