import threading
import time
import requests
import imgui


class AddonUpdateChecker:
    """Checks for addon updates via a JSON manifest on GitHub."""

    MANIFEST_URL = (
        "https://raw.githubusercontent.com/ack00gar/"
        "FunGen-AI-Powered-Funscript-Generator/main/addon_versions.json"
    )

    # Status strip ad: show for 12s, start 8s after launch to avoid clashing
    _AD_INITIAL_DELAY = 8.0
    _AD_DISPLAY_DURATION = 12.0
    _AD_CYCLE_INTERVAL = 15.0  # seconds between cycling to next addon

    def __init__(self, app):
        self.app = app
        self.updates_available = []   # outdated installed addons
        self.unowned_addons = []      # addons not installed locally
        self.show_dialog = False
        self._checked = False
        self._ad_index = 0
        self._ad_next_time = 0.0      # when to show next ad
        self._ad_count = 0            # how many ads we've shown (cap it)

    def check_for_updates_async(self):
        threading.Thread(
            target=self._check_worker, daemon=True, name="AddonUpdateCheckThread"
        ).start()

    def _check_worker(self):
        try:
            resp = requests.get(
                self.MANIFEST_URL,
                headers={"User-Agent": "FunGen-Updater/1.0"},
                timeout=10,
            )
            resp.raise_for_status()
            manifest = resp.json()
        except Exception:
            return

        outdated = []
        unowned = []
        for addon_name, info in manifest.get("addons", {}).items():
            try:
                mod = __import__(addon_name)
                local_version = getattr(mod, "__version__", None)
                if local_version is None:
                    continue
            except ImportError:
                # Addon not installed — candidate for status strip ad
                unowned.append({
                    "name": addon_name,
                    "display_name": info.get("display_name", addon_name),
                    "remote_version": info.get("version", ""),
                    "changelog": info.get("changelog", ""),
                })
                continue

            remote_version = info.get("version", "")
            try:
                local_tuple = tuple(int(x) for x in local_version.split("."))
                remote_tuple = tuple(int(x) for x in remote_version.split("."))
            except (ValueError, AttributeError):
                continue

            if local_tuple < remote_tuple:
                outdated.append({
                    "name": addon_name,
                    "display_name": info.get("display_name", addon_name),
                    "local_version": local_version,
                    "remote_version": remote_version,
                    "changelog": info.get("changelog", ""),
                })

        if outdated:
            self.updates_available = outdated
            self.show_dialog = True

        if unowned:
            self.unowned_addons = unowned
            self._ad_next_time = time.time() + self._AD_INITIAL_DELAY

        self._checked = True

    # ------------------------------------------------------------------
    # Status strip ads for unowned addons (called each frame from GUI)
    # ------------------------------------------------------------------
    def tick_status_ads(self):
        """Call once per frame. Posts a status-strip message for unowned addons."""
        if not self.unowned_addons or self._ad_count >= len(self.unowned_addons):
            return
        now = time.time()
        if now < self._ad_next_time:
            return
        # Don't overwrite an existing active status message
        app_state = getattr(self.app, "app_state_ui", None)
        if app_state is None:
            return
        if app_state.status_message and now < app_state.status_message_time:
            return

        addon = self.unowned_addons[self._ad_index % len(self.unowned_addons)]
        msg = (
            f"\u2605 {addon['display_name']} v{addon['remote_version']}"
            f" available — ko-fi.com/k00gar"
        )
        self.app.set_status_message(msg, duration=self._AD_DISPLAY_DURATION)

        self._ad_index += 1
        self._ad_count += 1
        self._ad_next_time = now + self._AD_CYCLE_INTERVAL

    # ------------------------------------------------------------------
    # Modal dialog for outdated installed addons
    # ------------------------------------------------------------------
    def render_update_dialog(self):
        if self.show_dialog:
            imgui.open_popup("Addon Updates Available")
            self.show_dialog = False

        if not imgui.is_popup_open("Addon Updates Available"):
            return

        main_viewport = imgui.get_main_viewport()
        popup_pos = (
            main_viewport.pos[0] + main_viewport.size[0] * 0.5,
            main_viewport.pos[1] + main_viewport.size[1] * 0.5,
        )
        imgui.set_next_window_position(
            popup_pos[0], popup_pos[1], pivot_x=0.5, pivot_y=0.5, condition=imgui.APPEARING
        )
        imgui.set_next_window_size(480, 0, condition=imgui.APPEARING)

        opened, _ = imgui.begin_popup_modal(
            "Addon Updates Available", True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE
        )
        if opened:
            imgui.text("Addon Updates Available")
            imgui.separator()
            imgui.spacing()

            for update in self.updates_available:
                imgui.text(
                    f"{update['display_name']}: v{update['local_version']}"
                    f" -> v{update['remote_version']}"
                )
                if update.get("changelog"):
                    imgui.push_style_color(imgui.COLOR_TEXT, 0.6, 0.6, 0.6, 1.0)
                    imgui.text_wrapped(f"  {update['changelog']}")
                    imgui.pop_style_color()
                imgui.spacing()

            imgui.separator()
            imgui.spacing()
            imgui.text_wrapped(
                "Download from Ko-fi (ko-fi.com/k00gar) or DM k00gar on Discord"
                " if registered with the bot."
            )
            imgui.spacing()

            button_width = 120
            window_width = imgui.get_window_width()
            imgui.set_cursor_pos_x((window_width - button_width) * 0.5)
            if imgui.button("OK", width=button_width):
                self.updates_available = []
                imgui.close_current_popup()

            imgui.end_popup()
