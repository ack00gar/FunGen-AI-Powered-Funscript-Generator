import threading
import time
import requests


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

        # Show outdated addon updates as toast notifications
        for update in outdated:
            msg = f"{update['display_name']}: v{update['local_version']} -> v{update['remote_version']}"
            if hasattr(self.app, 'notify'):
                self.app.notify(msg, "warning", 8.0)

        # Show available addons as info toasts (one per addon, once)
        for addon in unowned:
            msg = f"{addon['display_name']} v{addon['remote_version']} available at paypal.me/k00gar"
            if hasattr(self.app, 'notify'):
                self.app.notify(msg, "info", 6.0)

        self._checked = True

    # ------------------------------------------------------------------
    # Status strip ads for unowned addons (called each frame from GUI)
    # ------------------------------------------------------------------
    def tick_status_ads(self):
        """No-op. Addon notifications now use toast system."""
        pass

    def render_update_dialog(self):
        """No-op. Addon updates now shown as toast notifications."""
        pass
