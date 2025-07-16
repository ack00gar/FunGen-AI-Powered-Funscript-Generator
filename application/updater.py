import threading
import subprocess
import os
import sys
import requests
import imgui
import time


class AutoUpdater:
    """
    Handles checking for and applying updates from a Git repository.
    """
    REPO_OWNER = "ack00gar"
    REPO_NAME = "FunGen-AI-Powered-Funscript-Generator"
    BRANCH = "v0.5.0"
    API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/commits/{BRANCH}"

    def __init__(self, app_logic):
        self.app = app_logic
        self.logger = self.app.logger
        self.local_commit_hash = ""
        self.remote_commit_hash = ""
        self.update_available = False
        self.update_check_complete = False
        self.status_message = "Checking for updates..."
        self.show_update_dialog = False
        self.update_in_progress = False

    def _get_local_commit_hash(self) -> str | None:
        """Gets the commit hash of the local repository's HEAD."""
        try:
            if not os.path.isdir('.git'):
                self.logger.warning("Not a git repository. Skipping update check.")
                return None
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.error(f"Could not get local git hash: {e}")
            self.status_message = "Could not determine local version (Git not found or not a repo)."
            return None

    def _get_remote_commit_hash(self) -> str | None:
        """Gets the latest commit hash from the remote repository via GitHub API."""
        try:
            response = requests.get(self.API_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('sha')
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch remote version: {e}")
            self.status_message = "Could not connect to check for updates."
            return None

    def _check_worker(self):
        """Worker thread to check for updates without blocking the UI."""
        self.local_commit_hash = self._get_local_commit_hash()
        if self.local_commit_hash:
            self.remote_commit_hash = self._get_remote_commit_hash()

        if self.local_commit_hash and self.remote_commit_hash:
            if self.local_commit_hash != self.remote_commit_hash:
                self.logger.info("Update available.")
                self.status_message = "A new update is available!"
                self.update_available = True
                self.show_update_dialog = True
            else:
                self.logger.info("Application is up to date.")
                self.status_message = "You are on the latest version."

        self.update_check_complete = True

    def check_for_updates_async(self):
        """Starts the update check in a background thread."""
        threading.Thread(target=self._check_worker, daemon=True).start()

    def apply_update_and_restart(self):
        """Pulls the latest changes from git and restarts the application."""
        self.update_in_progress = True
        self.status_message = "Pulling updates..."
        self.logger.info("Attempting to pull updates from origin...")

        try:
            pull_result = subprocess.run(
                ['git', 'pull', 'origin', self.BRANCH],
                check=True, capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            self.logger.info(f"Git pull successful: {pull_result.stdout}")
            self.status_message = "Update complete. Restarting..."

            # Give a moment for the message to be seen (optional)
            time.sleep(2)

            # Restart the application
            os.execl(sys.executable, sys.executable, *sys.argv)

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Update failed during 'git pull': {e.stderr}")
            self.status_message = "Update failed. Please check console or update manually."
            self.update_in_progress = False
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during update: {e}")
            self.status_message = "An unexpected error occurred. See logs."
            self.update_in_progress = False

    def render_update_dialog(self):
        """Renders the ImGui popup for the update confirmation."""
        if self.show_update_dialog:
            imgui.open_popup("Update Available")
            self.show_update_dialog = False  # Prevent re-opening every frame

        main_viewport = imgui.get_main_viewport()
        popup_pos = (main_viewport.pos[0] + main_viewport.size[0] * 0.5,
                     main_viewport.pos[1] + main_viewport.size[1] * 0.5)
        imgui.set_next_window_position(popup_pos[0], popup_pos[1], pivot_x=0.5, pivot_y=0.5)

        if imgui.begin_popup_modal("Update Available", True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            if self.update_in_progress:
                imgui.text(self.status_message)
                spinner_chars = "|/-\\"
                spinner_index = int(time.time() * 4) % 4
                imgui.text(f"Processing... {spinner_chars[spinner_index]}")
            else:
                imgui.text("A new version is available for FunGen.")
                imgui.text("Would you like to update and restart the application?")
                imgui.separator()
                imgui.text_wrapped(f"Your Version: {self.local_commit_hash[:7] if self.local_commit_hash else 'N/A'}")
                imgui.text_wrapped(
                    f"Latest Version: {self.remote_commit_hash[:7] if self.remote_commit_hash else 'N/A'}")
                imgui.separator()

                if imgui.button("Update and Restart", width=150):
                    self.apply_update_and_restart()

                imgui.same_line()

                if imgui.button("Later", width=100):
                    self.update_available = False  # Don't ask again this session
                    imgui.close_current_popup()

            imgui.end_popup()
