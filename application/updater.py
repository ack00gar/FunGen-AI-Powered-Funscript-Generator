import threading
import subprocess
import os
import sys
import requests
import imgui
import time


class AutoUpdater:
    """
    Handles checking for and applying updates from a Git repository, with
    intelligent handling of local user changes and an improved UI.
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
        self.last_check_time = 0
        self.update_changelog = []

        self.update_in_progress = False
        self.has_local_changes = False
        self.show_local_changes_prompt = False
        self.update_error_message = ""

    def _run_git_command(self, command: list[str]) -> subprocess.CompletedProcess:
        """Helper to run git commands without a console window on Windows."""
        return subprocess.run(
            command,
            capture_output=True, text=True, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )

    def _get_local_commit_hash(self) -> str | None:
        """Gets the commit hash of the local repository's HEAD."""
        try:
            if not os.path.isdir('.git'):
                self.logger.warning("Not a git repository. Skipping update check.")
                return None
            result = self._run_git_command(['git', 'rev-parse', 'HEAD'])
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

    def _get_commit_diff(self, local_hash: str, remote_hash: str) -> list[str]:
        """Gets commit messages between local and remote versions."""
        compare_url = f"https://api.github.com/repos/{self.REPO_OWNER}/{self.REPO_NAME}/compare/{local_hash}...{remote_hash}"
        changelog = []
        try:
            response = requests.get(compare_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            commits = data.get('commits', [])
            for commit_data in commits:
                message = commit_data.get('commit', {}).get('message', 'No commit message.')
                changelog.append(message)
            return changelog
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch commit diff: {e}")
            return ["Could not retrieve update details."]

    def _check_for_local_changes(self) -> bool:
        """Checks if there are uncommitted changes in the local repository."""
        try:
            result = self._run_git_command(['git', 'status', '--porcelain'])
            return bool(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error("Could not check for local changes (git command failed).")
            return False

    def _check_worker(self):
        """Worker thread to check for updates and fetch changelog."""
        self.local_commit_hash = self._get_local_commit_hash()
        if self.local_commit_hash:
            self.remote_commit_hash = self._get_remote_commit_hash()

        if self.local_commit_hash and self.remote_commit_hash:
            if self.local_commit_hash != self.remote_commit_hash:
                self.logger.info("Update available.")
                self.status_message = "A new update is available!"
                self.update_available = True
                self.update_changelog = self._get_commit_diff(self.local_commit_hash, self.remote_commit_hash)
                if not self.app.app_settings.get("updater_suppress_popup", False):
                    self.show_update_dialog = True
            else:
                self.logger.info("Application is up to date.")
                self.status_message = "You are on the latest version."
                self.update_changelog = []
        self.update_check_complete = True

    def check_for_updates_async(self):
        """Starts the update check in a background thread."""
        self.last_check_time = time.time()
        threading.Thread(target=self._check_worker, daemon=True).start()

    def initiate_update(self):
        """First step of the update process. Checks for local changes."""
        self.update_error_message = ""
        self.has_local_changes = self._check_for_local_changes()
        if self.has_local_changes:
            self.logger.info("Local changes detected. Prompting user for action.")
            self.show_local_changes_prompt = True
        else:
            self.logger.info("No local changes detected. Proceeding with standard update.")
            self.apply_update_and_restart()

    def apply_update_and_restart(self, stash_changes: bool = False, discard_changes: bool = False):
        """Pulls latest changes and restarts. This is threaded to not block the UI."""
        self.update_in_progress = True
        self.show_local_changes_prompt = False
        args = (stash_changes, discard_changes)
        threading.Thread(target=self._apply_update_worker, args=args, daemon=True).start()

    def _apply_update_worker(self, stash_changes: bool, discard_changes: bool):
        """Worker thread for performing the git operations with rollback safety."""
        original_commit_hash = self.local_commit_hash

        try:
            if discard_changes:
                self.status_message = "Discarding local changes..."
                self.logger.info("Discarding local changes with 'git reset --hard'")
                self._run_git_command(['git', 'reset', '--hard', 'HEAD'])

            elif stash_changes:
                self.status_message = "Stashing local changes..."
                self.logger.info("Stashing local changes with 'git stash push'")
                self._run_git_command(['git', 'stash', 'push', '-m', 'FunGen Auto-Stash'])

            self.status_message = "Pulling updates from remote..."
            self.logger.info(f"Attempting to pull updates from origin/{self.BRANCH}...")
            self._run_git_command(['git', 'pull', 'origin', self.BRANCH])

            if stash_changes:
                self.status_message = "Re-applying local changes..."
                self.logger.info("Attempting to re-apply stashed changes with 'git stash pop'...")
                try:
                    self._run_git_command(['git', 'stash', 'pop'])
                    self.logger.info("Stashed changes re-applied successfully.")
                except subprocess.CalledProcessError as stash_err:
                    self.logger.error(f"Could not auto-apply stashed changes: {stash_err.stderr}. Rolling back update.")
                    self.status_message = "Conflict detected. Rolling back update..."

                    # --- ATOMIC ROLLBACK LOGIC ---
                    self._run_git_command(['git', 'reset', '--hard', original_commit_hash])
                    self._run_git_command(['git', 'stash', 'pop'])  # This should now succeed.
                    self.logger.info("Successfully rolled back update and restored local changes.")

                    self.update_error_message = ("Update failed due to a conflict with your local changes.\n\n"
                                                 "The update has been cancelled, and your changes have been safely restored.\n"
                                                 "Please commit your changes or contact support if you need help.")
                    self.update_in_progress = False
                    return

            self.status_message = "Update complete. Restarting..."
            self.logger.info("Restarting application.")
            time.sleep(2)
            os.execl(sys.executable, sys.executable, *sys.argv)

        except subprocess.CalledProcessError as e:
            error_details = e.stderr or e.stdout
            self.logger.error(f"Update failed during git operation: {error_details}")
            self.update_error_message = f"Update failed. Please check logs or update manually.\n\nDetails: {error_details.splitlines()[0]}"
            self.update_in_progress = False
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during update: {e}", exc_info=True)
            self.update_error_message = "An unexpected critical error occurred. See logs."
            self.update_in_progress = False

    def render_update_dialog(self):
        """Renders the ImGui popup for the update, with an improved layout and states."""
        if self.show_update_dialog:
            imgui.open_popup("Update Available")
            self.show_update_dialog = False
            self.update_in_progress = False
            self.show_local_changes_prompt = False
            self.update_error_message = ""

        main_viewport = imgui.get_main_viewport()
        # Set size to 40% of viewport width and 60% of viewport height.
        imgui.set_next_window_size(main_viewport.size[0] * 0.4, main_viewport.size[1] * 0.6, condition=imgui.APPEARING)
        imgui.set_next_window_position(main_viewport.work_pos[0] + main_viewport.work_size[0] * 0.5,
                                       main_viewport.work_pos[1] + main_viewport.work_size[1] * 0.5,
                                       pivot_x=0.5, pivot_y=0.5, condition=imgui.APPEARING)

        popup_opened, _ = imgui.begin_popup_modal("Update Available", True)

        if popup_opened:
            window_width = imgui.get_content_region_available()[0]

            if self.update_in_progress:
                imgui.text(self.status_message)
                spinner_chars = "|/-\\"
                spinner_index = int(time.time() * 4) % 4
                imgui.text(f"Processing... {spinner_chars[spinner_index]}")

            elif self.update_error_message:
                imgui.text("Update Error")
                imgui.separator()
                imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.8, 0.8, 1.0)
                imgui.text_wrapped(self.update_error_message)
                imgui.pop_style_color()
                imgui.separator()
                if imgui.button("Close", width=window_width):
                    self.update_error_message = ""
                    self.update_available = False
                    imgui.close_current_popup()

            elif self.show_local_changes_prompt:
                imgui.text("Local Changes Detected!")
                imgui.separator()
                imgui.text_wrapped("You have uncommitted changes in your local files. "
                                   "How would you like to proceed with the update?")
                imgui.spacing()

                if imgui.button("Stash and Update", width=window_width):
                    self.logger.info("User chose to Stash and Update.")
                    self.apply_update_and_restart(stash_changes=True)
                imgui.text_wrapped("Recommended: Saves your changes, updates, then tries to re-apply them. "
                                   "If there's a conflict, the update will be safely cancelled.")
                imgui.spacing()

                if imgui.button("Discard and Update", width=window_width):
                    self.logger.info("User chose to Discard and Update.")
                    self.apply_update_and_restart(discard_changes=True)
                imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 0.6, 1.0)
                imgui.text_wrapped("Warning: This will permanently delete your local changes.")
                imgui.pop_style_color()
                imgui.spacing()
                imgui.separator()

                if imgui.button("Cancel", width=120):
                    self.show_local_changes_prompt = False

            else:
                imgui.text("A new version of FunGen is available!")
                imgui.text("Would you like to download and install the update?")
                imgui.separator()

                if self.update_changelog:
                    imgui.text("Changes in this update:")
                    # This child window will now correctly scroll within the parent's fixed size
                    imgui.begin_child("Changelog", 0, -50, border=True)
                    for message in self.update_changelog:
                        imgui.bullet()
                        imgui.same_line()
                        imgui.text_wrapped(message)
                    imgui.end_child()

                imgui.text(f"Your Version:   {self.local_commit_hash[:7] if self.local_commit_hash else 'N/A'}")
                imgui.text(f"Latest Version: {self.remote_commit_hash[:7] if self.remote_commit_hash else 'N/A'}")
                imgui.separator()

                # --- Button Layout at the bottom ---
                button_width = (window_width - imgui.get_style().item_spacing[0]) / 2

                if imgui.button("Update and Restart", width=button_width):
                    self.initiate_update()

                imgui.same_line()

                if imgui.button("Later", width=button_width):
                    self.update_available = False
                    imgui.close_current_popup()

            imgui.end_popup()

        if not popup_opened:
            self.update_available = False
            self.show_local_changes_prompt = False
            self.update_in_progress = False
