import threading
import subprocess
import os
import sys
import requests
import imgui
import time
import unicodedata
import re
from typing import List, Dict
from application.utils.github_token_manager import GitHubTokenManager


class AutoUpdater:
    """
    Handles checking for and applying updates from a Git repository.
    
    Features:
    - Automatic update checking against the configured branch (v0.5.0)
    - Manual version selection from available commits
    - Changelog generation for selected commits
    - Background async operations to avoid UI blocking
    
    Version Picker Usage:
    - Access via Updates menu -> "Select Update Commit"
    - Shows all available commits from the current branch
    - Highlights current version in green
    - Allows switching to any commit (upgrade or downgrade)
    - Shows commit details and messages
    - Automatically restarts application after version change
    
    Requirements:
    - Git must be installed and in PATH
    - Repository must be a valid Git repository
    - Internet connection for GitHub API calls
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
        
        # Commit dates for display (cached to avoid repeated API calls)
        self.local_commit_date = None
        self.remote_commit_date = None
        
        # Version picker state
        self.available_versions = []
        self.selected_version = None
        self.version_picker_loading = False

        # Inline changelog state for each commit
        self.expanded_commits = set()  # Set of commit hashes that are expanded
        self.commit_changelogs = {}    # Cache for commit changelog data
        
        # GitHub token manager
        self.token_manager = GitHubTokenManager()

    def _get_local_commit_hash(self) -> str | None:
        """Gets the commit hash of the local repository's target branch (v0.5.0)."""
        try:
            if not os.path.isdir('.git'):
                self.logger.warning("Not a git repository. Skipping update check.")
                return None
            result = subprocess.run(
                ['git', 'rev-parse', self.BRANCH],
                capture_output=True, text=True, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.error(f"Could not get local git hash for branch {self.BRANCH}: {e}")
            self.status_message = f"Could not determine local version for branch {self.BRANCH}."
            return None

    def _get_remote_commit_hash(self) -> str | None:
        """Gets the latest commit hash from the remote repository via GitHub API."""
        try:
            headers = {
                'User-Agent': 'FunGen-Updater/1.0',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            github_token = self.token_manager.get_token()
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            
            response = requests.get(self.API_URL, headers=headers, timeout=10)
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
            headers = {
                'User-Agent': 'FunGen-Updater/1.0',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            github_token = self.token_manager.get_token()
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            
            response = requests.get(compare_url, headers=headers, timeout=10)
            
            # Handle 404 errors (usually when comparing commits from different branches)
            if response.status_code == 404:
                self.logger.warning(f"Could not compare commits {local_hash[:7]} and {remote_hash[:7]} - they may be from different branches")
                # Fallback: get recent commits from the target branch
                return self._get_recent_commits_from_branch()
            
            response.raise_for_status()
            data = response.json()
            commits = data.get('commits', [])
            for commit_data in commits:
                message = commit_data.get('commit', {}).get('message', 'No commit message.')
                # Get the first line of the commit message for brevity
                changelog.append(message)
            return changelog
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch commit diff: {e}")
            return ["Could not retrieve update details."]

    def _get_recent_commits_from_branch(self) -> list[str]:
        """Gets recent commits from the target branch as a fallback when direct comparison fails."""
        try:
            commits_url = f"https://api.github.com/repos/{self.REPO_OWNER}/{self.REPO_NAME}/commits?sha={self.BRANCH}&per_page=10"
            
            headers = {
                'User-Agent': 'FunGen-Updater/1.0',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            github_token = self.token_manager.get_token()
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            
            response = requests.get(commits_url, headers=headers, timeout=10)
            response.raise_for_status()
            commits_data = response.json()
            
            changelog = [f"Recent commits from {self.BRANCH} branch:"]
            for commit in commits_data[:5]:  # Show last 5 commits
                message = commit.get('commit', {}).get('message', 'No commit message.')
                # Get the first line of the commit message
                first_line = message.split('\n')[0]
                changelog.append(f"- {first_line}")
            
            return changelog
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch recent commits: {e}")
            return ["Could not retrieve recent commit information."]

    def _get_commit_date(self, commit_hash: str) -> str:
        """Gets the commit date for a given commit hash."""
        try:
            commit_url = f"https://api.github.com/repos/{self.REPO_OWNER}/{self.REPO_NAME}/commits/{commit_hash}"
            
            headers = {
                'User-Agent': 'FunGen-Updater/1.0',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            github_token = self.token_manager.get_token()
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            
            response = requests.get(commit_url, headers=headers, timeout=10)
            response.raise_for_status()
            commit_data = response.json()
            
            # Extract date from commit info
            commit_info = commit_data.get('commit', {})
            author_info = commit_info.get('author', {})
            date_str = author_info.get('date', 'Unknown date')
            
            # Parse and format the date
            try:
                from datetime import datetime
                # GitHub dates are in ISO format: 2024-01-15T10:30:00Z
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                # Format as YYYY-MM-DD HH:MM
                return date_obj.strftime('%Y-%m-%d %H:%M')
            except ValueError:
                # If date parsing fails, return the original string
                return date_str
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch commit date for {commit_hash[:7]}: {e}")
            return 'Unknown date'
        except Exception as e:
            self.logger.error(f"Unexpected error getting commit date: {e}")
            return 'Unknown date'

    def _check_worker(self):
        """Worker thread to check for updates and fetch changelog."""
        self.local_commit_hash = self._get_local_commit_hash()
        if self.local_commit_hash:
            self.remote_commit_hash = self._get_remote_commit_hash()

        if self.local_commit_hash and self.remote_commit_hash:
            # Cache commit dates to avoid repeated API calls in the UI
            self.local_commit_date = self._get_commit_date(self.local_commit_hash)
            self.remote_commit_date = self._get_commit_date(self.remote_commit_hash)
            
            if self.local_commit_hash != self.remote_commit_hash:
                self.logger.info("Update available.")
                self.status_message = "A new update is available!"
                self.update_available = True
                # Fetch the changelog
                self.update_changelog = self._get_commit_diff(self.local_commit_hash, self.remote_commit_hash)
                if not self.app.app_settings.get("updater_suppress_popup", False):
                    self.show_update_dialog = True
            else:
                self.logger.info("Application is up to date.")
                self.status_message = "You are on the latest version."
                self.update_changelog = []

        self.update_check_complete = True

    def check_for_updates_async(self):
        """Starts the update check in a background thread and updates the timestamp."""
        self.last_check_time = time.time() # Update time when a check is initiated
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

            # Give a moment for the message to be seen
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
            self.show_update_dialog = False

        # Set initial position for first time
        if not hasattr(self, '_update_dialog_pos'):
            main_viewport = imgui.get_main_viewport()
            popup_pos = (main_viewport.pos[0] + main_viewport.size[0] * 0.5,
                         main_viewport.pos[1] + main_viewport.size[1] * 0.5)
            self._update_dialog_pos = (popup_pos[0] - 250, popup_pos[1] - 150)  # Center the window

        # Allow width to grow/shrink but keep a minimum width; let height be auto
        imgui.set_next_window_size_constraints((500, 0), (float("inf"), float("inf")))
        imgui.set_next_window_position(*self._update_dialog_pos, condition=imgui.ONCE)

        # Begin popup modal (still with auto-resize flag for height)
        if imgui.begin_popup_modal("Update Available", True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            # Save window position for persistence
            window_pos = imgui.get_window_position()
            if window_pos[0] > 0 and window_pos[1] > 0:
                self._update_dialog_pos = window_pos
            
            if self.update_in_progress:
                imgui.text(self.status_message)
                spinner_chars = "|/-\\"
                spinner_index = int(time.time() * 4) % 4
                imgui.text(f"Processing... {spinner_chars[spinner_index]}")
            else:
                imgui.text("A new version is available for FunGen.")
                imgui.text("Would you like to update and restart the application?")
                imgui.separator()

                if self.update_changelog:
                    imgui.text("Changes in this update:")

                    # Fill width, scroll if needed
                    child_width = imgui.get_content_region_available()[0]
                    child_height = 180
                    imgui.begin_child("Changelog", child_width, child_height, border=True, flags=imgui.WINDOW_HORIZONTAL_SCROLLING_BAR
                            | imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
                    for message in self.update_changelog:
                        imgui.text_wrapped(self.clean_text(f"- {message}"))
                    imgui.end_child()

                # Use cached commit dates to avoid repeated API calls
                local_date = self.local_commit_date if self.local_commit_date else 'N/A'
                remote_date = self.remote_commit_date if self.remote_commit_date else 'N/A'
                
                imgui.text_wrapped(f"Your Version: {self.local_commit_hash[:7] if self.local_commit_hash else 'N/A'} ({local_date})")
                imgui.text_wrapped(
                    f"Latest Version: {self.remote_commit_hash[:7] if self.remote_commit_hash else 'N/A'} ({remote_date})")
                imgui.separator()
                if imgui.button("Update and Restart", width=200):
                    self.apply_update_and_restart()
                imgui.same_line()

                if imgui.button("Later", width=100):
                    self.update_available = False
                    imgui.close_current_popup()

            imgui.end_popup()

    def _get_current_branch_name(self) -> str:
        """Gets the current branch name from git."""
        try:
            if not os.path.isdir('.git'):
                self.logger.warning("Not a git repository.")
                return self.BRANCH  # Fallback to default
            
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                capture_output=True, text=True, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.error(f"Could not get current branch: {e}")
            return self.BRANCH  # Fallback to default

    def _get_available_versions(self) -> List[Dict]:
        """Fetches available commits from the configured branch (v0.5.0)."""
        versions = []

        try:
            # Use the configured branch instead of current branch
            target_branch = self.BRANCH
            self.logger.info(f"Fetching versions from branch: {target_branch}")
            
            # Fetch commits from the configured branch
            commits_url = f"https://api.github.com/repos/{self.REPO_OWNER}/{self.REPO_NAME}/commits?sha={target_branch}"
            
            # Add headers for better rate limit handling
            headers = {
                'User-Agent': 'FunGen-Updater/1.0',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            # Check if we have a GitHub token for higher rate limits
            github_token = self.token_manager.get_token()
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            
            response = requests.get(commits_url, headers=headers, timeout=10)
            
            # Handle rate limiting
            if response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
                self.logger.error("GitHub API rate limit exceeded")
                return [{'name': 'Rate limit exceeded. Please set GitHub token in Settings.', 'commit_hash': 'rate_limit', 'type': 'error', 'date': '', 'full_message': ''}]
            
            response.raise_for_status()
            commits_data = response.json()
            
            for commit in commits_data:
                try:
                    sha = commit.get('sha')
                    if not sha:
                        continue
                    
                    # Get commit details
                    commit_info = commit.get('commit', {})
                    author_info = commit_info.get('author', {})
                    date = author_info.get('date')
                    message = commit_info.get('message', 'No commit message')
                    
                    # Get first line of commit message for display
                    first_line = message.split('\n')[0] if message else 'No commit message'
                    
                    if sha and date:
                        versions.append({
                            'name': first_line,
                            'commit_hash': sha,
                            'type': 'commit',
                            'date': date,
                            'full_message': message
                        })
                except (KeyError, TypeError) as e:
                    self.logger.warning(f"Skipping malformed commit data: {e}")
                    continue
            
            # Sort by date (newest first)
            versions.sort(key=lambda x: x.get('date', 'Unknown'), reverse=True)
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch available versions: {e}")
            return []

        return versions

    def _checkout_version(self, commit_hash: str) -> bool:
        """Checks out a specific commit."""
        try:
            # Check if we're in a git repository
            if not os.path.isdir('.git'):
                self.logger.error("Not a git repository. Cannot checkout version.")
                return False
            
            # Fetch latest changes first
            subprocess.run(
                ['git', 'fetch', 'origin'],
                check=True, capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            # Checkout the specific commit
            subprocess.run(
                ['git', 'checkout', commit_hash],
                check=True, capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to checkout commit {commit_hash}: {e.stderr}")
            return False
        except FileNotFoundError:
            self.logger.error("Git not found. Please ensure Git is installed and in PATH.")
            return False

    def clean_text(self, s: str) -> str:
        if not s:
            return ""
            
        # Normalize to NFC (composed form), then remove problematic characters
        cleaned = unicodedata.normalize("NFC", s)

        # Remove replacement characters (), BOM, ZWSP, NBSP, etc.
        cleaned = cleaned.replace('\ufffd', '')  # Replacement character
        cleaned = cleaned.replace('\ufeff', '')  # BOM
        cleaned = cleaned.replace('\u200b', '')  # ZWSP
        cleaned = cleaned.replace('\u00a0', ' ')  # NBSP → space

        # Remove other control characters and question marks that might be replacement chars
        cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned)
        cleaned = re.sub(r'^\?+', '', cleaned)  # Remove leading question marks
        cleaned = re.sub(r'\?+$', '', cleaned)  # Remove trailing question marks

        # Strip any remaining invalid UTF-8 bytes silently
        cleaned = cleaned.encode('utf-8', 'ignore').decode('utf-8', 'ignore')

        # Final cleanup - remove any remaining problematic characters
        cleaned = re.sub(r'[^\x20-\x7E\xA0-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF\u2C60-\u2C7F\uA720-\uA7FF]', '', cleaned)

        return cleaned.strip()


    def _get_version_diff(self, target_hash: str) -> List[str]:
        """Gets commit details for the specified commit hash."""
        current_hash = self._get_local_commit_hash()
        if not current_hash:
            return ["Could not determine current version."]
        
        # Get the selected commit details (even if it's the current commit)
        try:
            commit_url = f"https://api.github.com/repos/{self.REPO_OWNER}/{self.REPO_NAME}/commits/{target_hash}"
            
            # Add headers for better rate limit handling
            headers = {
                'User-Agent': 'FunGen-Updater/1.0',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            # Check if we have a GitHub token for higher rate limits
            github_token = self.token_manager.get_token()
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            
            response = requests.get(commit_url, headers=headers, timeout=10)
            
            # Handle rate limiting
            if response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
                return ["GitHub API rate limit exceeded. Please wait or set GitHub token in Settings."]
            
            response.raise_for_status()
            commit_data = response.json()

            # Get author info - prefer GitHub username, fallback to commit author name
            author_data = commit_data.get('author')
            commit_info = commit_data.get('commit', {})
            commit_author = commit_info.get('author', {})
            
            # Use GitHub username if available, otherwise use commit author name
            author = author_data.get('login') if author_data else commit_author.get('name', 'Unknown')
            
            message = commit_info.get('message', 'No commit message')
            date = commit_author.get('date', 'Unknown date')

            # Split message into lines and format nicely
            message_lines = message.split('\n')
            changelog = []
            
            # Add current version indicator if this is the current commit
            if current_hash == target_hash:
                changelog.append("*** CURRENT VERSION ***")
                changelog.append("")
            
            changelog.append(f"Commit: {target_hash[:7]}")
            changelog.append(f"Author: {self.clean_text(author)}")
            changelog.append(f"Date:   {date}")
            changelog.append("")
            changelog.append("Message: ")
            for line in message_lines:
                cleaned_line = self.clean_text(line)
                if cleaned_line.strip():
                    changelog.append(f"  {cleaned_line}")

            return changelog
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch commit details: {e}")
            return [f"Could not retrieve commit details: {str(e)}"]
        except Exception as e:
            self.logger.error(f"Unexpected error in _get_version_diff: {e}")
            return [f"Error retrieving commit details: {str(e)}"]

    def load_available_versions_async(self):
        """Loads available versions in a background thread."""
        self.logger.info("Starting async version loading")
        self.version_picker_loading = True
        threading.Thread(target=self._load_versions_worker, daemon=True).start()

    def _load_versions_worker(self):
        """Worker thread to load available versions."""
        self.logger.info("Loading versions in worker thread")
        self.available_versions = self._get_available_versions()
        self.logger.info(f"Loaded {len(self.available_versions)} versions")
        self.version_picker_loading = False

    def apply_version_change(self, commit_hash: str, commit_message: str):
        """Applies the selected version change and restarts the application."""
        self.update_in_progress = True
        self.status_message = f"Switching to commit: {commit_message[:50]}..."
        self.logger.info(f"Attempting to checkout commit {commit_hash}")

        try:
            success = self._checkout_version(commit_hash)
            if success:
                self.logger.info(f"Successfully checked out commit {commit_hash}")
                self.status_message = "Version change complete. Restarting..."
                time.sleep(2)
                os.execl(sys.executable, sys.executable, *sys.argv)
            else:
                self.status_message = "Version change failed. Please check console."
                self.update_in_progress = False
                
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during version change: {e}")
            self.status_message = "An unexpected error occurred. See logs."
            self.update_in_progress = False

 