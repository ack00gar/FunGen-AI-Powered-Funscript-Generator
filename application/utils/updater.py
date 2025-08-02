import threading
import subprocess
import os
import sys
import requests
import imgui
import time
import unicodedata
import re
from typing import List, Dict, Optional
from application.utils.github_token_manager import GitHubTokenManager
from config.constants import DEFAULT_COMMIT_FETCH_COUNT


class GitHubAPIClient:
    """Centralized GitHub API client to reduce code duplication."""
    
    def __init__(self, repo_owner: str, repo_name: str, token_manager: GitHubTokenManager):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.token_manager = token_manager
        self.base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        
    def _get_headers(self) -> Dict[str, str]:
        """Get common headers for GitHub API requests."""
        headers = {
            'User-Agent': 'FunGen-Updater/1.0',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        github_token = self.token_manager.get_token()
        if github_token:
            headers['Authorization'] = f'token {github_token}'
            
        return headers
    
    def _make_request(self, endpoint: str, timeout: int = 10) -> Optional[Dict]:
        """Make a GitHub API request with common error handling."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, headers=self._get_headers(), timeout=timeout)
            
            # Check rate limit headers
            remaining = response.headers.get('X-RateLimit-Remaining')
            limit = response.headers.get('X-RateLimit-Limit')
            reset_time = response.headers.get('X-RateLimit-Reset')
            
            # Log rate limit info for debugging
            if remaining and limit:
                print(f"GitHub API: {remaining}/{limit} requests remaining")
            
            # Handle rate limiting properly
            if response.status_code == 403:
                if remaining == '0':
                    # Rate limit truly exceeded
                    reset_timestamp = int(reset_time) if reset_time else None
                    if reset_timestamp:
                        from datetime import datetime
                        reset_time_str = datetime.fromtimestamp(reset_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                        error_msg = f"GitHub API rate limit exceeded. Reset at {reset_time_str}"
                    else:
                        error_msg = "GitHub API rate limit exceeded"
                    raise requests.RequestException(error_msg)
                else:
                    # 403 but not rate limit - could be other issues
                    raise requests.RequestException(f"GitHub API 403 error: {response.text}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            # Log error but don't raise - let caller handle None return
            return None
    
    def get_branch_commit(self, branch: str) -> Optional[Dict]:
        """Get the latest commit for a specific branch."""
        return self._make_request(f"/commits/{branch}")
    
    def get_commit_details(self, commit_hash: str) -> Optional[Dict]:
        """Get detailed information for a specific commit."""
        return self._make_request(f"/commits/{commit_hash}")
    
    def get_commits_list(self, branch: str, per_page: int = None) -> Optional[List[Dict]]:
        """Get a list of commits for a branch."""
        if per_page is None:
            per_page = DEFAULT_COMMIT_FETCH_COUNT
        return self._make_request(f"/commits?sha={branch}&per_page={per_page}")
    
    def compare_commits(self, base_hash: str, head_hash: str) -> Optional[Dict]:
        """Compare two commits."""
        return self._make_request(f"/compare/{base_hash}...{head_hash}")


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
        
        # GitHub token manager and API client
        self.token_manager = GitHubTokenManager()
        self.github_api = GitHubAPIClient(self.REPO_OWNER, self.REPO_NAME, self.token_manager)

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
        commit_data = self.github_api.get_branch_commit(self.BRANCH)
        if commit_data:
            return commit_data.get('sha')
        else:
            self.logger.error("Failed to fetch remote version")
            self.status_message = "Could not connect to check for updates."
            return None

    def _get_commit_diff(self, local_hash: str, remote_hash: str) -> list[str]:
        """Gets commit messages between local and remote versions."""
        compare_data = self.github_api.compare_commits(local_hash, remote_hash)
        
        if compare_data is None:
            # Handle 404 errors (usually when comparing commits from different branches)
            self.logger.warning(f"Could not compare commits {local_hash[:7]} and {remote_hash[:7]} - they may be from different branches")
            # Fallback: get recent commits from the target branch
            return self._get_recent_commits_from_branch()

        changelog = []
        commits = compare_data.get('commits', [])
        for commit_data in commits:
            message = commit_data.get('commit', {}).get('message', 'No commit message.')
            # Get the first line of the commit message for brevity
            changelog.append(message)
        return changelog

    def _get_recent_commits_from_branch(self) -> list[str]:
        """Gets recent commits from the target branch as a fallback when direct comparison fails."""
        commits_data = self.github_api.get_commits_list(self.BRANCH, per_page=DEFAULT_COMMIT_FETCH_COUNT)
        
        if commits_data is None:
            self.logger.error("Failed to fetch recent commits")
            return ["Could not retrieve recent commit information."]
        
        changelog = [f"Recent commits from {self.BRANCH} branch:"]
        for commit in commits_data[:5]:  # Show last 5 commits
            message = commit.get('commit', {}).get('message', 'No commit message.')
            # Get the first line of the commit message
            first_line = message.split('\n')[0]
            changelog.append(f"- {first_line}")
        
        return changelog

    def _get_commit_date(self, commit_hash: str) -> str:
        """Gets the commit date for a given commit hash."""
        commit_data = self.github_api.get_commit_details(commit_hash)
        
        if commit_data is None:
            self.logger.error(f"Failed to fetch commit date for {commit_hash[:7]}")
            return 'Unknown date'
        
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

    def _get_available_versions(self, custom_count: int = None) -> List[Dict]:
        """Fetches available commits from the configured branch (v0.5.0)."""
        versions = []

        try:
            # Use the configured branch instead of current branch
            target_branch = self.BRANCH
            self.logger.info(f"Fetching versions from branch: {target_branch}")
            
            commits_data = self.github_api.get_commits_list(target_branch, per_page=custom_count)
            
            if commits_data is None:
                self.logger.error("Failed to fetch commits from GitHub API")
                return [{'name': 'Failed to fetch commits. Check network connection or GitHub token.', 'commit_hash': 'error', 'type': 'error', 'date': '', 'full_message': ''}]
            
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
            
        except Exception as e:
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
        commit_data = self.github_api.get_commit_details(target_hash)
        
        if commit_data is None:
            return ["Failed to fetch commit details. Check network connection or GitHub token."]

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

    def load_available_versions_async(self, custom_count: int = None):
        """Loads available versions in a background thread."""
        self.logger.info("Starting async version loading")
        self.version_picker_loading = True
        threading.Thread(target=self._load_versions_worker, args=(custom_count,), daemon=True).start()

    def _load_versions_worker(self, custom_count: int = None):
        """Worker thread to load available versions."""
        self.logger.info("Loading versions in worker thread")
        self.available_versions = self._get_available_versions(custom_count)
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

    def render_update_settings_dialog(self):
        """Renders the combined update commit & GitHub token dialog with tabs."""
        if self.app.app_state_ui.show_update_settings_dialog:
            imgui.open_popup("Updates & GitHub Token")
            self.app.app_state_ui.show_update_settings_dialog = False
            # Load versions when dialog opens
            self.load_available_versions_async()

        # Initialize buffers if needed
        if not hasattr(self, '_github_token_buffer'):
            self._github_token_buffer = self.token_manager.get_token()
        if not hasattr(self, '_updates_active_tab'):
            self._updates_active_tab = 0  # 0 = Version, 1 = Token

        # Set initial size and make resizable
        if not hasattr(self, '_update_settings_window_size'):
            self._update_settings_window_size = (800, 600)
        
        # Set initial position for first time
        if not hasattr(self, '_update_settings_window_pos'):
            main_viewport = imgui.get_main_viewport()
            popup_pos = (main_viewport.pos[0] + main_viewport.size[0] * 0.5,
                         main_viewport.pos[1] + main_viewport.size[1] * 0.5)
            self._update_settings_window_pos = (popup_pos[0] - 400, popup_pos[1] - 300)  # Center the window
        
        imgui.set_next_window_size(*self._update_settings_window_size, condition=imgui.ONCE)
        imgui.set_next_window_size_constraints((600, 400), (1200, 800))
        imgui.set_next_window_position(*self._update_settings_window_pos, condition=imgui.ONCE)

        if imgui.begin_popup_modal("Updates & GitHub Token", True)[0]:
            # Save window size and position for persistence
            window_size = imgui.get_window_size()
            window_pos = imgui.get_window_position()
            if window_size[0] > 0 and window_size[1] > 0:
                self._update_settings_window_size = window_size
            if window_pos[0] > 0 and window_pos[1] > 0:
                self._update_settings_window_pos = window_pos
            
            # Tab bar
            if imgui.begin_tab_bar("Updates & GitHub Token Tabs"):
                # Version Selection Tab
                if imgui.begin_tab_item("Choose FunGen Version")[0]:
                    self._updates_active_tab = 0
                    imgui.end_tab_item()
                
                # GitHub Token Tab
                if imgui.begin_tab_item("GitHub Token")[0]:
                    self._updates_active_tab = 1
                    imgui.end_tab_item()
                
                imgui.end_tab_bar()

            # Tab content
            if self._updates_active_tab == 0:
                # Version Selection Tab
                self._render_version_picker_content()
            else:
                # GitHub Token Tab
                self._render_github_token_content()

            imgui.separator()
            
            # Close button positioned at bottom right
            close_button_width = 80
            imgui.set_cursor_pos_x(imgui.get_window_width() - close_button_width - 10)  # Position from right edge
            if imgui.button("Close", width=close_button_width):
                imgui.close_current_popup()

            imgui.end_popup()

    def _render_version_picker_content(self):
        """Renders the version picker content within the tabbed dialog."""
        if self.version_picker_loading:
            imgui.text("Loading available commits...")
            spinner_chars = "|/-\\"
            spinner_index = int(time.time() * 4) % 4
            imgui.text(f"Please wait... {spinner_chars[spinner_index]}")
        elif self.update_in_progress:
            imgui.text(self.status_message)
            spinner_chars = "|/-\\"
            spinner_index = int(time.time() * 4) % 4
            imgui.text(f"Processing... {spinner_chars[spinner_index]}")
        else:
            target_branch = self.BRANCH
            imgui.text(f"Select a commit from branch '{target_branch}' to switch to:")
            imgui.separator()

            # Version list with inline changelogs
            child_width = imgui.get_content_region_available()[0]
            child_height = 400
            imgui.begin_child("VersionList", child_width, child_height, border=True, flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)

            current_hash = self._get_local_commit_hash()
            
            for version in self.available_versions:
                commit_hash = version['commit_hash']
                is_expanded = commit_hash in self.expanded_commits
                
                # Highlight current version
                is_current = current_hash and commit_hash.startswith(current_hash[:7])
                
                if is_current:
                    from config.element_group_colors import AppGUIColors
                    imgui.push_style_color(imgui.COLOR_TEXT, *AppGUIColors.VERSION_CURRENT_HIGHLIGHT)
                
                # Create expand/collapse button
                expand_icon = "▼" if is_expanded else "▶"
                button_text = f"{expand_icon} {commit_hash[:7]}"
                if imgui.button(button_text, width=80):
                    if is_expanded:
                        self.expanded_commits.discard(commit_hash)
                    else:
                        self.expanded_commits.add(commit_hash)
                        # Load changelog if not cached
                        if commit_hash not in self.commit_changelogs:
                            try:
                                changelog = self._get_version_diff(commit_hash)
                                self.commit_changelogs[commit_hash] = changelog
                            except Exception as e:
                                self.logger.error(f"Error loading changelog for {commit_hash[:7]}: {e}")
                                self.commit_changelogs[commit_hash] = [f"Error loading changelog: {str(e)}"]
                
                imgui.same_line()

                # Add commit date between hash and title
                commit_date = version.get('date', 'Unknown date')
                if commit_date != 'Unknown date':
                    try:
                        from datetime import datetime
                        # Parse GitHub date format and format as date only
                        date_obj = datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
                        commit_date = date_obj.strftime('%Y-%m-%d')
                    except:
                        commit_date = 'Unknown date'

                imgui.text(f"({commit_date})")
                imgui.same_line()
                
                # Commit message
                commit_msg = version['name']
                if len(commit_msg) > 60:
                    commit_msg = commit_msg[:57] + "..."
                
                # Calculate space for checkbox on the right
                window_width = imgui.get_window_width()
                checkbox_width = 20
                text_width = window_width - 200 - checkbox_width  # Leave space for button and checkbox
                
                if imgui.selectable(commit_msg, self.selected_version == version, flags=imgui.SELECTABLE_SPAN_ALL_COLUMNS)[0]:
                    self.selected_version = version
                
                # Add (Current) indicator after commit message
                if is_current:
                    imgui.same_line()
                    imgui.text("(Current)")
                    imgui.pop_style_color()
                
                # Add checkbox on the right (with proper positioning)
                imgui.same_line()
                # Position checkbox at the right edge with proper spacing
                checkbox_x = imgui.get_window_width() - 110  # pixels from right edge
                imgui.set_cursor_pos_x(checkbox_x)
                
                # Initialize ignored state if not present
                if 'ignored' not in version:
                    version['ignored'] = False
                
                # Make checkbox interactive
                changed, version['ignored'] = imgui.checkbox("##checkbox_" + commit_hash[:7], version['ignored'])
                imgui.same_line()
                imgui.text(f"Ignore")

                # Show inline changelog if expanded
                if is_expanded:
                    imgui.indent(30)
                    from config.element_group_colors import AppGUIColors
                    imgui.push_style_color(imgui.COLOR_TEXT, *AppGUIColors.VERSION_CHANGELOG_TEXT)
                    
                    changelog = self.commit_changelogs.get(commit_hash, [])
                    if not changelog:
                        imgui.text_wrapped("Loading changelog...")
                    else:
                        for line in changelog:
                            imgui.text_wrapped(self.clean_text(line))
                    
                    imgui.pop_style_color()
                    imgui.unindent(30)
                    imgui.separator()

            imgui.end_child()
            imgui.separator()

            # Action buttons and commit count controls
            if self.selected_version:
                if imgui.button("Switch to Commit", width=200):
                    self.apply_version_change(self.selected_version['commit_hash'], self.selected_version['name'])
            else:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                imgui.button("Switch to Commit", width=160)
                imgui.pop_style_var()
            
            # Add commit count controls on the same line, aligned to the right
            imgui.same_line()
            imgui.set_cursor_pos_x(imgui.get_window_width() - 235)
            
            imgui.text("Fetch commits:")
            imgui.same_line()
            
            # Initialize custom commit count if not set
            if not hasattr(self, '_custom_commit_count'):
                from config.constants import DEFAULT_COMMIT_FETCH_COUNT
                self._custom_commit_count = str(DEFAULT_COMMIT_FETCH_COUNT)
            
            # Input for custom commit count (narrow width)
            imgui.push_item_width(30)  # Set a narrow width for the input
            changed, self._custom_commit_count = imgui.input_text("##commit_count", self._custom_commit_count, 3, imgui.INPUT_TEXT_CHARS_DECIMAL)
            imgui.pop_item_width()
            imgui.same_line()
            
            if imgui.button("Apply", width=80):
                try:
                    count = int(self._custom_commit_count)
                    if count > 0 and count <= 100:  # Reasonable limits
                        self.load_available_versions_async(count)
                    else:
                        # Reset to default if invalid
                        from config.constants import DEFAULT_COMMIT_FETCH_COUNT
                        self._custom_commit_count = str(DEFAULT_COMMIT_FETCH_COUNT)
                except ValueError:
                    # Reset to default if invalid
                    from config.constants import DEFAULT_COMMIT_FETCH_COUNT
                    self._custom_commit_count = str(DEFAULT_COMMIT_FETCH_COUNT)

    def _render_github_token_content(self):
        """Renders the GitHub token content within the tabbed dialog."""
        imgui.text("GitHub Personal Access Token")
        imgui.text_wrapped("A GitHub token increases the API rate limit from 60 to 5000 requests per hour.")
        imgui.separator()
        
        current_token = self.token_manager.get_token()
        
        # Show current token status
        if current_token:
            masked_token = self.token_manager.get_masked_token()
            imgui.text(f"Current token: {masked_token}")
            from config.element_group_colors import UpdateSettingsColors
            imgui.text_colored("✓ Token is set", *UpdateSettingsColors.TOKEN_SET)
        else:
            from config.element_group_colors import UpdateSettingsColors
            imgui.text_colored("No token set", *UpdateSettingsColors.TOKEN_NOT_SET)
        
        imgui.separator()
        
        # Token input field
        imgui.text("Enter GitHub Personal Access Token:")
        imgui.text_wrapped("Get a token from: GitHub → Settings → Developer settings → Personal access tokens")
        imgui.text_wrapped("Required scope: public_repo (for public repositories)")
        
        changed, self._github_token_buffer = imgui.input_text("Token", self._github_token_buffer, 100, imgui.INPUT_TEXT_PASSWORD)
        
        imgui.separator()
        
        # Buttons
        if imgui.button("Save Token", width=120):
            self.token_manager.set_token(self._github_token_buffer)
        
        imgui.same_line()
        
        if imgui.button("Test Token", width=120):
            # Test the current token in the buffer
            test_token = self._github_token_buffer if self._github_token_buffer else self.token_manager.get_token()
            validation_result = self.token_manager.validate_token(test_token)
            
            if validation_result['valid']:
                imgui.open_popup("Token Validation")
                self._token_validation_result = validation_result
            else:
                imgui.open_popup("Token Validation")
                self._token_validation_result = validation_result
        
        imgui.same_line()
        
        if imgui.button("Remove Token", width=120):
            self.token_manager.remove_token()
            self._github_token_buffer = ""
        
        # Token validation result popup
        if hasattr(self, '_token_validation_result'):
            if imgui.begin_popup_modal("Token Validation", True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
                result = self._token_validation_result
                
                if result['valid']:
                    from config.element_group_colors import UpdateSettingsColors
                    imgui.text_colored("Token is valid!", *UpdateSettingsColors.TOKEN_VALID)
                    if result['user_info']:
                        imgui.text(f"Username: {result['user_info'].get('login', 'Unknown')}")
                else:
                    from config.element_group_colors import UpdateSettingsColors
                    imgui.text_colored("✗ Token validation failed", *UpdateSettingsColors.TOKEN_INVALID)
                    imgui.text(result['message'])
                
                imgui.separator()
                
                if imgui.button("OK", width=100):
                    imgui.close_current_popup()
                    delattr(self, '_token_validation_result')
                
                imgui.end_popup()
