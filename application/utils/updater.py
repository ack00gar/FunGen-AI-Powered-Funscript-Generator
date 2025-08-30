import threading
import subprocess
import os
import sys
import requests
import imgui
import time
import unicodedata
import re
import json
from typing import List, Dict, Optional
from datetime import datetime
from application.utils import GitHubTokenManager, format_github_date, check_internet_connection
from config.constants import DEFAULT_COMMIT_FETCH_COUNT
from config.element_group_colors import AppGUIColors, UpdateSettingsColors

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
            
            remaining = response.headers.get('X-RateLimit-Remaining')
            limit = response.headers.get('X-RateLimit-Limit')
            reset_time = response.headers.get('X-RateLimit-Reset')
            
            if remaining and limit:
                print(f"GitHub API: {remaining}/{limit} requests remaining")
            
            if response.status_code == 403:
                if remaining == '0':
                    reset_timestamp = int(reset_time) if reset_time else None
                    if reset_timestamp:
                        reset_time_str = datetime.fromtimestamp(reset_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                        error_msg = f"GitHub API rate limit exceeded. Reset at {reset_time_str}"
                    else:
                        error_msg = "GitHub API rate limit exceeded"
                    raise requests.RequestException(error_msg)
                else:
                    raise requests.RequestException(f"GitHub API 403 error: {response.text}")
            
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return None
    
    def get_branch_commit(self, branch: str) -> Optional[Dict]:
        """Get the latest commit for a specific branch."""
        return self._make_request(f"/commits/{branch}")
    
    def get_commit_details(self, commit_hash: str) -> Optional[Dict]:
        """Get detailed information for a specific commit."""
        return self._make_request(f"/commits/{commit_hash}")
    
    def get_commits_list(self, branch: str, per_page: int = None, page: int = 1) -> Optional[List[Dict]]:
        """Get a list of commits for a branch."""
        if per_page is None:
            per_page = DEFAULT_COMMIT_FETCH_COUNT
        return self._make_request(f"/commits?sha={branch}&per_page={per_page}&page={page}")
    
    def compare_commits(self, base_hash: str, head_hash: str) -> Optional[Dict]:
        """Compare two commits."""
        return self._make_request(f"/compare/{base_hash}...{head_hash}")

class AutoUpdater:
    """
    Handles checking for and applying updates from a Git repository.
    
    Features:
    - Automatic update checking against multiple branches (main, v0.5.0)
    - Intelligent branch selection using newest commit timestamps
    - Manual update selection from available commits
    - Changelog generation for selected commits
    - Background async operations to avoid UI blocking
    - Seamless branch migration support
    
    Update Picker Usage:
    - Access via Updates menu -> "Select Update Commit"
    - Shows all available commits from the active branch
    - Highlights current update in green
    - Allows switching to any commit (upgrade or downgrade)
    - Shows commit details and messages
    - Automatically restarts application after update change
    
    Requirements:
    - Git must be installed and in PATH
    - Repository must be a valid Git repository
    - Internet connection for GitHub API calls
    """
    REPO_OWNER = "ack00gar"
    REPO_NAME = "FunGen-AI-Powered-Funscript-Generator"
    
    # Multi-branch configuration for seamless migration
    PRIMARY_BRANCH = "main"      # Target branch for future updates
    FALLBACK_BRANCH = "v0.5.0"   # Legacy branch for compatibility
    MIGRATION_MODE = True        # Enable dual-branch checking
    
    # Backward compatibility
    BRANCH = FALLBACK_BRANCH  # Default to v0.5.0 initially

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
        self.show_update_error_dialog = False
        self.update_error_message = "Failed to check for updates."
        
        self.local_commit_date = None
        self.remote_commit_date = None
        
        self.available_updates = []
        self.selected_update = None
        self.update_picker_loading = False

        self.expanded_commits = set()
        self.commit_changelogs = {}
        self.skipped_commits = set()  # Set of commit hashes to skip
        self.skip_updates_file = "skip_updates.json"
        self.test_mode_enabled = False  # Manual test mode toggle
        
        self.token_manager = GitHubTokenManager()
        self.github_api = GitHubAPIClient(self.REPO_OWNER, self.REPO_NAME, self.token_manager)
        
        # Multi-branch migration support
        self.active_branch = self.FALLBACK_BRANCH  # Start with v0.5.0
        self.branch_transition_count = 0
        self.last_branch_check = 0
        self.branch_comparison_cache = {}  # Cache branch comparisons
        
        # Load saved skip settings
        self._load_skip_updates()
    
    def _get_branch_commit_with_date(self, branch: str) -> dict | None:
        """Get commit data with parsed date for a specific branch."""
        commit_data = self.github_api.get_branch_commit(branch)
        if commit_data:
            commit_info = commit_data.get('commit', {})
            author_info = commit_info.get('author', {})
            date_str = author_info.get('date', '')
            
            try:
                from application.utils import format_github_date
                parsed_date = format_github_date(date_str, return_datetime=True)
                return {
                    'data': commit_data,
                    'branch': branch,
                    'date_str': date_str,
                    'parsed_date': parsed_date,
                    'sha': commit_data.get('sha')
                }
            except Exception as e:
                self.logger.warning(f"Failed to parse date for branch {branch}: {e}")
                return {
                    'data': commit_data,
                    'branch': branch,
                    'date_str': date_str,
                    'parsed_date': None,
                    'sha': commit_data.get('sha')
                }
        return None
    
    def _determine_best_branch(self) -> tuple[str, dict | None]:
        """Determine the best branch to use based on newest commits."""
        if not self.MIGRATION_MODE:
            # Migration mode disabled - use primary branch only
            branch_data = self._get_branch_commit_with_date(self.PRIMARY_BRANCH)
            return self.PRIMARY_BRANCH, branch_data
        
        # Get commits from both branches
        primary_data = self._get_branch_commit_with_date(self.PRIMARY_BRANCH)
        fallback_data = self._get_branch_commit_with_date(self.FALLBACK_BRANCH)
        
        candidates = []
        if primary_data and primary_data['parsed_date']:
            candidates.append(primary_data)
        if fallback_data and fallback_data['parsed_date']:
            candidates.append(fallback_data)
            
        if not candidates:
            # Fallback to string comparison if date parsing fails
            if primary_data:
                candidates.append(primary_data)
            if fallback_data:
                candidates.append(fallback_data)
        
        if not candidates:
            self.logger.error("No accessible branches found")
            return self.FALLBACK_BRANCH, None
        
        # Select newest commit
        if len(candidates) == 1:
            selected = candidates[0]
        else:
            # Compare by parsed date if available, otherwise by date string
            selected = max(candidates, key=lambda x: x['parsed_date'] or x['date_str'])
        
        selected_branch = selected['branch']
        
        # Log branch transition if changed
        if hasattr(self, 'active_branch') and self.active_branch != selected_branch:
            self._log_branch_transition(self.active_branch, selected_branch)
        
        return selected_branch, selected
    
    def _log_branch_transition(self, from_branch: str, to_branch: str):
        """Log branch transitions for monitoring."""
        self.logger.info(f"Branch transition: {from_branch} → {to_branch}")
        self.branch_transition_count += 1
        
        # Notify user about significant transitions
        if to_branch == self.PRIMARY_BRANCH and from_branch == self.FALLBACK_BRANCH:
            self.logger.info("Migrated to main branch - now receiving latest updates")
        elif to_branch == self.FALLBACK_BRANCH and from_branch == self.PRIMARY_BRANCH:
            self.logger.info("Reverted to v0.5.0 branch - using stable release channel")
    
    def get_migration_status(self) -> dict:
        """Get current migration status information."""
        return {
            'migration_mode': self.MIGRATION_MODE,
            'active_branch': self.active_branch,
            'primary_branch': self.PRIMARY_BRANCH,
            'fallback_branch': self.FALLBACK_BRANCH,
            'transition_count': self.branch_transition_count,
            'is_on_primary': self.active_branch == self.PRIMARY_BRANCH,
            'is_on_fallback': self.active_branch == self.FALLBACK_BRANCH
        }
    
    def disable_migration_mode(self):
        """Disable migration mode - use primary branch only."""
        self.MIGRATION_MODE = False
        self.active_branch = self.PRIMARY_BRANCH
        self.logger.info("Migration mode disabled - using primary branch only")
    
    def enable_migration_mode(self):
        """Enable migration mode - check both branches."""
        self.MIGRATION_MODE = True
        self.logger.info("Migration mode enabled - checking both branches")
    
    def _load_skip_updates(self):
        """Load skipped commit hashes from file."""
        try:
            if os.path.exists(self.skip_updates_file):
                with open(self.skip_updates_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.skipped_commits = set(data)
                    else:
                        self.skipped_commits = set()
            else:
                self.skipped_commits = set()
        except (json.JSONDecodeError, IOError, OSError) as e:
            self.logger.warning(f"Failed to load skip settings: {e}")
            self.skipped_commits = set()
    
    def _save_skip_updates(self):
        """Save skipped commit hashes to file."""
        try:
            with open(self.skip_updates_file, 'w') as f:
                json.dump(list(self.skipped_commits), f)
        except (IOError, OSError) as e:
            self.logger.error(f"Failed to save skip settings: {e}")
    
    def _update_skip_state(self, commit_hash: str, skipped: bool):
        """Update the skip state for a commit hash."""
        if skipped:
            self.skipped_commits.add(commit_hash)
        else:
            self.skipped_commits.discard(commit_hash)

    def _get_current_branch(self) -> str | None:
        """Gets the current branch name."""
        try:
            if not os.path.isdir('.git'):
                self.logger.warning("Not a git repository.")
                return None
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True, text=True, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.error(f"Could not get current branch: {e}")
            return None

    def _get_local_commit_hash(self) -> str | None:
        """Gets the commit hash of the current HEAD commit."""
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
            self.logger.error(f"Could not get local git hash for HEAD: {e}")
            self.status_message = f"Could not determine current commit."
            return None

    def _get_remote_commit_hash(self) -> str | None:
        """Gets the latest commit hash from the best available branch."""
        best_branch, branch_data = self._determine_best_branch()
        
        if branch_data and branch_data['sha']:
            # Update active branch if it changed
            if self.active_branch != best_branch:
                self.active_branch = best_branch
            return branch_data['sha']
        else:
            self.logger.error(f"Failed to fetch remote update from branches: {self.PRIMARY_BRANCH}, {self.FALLBACK_BRANCH}")
            self.status_message = "Could not connect to check for updates."
            return None

    def _get_remote_commit_data(self) -> tuple[str | None, str | None]:
        """Gets the latest commit hash and date from the best available branch."""
        best_branch, branch_data = self._determine_best_branch()
        
        if branch_data and branch_data['data']:
            # Update active branch if it changed
            if self.active_branch != best_branch:
                self.active_branch = best_branch
                
            commit_hash = branch_data['sha']
            commit_date = branch_data['date_str']
            return commit_hash, commit_date
        else:
            self.logger.error(f"Failed to fetch remote update from branches: {self.PRIMARY_BRANCH}, {self.FALLBACK_BRANCH}")
            self.status_message = "Could not connect to check for updates."
            return None, None

    def _get_commit_diff(self, local_hash: str, remote_hash: str) -> tuple[list[str] | None, str | None]:
        """Gets detailed commit information between local and remote updates, plus local commit date."""
        compare_data = self.github_api.compare_commits(local_hash, remote_hash)
        
        if compare_data is None:
            self.logger.warning(f"Could not compare commits {local_hash[:7]} and {remote_hash[:7]} - they may be from different branches")
            return None, None

        # Extract local commit date from the comparison response
        local_commit_date = None
        base_commit = compare_data.get('base_commit', {})
        if base_commit:
            commit_info = base_commit.get('commit', {})
            author_info = commit_info.get('author', {})
            local_commit_date = author_info.get('date', 'Unknown date')

        changelog = []
        commits = compare_data.get('commits', [])
        
        if not commits:
            changelog.append("No commits found between the specified hashes.")
            return changelog, local_commit_date
            
        changelog.append(f"Changes from {local_hash[:7]} to {remote_hash[:7]}")
        changelog.append(f"Total commits: {len(commits)}")
        changelog.append("")
        
        for commit_data in commits:
            # Get commit details
            commit_info = commit_data.get('commit', {})
            author_info = commit_info.get('author', {})
            author_data = commit_data.get('author')
            
            # Use GitHub username if available, otherwise use commit author name
            author = author_data.get('login') if author_data else author_info.get('name', 'Unknown')
            message = commit_info.get('message', 'No commit message')
            date = author_info.get('date', 'Unknown date')
            commit_hash = commit_data.get('sha', 'Unknown')
            
            changelog.append(f"Commit: {commit_hash[:7]}")
            changelog.append(f"Author: {self.clean_text(author)}")
            changelog.append(f"Date:   {format_github_date(date, include_time=True)}")
            changelog.append("Message:")
            
            # Split message into lines and format nicely
            message_lines = message.split('\n')
            for line in message_lines:
                cleaned_line = self.clean_text(line)
                if cleaned_line.strip():
                    changelog.append(f"  {cleaned_line}")
            changelog.append("")
        return changelog, local_commit_date

    def _get_commit_date(self, commit_hash: str) -> str:
        """Gets the commit date for a given commit hash."""
        commit_data = self.github_api.get_commit_details(commit_hash)
        
        if commit_data is None:
            self.logger.error(f"Failed to fetch commit date for {commit_hash[:7]}")
            return 'Unknown date'
        
        commit_info = commit_data.get('commit', {})
        author_info = commit_info.get('author', {})
        date_str = author_info.get('date', 'Unknown date')
        return format_github_date(date_str, include_time=False)

    def _check_worker(self):
        """Worker thread to check for updates and fetch changelog."""
        self.local_commit_hash = self._get_local_commit_hash()
        if not self.local_commit_hash:
            self.logger.error("Could not determine local commit hash")
            self.update_error_message = "Could not determine local commit hash."
            self.show_update_error_dialog = True
            self.update_check_complete = True
            return

        # Get latest commit hash and date from repo (1 API call)
        self.remote_commit_hash, self.remote_commit_date = self._get_remote_commit_data()
        if not self.remote_commit_hash:
            # Check internet connection when GitHub API fails
            if not check_internet_connection():
                self.logger.error("No internet connection available")
                self.update_error_message = "No internet connection available. Please check your network connection."
            else:
                self.logger.error("Could not determine remote commit hash")
                self.update_error_message = "Could not connect to GitHub. Please check your GitHub token."
            self.show_update_error_dialog = True
            self.update_check_complete = True
            return

        # Compare latest repo hash with local hash
        if self.local_commit_hash == self.remote_commit_hash:
            # Same hash - no update needed
            self.logger.info("Application is up to date.")
            self.status_message = "You are on the latest update."
            self.update_available = False
            self.update_changelog = []
            self.update_check_complete = True
            return

        # Different hash - check if it's in skip list
        if self.remote_commit_hash in self.skipped_commits:
            self.logger.info(f"Update {self.remote_commit_hash[:7]} is marked as skipped, ignoring.")
            self.status_message = "You are on the latest update (skipped updates ignored)."
            self.update_available = False
            self.update_changelog = []
            self.update_check_complete = True
            return

        # Hash is different and not skipped - get changelog and local commit date (1 API call)
        self.update_changelog, self.local_commit_date = self._get_commit_diff(self.local_commit_hash, self.remote_commit_hash)
        
        if self.update_changelog is None:
            # Failed to fetch changelog - show error popup with specific error message
            self.update_error_message = f"Could not compare commits {self.local_commit_hash[:7]} and {self.remote_commit_hash[:7]} - they may be from different branches"
            self.show_update_error_dialog = True
            self.update_check_complete = True
            return

        if self._is_remote_commit_newer():
            self.logger.info("Update available.")
            self.status_message = "A new update is available!"
            self.update_available = True
            # Always show popup when startup checking is enabled and update is available
            self.show_update_dialog = True
        else:
            self.logger.info("Remote commit is same or older than local commit.")
            # self.status_message = "You are on the latest update."
            self.update_available = False
            self.update_changelog = []

        self.update_check_complete = True

    def _is_remote_commit_newer(self) -> bool:
        """Compares local and remote commit timestamps to determine if remote is newer."""
        try:
            # Parse the date strings to datetime objects for comparison using time utility
            local_date = format_github_date(self.local_commit_date, return_datetime=True)
            remote_date = format_github_date(self.remote_commit_date, return_datetime=True)
            
            if local_date is None or remote_date is None:
                self.logger.warning("Could not parse commit dates")
                return True
            
            # Return True if remote commit is newer than local commit
            return remote_date > local_date
        except (ValueError, AttributeError) as e:
            self.logger.warning(f"Could not compare commit dates: {e}")
            # If we can't compare dates, assume remote is newer (safer default)
            return True

    def _restart_application(self):
        """Restarts the application with proper cleanup to prevent zombie processes."""
        try:
            main_script = "main.py"
            
            if not os.path.exists(main_script):
                self.logger.error(f"Main script {main_script} not found")
                return
            
            # Create the command to restart the application
            cmd = [sys.executable, main_script]
            
            # Add any original arguments that aren't the script name
            original_args = sys.argv[1:]
            if original_args:
                cmd.extend(original_args)
            
            # Start the new process
            if sys.platform == 'win32':
                # On Windows, use subprocess.Popen with inherited console context
                # This prevents CMD window proliferation while maintaining proper process inheritance
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0  # SW_HIDE = 0 (hide console window)
                subprocess.Popen(cmd, startupinfo=startupinfo)  # Remove DETACHED_PROCESS for better UX
            else:
                # On Unix-like systems, use subprocess.Popen
                subprocess.Popen(cmd)

            time.sleep(0.1)

            self.logger.info("Restarting application...")
            os._exit(0)

        except Exception as e:
            self.logger.error(f"Failed to restart application: {e}")
            # Fallback to os.execl if the proper restart fails
            os.execl(sys.executable, sys.executable, *sys.argv)

    def test_restart(self):
        """Test the restart mechanism without making any actual changes.
        This triggers the exact same restart procedure as a real update."""
        self.logger.info("Testing restart mechanism (no changes made)...")
        self._restart_application()

    def check_for_updates_async(self):
        """Starts the update check in a background thread and updates the timestamp."""
        self.last_check_time = time.time() # Update time when a check is initiated
        threading.Thread(target=self._check_worker, daemon=True, name="UpdaterCheckThread").start()

    def _apply_update(self, target_hash: str = None, use_pull: bool = True):
        """Unified method to apply updates using either a git update or git checkout."""
        self.update_in_progress = True
        
        # CRITICAL SELF-BOOTSTRAPPING: Always update current branch first
        # This ensures users with old updater code get the latest fixes before migration
        try:
            self.status_message = "Bootstrapping updater..."
            self.logger.info("Self-bootstrapping: updating current branch to latest version")
            
            # Get current branch
            current_branch_result = subprocess.run(
                ['git', 'branch', '--show-current'],
                capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            current_branch = current_branch_result.stdout.strip()
            
            if current_branch:  # Only if we're on a branch (not detached HEAD)
                # Stash changes first
                try:
                    subprocess.run(
                        ['git', 'stash', 'push', '-m', 'Auto-stash for updater bootstrap'],
                        capture_output=True, text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )
                except:
                    pass  # Continue even if stash fails
                
                # Fetch and update current branch
                try:
                    subprocess.run(
                        ['git', 'fetch', 'origin', current_branch],
                        check=True, capture_output=True, text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )
                    subprocess.run(
                        ['git', 'reset', '--hard', f'origin/{current_branch}'],
                        check=True, capture_output=True, text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )
                    self.logger.info(f"Self-bootstrap successful: updated {current_branch} to latest")
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"Self-bootstrap failed: {e} - continuing anyway")
        except Exception as e:
            self.logger.warning(f"Self-bootstrap error: {e} - continuing anyway")
        
        # Special pre-migration handling for v0.5.0 to main transition
        if self.MIGRATION_MODE and self.active_branch != self.FALLBACK_BRANCH and not use_pull:
            self.logger.info("Detected migration update - preparing working directory")
            self.status_message = "Preparing for migration..."
            
            # Stash any local changes to ensure clean migration
            try:
                stash_result = subprocess.run(
                    ['git', 'stash', 'push', '-m', 'Auto-stash before migration update'],
                    capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                if "No local changes" not in stash_result.stdout:
                    self.logger.info("Stashed local changes before migration")
            except Exception as e:
                self.logger.warning(f"Could not stash changes: {e}")
        
        if use_pull:
            self.status_message = "Pulling updates..."
            self.logger.info("Attempting to pull updates from origin...")
        else:
            self.status_message = f"Switching to commit: {target_hash[:7]}..."
            self.logger.info(f"Attempting to checkout commit {target_hash}")

        # Check if test mode is enabled
        if self.test_mode_enabled:
            current_branch = self._get_current_branch()
            if use_pull:
                self.logger.info(f"Running in test mode (current branch: {current_branch}, target branch: {self.BRANCH})")
                self.status_message = "TEST MODE: Would pull latest updates"
            else:
                self.logger.info(f"Running in test mode (current branch: {current_branch}, target commit: {target_hash[:7]})")
                self.status_message = f"TEST MODE: Would switch to commit {target_hash[:7]}"
            time.sleep(2)
            self.update_in_progress = False
            return

        try:
            if use_pull:
                success = self._perform_git_pull()
            else:
                success = self._perform_git_checkout(target_hash)
                
            if success:
                if use_pull:
                    self.logger.info("Update pull successful")
                else:
                    self.logger.info(f"Successfully checked out commit {target_hash}")
                self.status_message = "Update complete. Restarting..."
                time.sleep(2)
                self._restart_application()
            else:
                if use_pull:
                    self.status_message = "Update failed. Please check console or update manually."
                else:
                    self.status_message = "Update change failed. Please check console."
                self.update_in_progress = False
                
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during update: {e}")
            self.status_message = "An unexpected error occurred. See logs."
            self.update_in_progress = False

    def _perform_git_pull(self) -> bool:
        """Performs a git fetch and hard reset to handle forced updates gracefully."""
        try:
            # Fetch the latest updates from the remote without trying to merge or rebase
            fetch_result = subprocess.run(
                ['git', 'fetch', 'origin', self.BRANCH],
                check=True, capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            self.logger.info(f"Git fetch successful: {fetch_result.stdout}")

            # Reset the local branch to exactly match the remote branch
            # This is a robust way to handle force pushes and ensures a clean update
            reset_result = subprocess.run(
                ['git', 'reset', '--hard', f'origin/{self.BRANCH}'],
                check=True, capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            self.logger.info(f"Git reset successful: {reset_result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Update failed during 'git fetch' or 'git reset': {e.stderr}")
            return False

    def _perform_git_checkout(self, commit_hash: str) -> bool:
        """Performs git checkout operation with branch migration support."""
        try:
            # CRITICAL: Ensure commit exists locally before attempting checkout
            # This is essential for migration where commit might be from different branch
            try:
                self.logger.info(f"Ensuring commit {commit_hash[:7]} exists locally...")
                
                # Fetch all branches to ensure we have the commit
                subprocess.run(
                    ['git', 'fetch', 'origin'],
                    check=True, capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                
                # Verify commit exists after fetch
                commit_check = subprocess.run(
                    ['git', 'cat-file', '-e', commit_hash],
                    capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                
                if commit_check.returncode != 0:
                    self.logger.error(f"Commit {commit_hash[:7]} does not exist even after fetch")
                    return False
                    
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"Failed to verify commit existence: {e}")
                # Continue anyway, might still work
            
            # Enhanced checkout logic: if we're migrating branches, ensure proper branch setup
            if self.MIGRATION_MODE and self.active_branch != self.FALLBACK_BRANCH:
                self.logger.info(f"Migration mode: Moving from {self.FALLBACK_BRANCH} to {self.active_branch}")
                
                # CRITICAL: For migration, we need to be more aggressive
                # First, stash any local changes to prevent conflicts
                try:
                    stash_result = subprocess.run(
                        ['git', 'stash', 'push', '-m', 'Auto-stash before v0.5.0 to main migration'],
                        capture_output=True, text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )
                    if "No local changes" not in stash_result.stdout:
                        self.logger.info("Stashed local changes before migration")
                except Exception as e:
                    self.logger.warning(f"Could not stash changes: {e}")
                
                # Ensure we have the latest main branch data (additional fetch for safety)
                try:
                    self.logger.info("Fetching latest main branch from origin...")
                    subprocess.run(
                        ['git', 'fetch', 'origin', 'main'],
                        check=True, capture_output=True, text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"Fetch warning during migration: {e}")
                
                # Try to create/update local main branch
                if self._ensure_local_branch_exists(self.active_branch):
                    # Use force checkout for migration to bypass any issues
                    try:
                        # First try normal checkout
                        checkout_result = subprocess.run(
                            ['git', 'checkout', self.active_branch],
                            capture_output=True, text=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                        )
                        if checkout_result.returncode != 0:
                            # Force checkout if normal fails
                            self.logger.warning("Normal checkout failed, using force checkout for migration")
                            subprocess.run(
                                ['git', 'checkout', '-f', self.active_branch],
                                check=True, capture_output=True, text=True,
                                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                            )
                        
                        # Now reset to the target commit
                        reset_result = subprocess.run(
                            ['git', 'reset', '--hard', commit_hash],
                            check=True, capture_output=True, text=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                        )
                        self.logger.info(f"Git migration successful - now on branch {self.active_branch} at commit {commit_hash[:7]}")
                        return True
                        
                    except subprocess.CalledProcessError as e:
                        self.logger.error(f"Migration checkout failed: {e}")
                        # Last resort: force everything
                        try:
                            self.logger.warning("Attempting force migration as last resort...")
                            # Force checkout to main
                            subprocess.run(
                                ['git', 'checkout', '-f', self.active_branch],
                                check=True, capture_output=True, text=True,
                                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                            )
                            # Reset hard to target
                            subprocess.run(
                                ['git', 'reset', '--hard', commit_hash],
                                check=True, capture_output=True, text=True,
                                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                            )
                            self.logger.info("Force migration successful")
                            return True
                        except subprocess.CalledProcessError as final_e:
                            self.logger.error(f"Force migration also failed: {final_e}")
                            return False
                else:
                    self.logger.error(f"Failed to ensure local branch {self.active_branch} exists")
                    return False
            else:
                # Standard checkout for v0.5.0 or when not migrating
                checkout_result = subprocess.run(
                    ['git', 'checkout', commit_hash],
                    check=True, capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                self.logger.info(f"Git checkout successful: {checkout_result.stdout}")
                return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Update failed during 'git checkout': {e.stderr}")
            return False

    def _ensure_local_branch_exists(self, branch_name: str) -> bool:
        """Ensure a local branch exists and tracks the corresponding remote branch."""
        try:
            # Check if local branch exists
            result = subprocess.run(
                ['git', 'branch', '--list', branch_name], 
                capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            if branch_name not in result.stdout:
                # Local branch doesn't exist - fetch from remote first to ensure we have the branch data
                self.logger.info(f"Local branch '{branch_name}' doesn't exist, fetching from remote...")
                
                # First, always fetch all remote branches to ensure we have the latest
                try:
                    subprocess.run(
                        ['git', 'fetch', 'origin'],
                        check=True, capture_output=True, text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )
                    self.logger.info("Fetched latest from origin")
                except subprocess.CalledProcessError as fetch_err:
                    self.logger.warning(f"Fetch warning: {fetch_err}")
                
                # Check if the remote branch exists
                remote_check = subprocess.run(
                    ['git', 'ls-remote', '--heads', 'origin', branch_name],
                    capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                
                if not remote_check.stdout.strip():
                    self.logger.error(f"Remote branch 'origin/{branch_name}' does not exist!")
                    return False
                
                # Try to create local branch from remote
                try:
                    # Method 1: Direct fetch to create branch
                    subprocess.run(
                        ['git', 'fetch', 'origin', f'{branch_name}:{branch_name}'],
                        check=True, capture_output=True, text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )
                    self.logger.info(f"Successfully fetched and created local branch '{branch_name}'")
                except subprocess.CalledProcessError as e1:
                    self.logger.info(f"Direct fetch failed, trying checkout -b approach...")
                    
                    # Method 2: Create branch with checkout -b
                    try:
                        subprocess.run(
                            ['git', 'checkout', '-b', branch_name, f'origin/{branch_name}'], 
                            check=True, capture_output=True, text=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                        )
                        self.logger.info(f"Created local branch '{branch_name}' tracking 'origin/{branch_name}'")
                    except subprocess.CalledProcessError as e2:
                        # Method 3: Force create branch
                        self.logger.warning(f"Checkout -b failed, trying force create...")
                        try:
                            # Delete any existing ref that might be conflicting
                            subprocess.run(
                                ['git', 'branch', '-D', branch_name],
                                capture_output=True, text=True,
                                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                            )
                            # Create fresh branch from remote
                            subprocess.run(
                                ['git', 'branch', branch_name, f'origin/{branch_name}'],
                                check=True, capture_output=True, text=True,
                                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                            )
                            self.logger.info(f"Force created local branch '{branch_name}' from 'origin/{branch_name}'")
                        except subprocess.CalledProcessError as e3:
                            self.logger.error(f"All methods to create local branch failed")
                            self.logger.error(f"Final error stderr: {e3.stderr if hasattr(e3, 'stderr') else 'No stderr'}")
                            return False
            else:
                self.logger.info(f"Local branch '{branch_name}' already exists")
                # Update the branch to track remote if it doesn't already
                try:
                    subprocess.run(
                        ['git', 'branch', '--set-upstream-to', f'origin/{branch_name}', branch_name],
                        capture_output=True, text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )
                except subprocess.CalledProcessError:
                    pass  # Branch might already be tracking, that's fine
            
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to ensure local branch '{branch_name}' exists: {e}")
            self.logger.error(f"Error details: {e.stderr if hasattr(e, 'stderr') else 'No stderr'}")
            return False

    def _switch_to_branch(self, branch_name: str) -> bool:
        """Switch to a specific branch, creating it locally if needed."""
        # Ensure the local branch exists
        if not self._ensure_local_branch_exists(branch_name):
            return False
        
        try:
            # Switch to the branch
            subprocess.run(
                ['git', 'checkout', branch_name], 
                check=True, capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            self.logger.info(f"Switched to branch '{branch_name}'")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to switch to branch '{branch_name}': {e}")
            self.logger.error(f"Git stderr: {e.stderr}")
            
            # Try to handle common failure scenarios
            if "Your local changes" in e.stderr or "would be overwritten" in e.stderr:
                self.logger.warning("Local changes detected, attempting to stash and retry...")
                try:
                    # Stash any local changes
                    subprocess.run(
                        ['git', 'stash', 'push', '-m', 'Auto-stash before branch switch'],
                        check=True, capture_output=True, text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )
                    self.logger.info("Stashed local changes")
                    
                    # Try checkout again
                    subprocess.run(
                        ['git', 'checkout', branch_name],
                        check=True, capture_output=True, text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )
                    self.logger.info(f"Successfully switched to branch '{branch_name}' after stashing")
                    return True
                    
                except subprocess.CalledProcessError as stash_error:
                    self.logger.error(f"Failed to stash and switch: {stash_error}")
                    self.logger.error(f"Stash stderr: {stash_error.stderr}")
                    
                    # Last resort: force checkout with potential data loss warning
                    self.logger.warning("Attempting force checkout - uncommitted changes may be lost!")
                    try:
                        subprocess.run(
                            ['git', 'checkout', '-f', branch_name],
                            check=True, capture_output=True, text=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                        )
                        self.logger.info(f"Force switched to branch '{branch_name}'")
                        return True
                    except subprocess.CalledProcessError as force_error:
                        self.logger.error(f"Force checkout also failed: {force_error}")
                        self.logger.error(f"Force checkout stderr: {force_error.stderr}")
            
            return False

    def apply_update_and_restart(self):
        """Pulls the latest changes from git and restarts the application."""
        # During migration, use checkout method to trigger migration logic
        if self.MIGRATION_MODE and self.active_branch != self.FALLBACK_BRANCH:
            # Use checkout with target commit hash to trigger migration logic
            target_hash = self.remote_commit_hash
            if target_hash:
                self._apply_update(target_hash=target_hash, use_pull=False)
            else:
                self._apply_update(use_pull=True)
        else:
            self._apply_update(use_pull=True)

    def _get_spinner_text(self) -> str:
        """Returns the current spinner animation text."""
        spinner_chars = "|/-\\"
        spinner_index = int(time.time() * 4) % 4
        return spinner_chars[spinner_index]

    def render_update_dialog(self):
        """Renders the ImGui popup for the update confirmation."""
        if self.show_update_dialog:
            imgui.open_popup("Update Available")
            self.show_update_dialog = False

        if not hasattr(self, '_update_dialog_pos'):
            main_viewport = imgui.get_main_viewport()
            popup_pos = (main_viewport.pos[0] + main_viewport.size[0] * 0.5,
                         main_viewport.pos[1] + main_viewport.size[1] * 0.5)
            self._update_dialog_pos = (popup_pos[0] - 250, popup_pos[1] - 150)  # Center the window

        imgui.set_next_window_size_constraints((500, 0), (float("inf"), float("inf")))
        imgui.set_next_window_position(*self._update_dialog_pos, condition=imgui.ONCE)

        if imgui.begin_popup_modal("Update Available", True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
            window_pos = imgui.get_window_position()
            if window_pos[0] > 0 and window_pos[1] > 0:
                self._update_dialog_pos = window_pos
            
            if self.update_in_progress:
                imgui.text(self.status_message)
                imgui.text(f"Processing... {self._get_spinner_text()}")
            else:
                # Show branch information in the update dialog
                branch_info = f"from {self.active_branch} branch"
                if self.MIGRATION_MODE and self.active_branch != self.FALLBACK_BRANCH:
                    branch_info = f"from {self.active_branch} branch (migrated from {self.FALLBACK_BRANCH})"
                
                imgui.text(f"A new update is available for FunGen {branch_info}.")
                imgui.text("Would you like to update and restart the application?")
                imgui.separator()

                if self.update_changelog:
                    imgui.text("Changes in this update:")

                    child_width = imgui.get_content_region_available()[0]
                    child_height = 180
                    imgui.begin_child("Changelog", child_width, child_height, border=True, flags=imgui.WINDOW_HORIZONTAL_SCROLLING_BAR
                            | imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
                    for message in self.update_changelog:
                        imgui.text_wrapped(self.clean_text(message))
                    imgui.end_child()

                local_date = self.local_commit_date if self.local_commit_date else 'N/A'
                remote_date = self.remote_commit_date if self.remote_commit_date else 'N/A'
                
                imgui.text_wrapped(f"Your Update: {self.local_commit_hash[:7] if self.local_commit_hash else 'N/A'} ({local_date})")
                
                # Show which branch the latest update is from
                remote_branch_info = f"[{self.active_branch}]"
                if self.MIGRATION_MODE and self.active_branch != self.FALLBACK_BRANCH:
                    remote_branch_info = f"[{self.active_branch}] (migrated from {self.FALLBACK_BRANCH})"
                
                imgui.text_wrapped(
                    f"Latest Update: {self.remote_commit_hash[:7] if self.remote_commit_hash else 'N/A'} ({remote_date}) {remote_branch_info}")
                imgui.separator()
                if imgui.button("Update and Restart", width=200):
                    self.apply_update_and_restart()
                imgui.same_line()

                if imgui.button("Later", width=100):
                    imgui.close_current_popup()

            imgui.end_popup()

    def render_update_error_dialog(self):
        """Renders a simple error popup when update checking fails."""
        if self.show_update_error_dialog:
            imgui.open_popup("Update Check Failed")
            self.show_update_error_dialog = False

        if not hasattr(self, '_update_error_dialog_pos'):
            main_viewport = imgui.get_main_viewport()
            popup_pos = (main_viewport.pos[0] + main_viewport.size[0] * 0.5,
                         main_viewport.pos[1] + main_viewport.size[1] * 0.5)
            self._update_error_dialog_pos = (popup_pos[0] - 200, popup_pos[1] - 100)  # Center the window

        imgui.set_next_window_size(400, 150, condition=imgui.ONCE)
        imgui.set_next_window_position(*self._update_error_dialog_pos, condition=imgui.ONCE)

        if imgui.begin_popup_modal("Update Check Failed", True)[0]:
            window_pos = imgui.get_window_position()
            if window_pos[0] > 0 and window_pos[1] > 0:
                self._update_error_dialog_pos = window_pos

            imgui.text_wrapped(self.update_error_message)
            
            imgui.separator()
            close_button_width = 80
            imgui.set_cursor_pos_x((imgui.get_window_width() - close_button_width) * 0.5)
            if imgui.button("Close", width=close_button_width):
                imgui.close_current_popup()
            imgui.end_popup()

    def _get_available_updates(self, custom_count: int = None) -> List[Dict]:
        """Fetches available commits (merge commits and direct pushes) from the configured branch (v0.5.0)."""
        updates = []

        try:
            # Use the configured branch instead of current branch
            target_branch = self.BRANCH
            self.logger.info(f"Fetching commits from branch: {target_branch}")
            
            target_commit_count = custom_count if custom_count else DEFAULT_COMMIT_FETCH_COUNT
            page = 1
            per_page = 30  # GitHub API default
            
            while len(updates) < target_commit_count:
                commits_data = self.github_api.get_commits_list(target_branch, per_page=per_page, page=page)
                
                if commits_data is None:
                    self.logger.error("Failed to fetch commits from GitHub API")
                    return [{'name': 'Failed to fetch commits. Check network connection or GitHub token.', 'commit_hash': 'error', 'type': 'error', 'date': '', 'full_message': ''}]
                
                if not commits_data:
                    break
                
                for commit in commits_data:
                    try:
                        sha = commit.get('sha')
                        if not sha:
                            continue
                        
                        commit_info = commit.get('commit', {})
                        author_info = commit_info.get('author', {})
                        date = author_info.get('date')
                        message = commit_info.get('message', 'No commit message')
                        
                        parents = commit.get('parents', [])
                        
                        if parents:
                            first_line = message.split('\n')[0] if message else 'No commit message'
                            
                            if sha and date:
                                updates.append({
                                    'name': first_line,
                                    'commit_hash': sha,
                                    'type': 'commit',
                                    'date': date,
                                    'full_message': message
                                })
                                if len(updates) >= target_commit_count:
                                    break
                    except (KeyError, TypeError) as e:
                        self.logger.warning(f"Skipping malformed commit data: {e}")
                        continue

                page += 1
                if page > 50:  # Maximum 50 pages
                    self.logger.warning(f"Reached maximum page limit. Found {len(updates)} commits, requested {target_commit_count}")
                    break
            
            # Sort by date (newest first)
            updates.sort(key=lambda x: x.get('date', 'Unknown'), reverse=True)
            
            self.logger.info(f"Found {len(updates)} commits out of {target_commit_count} requested")
            
        except Exception as e:
            self.logger.error(f"Failed to fetch available updates: {e}")
            return []
        return updates

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

    def _get_update_diff(self, target_hash: str) -> List[str]:
        """Gets commit details for the specified commit hash."""
        current_hash = self._get_local_commit_hash()
        if not current_hash:
            return ["Could not determine current update."]
        
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
        
        # Add current update indicator if this is the current commit
        if current_hash == target_hash:
            changelog.append("*** CURRENT UPDATE ***")
            changelog.append("")
        
        changelog.append(f"Commit: {target_hash[:7]}")
        changelog.append(f"Author: {self.clean_text(author)}")
        changelog.append(f"Date:   {format_github_date(date, include_time=True)}")
        changelog.append("")
        changelog.append("Message: ")
        for line in message_lines:
            cleaned_line = self.clean_text(line)
            if cleaned_line.strip():
                changelog.append(f"  {cleaned_line}")
        return changelog

    def load_available_updates_async(self, custom_count: int = None):
        """Loads available updates in a background thread."""
        self.logger.info("Starting async update loading")
        self.update_picker_loading = True
        threading.Thread(target=self._load_updates_worker, args=(custom_count,), daemon=True, name="UpdaterLoadUpdatesThread").start()

    def _load_updates_worker(self, custom_count: int = None):
        """Worker thread to load available updates."""
        self.logger.info("Loading updates in worker thread")
        self.available_updates = self._get_available_updates(custom_count)
        self.logger.info(f"Loaded {len(self.available_updates)} commits")
        self.update_picker_loading = False
        self._updates_last_loaded = time.time()  # Track when updates were last loaded
        
        # Clear caches when update list changes to ensure data consistency
        self._clear_render_caches()
    
    def _clear_render_caches(self):
        """Clear cached data used for rendering to ensure consistency."""
        if hasattr(self, '_cached_formatted_dates'):
            self._cached_formatted_dates.clear()
        if hasattr(self, '_cached_cleaned_changelogs'):
            self._cached_cleaned_changelogs.clear()
        # Clear current hash cache when update list changes
        if hasattr(self, '_cached_current_hash'):
            delattr(self, '_cached_current_hash')
            delattr(self, '_cached_current_hash_time')
    
    def _load_changelog_async(self, commit_hash: str):
        """Load changelog for a commit in a background thread."""
        try:
            # Only load if not already cached
            if commit_hash not in self.commit_changelogs or self.commit_changelogs[commit_hash] == ["Loading changelog..."]:
                changelog = self._get_update_diff(commit_hash)
                self.commit_changelogs[commit_hash] = changelog
                # Clear cleaned changelog cache for this commit to force regeneration
                if hasattr(self, '_cached_cleaned_changelogs') and commit_hash in self._cached_cleaned_changelogs:
                    del self._cached_cleaned_changelogs[commit_hash]
        except Exception as e:
            self.logger.error(f"Error loading changelog for {commit_hash[:7]}: {e}")
            self.commit_changelogs[commit_hash] = [f"Error loading changelog: {str(e)}"]

    def apply_update_change(self, commit_hash: str, commit_message: str = ""):
        """Applies the selected update change and restarts the application."""
        self._apply_update(target_hash=commit_hash, use_pull=False)

    def render_update_settings_dialog(self):
        """Renders the combined update commit & GitHub token dialog with tabs."""
        if self.app.app_state_ui.show_update_settings_dialog:
            imgui.open_popup("Updates & GitHub Token")
            self.app.app_state_ui.show_update_settings_dialog = False
            # Load updates only if not already loaded or if cache is stale
            if not hasattr(self, '_updates_last_loaded') or time.time() - self._updates_last_loaded > 300.0:  # 5 minute cache
                self.load_available_updates_async()
            # Initialize commit count to default if not set
            if not hasattr(self, '_custom_commit_count'):
                self._custom_commit_count = str(DEFAULT_COMMIT_FETCH_COUNT)

        # Initialize buffers if needed
        if not hasattr(self, '_github_token_buffer'):
            self._github_token_buffer = self.token_manager.get_token()
        if not hasattr(self, '_updates_active_tab'):
            self._updates_active_tab = 0  # 0 = Update, 1 = Token

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

        # Track if popup is open
        popup_open = imgui.begin_popup_modal("Updates & GitHub Token", True)[0]
        
        if popup_open:
            # Save window size and position for persistence
            window_size = imgui.get_window_size()
            window_pos = imgui.get_window_position()
            if window_size[0] > 0 and window_size[1] > 0:
                self._update_settings_window_size = window_size
            if window_pos[0] > 0 and window_pos[1] > 0:
                self._update_settings_window_pos = window_pos
            
            # Tab bar
            if imgui.begin_tab_bar("Updates & GitHub Token Tabs"):
                # Update Selection Tab
                if imgui.begin_tab_item("Choose FunGen Update")[0]:
                    self._updates_active_tab = 0
                    imgui.end_tab_item()
                
                # GitHub Token Tab
                if imgui.begin_tab_item("GitHub Token")[0]:
                    self._updates_active_tab = 1
                    imgui.end_tab_item()
                
                imgui.end_tab_bar()

            # Tab content
            if self._updates_active_tab == 0:
                # Update Selection Tab
                self._render_update_picker_content()
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
            
            # Save settings when popup closes (works for both X button and Close button)
            self._save_skip_updates()

    def _render_update_picker_content(self):
        """Renders the update picker content within the tabbed dialog."""
        if self.update_picker_loading:
            imgui.text("Loading available commits...")
            imgui.text(f"Please wait... {self._get_spinner_text()}")
        elif self.update_in_progress:
            imgui.text(self.status_message)
            imgui.text(f"Processing... {self._get_spinner_text()}")
        else:
            target_branch = self.active_branch
            branch_info = f"'{target_branch}'"
            if self.MIGRATION_MODE and target_branch != self.FALLBACK_BRANCH:
                branch_info += f" (migrated from {self.FALLBACK_BRANCH})"
            imgui.text(f"Select a commit from branch {branch_info} to switch to:")
            
            # Show branch status info
            if self.MIGRATION_MODE:
                imgui.text_colored(f"Migration Mode: Active | Checking {self.PRIMARY_BRANCH} & {self.FALLBACK_BRANCH}", 0.7, 0.9, 0.7, 1.0)
                if self.branch_transition_count > 0:
                    imgui.text_colored(f"Branch transitions: {self.branch_transition_count}", 0.6, 0.8, 1.0, 1.0)
                    
                # Add migration status button
                if imgui.button("Migration Status", width=120):
                    status = self.get_migration_status()
                    print(f"\n=== MIGRATION STATUS ===")
                    for key, value in status.items():
                        print(f"{key}: {value}")
                    print("========================\n")
            imgui.separator()

            # Update list with inline changelogs
            child_width = imgui.get_content_region_available()[0]
            child_height = 400
            imgui.begin_child("UpdateList", child_width, child_height, border=True, flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)

            # Cache current hash - only refresh when actually needed (commit changes)
            if not hasattr(self, '_cached_current_hash'):
                self._cached_current_hash = self._get_local_commit_hash()
                self._cached_current_hash_time = time.time()
            
            current_hash = self._cached_current_hash
            
            # Cache formatted dates to avoid repeated formatting
            if not hasattr(self, '_cached_formatted_dates'):
                self._cached_formatted_dates = {}
            
            for update in self.available_updates:
                commit_hash = update['commit_hash']
                is_expanded = commit_hash in self.expanded_commits
                
                # Highlight current update (use cached hash)
                is_current = current_hash and commit_hash.startswith(current_hash[:7])
                if is_current:
                    imgui.push_style_color(imgui.COLOR_TEXT, *AppGUIColors.VERSION_CURRENT_HIGHLIGHT)
                
                # Create expand/collapse button
                expand_icon = "v" if is_expanded else ">"
                button_text = f"{expand_icon} {commit_hash[:7]}"
                if imgui.button(button_text, width=80):
                    if is_expanded:
                        self.expanded_commits.discard(commit_hash)
                    else:
                        self.expanded_commits.add(commit_hash)
                        # Load changelog if not cached (async to avoid blocking UI)
                        if commit_hash not in self.commit_changelogs:
                            self.commit_changelogs[commit_hash] = ["Loading changelog..."]
                            threading.Thread(target=self._load_changelog_async, args=(commit_hash,), daemon=True, name=f"UpdaterLoadChangelog-{commit_hash[:7]}").start()

                imgui.same_line()
                
                # Cache formatted date
                date_key = update.get('date', 'Unknown date')
                if date_key not in self._cached_formatted_dates:
                    self._cached_formatted_dates[date_key] = format_github_date(date_key, include_time=False)
                commit_date = self._cached_formatted_dates[date_key]
                imgui.text(f"({commit_date})")
                imgui.same_line()

                is_skipped = commit_hash in self.skipped_commits

                # Position checkbox and label at the right edge first (before selectable)
                imgui.same_line()
                imgui.set_cursor_pos_x(imgui.get_window_width() - 90)

                checkbox_id = f"##skip_update_{commit_hash[:7]}"
                changed, is_skipped = imgui.checkbox(checkbox_id, is_skipped)

                if changed:
                    self._update_skip_state(commit_hash, is_skipped)
                    self._save_skip_updates()
                imgui.same_line()
                imgui.text("Skip")
                imgui.same_line()
                imgui.set_cursor_pos_x(190)  # Position after the expand button and date with more space
                
                # Cache truncated commit message
                commit_msg = update['name']
                if len(commit_msg) > 60:
                    commit_msg = commit_msg[:57] + "..."

                if imgui.selectable(commit_msg, self.selected_update == update)[0]:
                    self.selected_update = update

                if is_current:
                    imgui.same_line()
                    imgui.text("(Current)")
                    imgui.pop_style_color()

                if is_expanded:
                    imgui.indent(30)
                    imgui.push_style_color(imgui.COLOR_TEXT, *AppGUIColors.VERSION_CHANGELOG_TEXT)
                    
                    changelog = self.commit_changelogs.get(commit_hash, [])
                    if not changelog:
                        imgui.text_wrapped("Loading changelog...")
                    else:
                        # Cache cleaned changelog lines to avoid repeated cleaning
                        if not hasattr(self, '_cached_cleaned_changelogs'):
                            self._cached_cleaned_changelogs = {}
                        
                        if commit_hash not in self._cached_cleaned_changelogs:
                            self._cached_cleaned_changelogs[commit_hash] = [self.clean_text(line) for line in changelog]
                        
                        for line in self._cached_cleaned_changelogs[commit_hash]:
                            imgui.text_wrapped(line)
                    
                    imgui.pop_style_color()
                    imgui.unindent(30)
                    imgui.separator()

            imgui.end_child()
            imgui.separator()

            # Test mode toggle
            imgui.text("Test Mode:")
            imgui.same_line()
            changed, self.test_mode_enabled = imgui.checkbox("Enable Test Mode", self.test_mode_enabled)
            if imgui.is_item_hovered():
                imgui.set_tooltip("When enabled, commit switching will only simulate the action without actually changing commits. Useful for testing the update system.")
            
            if self.test_mode_enabled:
                imgui.same_line()
                if imgui.button("Test Restart", width=120):
                    self.test_restart()
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Test the restart mechanism without making any changes. This triggers the exact same restart procedure as a real update.")

            imgui.separator()
            if self.selected_update:
                button_text = "Switch to Commit" if not self.test_mode_enabled else "Test Switch to Commit"
                if imgui.button(button_text, width=200):
                    self.apply_update_change(self.selected_update['commit_hash'], self.selected_update['name'])
            else:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.5)
                button_text = "Switch to Commit" if not self.test_mode_enabled else "Test Switch to Commit"
                imgui.button(button_text, width=160)
                imgui.pop_style_var()

            imgui.same_line()
            imgui.set_cursor_pos_x(imgui.get_window_width() - 280)
            
            imgui.text("Fetch commits:")
            imgui.same_line()

            if not hasattr(self, '_custom_commit_count'):
                self._custom_commit_count = str(DEFAULT_COMMIT_FETCH_COUNT)

            def adjust_commit_count(delta: int) -> None:
                try:
                    current_count = int(self._custom_commit_count)
                    new_count = current_count + delta
                    if 1 <= new_count <= 100:  # Valid range
                        self._custom_commit_count = str(new_count)
                    else:
                        self._custom_commit_count = str(DEFAULT_COMMIT_FETCH_COUNT)
                except ValueError:
                    self._custom_commit_count = str(DEFAULT_COMMIT_FETCH_COUNT)

            imgui.push_item_width(30)
            changed, self._custom_commit_count = imgui.input_text("##commit_count", self._custom_commit_count, 3, imgui.INPUT_TEXT_CHARS_DECIMAL)
            imgui.pop_item_width()
            
            imgui.same_line()
            if imgui.button("-", width=15):
                adjust_commit_count(-1)
            
            imgui.same_line()
            if imgui.button("+", width=15):
                adjust_commit_count(1)
            
            imgui.same_line()
            if imgui.button("Apply", width=80):
                try:
                    count = int(self._custom_commit_count)
                    if 1 <= count <= 100:
                        self.load_available_updates_async(count)
                    else:
                        self._custom_commit_count = str(DEFAULT_COMMIT_FETCH_COUNT)
                except ValueError:
                    self._custom_commit_count = str(DEFAULT_COMMIT_FETCH_COUNT)

    def _render_github_token_content(self):
        """Renders the GitHub token content within the tabbed dialog."""
        imgui.text("GitHub Personal Access Token")
        imgui.text_wrapped("A GitHub token increases the API rate limit from 60 to 5000 requests per hour.")
        imgui.separator()

        current_token = self.token_manager.get_token()

        if current_token:
            masked_token = self.token_manager.get_masked_token()
            imgui.text(f"Current token: {masked_token}")
            imgui.text_colored("Token is set", *UpdateSettingsColors.TOKEN_SET)
        else:
            imgui.text_colored("No token set", *UpdateSettingsColors.TOKEN_NOT_SET)

        imgui.separator()
        imgui.text("Enter GitHub Personal Access Token:")
        imgui.text_wrapped("Get a token from: GitHub → Settings → Developer settings → Personal access tokens")
        imgui.text_wrapped("Required scope: public_repo (for public repositories)")
        changed, self._github_token_buffer = imgui.input_text("Token", self._github_token_buffer, 100, imgui.INPUT_TEXT_PASSWORD)

        imgui.separator()
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
                    imgui.text_colored("Token is valid!", *UpdateSettingsColors.TOKEN_VALID)
                    if result['user_info']:
                        imgui.text(f"Username: {result['user_info'].get('login', 'Unknown')}")
                else:
                    imgui.text_colored("✗ Token validation failed", *UpdateSettingsColors.TOKEN_INVALID)
                    imgui.text(result['message'])

                imgui.separator()

                if imgui.button("OK", width=100):
                    imgui.close_current_popup()
                    delattr(self, '_token_validation_result')

                imgui.end_popup()
