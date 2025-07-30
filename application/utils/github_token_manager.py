import os
import configparser
import logging


class GitHubTokenManager:
    """Manages GitHub token storage in a separate INI file."""
    
    def __init__(self, token_file_path: str = "github_token.ini"):
        self.token_file_path = token_file_path
        self.logger = logging.getLogger(__name__)
        self._config = configparser.ConfigParser()
        self._load_token_file()
    
    def _load_token_file(self):
        """Load the token file if it exists."""
        if os.path.exists(self.token_file_path):
            try:
                self._config.read(self.token_file_path)
                if 'GitHub' not in self._config:
                    self._config['GitHub'] = {}
            except Exception as e:
                self.logger.error(f"Error loading GitHub token file: {e}")
                self._config['GitHub'] = {}
        else:
            self._config['GitHub'] = {}
    
    def _save_token_file(self):
        """Save the token file."""
        try:
            with open(self.token_file_path, 'w') as f:
                self._config.write(f)
            self.logger.info(f"GitHub token saved to {self.token_file_path}")
        except Exception as e:
            self.logger.error(f"Error saving GitHub token file: {e}")
    
    def get_token(self) -> str:
        """Get the stored GitHub token."""
        return self._config.get('GitHub', 'token', fallback='')
    
    def set_token(self, token: str):
        """Set the GitHub token."""
        self._config['GitHub']['token'] = token
        self._save_token_file()
    
    def remove_token(self):
        """Remove the GitHub token."""
        if 'token' in self._config['GitHub']:
            del self._config['GitHub']['token']
            self._save_token_file()
    
    def has_token(self) -> bool:
        """Check if a token is stored."""
        return bool(self.get_token())
    
    def get_masked_token(self) -> str:
        """Get a masked version of the token for display."""
        token = self.get_token()
        if not token:
            return ""
        if len(token) <= 8:
            return "***"
        return token[:4] + "*" * (len(token) - 8) + token[-4:] 