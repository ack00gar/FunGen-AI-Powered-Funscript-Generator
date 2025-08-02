import os
import configparser
import logging
import requests


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
    
    def validate_token(self, token: str = None) -> dict:
        """
        Validate a GitHub token by making a test API call.
        
        Args:
            token: The token to validate. If None, uses the stored token.
            
        Returns:
            dict: {
                'valid': bool,
                'message': str,
                'user_info': dict or None
            }
        """
        if token is None:
            token = self.get_token()
        
        if not token:
            return {
                'valid': False,
                'message': 'No token provided',
                'user_info': None
            }
        
        try:
            headers = {
                'User-Agent': 'FunGen-Updater/1.0',
                'Accept': 'application/vnd.github.v3+json',
                'Authorization': f'token {token}'
            }
            
            # Make a simple API call to get user info
            response = requests.get('https://api.github.com/user', headers=headers, timeout=10)
            
            if response.status_code == 200:
                user_info = response.json()
                return {
                    'valid': True,
                    'message': f'Token is valid: {user_info.get("login", "Unknown")}',
                    'user_info': user_info
                }
            elif response.status_code == 401:
                return {
                    'valid': False,
                    'message': 'Invalid token - unauthorized',
                    'user_info': None
                }
            elif response.status_code == 403:
                return {
                    'valid': False,
                    'message': 'Token lacks required permissions',
                    'user_info': None
                }
            else:
                return {
                    'valid': False,
                    'message': f'Token validation failed (HTTP {response.status_code})',
                    'user_info': None
                }
                
        except requests.RequestException as e:
            return {
                'valid': False,
                'message': f'Network error: {str(e)}',
                'user_info': None
            }
        except Exception as e:
            return {
                'valid': False,
                'message': f'Validation error: {str(e)}',
                'user_info': None
            } 