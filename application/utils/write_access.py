import os
import tempfile
from typing import Union, Sequence

class WriteAccessError(Exception):
    pass

def check_write_access(paths: Union[str, Sequence[str]]):
    """
    Checks if the given path(s) (file or directory) are writable.
    Creates directories if they do not exist.
    Raises WriteAccessError if any are not writable.
    """
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        if os.path.isdir(path):
            # Try to create and delete a temp file in the directory
            try:
                with tempfile.TemporaryFile(dir=path):
                    pass
            except Exception as e:
                raise WriteAccessError(f"No write access to directory: {path} ({e})")
        else:
            dir_path = os.path.dirname(path) or "."
            # Create the directory if it does not exist
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except Exception as e:
                    raise WriteAccessError(f"Could not create directory: {dir_path} ({e})")
            # Try to create and delete a temp file in the directory
            try:
                with tempfile.TemporaryFile(dir=dir_path):
                    pass
            except Exception as e:
                raise WriteAccessError(f"No write access to directory: {dir_path} ({e})")
            # If file exists, check if it can be opened for writing
            if os.path.exists(path):
                try:
                    with open(path, "a"):
                        pass
                except Exception as e:
                    raise WriteAccessError(f"No write access to file: {path} ({e})") 