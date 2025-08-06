import socket
from config.constants import INTERNET_TEST_HOSTS


def check_internet_connection() -> bool:
    """
    Check if internet connection is available by testing connectivity to reliable hosts.
    
    Returns:
        bool: True if internet connection is available, False otherwise
    """
    for host, port in INTERNET_TEST_HOSTS:
        try:
            socket.create_connection((host, port), timeout=3)
            return True
        except (socket.timeout, socket.error, OSError):
            continue
    return False