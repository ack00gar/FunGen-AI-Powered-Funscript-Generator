"""
Config package initialization.
Ensures color pipeline is initialized early to avoid import timing issues.
"""

# Import the entire color pipeline to ensure all constants are properly initialized
from .constants_colors import *
from .element_group_colors import *
from .constants import *