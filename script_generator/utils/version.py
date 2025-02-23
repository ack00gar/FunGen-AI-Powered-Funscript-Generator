import re
from script_generator.debug.logger import log_fun


def sanitize_version(version: str) -> str:
    """
    Extracts the leading semantic version (digits and dots) from a version string.
    For example, "0.0.1_25-01-16" becomes "0.0.1".
    """

    match = re.match(r"(\d+(?:\.\d+)*)", version)
    return match.group(1) if match else version


def version_is_less_than(version_a: str, version_b: str) -> bool:
    """
    Compare two semantic version strings (e.g., "0.2.1" and "0.10.0")
    and return True if version_a < version_b.
    """

    # Sanitize the version strings by extracting the semantic version part.
    sanitized_a = sanitize_version(version_a)
    sanitized_b = sanitize_version(version_b)

    log_fun.debug(f"Comparing versions: {sanitized_a} < {sanitized_b}")

    parts_a = [int(x) for x in sanitized_a.split('.')]
    parts_b = [int(x) for x in sanitized_b.split('.')]

    return parts_a < parts_b