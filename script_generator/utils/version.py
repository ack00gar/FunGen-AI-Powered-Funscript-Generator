def version_is_less_than(version_a: str, version_b: str) -> bool:
    """
    Compare two semantic version strings (e.g., "0.2.1" and "0.10.0")
    and return True if version_a < version_b.
    """
    parts_a = [int(x) for x in version_a.split('.')]
    parts_b = [int(x) for x in version_b.split('.')]

    return parts_a < parts_b