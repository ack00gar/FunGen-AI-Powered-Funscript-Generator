import platform
from functools import reduce


def is_mac():
    return platform.system() == "Darwin"

def is_windows():
    return platform.system() == "Windows"

def to_int_or_none(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def to_int_or_fallback(value, fallback):
    try:
        return int(value)
    except (ValueError, TypeError):
        return fallback

def optional_chain(obj, *attrs):
    """Use similarly to Javascript optional chain ?."""
    return reduce(lambda acc, attr: getattr(acc, attr, None) if acc else None, attrs, obj)

def optional_chain_fallback(obj, fallback, *attrs):
    """Use similarly to Javascript optional chain ?."""
    return reduce(lambda acc, attr: getattr(acc, attr, fallback) if acc else fallback, attrs, obj)