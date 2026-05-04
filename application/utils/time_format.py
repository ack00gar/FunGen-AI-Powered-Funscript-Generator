from datetime import datetime, timedelta
from functools import lru_cache


@lru_cache(maxsize=512)
def _format_ms(total_ms: int) -> str:
    if total_ms < 0:
        total_ms = 0
    s, ms = divmod(total_ms, 1000)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02}:{m:02}:{s:02}.{ms:03d}"


@lru_cache(maxsize=512)
def _format_ms_short(total_ms: int) -> str:
    if total_ms < 0:
        total_ms = 0
    s = total_ms // 1000
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02}:{m:02}:{s:02}"


def _format_time(self, time_seconds: float, with_ms: bool = True) -> str:
    if time_seconds is None:
        return "00:00:00.000" if with_ms else "00:00:00"
    if time_seconds < 0:
        time_seconds = 0
    total_ms = int(time_seconds * 1000.0 + 0.5)
    return _format_ms(total_ms) if with_ms else _format_ms_short(total_ms)

def format_github_date(date_str: str, include_time: bool = False, return_datetime: bool = False):
    """Formats a GitHub date string to either YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format, or returns datetime object."""
    if date_str == 'Unknown date':
        return date_str if not return_datetime else None
    try:
        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        if return_datetime:
            return date_obj
        elif include_time:
            return date_obj.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return date_str if not return_datetime else None