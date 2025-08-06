from datetime import datetime, timedelta

def _format_time(self, time_seconds: float) -> str:
    if time_seconds < 0: time_seconds = 0

    time_seconds = float(time_seconds)
    try:
        td = timedelta(seconds=time_seconds)
        total_seconds_int = int(td.total_seconds())
        hours, remainder = divmod(total_seconds_int, 3600)
        minutes, seconds_part = divmod(remainder, 60)
        milliseconds = int(td.microseconds / 1000)
        return f"{hours:02}:{minutes:02}:{seconds_part:02}.{milliseconds:03d}"
    except OverflowError:  # Handle very large time_seconds if that's a possibility
        return "Time Overflow"

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