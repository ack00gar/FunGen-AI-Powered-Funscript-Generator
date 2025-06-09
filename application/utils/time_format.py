import datetime

def _format_time(self, time_seconds: float) -> str:
    if time_seconds < 0: time_seconds = 0
    # Ensure time_seconds is a standard float, not numpy float if it causes issues
    time_seconds = float(time_seconds)
    try:
        td = datetime.timedelta(seconds=time_seconds)
        total_seconds_int = int(td.total_seconds())
        hours, remainder = divmod(total_seconds_int, 3600)
        minutes, seconds_part = divmod(remainder, 60)
        milliseconds = int(td.microseconds / 1000)
        return f"{hours:02}:{minutes:02}:{seconds_part:02}.{milliseconds:03d}"
    except OverflowError:  # Handle very large time_seconds if that's a possibility
        return "Time Overflow"