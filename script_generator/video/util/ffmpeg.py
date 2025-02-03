import os
import shutil
import platform

from script_generator.utils.helpers import is_mac


def get_ffmpeg_paths():
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if ffmpeg_path and ffprobe_path:
        return ffmpeg_path, ffprobe_path

    common_paths = []

    if platform.system() == "Windows":
        common_paths = [
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Users\{user}\ffmpeg\bin\ffmpeg.exe".format(user=os.getlogin()),
        ]

    elif is_mac():
        common_paths = [
            "/usr/local/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg",
            "/usr/bin/ffmpeg",
            "/usr/local/ffmpeg/bin/ffmpeg",
        ]

    elif platform.system() == "Linux":
        common_paths = [
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/snap/bin/ffmpeg",
            "/var/lib/flatpak/exports/bin/ffmpeg",
        ]

    for path in common_paths:
        if os.path.exists(path):
            if "ffmpeg" in path and not ffmpeg_path:
                ffmpeg_path = path
            elif "ffprobe" in path and not ffprobe_path:
                ffprobe_path = path

    return ffmpeg_path, ffprobe_path
