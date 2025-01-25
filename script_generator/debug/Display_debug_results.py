from script_generator.utils.file import get_output_file_path
from utils.lib_Debugger import Debugger

video = "/Users/k00gar/Downloads/703-czechvr-3d-2160x1080-60fps-smartphone_hq.mp4"

#frame = 32280  # int(600 * 59.94)

frame = (44 * 60 + 26) * 60

log_file, _ = get_output_file_path(video, "_debug_logs.json")
debugger = Debugger(video, is_vr=True, video_reader="FFmpeg", log_file=log_file)

debugger.load_logs()


#debugger.play_video(frame)

debugger.play_debug_video(frame, downsize_ratio=1) #), duration=4)