from utils.lib_Debugger import Debugger

video = "/Users/k00gar/Downloads/703-czechvr-3d-2160x1080-60fps-smartphone_hq.mp4"

#frame = 32280  # int(600 * 59.94)

frame = (44 * 60 + 26) * 60

debugger = Debugger(video, isVR=True, video_reader="FFmpeg", output_dir=video[:-4])

debugger.load_logs()

#debugger.display_frame(frame)

#debugger.play_video(frame)

debugger.play_video(frame, record=True, downsize_ratio=1, duration=4)