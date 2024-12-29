from utils.lib_Debugger import Debugger

video = "/Users/k00gar/Downloads/730-czechvr-3d-7680x3840-60fps-oculusrift_uhq_h265.mp4"

frame = 8*60*60

debugger = Debugger(video, video[:-4])

debugger.load_logs()

#debugger.display_frame(frame)

#debugger.play_video(frame)

debugger.play_video(frame, record=False, downsize_ratio=1)  # , duration=10)