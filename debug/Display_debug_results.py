from utils.lib_Debugger import Debugger

video = "/Users/k00gar/Downloads/SLR_SLR Originals_Vote for me_1920p_51071_FISHEYE190_alpha.mp4"

frame = 54000

debugger = Debugger(video, video[:-4])

debugger.load_logs()

#debugger.display_frame(frame)

#debugger.play_video(frame)

debugger.play_video(frame, record=False, downsize_ratio=1)  # , duration=10)