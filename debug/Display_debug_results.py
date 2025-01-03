from utils.lib_Debugger import Debugger

#video = "/Users/k00gar/Downloads/SLR_SLR Originals_Vote for me_1920p_51071_FISHEYE190_alpha.mp4"

#video = "/Users/k00gar/Downloads/EBVR-093-C_reenc.mp4"

#video = "/Users/k00gar/Downloads/VRCONK_Kiara Cole_game_of_thrones_daenerys_targaryen_a_porn_parody_8K_180x180_3dh.mp4"
video = "/Users/k00gar/Downloads/OrgyVR -  Linda Lan, Lulu Chu, Ember Snow, Nicole Doshi - The Pussycat Girls.mp4"


frame = 150000  # int(600 * 59.94)

debugger = Debugger(video, video[:-4])

debugger.load_logs()

debugger.display_frame(frame)

#debugger.play_video(frame)

#debugger.play_video(frame, record=True, downsize_ratio=1, duration=4)