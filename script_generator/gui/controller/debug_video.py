import threading
from tkinter import messagebox

from script_generator.debug.debug_data import get_metrics_file_info
from script_generator.debug.logger import log
from script_generator.debug.video_player.play import play_debug_video
from script_generator.state.app_state import AppState
from script_generator.utils.helpers import is_mac


def debug_video(state: AppState):
    if not state.video_path:
        messagebox.showerror("Error", "Please select a video file.")
        return

    def run():
        state.set_video_info()

        exists, path, filename = get_metrics_file_info(state)
        if exists:
            play_debug_video(state=state, start_frame=state.frame_start)
        else:
            log.error(f"Debug logs file not found: {path}")
            messagebox.showinfo("Info", f"Debug logs file not found: {filename}")

    # Mac needs to run the gui on the same thread, this will block the main GUI so we don't do it on windows
    if is_mac():
        run()
    else:
        processing_thread = threading.Thread(target=run)
        processing_thread.start()
