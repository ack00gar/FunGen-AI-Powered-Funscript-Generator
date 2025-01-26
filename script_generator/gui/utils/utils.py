def disable_widgets(buttons):
    for btn in buttons:
        btn.config(state='disabled')

def enable_widgets(buttons):
    for btn in buttons:
        btn.config(state='normal')

def set_progressbars_done(progress_bars):
    for progress, label in progress_bars:
        progress["value"] = progress["maximum"]
        label = "Done"

def reset_progressbars(progress_bars):
    for progress, label in progress_bars:
        progress["value"] = 0
        label = ""