def disable_widgets(widgets):
    for widget in widgets:
        widget.config(state='disabled')

def enable_widgets(widgets):
    for widget in widgets:
        widget.config(state='normal')

def set_progressbars_done(progress_bars):
    for progress, label in progress_bars:
        progress["value"] = progress["maximum"]
        label.config(text="Done")

def reset_progressbars(progress_bars):
    for progress, label in progress_bars:
        progress["value"] = 0
        label.config(text="")