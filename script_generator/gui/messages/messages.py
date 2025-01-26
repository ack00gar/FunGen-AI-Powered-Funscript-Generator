class UIMessage:
    pass

class ProgressMessage(UIMessage):
    def __init__(self, process, frames_processed, total_frames, eta):
        self.process = process
        self.frames_processed = frames_processed
        self.total_frames = total_frames
        self.eta = eta

class UpdateGUIState(UIMessage):
    def __init__(self, attr, value):
        self.attr = attr
        self.value = value