import ctypes
import os
import tkinter as tk

from config import LOGO, ICON
from script_generator.gui.views.funscript_generator import FunscriptGeneratorPage
from script_generator.state.app_state import AppState

# TODO this is a workaround and needs to be fixed properly
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        if hasattr(ctypes, "windll"):
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # For Windows DPI scaling
        # self.tk.call('tk', 'scaling', 1.0)
        self.title("VR funscript generation")
        self.geometry("700x920")
        self.resizable(False, False)

        self.iconphoto(False, tk.PhotoImage(file=LOGO))
        self.iconbitmap(ICON)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.container = tk.Frame(self)
        self.container.grid(row=0, column=0, sticky="nsew")
        self.container.grid_columnconfigure(0, weight=1)

        self.state = AppState(is_cli=False)

        # Dictionary to store pages
        self.frames = {}


    def show_frame(self, page_name):
        """Show a frame, creating it if necessary."""
        if page_name not in self.frames:
            # Dynamically create the page
            frame = self.create_page(page_name)
            if frame is not None:
                self.frames[page_name] = frame
                frame.grid(row=0, column=0, sticky="nsew")
            else:
                print(f"Page '{page_name}' not found!")
                return

        # Show the requested page
        frame = self.frames[page_name]
        frame.tkraise()

    def create_page(self, page_name):
        if page_name == PageNames.FUNSCRIPT_GENERATOR:
            return FunscriptGeneratorPage(parent=self.container, controller=self)

        return None

class PageNames:
    FUNSCRIPT_GENERATOR = "Funscript generator"

def start_app():
    app = App()
    app.show_frame(PageNames.FUNSCRIPT_GENERATOR)
    app.mainloop()