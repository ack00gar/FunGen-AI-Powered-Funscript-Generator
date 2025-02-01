import ctypes
import os
import tkinter as tk

from script_generator.constants import LOGO, ICON
from script_generator.gui.views.funscript_generator import FunscriptGeneratorPage
from script_generator.gui.views.settings import SettingsPage
from script_generator.state.app_state import AppState
from script_generator.utils.helpers import is_mac
from script_generator.debug.logger import log
from script_generator.constants import VERSION

# TODO this is a workaround and needs to be fixed properly
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        if hasattr(ctypes, "windll"):
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # For Windows DPI scaling
        # self.tk.call('tk', 'scaling', 1.0)
        self.title(f"VR & 2D POV Funscript AI Generator - v" + VERSION)
        self.geometry('800x900' if is_mac() else '700x900')
        self.resizable(False, False)

        self.iconphoto(False, tk.PhotoImage(file=LOGO))
        self.iconbitmap(ICON)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.container = tk.Frame(self)
        self.container.grid(row=0, column=0, sticky="nsew")
        self.container.grid_columnconfigure(0, weight=1)

        menu_bar = tk.Menu(self)

        # File Menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        # file_menu.add_command(label="Exit", command=self.quit)
        # menu_bar.add_cascade(label="File", menu=file_menu)
        menu_bar.add_command(label="Funscript generator", command=lambda: self.show_frame(PageNames.FUNSCRIPT_GENERATOR))
        menu_bar.add_command(label="Settings", command=lambda: self.show_frame(PageNames.SETTINGS))
        self.config(menu=menu_bar)

        self.state = AppState()
        self.state.set_is_cli(False)

        # Dictionary to store pages
        self.frames = {}

        # Ensure the window is always on top
        self.attributes("-topmost", True)
        self.after(200, self.reset_topmost)

    def reset_topmost(self):
        self.focus_force()
        self.attributes("-topmost", False)

    def show_frame(self, page_name):
        """Show a frame, creating it if necessary."""
        if page_name not in self.frames:
            # Dynamically create the page
            frame = self.create_page(page_name)
            if frame is not None:
                self.frames[page_name] = frame
                frame.grid(row=0, column=0, sticky="nsew")
            else:
                log.info(f"Page '{page_name}' not found!")
                return

        # Show the requested page
        frame = self.frames[page_name]
        frame.tkraise()

    def create_page(self, page_name):
        if page_name == PageNames.FUNSCRIPT_GENERATOR:
            return FunscriptGeneratorPage(parent=self.container, controller=self)
        elif page_name == PageNames.SETTINGS:
            return SettingsPage(parent=self.container, controller=self)
        return None


class PageNames:
    FUNSCRIPT_GENERATOR = "Funscript generator"
    SETTINGS = "Settings"


def start_app():
    app = App()
    app.show_frame(PageNames.FUNSCRIPT_GENERATOR)
    app.mainloop()
