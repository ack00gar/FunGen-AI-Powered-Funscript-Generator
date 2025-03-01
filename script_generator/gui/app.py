import ctypes
import os
import platform
import tkinter as tk
from ultralytics import settings

from script_generator.constants import LOGO, ICON
from script_generator.gui.controller.stop_processing import stop_processing
from script_generator.gui.views.funscript_generator import FunscriptGeneratorPage
from script_generator.gui.views.help import HelpPage
from script_generator.gui.views.help_debug_video import HelpDebugVideoPage
from script_generator.gui.views.settings import SettingsPage
from script_generator.state.app_state import AppState
from script_generator.utils.helpers import is_mac
from script_generator.debug.logger import log
from script_generator.constants import VERSION

# TODO this is a workaround and needs to be fixed properly
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
settings.update({"sync": False}) # Disable analytics and crash reporting

class App(tk.Tk):
    def __init__(self, state: AppState= None):
        super().__init__()
        if hasattr(ctypes, "windll"):
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # For Windows DPI scaling
        # self.tk.call('tk', 'scaling', 1.0)
        self.title(f"FunGen - AI-Powered FunScript Generation v{VERSION}")
        # Base size to prevent major resizing
        self.geometry('780x1000' if is_mac() else '708x975')

        self.iconphoto(False, tk.PhotoImage(file=LOGO))
        if platform.system() == "Windows":
            self.iconbitmap(ICON) # Ensure icon works for Windows and is skipped on Linux

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.container = tk.Frame(self)
        self.container.grid(row=0, column=0, sticky="nsew")
        self.container.grid_columnconfigure(0, weight=1)

        # App menu
        menu_bar = tk.Menu(self)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        view_menu = tk.Menu(menu_bar, tearoff=0)
        view_menu.add_command(label="Funscript generator", command=lambda: self.show_frame(PageNames.FUNSCRIPT_GENERATOR))
        view_menu.add_command(label="Settings", command=lambda: self.show_frame(PageNames.SETTINGS))
        menu_bar.add_cascade(label="View", menu=view_menu)
        help_menu = tk.Menu(menu_bar, tearoff=0)
        # TODO add options menu?
        # options_menu = tk.Menu(menu_bar, tearoff=0)
        # self.show_funscript_settings = tk.BooleanVar(value=False)
        # self.show_debug_options = tk.BooleanVar(value=False)
        # options_menu.add_checkbutton(
        #     label="Show Funscript options",
        #     variable=self.show_funscript_settings,
        #     onvalue=True,
        #     offvalue=False
        # )
        # options_menu.add_checkbutton(
        #     label="Show debug options",
        #     variable=self.show_debug_options,
        #     onvalue=True,
        #     offvalue=False
        # )
        # menu_bar.add_cascade(label="Options", menu=options_menu)
        help_menu.add_command(label="General", command=lambda: self.show_frame(PageNames.HELP))
        # help_menu.add_command(label="Debug video", command=lambda: self.show_frame(PageNames.HELP_DEBUG_VIDEO))
        menu_bar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menu_bar)

        self.state = state if state else AppState()
        self.state.set_is_cli(False)
        self.state.set_root(self)

        # Dictionary to store pages
        self.frames = {}

        # Ensure the window is always on top
        self.attributes("-topmost", True)
        self.after(200, self.reset_topmost)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Initialize with the default page and scale to contents
        self.show_frame(PageNames.FUNSCRIPT_GENERATOR)
        self.scale_to_contents()

    def reset_topmost(self):
        self.focus_force()
        self.attributes("-topmost", False)

    def scale_to_contents(self):
        """Force the window to update and scale to its contents."""
        self.update_idletasks()
        width = self.winfo_reqwidth()
        height = self.winfo_reqheight()
        self.geometry(f"{width}x{height}")
        self.resizable(False, False)

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
        """Factory method to create pages dynamically."""
        pages = {
            PageNames.FUNSCRIPT_GENERATOR: FunscriptGeneratorPage,
            PageNames.SETTINGS: SettingsPage,
            PageNames.HELP: HelpPage,
            PageNames.HELP_DEBUG_VIDEO: HelpDebugVideoPage,
        }
        return pages.get(page_name, lambda *args, **kwargs: None)(parent=self.container, controller=self)

    def on_close(self):
        log.info("Application is closing...")
        stop_processing(self.state)
        self.destroy()


class PageNames:
    FUNSCRIPT_GENERATOR = "Funscript generator"
    SETTINGS = "Settings"
    HELP = "Help"
    HELP_DEBUG_VIDEO = "Help: Debug video"


def start_app(state: AppState= None):
    app = App(state)
    app.show_frame(PageNames.FUNSCRIPT_GENERATOR)
    app.mainloop()
