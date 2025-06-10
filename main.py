import multiprocessing
import platform
from application.gui_components.app_gui import GUI
from application.logic.app_logic import ApplicationLogic

if __name__ == "__main__":
    if platform.system() != "Windows":
        multiprocessing.set_start_method('spawn', force=True)

    core_app = ApplicationLogic()
    gui = GUI(app_logic=core_app)
    core_app.gui_instance = gui
    gui.run()