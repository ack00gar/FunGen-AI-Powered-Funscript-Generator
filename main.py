from application.gui_components.app_gui import GUI
from application.logic.app_logic import ApplicationLogic

if __name__ == "__main__":
    core_app = ApplicationLogic()
    gui = GUI(app_logic=core_app)
    core_app.gui_instance = gui
    gui.run()