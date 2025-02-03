import time

from script_generator.debug.logger import log
from script_generator.state.app_state import AppState


def stop_processing(state: AppState):
    if state.analyze_task:
        log.debug("Stopping funscript generation task")
        state.analyze_task.stop()
        time.sleep(2)

    # TODO stop tracking logic

