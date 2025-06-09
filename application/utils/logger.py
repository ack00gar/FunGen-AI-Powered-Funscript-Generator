import logging
import inspect


class StatusMessageHandler(logging.Handler):
    """
    A custom logging handler to send messages to the application's status bar.
    """
    # Default durations for status messages based on log level
    DEFAULT_LEVEL_DURATIONS = {
        logging.INFO: 3.0,
        logging.WARNING: 5.0,
        logging.ERROR: 7.0,
        logging.CRITICAL: 10.0,
    }

    def __init__(self, set_status_message_func, level_durations=None):
        super().__init__()
        self.set_status_message_func = set_status_message_func
        self.level_durations = level_durations if level_durations is not None else self.DEFAULT_LEVEL_DURATIONS

    def emit(self, record):
        try:
            show_in_status = False
            final_duration = None

            # Priority 1: Explicit 'status_message' in extra
            # Allows forcing a message (e.g., DEBUG) to status or suppressing one.
            if hasattr(record, 'status_message'):
                if record.status_message is True:
                    show_in_status = True
                    # Use duration from extra, or from level config, or a fallback default (e.g., 3s for INFO-like forced messages)
                    default_fallback_duration = self.level_durations.get(record.levelno, 3.0)
                    final_duration = record.__dict__.get('duration', default_fallback_duration)
                elif record.status_message is False:
                    return  # Explicitly do not show in status

            # Priority 2: Implicitly via log level being in configured level_durations (if not handled by Priority 1)
            if not show_in_status and record.levelno in self.level_durations:
                show_in_status = True
                final_duration = self.level_durations[record.levelno]
                # Allow 'duration' in extra to override the level-based duration
                if hasattr(record, 'duration'):
                    final_duration = record.duration

            if show_in_status and final_duration is not None:
                message_to_display = record.getMessage()  # Get the raw message
                self.set_status_message_func(message_to_display, final_duration)
        except Exception:
            self.handleError(record)  # Delegate to superclass's error handling


class ColoredFormatter(logging.Formatter):
    """
    A custom formatter to add colors to log messages based on log level.
    """
    # Define ANSI escape codes for colors (using more standard codes)
    GREY = "\x1b[90m"  # Bright Black (often renders as grey)
    GREEN = "\x1b[32m"  # Green
    YELLOW = "\x1b[33m"  # Yellow
    RED = "\x1b[31m"  # Red
    BOLD_RED = "\x1b[31;1m"  # Bold Red
    RESET = "\x1b[0m"  # Reset all attributes

    log_format_base = "%(asctime)s - %(name)s - %(levelname)-8s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s"

    FORMATS = {
        logging.DEBUG: GREY + log_format_base + RESET,
        logging.INFO: GREEN + log_format_base + RESET,
        logging.WARNING: YELLOW + log_format_base + RESET,
        logging.ERROR: RED + log_format_base + RESET,
        logging.CRITICAL: BOLD_RED + log_format_base + RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


class AppLogger:
    """
    A wrapper class to simplify the creation and configuration of a logger.
    """

    def __init__(self, logger_name=None, level=logging.DEBUG, log_file=None,
                 app_logic_instance=None, status_level_durations=None):
        """
        Initializes the logger.

        Args:
            logger_name (str, optional): Name of the logger.
                                         Defaults to the name of the calling module.
            level (int, optional): The logging level for console and file. Defaults to logging.DEBUG.
            log_file (str, optional): Path to a file to save logs.
                                      If None, logs are only printed to console.
            app_logic_instance (object, optional): Instance of ApplicationLogic (or any class
                                                   with a 'set_status_message(msg, duration)' method).
            status_level_durations (dict, optional): Dict mapping log levels (e.g., logging.INFO)
                                                     to durations for status messages. If None, uses defaults
                                                     from StatusMessageHandler.
        """
        if logger_name is None:
            # Automatically get the name of the module that is creating the logger
            frm = inspect.stack()[1]
            mod = inspect.getmodule(frm[0])
            logger_name = mod.__name__ if mod else '__main__'

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)  # Master level for the logger

        # Prevent duplicate handlers if logger already exists
        if not self.logger.handlers:
            # Console Handler with colors
            ch = logging.StreamHandler()
            ch.setLevel(level)  # Handler respects the master level
            ch.setFormatter(ColoredFormatter())
            self.logger.addHandler(ch)

            # File Handler (optional)
            if log_file:
                fh = logging.FileHandler(log_file)
                fh.setLevel(level)  # Handler respects the master level
                file_formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)-8s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                fh.setFormatter(file_formatter)
                self.logger.addHandler(fh)

            # Status Message Handler (optional)
            if app_logic_instance and hasattr(app_logic_instance, 'set_status_message'):
                smh = StatusMessageHandler(app_logic_instance.set_status_message,
                                           level_durations=status_level_durations)
                # The StatusMessageHandler will decide to show a message based on its
                # internal logic (level_durations config, or 'status_message' in extra).
                # Set its level to the logger's main level so it sees all relevant messages.
                smh.setLevel(level)
                self.logger.addHandler(smh)

    def get_logger(self):
        """
        Returns the configured logger instance.
        """
        return self.logger
