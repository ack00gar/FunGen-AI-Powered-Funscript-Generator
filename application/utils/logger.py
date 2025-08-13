import logging
import inspect
from logging.handlers import RotatingFileHandler
from config import constants


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

    def __init__(self, level=logging.DEBUG, log_file=None,
                 app_logic_instance=None, status_level_durations=None):
        """
        Initializes and configures the root logger for the entire application.

        Args:
            level (int, optional): The master logging level. Defaults to logging.DEBUG.
            log_file (str, optional): Path to a file to save logs.
            app_logic_instance (object, optional): Instance for status messages.
            status_level_durations (dict, optional): Durations for status messages.
        """
        # Configure the root logger directly to ensure consistency.
        self.logger = logging.getLogger()  # Get the root logger
        self.logger.setLevel(level)

        # Clear any existing handlers to prevent duplicates from previous runs or basicConfig.
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Console Handler with colors
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(ColoredFormatter())
        self.logger.addHandler(ch)

        # File Handler (optional)
        if log_file:
            # Use rotating file handler to keep log size manageable
            fh = RotatingFileHandler(
                log_file,
                mode='a',
                maxBytes=getattr(constants, 'LOG_MAX_BYTES', 5 * 1024 * 1024),
                backupCount=getattr(constants, 'LOG_BACKUP_COUNT', 3),
                encoding='utf-8'
            )
            fh.setLevel(level)
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
            smh.setLevel(level)
            self.logger.addHandler(smh)

    def get_logger(self):
        """
        Returns the configured logger instance.
        """
        # Since we configured the root, any call to getLogger will use this config.
        return logging.getLogger(self.get_caller_name())

    def get_caller_name(self):
        """Helper to get the name of the module that called get_logger."""
        try:
            frm = inspect.stack()[2] # 2 levels up to get the caller of get_logger
            mod = inspect.getmodule(frm[0])
            return mod.__name__ if mod else '__main__'
        except IndexError:
            return 'unknown_module'
