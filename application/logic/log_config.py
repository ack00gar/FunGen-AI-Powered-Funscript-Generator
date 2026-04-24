"""Logging configuration: purge old entries, silence noisy third parties, set level."""
from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime, timedelta


# Third-party loggers we quiet at startup. Keys are logger names, values
# the level we clamp them to.
_THIRD_PARTY_LEVELS = {
    'coremltools': logging.ERROR,
    'ultralytics': logging.WARNING,
    'torch': logging.WARNING,
    'torchvision': logging.WARNING,
    'requests': logging.WARNING,
    'urllib3': logging.WARNING,
    'PIL': logging.WARNING,
    'matplotlib': logging.WARNING,
}


def purge_old_log_entries(log_file_path: str, max_age_days: int = 7) -> None:
    """Drop log lines older than `max_age_days` from `log_file_path`.

    No-op if the file is missing or already trimmed. Atomic rewrite via
    sibling tmp file + os.replace, safe against daemon-thread kill on exit.
    """
    try:
        if not os.path.exists(log_file_path):
            return
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()

        first_line_to_keep_index = -1
        for i, line in enumerate(all_lines):
            try:
                line_date = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
                if line_date >= cutoff_date:
                    first_line_to_keep_index = i
                    break
            except (ValueError, IndexError):
                continue

        # Skip the rewrite if nothing would change — avoids needlessly
        # touching the log file on every startup.
        if first_line_to_keep_index <= 0:
            return

        lines_to_keep = all_lines[first_line_to_keep_index:]
        tmp_path = log_file_path + ".purge-tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.writelines(lines_to_keep)
        os.replace(tmp_path, log_file_path)
    except Exception as e:
        logging.getLogger(__name__).debug(f"Log purge failed: {e}")


def configure_third_party_logging() -> None:
    """Clamp noisy third-party loggers and suppress known stray warnings."""
    warnings.filterwarnings(
        "ignore", message="scikit-learn version .* is not supported")
    for logger_name, level in _THIRD_PARTY_LEVELS.items():
        logging.getLogger(logger_name).setLevel(level)
    # Extra: ultralytics model-loading warnings are chatty even at WARNING
    logging.getLogger('ultralytics').setLevel(logging.ERROR)


def set_application_logging_level(app, level_name: str) -> None:
    """Set app-wide log level. `app` is the ApplicationLogic instance."""
    numeric_level = getattr(logging, level_name.upper(), None)
    if numeric_level is not None and hasattr(app, '_logger_instance'):
        app._logger_instance.set_level(numeric_level)
        app.logging_level_setting = level_name
        # Persist so subsequent CLI/batch runs pick up the same level.
        try:
            app.app_settings.config.logging.level = level_name
        except Exception as e:
            app.logger.debug(f"Could not persist logging_level: {e}")
        app.logger.info(f"Logging level changed to: {level_name}",
                        extra={'status_message': True})
    else:
        app.logger.warning(f"Failed to set logging level or invalid level: {level_name}")
