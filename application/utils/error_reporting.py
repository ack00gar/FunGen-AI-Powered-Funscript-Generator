from __future__ import annotations

from typing import Optional, Union


def report_error(
    app,
    user_msg: str,
    detail: Optional[Union[str, BaseException]] = None,
    severity: str = "error",
) -> None:
    if severity not in ("warning", "error", "critical"):
        severity = "error"

    logger = getattr(app, "logger", None)
    if logger is not None:
        if isinstance(detail, BaseException):
            log_msg = f"{user_msg}: {detail}" if detail else user_msg
            log_fn = logger.warning if severity == "warning" else logger.error
            log_fn(log_msg, exc_info=detail)
        else:
            log_msg = f"{user_msg} ({detail})" if detail else user_msg
            if severity == "warning":
                logger.warning(log_msg)
            else:
                logger.error(log_msg)

    notify = getattr(app, "notify", None)
    if notify is not None:
        toast_type = "warning" if severity == "warning" else "error"
        duration = {"warning": 4.0, "error": 6.0, "critical": 10.0}[severity]
        notify(user_msg, toast_type, duration)
