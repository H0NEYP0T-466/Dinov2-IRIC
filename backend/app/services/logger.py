"""Centralized logging setup.

Configures the root logger with two handlers:

* a console handler at INFO level (stdout), and
* a file handler at DEBUG level, written to a timestamped file under
  ``backend/logs/``.

Noisy third-party loggers (e.g. ``uvicorn.access``) are silenced to WARNING so
that only the application's own detailed logs dominate the output.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

# Directory for log files, relative to backend/.
LOG_DIR: Path = Path(__file__).resolve().parent.parent.parent / "logs"

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> Path | None:
    """Initialize root + third-party logging. Idempotent.

    Returns the path of the active log file (or ``None`` if file logging is
    unavailable, e.g. read-only filesystem).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return None

    log_file: Path | None = None
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"server_{stamp}.log"
    except OSError:
        # Cannot create log dir (e.g. restricted FS); proceed console-only.
        log_file = None

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler -> INFO and above.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # File handler -> DEBUG and above (captures everything).
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Silence noisy third-party loggers.
    for noisy in ("uvicorn.access", "uvicorn.error", "watchfiles.main", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True

    boot = logging.getLogger("app.boot")
    boot.info("Logging initialized | level=%s | file=%s", logging.getLevelName(level), log_file)
    return log_file


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Call ``setup_logging`` at startup first."""
    return logging.getLogger(name)
