"""Logging helpers."""

import logging
from pathlib import Path
from datetime import datetime

def init_logging(
    log_file: Path | None = None,
    display_pid: bool = False,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
):
    def custom_format(record: logging.LogRecord) -> str:
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fnameline = f"{record.pathname}:{record.lineno}"

        # NOTE: Display PID is useful for multi-process logging.
        if display_pid:
            pid_str = f"[PID: {os.getpid()}]"
            message = f"{record.levelname} {pid_str} {dt} {fnameline[-15:]:>15} {record.getMessage()}"
        else:
            message = f"{record.levelname} {dt} {fnameline[-15:]:>15} {record.getMessage()}"
        return message

    formatter = logging.Formatter()
    formatter.format = custom_format

    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)  # Set the logger to the lowest level to capture all messages

    # Remove unused default handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Write logs to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level.upper())
    logger.addHandler(console_handler)

    # Additionally write logs to file
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level.upper())
        logger.addHandler(file_handler)

def create_logger(name: str, log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
