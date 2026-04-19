"""
Unified logging config for vox-agent.

Applied via uvicorn's log_config parameter so every logger — uvicorn's
own lifecycle and error loggers, uvicorn.access HTTP logs, and our
application logger — share a single consistent format.

Uses uvicorn's DefaultFormatter/AccessFormatter for colourised log
levels (auto-disabled when output is not a TTY). The uvicorn.error
logger is renamed to 'uvicorn' on output because its records are
informational, not error-level — the name is a historical uvicorn
quirk.
"""
import logging


class RenameUvicornErrorFilter(logging.Filter):
    """Rename 'uvicorn.error' to 'uvicorn' in log output.

    The uvicorn.error logger is uvicorn's internal channel for lifecycle
    messages ("Started server process", "Application startup complete").
    These are informational, not errors — the name is misleading.
    Renaming on output avoids confusion for anyone reading the logs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "uvicorn.error":
            record.name = "uvicorn"
        return True


LOG_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "rename_uvicorn": {
            "()": "voxagent.logging_config.RenameUvicornErrorFilter",
        },
    },
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "format": "%(asctime)s %(levelprefix)s %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": None,  # auto-detect TTY
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "format": (
                '%(asctime)s %(levelprefix)s %(name)s - %(client_addr)s - '
                '"%(request_line)s" %(status_code)s'
            ),
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": None,
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "filters": ["rename_uvicorn"],
            "stream": "ext://sys.stderr",
        },
        "access": {
            "class": "logging.StreamHandler",
            "formatter": "access",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
        },
        "uvicorn": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.error": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["access"],
            "level": "INFO",
            "propagate": False,
        },
        "voxagent": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
