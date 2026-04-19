"""
Unified logging config for vox-agent.

Applied via uvicorn's log_config parameter so every logger — uvicorn's
own lifecycle and error loggers, uvicorn.access HTTP logs, and our
application logger — share a single consistent format.

The uvicorn.access logger gets its own formatter because access log
records carry extra fields (client_addr, request_line, status_code)
that don't exist on regular log records.
"""

LOG_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)s %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "access": {
            "format": (
                '%(asctime)s %(levelname)s %(name)s - %(client_addr)s - '
                '"%(request_line)s" %(status_code)s'
            ),
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "class": "logging.StreamHandler",
            "formatter": "access",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        # Root fallback — anything without a specific config lands here
        "": {
            "handlers": ["default"],
            "level": "INFO",
        },
        # Uvicorn lifecycle and error loggers
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
        # HTTP access log — separate formatter for access-log fields
        "uvicorn.access": {
            "handlers": ["access"],
            "level": "INFO",
            "propagate": False,
        },
        # Application logger
        "voxagent": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
