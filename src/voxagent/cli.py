"""
Console entry point for vox-agent.

Registered in pyproject.toml as `voxagent`. After `uv sync`, run with:

    uv run voxagent

This calls uvicorn.run() programmatically so we can pass our log_config
dict. The raw `uv run uvicorn voxagent.main:app --reload` command still
works but falls back to uvicorn's default log format.
"""
import uvicorn
from voxagent.config import settings
from voxagent.logging_config import LOG_CONFIG


def main() -> None:
    uvicorn.run(
        "voxagent.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_config=LOG_CONFIG,
    )


if __name__ == "__main__":
    main()