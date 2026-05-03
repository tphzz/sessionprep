"""GUI-facing Pro Tools connection helpers."""

from __future__ import annotations

from sessionpreplib.daw_processors.ptsl_connection import (
    DEFAULT_PTSL_HANDSHAKE_TIMEOUT_SECONDS,
    DEFAULT_PTSL_HOST,
    DEFAULT_PTSL_HOST_READY_TIMEOUT_SECONDS,
    DEFAULT_PTSL_PORT,
    ProToolsConnectionSettings,
    connection_failure_message,
    connection_failure_message_from_text,
    connection_timeout_message,
    create_ptsl_engine,
)

PTSL_HOST = DEFAULT_PTSL_HOST
PTSL_PORT = DEFAULT_PTSL_PORT
PTSL_PREFLIGHT_TIMEOUT_SECONDS = 1.0
PTSL_HANDSHAKE_TIMEOUT_SECONDS = DEFAULT_PTSL_HANDSHAKE_TIMEOUT_SECONDS
PTSL_CONNECT_TIMEOUT_MS = 15000
PTSL_READ_TIMEOUT_MS = 10000
PTSL_MUTATION_TIMEOUT_MS = 15000
PTSL_HOST_READY_TIMEOUT_SECONDS = DEFAULT_PTSL_HOST_READY_TIMEOUT_SECONDS


def connection_button_state(state: str) -> tuple[str, str]:
    """Return button text and color for a connection state."""
    if state == "connected":
        return "Pro Tools: Connected", "#4caf50"
    if state == "checking":
        return "Pro Tools: Checking...", "#aaa"
    if state == "connecting":
        return "Pro Tools: Connecting...", "#aaa"
    if state == "failed":
        return "Pro Tools: Offline", "#f44336"
    return "Pro Tools: Offline", "#aaa"


def create_ptsl_engine_with_timeout(
    settings: ProToolsConnectionSettings | None = None,
):
    """Compatibility wrapper for older GUI callers."""
    return create_ptsl_engine(settings)
