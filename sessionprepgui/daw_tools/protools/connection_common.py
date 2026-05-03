"""Shared Pro Tools connection helpers.

This module intentionally avoids Qt imports so it can be used by the GUI and
by the isolated connection probe process.
"""

from __future__ import annotations

import logging

PTSL_HOST = "127.0.0.1"
PTSL_PORT = 31416
PTSL_PREFLIGHT_TIMEOUT_SECONDS = 1.0
PTSL_HANDSHAKE_TIMEOUT_SECONDS = 2.0
PTSL_CONNECT_TIMEOUT_MS = 15000
PTSL_READ_TIMEOUT_MS = 10000
PTSL_MUTATION_TIMEOUT_MS = 15000
PTSL_HOST_READY_TIMEOUT_SECONDS = 5.0

log = logging.getLogger(__name__)


def connection_failure_message(exc: Exception) -> tuple[str, str]:
    """Return a compact user-facing connection failure title and hint."""
    if isinstance(exc, ImportError):
        return (
            "py-ptsl is not installed",
            "Install the Pro Tools scripting dependency, then click Connect.",
        )

    return connection_failure_message_from_text(str(exc))


def connection_failure_message_from_text(text: str) -> tuple[str, str]:
    """Return a compact user-facing connection failure for an error string."""
    lowered = text.lower()
    unavailable_markers = (
        "statuscode.unavailable",
        "connection refused",
        "failed to connect to all addresses",
        "connectex",
        "deadline_exceeded",
        "still starting",
        "timed out",
        "timeout",
    )
    if any(marker in lowered for marker in unavailable_markers):
        return (
            "Pro Tools not available",
            "Start Pro Tools and make sure scripting is enabled, then click Connect.",
        )

    return (
        "Connection failed",
        "Check Pro Tools and PTSL, then click Connect to try again.",
    )


def connection_timeout_message() -> tuple[str, str, str]:
    """Return error, title, and hint for a timed-out connection attempt."""
    return (
        "Timed out while connecting to Pro Tools.",
        "Pro Tools not available",
        "Start Pro Tools and wait until it has finished launching, then click Connect.",
    )


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


def create_ptsl_engine_with_timeout():
    """Create a PTSL engine with bounded synchronous gRPC request timeouts."""
    log.debug(
        "Creating PTSL engine with handshake timeout: host=%s port=%s timeout=%.1fs",
        PTSL_HOST,
        PTSL_PORT,
        PTSL_HANDSHAKE_TIMEOUT_SECONDS,
    )
    import ptsl
    from ptsl import Engine

    original_send = ptsl.Client._send_sync_request

    def send_with_timeout(client, command_id, request_body_json, task_id=""):
        log.debug(
            "Sending PTSL handshake/request with timeout: command_id=%r "
            "task_id=%r timeout=%.1fs",
            command_id,
            task_id,
            PTSL_HANDSHAKE_TIMEOUT_SECONDS,
        )
        request = ptsl.PTSL_pb2.Request(
            header=ptsl.PTSL_pb2.RequestHeader(
                task_id=task_id,
                session_id=client.session_id,
                command=command_id,
                version=ptsl.client.PTSL_VERSION,
            ),
            request_body_json=request_body_json,
        )
        return client.raw_client.SendGrpcRequest(
            request,
            timeout=PTSL_HANDSHAKE_TIMEOUT_SECONDS,
        )

    try:
        ptsl.Client._send_sync_request = send_with_timeout
        engine = Engine(
            company_name="SessionPrep",
            application_name="Pro Tools Utils",
        )
        log.debug("Created PTSL engine for Pro Tools Utils")
        return engine
    finally:
        ptsl.Client._send_sync_request = original_send
