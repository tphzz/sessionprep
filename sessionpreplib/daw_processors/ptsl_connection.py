"""Shared Pro Tools PTSL connection helpers.

This module is intentionally Qt-free.  GUI code may use it through a worker
process, while DAW processors can use it directly from background threads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sessionpreplib.log import dbg

log = logging.getLogger(__name__)

DEFAULT_PTSL_HOST = "localhost"
DEFAULT_PTSL_PORT = 31416
DEFAULT_PTSL_COMPANY_NAME = "github.com"
DEFAULT_PTSL_APPLICATION_NAME = "sessionprep"
DEFAULT_PTSL_HANDSHAKE_TIMEOUT_SECONDS = 2.0
DEFAULT_PTSL_HOST_READY_TIMEOUT_SECONDS = 25.0
DEFAULT_PTSL_COMMAND_DELAY_SECONDS = 0.5
MIN_PTSL_PROTOCOL_VERSION = 2025


@dataclass(frozen=True)
class ProToolsConnectionSettings:
    """Configuration required to create a py-ptsl engine."""

    host: str = DEFAULT_PTSL_HOST
    port: int = DEFAULT_PTSL_PORT
    company_name: str = DEFAULT_PTSL_COMPANY_NAME
    application_name: str = DEFAULT_PTSL_APPLICATION_NAME
    handshake_timeout: float = DEFAULT_PTSL_HANDSHAKE_TIMEOUT_SECONDS
    host_ready_timeout: float = DEFAULT_PTSL_HOST_READY_TIMEOUT_SECONDS
    command_delay: float = DEFAULT_PTSL_COMMAND_DELAY_SECONDS

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "ProToolsConnectionSettings":
        """Build settings from either flat or structured SessionPrep config."""
        values = _protools_config_values(config or {})
        return cls(
            host=_non_empty_str(values.get("protools_host"), DEFAULT_PTSL_HOST),
            port=_int_in_range(values.get("protools_port"), DEFAULT_PTSL_PORT),
            company_name=_non_empty_str(
                values.get("protools_company_name"),
                DEFAULT_PTSL_COMPANY_NAME,
            ),
            application_name=_non_empty_str(
                values.get("protools_application_name"),
                DEFAULT_PTSL_APPLICATION_NAME,
            ),
            handshake_timeout=_positive_float(
                values.get("protools_handshake_timeout"),
                DEFAULT_PTSL_HANDSHAKE_TIMEOUT_SECONDS,
            ),
            host_ready_timeout=_positive_float(
                values.get("protools_host_ready_timeout"),
                DEFAULT_PTSL_HOST_READY_TIMEOUT_SECONDS,
            ),
            command_delay=_positive_float(
                values.get("protools_command_delay"),
                DEFAULT_PTSL_COMMAND_DELAY_SECONDS,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable settings dictionary."""
        return {
            "protools_host": self.host,
            "protools_port": self.port,
            "protools_company_name": self.company_name,
            "protools_application_name": self.application_name,
            "protools_handshake_timeout": self.handshake_timeout,
            "protools_host_ready_timeout": self.host_ready_timeout,
            "protools_command_delay": self.command_delay,
        }


def _protools_config_values(config: dict[str, Any]) -> dict[str, Any]:
    """Return the Pro Tools config section from flat or structured config."""
    daw_processors = config.get("daw_processors")
    if isinstance(daw_processors, dict):
        protools = daw_processors.get("protools")
        if isinstance(protools, dict):
            merged = dict(config)
            merged.update(protools)
            return merged
    return config


def _non_empty_str(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _int_in_range(value: Any, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    if 1 <= result <= 65535:
        return result
    return default


def _positive_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result > 0:
        return result
    return default


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


def create_ptsl_engine(
    settings: ProToolsConnectionSettings | None = None,
    *,
    handshake_timeout: float | None = None,
):
    """Create a PTSL engine with bounded synchronous handshake requests."""
    settings = settings or ProToolsConnectionSettings()
    timeout = handshake_timeout or settings.handshake_timeout
    log.debug(
        "Creating PTSL engine: host=%s port=%s company=%r application=%r "
        "handshake_timeout=%.1fs",
        settings.host,
        settings.port,
        settings.company_name,
        settings.application_name,
        timeout,
    )
    dbg(
        "Creating PTSL engine: "
        f"address={settings.address!r} application={settings.application_name!r} "
        f"handshake_timeout={timeout:.1f}s"
    )

    import ptsl
    from ptsl import Engine

    original_send = getattr(getattr(ptsl, "Client", None), "_send_sync_request", None)
    if original_send is None:
        engine = Engine(
            company_name=settings.company_name,
            application_name=settings.application_name,
            address=settings.address,
        )
        log.debug("Created PTSL engine without handshake patch")
        return engine

    def send_with_timeout(client, command_id, request_body_json, task_id=""):
        log.debug(
            "Sending PTSL handshake/request with timeout: command_id=%r "
            "task_id=%r timeout=%.1fs",
            command_id,
            task_id,
            timeout,
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
        return client.raw_client.SendGrpcRequest(request, timeout=timeout)

    try:
        ptsl.Client._send_sync_request = send_with_timeout
        engine = Engine(
            company_name=settings.company_name,
            application_name=settings.application_name,
            address=settings.address,
        )
        log.debug("Created PTSL engine")
        return engine
    finally:
        ptsl.Client._send_sync_request = original_send


def check_connectivity(
    settings: ProToolsConnectionSettings,
    *,
    ready_timeout: float | None = None,
    minimum_protocol_version: int = MIN_PTSL_PROTOCOL_VERSION,
) -> tuple[bool, str]:
    """Check whether Pro Tools is reachable and ready."""
    from . import ptsl_helpers

    engine = None
    try:
        engine = create_ptsl_engine(settings)
        version = engine.ptsl_version()
        log.debug("PTSL protocol version detected: %s", version)
        if version < minimum_protocol_version:
            return False, f"Protocol {minimum_protocol_version} or newer required"

        timeout = ready_timeout or settings.host_ready_timeout
        if not ptsl_helpers.wait_for_host_ready(
            engine,
            timeout=timeout,
            sleep_time=settings.command_delay,
        ):
            return (
                False,
                "Connected, but Pro Tools is busy or not ready. Please bring its window to the front.",
            )

        return True, f"Protocol: {version}"
    except ImportError:
        return False, "py-ptsl package not installed"
    except Exception as exc:
        return False, str(exc)
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception:
                log.debug("Failed to close PTSL engine after connectivity check")


class ProToolsConnection:
    """Small lifecycle wrapper around a py-ptsl engine."""

    def __init__(self, settings: ProToolsConnectionSettings | None = None):
        self.settings = settings or ProToolsConnectionSettings()
        self.engine = None

    def open(self):
        if self.engine is None:
            self.engine = create_ptsl_engine(self.settings)
        return self.engine

    def wait_until_ready(self, *, timeout: float | None = None) -> bool:
        from . import ptsl_helpers

        engine = self.open()
        return ptsl_helpers.wait_for_host_ready(
            engine,
            timeout=timeout or self.settings.host_ready_timeout,
            sleep_time=self.settings.command_delay,
        )

    def close(self) -> None:
        if self.engine is None:
            return
        try:
            self.engine.close()
        finally:
            self.engine = None
