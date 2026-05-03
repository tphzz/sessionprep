from __future__ import annotations

from sessionpreplib.daw_processors.ptsl_connection import (
    DEFAULT_PTSL_APPLICATION_NAME,
    DEFAULT_PTSL_HOST,
    DEFAULT_PTSL_PORT,
    ProToolsConnectionSettings,
    connection_failure_message,
    connection_timeout_message,
)


def test_connection_settings_from_flat_config():
    settings = ProToolsConnectionSettings.from_config(
        {
            "protools_host": "192.0.2.10",
            "protools_port": "31417",
            "protools_company_name": "Example Co",
            "protools_application_name": "Example App",
            "protools_command_delay": "0.25",
        }
    )

    assert settings.host == "192.0.2.10"
    assert settings.port == 31417
    assert settings.company_name == "Example Co"
    assert settings.application_name == "Example App"
    assert settings.command_delay == 0.25
    assert settings.address == "192.0.2.10:31417"


def test_connection_settings_from_structured_config():
    settings = ProToolsConnectionSettings.from_config(
        {
            "daw_processors": {
                "protools": {
                    "protools_host": "127.0.0.1",
                    "protools_port": 31416,
                    "protools_application_name": "sessionprep",
                }
            }
        }
    )

    assert settings.host == "127.0.0.1"
    assert settings.port == 31416
    assert settings.application_name == "sessionprep"


def test_connection_settings_fall_back_for_invalid_values():
    settings = ProToolsConnectionSettings.from_config(
        {
            "protools_host": " ",
            "protools_port": 70000,
            "protools_application_name": "",
        }
    )

    assert settings.host == DEFAULT_PTSL_HOST
    assert settings.port == DEFAULT_PTSL_PORT
    assert settings.application_name == DEFAULT_PTSL_APPLICATION_NAME


def test_connection_failure_message_for_missing_ptsl():
    title, hint = connection_failure_message(ImportError("No module named 'ptsl'"))

    assert title == "py-ptsl is not installed"
    assert "dependency" in hint


def test_connection_failure_message_for_grpc_unavailable():
    title, hint = connection_failure_message(
        RuntimeError("StatusCode.UNAVAILABLE: failed to connect to all addresses")
    )

    assert title == "Pro Tools not available"
    assert "Start Pro Tools" in hint


def test_connection_timeout_message():
    error, title, hint = connection_timeout_message()

    assert "Timed out" in error
    assert title == "Pro Tools not available"
    assert "finished launching" in hint
