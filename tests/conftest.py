import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

try:
    from ptsl import PTSL_pb2 as pt
except ImportError:
    pt = None


@pytest.fixture(scope="session")
def _font_app():
    """Keep the fallback face registered for the lifetime of headless Qt tests."""
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    original = app.font()
    font_id = -1
    windows_directory = os.environ.get("WINDIR")
    if windows_directory and app.platformName() == "offscreen" and not QFontDatabase.families():
        font_file = Path(windows_directory) / "Fonts" / "segoeui.ttf"
        font_id = QFontDatabase.addApplicationFont(str(font_file))
        if font_id >= 0:
            app.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 9))
    yield app
    if font_id >= 0:
        app.setFont(original)
        QFontDatabase.removeApplicationFont(font_id)


@pytest.fixture
def form_font(_font_app):
    """Restore the application's font after a geometry test changes its size."""
    from PySide6.QtGui import QFontDatabase, QFontInfo

    original = _font_app.font()
    if not QFontDatabase.families() or not QFontInfo(original).family():
        pytest.skip("Form geometry requires a resolved platform font")
    yield
    _font_app.setFont(original)


@pytest.fixture
def mock_engine():
    """Provides a MagicMock of a py-ptsl Engine.
    
    The raw_client.SendGrpcRequest method is mocked so test functions
    can assert what Request was sent, and configure what Response it
    should return.
    """
    engine = MagicMock()
    engine.client = MagicMock()
    engine.client.session_id = "test-session-123"
    engine.client.raw_client = MagicMock()
    
    # Default: returning an empty pt.Response just so it doesn't crash
    if pt:
        engine.client.raw_client.SendGrpcRequest.return_value = pt.Response(
            header=pt.ResponseHeader(status=pt.Completed),
            response_body_json="{}"
        )
    return engine

def make_ptsl_response(status, body_json="", error_json=""):
    """Helper to build a fake pt.Response protobuf."""
    if not pt:
        return None
        
    return pt.Response(
        header=pt.ResponseHeader(status=status),
        response_body_json=body_json,
        response_error_json=error_json
    )

@pytest.fixture
def ptsl_factory():
    """Factory fixture returning helper functions for creating protobuf responses."""
    
    class Factory:
        @staticmethod
        def ok(body_dict=None):
            if body_dict is None:
                body_dict = {}
            return make_ptsl_response(pt.Completed, json.dumps(body_dict))
            
        @staticmethod
        def fail(error_msg="Test error", error_type=pt.PT_UnknownError):
            # Pro Tools errors are usually a JSON serialized ResponseError protobuf
            from google.protobuf import json_format
            
            err = pt.ResponseError()
            e = err.errors.add()
            e.command_error_type = error_type
            e.command_error_message = error_msg
            
            err_json = json_format.MessageToJson(err, preserving_proto_field_name=True)
            return make_ptsl_response(pt.Failed, error_json=err_json)

    return Factory()
