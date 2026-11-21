"""Detail view subpackage: report, playback, and track detail controller."""

from .controller import TrackDetailController, resolve_audition_results
from .report import render_track_detail_html, render_summary_html
from .playback import PlaybackController

__all__ = [
    "TrackDetailController",
    "resolve_audition_results",
    "render_track_detail_html",
    "render_summary_html",
    "PlaybackController",
]
