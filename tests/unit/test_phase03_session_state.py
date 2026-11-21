"""Focused tests for Phase 03: typed SessionContext runtime state & summary decomposition."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from sessionpreplib.config import default_config
from sessionpreplib.detectors import default_detectors
from sessionpreplib.models import (
    ExecutionReceipts,
    FormatConsensus,
    SessionContext,
)
from sessionpreplib.pipeline import Pipeline, load_session
from sessionpreplib.processors import default_processors
from sessionpreplib.queue import SessionQueue
from sessionpreplib.rendering import build_diagnostic_summary as rendering_summary
from sessionpreplib.summary import build_diagnostic_summary


def _make_wav(path, sr=48000, freq=440.0, duration=0.5, amplitude=0.5):
    t = np.arange(int(sr * duration), dtype=np.float64) / sr
    samples = amplitude * np.sin(2.0 * np.pi * freq * t)
    sf.write(str(path), samples, sr, subtype="PCM_24")


def test_session_context_typed_runtime_fields_populated_by_pipeline(tmp_path):
    _make_wav(tmp_path / "kick.wav", sr=48000, freq=80.0, amplitude=0.6)
    _make_wav(tmp_path / "bass.wav", sr=48000, freq=110.0, amplitude=0.3)

    cfg = default_config()
    session = load_session(str(tmp_path), cfg)
    assert isinstance(session, SessionContext)
    assert isinstance(session.format_consensus, FormatConsensus)
    assert isinstance(session.gain_staging, dict)
    assert isinstance(session.execution_receipts, ExecutionReceipts)

    pipeline = Pipeline(
        detectors=default_detectors(),
        audio_processors=default_processors(),
        config=cfg,
    )
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    pipeline.plan(session)

    assert "format_consistency" in session.session_detector_results
    assert "length_consistency" in session.session_detector_results
    assert session.format_consensus.most_common_sr == 48000
    assert session.format_consensus.most_common_bd is not None

    out_dir = tmp_path / "processed"
    pipeline.execute(session, str(out_dir))
    assert len(session.execution_receipts.executed_files) == 2
    assert session.execution_receipts.execute_errors == []


def test_summary_module_matches_rendering_reexport(tmp_path):
    _make_wav(tmp_path / "vox.wav", sr=48000, freq=300.0, amplitude=0.4)

    cfg = default_config()
    session = load_session(str(tmp_path), cfg)
    pipeline = Pipeline(
        detectors=default_detectors(),
        audio_processors=default_processors(),
        config=cfg,
    )
    pipeline.analyze_phase1(session)
    pipeline.analyze_phase2(session)
    pipeline.plan(session)

    assert rendering_summary is build_diagnostic_summary
    summary = build_diagnostic_summary(session)
    assert isinstance(summary, dict)
    assert summary["total_ok"] == 1


def test_queue_uses_typed_session_runtime_fields(tmp_path):
    _make_wav(tmp_path / "guitar.wav", sr=48000, freq=220.0, amplitude=0.4)

    cfg = default_config()
    queue = SessionQueue(cfg)
    queue.add(str(tmp_path), config={"execute": True}, label="job1")

    def factory(c):
        return Pipeline(detectors=default_detectors(), audio_processors=default_processors(), config=c)

    job = queue.run_next(factory)
    assert job is not None
    assert job.error is None
    assert job.result is not None
    assert len(job.result.session.execution_receipts.executed_files) == 1
