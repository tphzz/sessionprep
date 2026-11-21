import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import soundfile as sf

import sessionprep as cli


def run_cli(monkeypatch, directory, *options):
    monkeypatch.setattr(sys, "argv", ["sessionprep", str(directory), *options])
    return cli.process_files()


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "Take.wav"
    sf.write(path, 0.1 * np.sin(2 * np.pi * 250 * np.arange(4800) / 48000), 48000)
    return path


@pytest.mark.parametrize("options", [
    ["--json", "../Take.wav"],
    ["--report", "Take.wav"],
    ["--report", "notes.txt", "--json", "./notes.txt"],
    ["--overwrite", "--backup", "."],
    ["--overwrite", "--json", "_originals/Take.wav"],
    ["--overwrite", "--json", "_originals/Take.wav.sessionprep-backup.json"],
    ["--output_folder", "."],
])
def test_aliases_fail_before_any_output_is_published(audio, monkeypatch, options):
    source_bytes = audio.read_bytes()
    result = run_cli(monkeypatch, audio.parent, "--execute", *options)
    assert result == 1
    assert audio.read_bytes() == source_bytes
    assert [p for p in audio.parent.rglob("*") if p.is_file()] == [audio]


def test_report_hard_link_to_input_is_rejected(audio, monkeypatch):
    source_bytes = audio.read_bytes()
    target = audio.parent / "processed"
    target.mkdir()
    os.link(audio, target / "analysis.txt")
    assert run_cli(monkeypatch, audio.parent, "--execute", "--report", "analysis.txt") == 1
    assert audio.read_bytes() == source_bytes
    assert not (target / audio.name).exists()


@pytest.mark.skipif(os.name != "nt", reason="Win32 ignores trailing dots in regular filenames")
def test_new_report_names_cannot_alias_on_windows(audio, monkeypatch):
    assert run_cli(monkeypatch, audio.parent, "--execute", "--report", "notes.", "--json", "notes") == 1
    assert not (audio.parent / "processed").exists()


def test_custom_report_cannot_overwrite_audio_from_an_earlier_run(audio, monkeypatch):
    prior_backup = audio.parent / "Copies" / "Take.wav"
    prior_backup.parent.mkdir()
    prior_backup.write_bytes(audio.read_bytes())
    assert run_cli(monkeypatch, audio.parent, "--execute", "--report", str(prior_backup)) == 1
    assert prior_backup.read_bytes() == audio.read_bytes()
    assert not (audio.parent / "processed").exists()


def test_backup_file_cannot_be_a_directory(audio, monkeypatch):
    original = audio.read_bytes()
    (audio.parent / "Copies" / audio.name).mkdir(parents=True)
    assert run_cli(monkeypatch, audio.parent, "--execute", "--overwrite", "--backup", "Copies") == 1
    assert audio.read_bytes() == original


@pytest.mark.parametrize("relocated", [False, True])
def test_report_cannot_replace_a_backup_receipt_from_an_earlier_run(audio, monkeypatch, relocated):
    options = ("--execute", "--overwrite", "--backup", "Copies")
    assert run_cli(monkeypatch, audio.parent, *options) == 0
    receipt = audio.parent / "Copies" / "Take.wav.sessionprep-backup.json"
    if relocated:
        actual = receipt.with_name("relocated.json")
        receipt.rename(actual)
        try:
            receipt.symlink_to(actual)
        except OSError as exc:
            pytest.skip(f"Creating a symlink requires OS support: {exc}")
    recorded = receipt.read_bytes()
    assert run_cli(monkeypatch, audio.parent, "--execute", "--report", str(receipt)) == 1
    assert receipt.read_bytes() == recorded
    assert run_cli(monkeypatch, audio.parent, *options) == 0


def test_distinct_custom_report_locations_work_with_in_place_backup(audio, monkeypatch):
    original = audio.read_bytes()
    reports = audio.parent / "reports elsewhere"
    reports.mkdir()
    result = run_cli(monkeypatch, audio.parent, "--execute", "--overwrite", "--backup", "Copies",
                     "--report", str(reports / "result.txt"), "--json", str(reports / "result.json"))
    assert result == 0
    assert (audio.parent / "Copies" / audio.name).read_bytes() == original
    assert audio.read_bytes() != original
    assert (reports / "result.txt").is_file() and (reports / "result.json").is_file()


def test_interrupted_backup_can_be_retried_without_losing_the_first_generation(audio, monkeypatch):
    original = audio.read_bytes()
    backup = audio.parent / "Copies" / audio.name
    options = ("--execute", "--overwrite", "--backup", "Copies")

    def interrupted_copy(source, destination):
        Path(destination).write_bytes(Path(source).read_bytes()[:32])
        raise OSError("interrupted backup")

    with monkeypatch.context() as fault:
        fault.setattr(shutil, "copy2", interrupted_copy)
        assert run_cli(monkeypatch, audio.parent, *options) == 1
    assert audio.read_bytes() == original
    assert not backup.exists()
    assert run_cli(monkeypatch, audio.parent, *options) == 0
    assert audio.read_bytes() != original
    assert backup.read_bytes() == original
    assert run_cli(monkeypatch, audio.parent, *options) == 0
    assert backup.read_bytes() == original


def test_existing_truncated_backup_prevents_overwrite(audio, monkeypatch):
    original = audio.read_bytes()
    backup = audio.parent / "Copies" / audio.name
    backup.parent.mkdir()
    backup.write_bytes(original[:64])
    assert run_cli(monkeypatch, audio.parent, "--execute", "--overwrite", "--backup", "Copies") == 1
    assert audio.read_bytes() == original
    assert backup.read_bytes() == original[:64]


@pytest.mark.parametrize("substituted", ["unverified backup", "verified backup", "source"])
def test_matching_audio_geometry_cannot_authorize_overwrite(audio, monkeypatch, substituted):
    backup = audio.parent / "Copies" / audio.name
    options = ("--execute", "--overwrite", "--backup", "Copies")
    if substituted == "unverified backup":
        backup.parent.mkdir()
    else:
        assert run_cli(monkeypatch, audio.parent, *options) == 0
    replacement = audio if substituted == "source" else backup
    sf.write(replacement, np.full(4800, 0.125), 48000, subtype="PCM_16")
    before = {p: p.read_bytes() for p in (audio, backup)}
    assert run_cli(monkeypatch, audio.parent, *options) == 1
    assert {p: p.read_bytes() for p in before} == before


def test_legacy_identical_backup_can_be_used_for_nested_source(audio):
    from sessionpreplib.audio import load_track
    from sessionpreplib.models import SessionContext
    from sessionpreplib.pipeline import Pipeline

    source = audio.parent / "drums" / audio.name
    source.parent.mkdir()
    audio.rename(source)
    backup = audio.parent / "Copies" / "drums" / audio.name
    backup.parent.mkdir(parents=True)
    backup.write_bytes(source.read_bytes())
    original = source.read_bytes()
    track = load_track(str(source))
    track.filename = "drums/" + audio.name
    track.manual_gain_db = -6
    session = SessionContext([track], {})
    Pipeline([], [], {}).execute(session, str(audio.parent), str(audio.parent / "Copies"), True)
    assert session.config["_execute_errors"] == []
    assert backup.read_bytes() == original
    assert source.read_bytes() != original


def test_backup_removed_during_render_aborts_the_overwrite(audio, monkeypatch):
    original = audio.read_bytes()
    backup = audio.parent / "Copies" / audio.name
    render = cli.Pipeline._render_track

    def remove_backup(pipeline, track):
        backup.unlink()
        return render(pipeline, track)

    monkeypatch.setattr(cli.Pipeline, "_render_track", remove_backup)
    assert run_cli(monkeypatch, audio.parent, "--execute", "--overwrite", "--backup", "Copies") == 1
    assert audio.read_bytes() == original


def test_cli_returns_failure_for_missing_and_empty_input(tmp_path, monkeypatch):
    assert run_cli(monkeypatch, tmp_path / "missing", "--execute") == 1
    assert run_cli(monkeypatch, tmp_path, "--execute") == 1
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["sessionprep.txt", "sessionprep.json"])
def test_new_hard_link_after_validation_cannot_truncate_an_input(audio, monkeypatch, name):
    original = audio.read_bytes()
    target = audio.parent / "processed" / name
    execute = cli.Pipeline.execute

    def link_after_render(pipeline, *args, **kwargs):
        session = execute(pipeline, *args, **kwargs)
        os.link(audio, target)
        return session

    monkeypatch.setattr(cli.Pipeline, "execute", link_after_render)
    assert run_cli(monkeypatch, audio.parent, "--execute") == 0
    assert audio.read_bytes() == original
    assert not target.samefile(audio)
    assert audio.name in target.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["sessionprep.txt", "sessionprep.json"])
def test_report_publish_failure_preserves_previous_file(audio, monkeypatch, name):
    output = audio.parent / "processed"
    output.mkdir()
    target = output / name
    target.write_text("last successful report", encoding="utf-8")
    replace = os.replace

    def locked_report(source, destination):
        if Path(destination) == target:
            raise PermissionError("Report is locked")
        replace(source, destination)

    monkeypatch.setattr(os, "replace", locked_report)
    assert run_cli(monkeypatch, audio.parent, "--execute") == 1
    assert target.read_text(encoding="utf-8") == "last successful report"
    assert not list(output.glob(".sp-report-*"))
