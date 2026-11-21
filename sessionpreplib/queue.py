from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import os
from typing import Any, Callable
from uuid import uuid4

from .models import SessionJob, SessionResult, JobStatus, SessionContext
from .pipeline import Pipeline, load_session
from .events import EventBus


class SessionQueue:
    """Ordered queue of session jobs. Processes sequentially."""

    def __init__(self, default_config: dict[str, Any] | None = None):
        self._jobs: list[SessionJob] = []
        self._default_config: dict[str, Any] = deepcopy(default_config or {})

    def add(
        self,
        source_dir: str,
        config: dict[str, Any] | None = None,
        priority: int = 0,
        label: str | None = None,
    ) -> SessionJob:
        """
        Enqueue a session. Config is merged over defaults
        (default < per-job overrides).
        """
        if label and any(job.job_id == label for job in self._jobs):
            raise ValueError(f"Duplicate job ID: {label}")
        merged = deepcopy({**self._default_config, **(config or {})})
        job = SessionJob(
            job_id=label or str(uuid4()),
            source_dir=source_dir,
            config=merged,
            priority=priority,
        )
        self._jobs.append(job)
        self._jobs.sort(key=lambda j: j.priority)
        return job

    def remove(self, job_id: str) -> bool:
        """Remove a pending job. Returns True if found and removed."""
        for i, job in enumerate(self._jobs):
            if job.job_id == job_id and job.status == JobStatus.PENDING:
                self._jobs.pop(i)
                return True
        return False

    def reorder(self, job_id: str, new_priority: int) -> None:
        """Change a pending job's priority and re-sort."""
        for job in self._jobs:
            if job.job_id == job_id and job.status == JobStatus.PENDING:
                job.priority = new_priority
                break
        self._jobs.sort(key=lambda j: j.priority)

    def cancel(self, job_id: str) -> None:
        """Mark a pending job as cancelled."""
        for job in self._jobs:
            if job.job_id == job_id and job.status == JobStatus.PENDING:
                job.status = JobStatus.CANCELLED
                break

    def pending(self) -> list[SessionJob]:
        return [j for j in self._jobs if j.status == JobStatus.PENDING]

    def completed(self) -> list[SessionJob]:
        return [j for j in self._jobs if j.status == JobStatus.COMPLETED]

    def all_jobs(self) -> list[SessionJob]:
        return list(self._jobs)

    def run_next(
        self,
        pipeline_factory: Callable[[dict[str, Any]], Pipeline],
        event_bus: EventBus | None = None,
    ) -> SessionJob | None:
        """
        Run the next pending job. Returns the completed/failed job,
        or None if queue is empty.
        """
        job = next(
            (j for j in self._jobs if j.status == JobStatus.PENDING), None
        )
        if not job:
            return None

        job.status = JobStatus.RUNNING
        if event_bus:
            event_bus.emit("job.start", job_id=job.job_id)

        try:
            if not os.path.isdir(job.source_dir):
                raise ValueError(f"Input directory does not exist: {job.source_dir}")
            pipeline = pipeline_factory(job.config)
            session = load_session(job.source_dir, job.config, event_bus=event_bus)
            if not session.tracks:
                raise ValueError(f"No audio files found in {job.source_dir}")
            job.result = SessionResult(session=session)
            pipeline.analyze_phase1(session)
            pipeline.analyze_phase2(session)
            pipeline.plan(session)

            if job.config.get("execute"):
                source_dir = job.source_dir
                if job.config.get("overwrite"):
                    output_dir = source_dir
                    backup_dir = os.path.join(source_dir, job.config.get("backup", "_originals"))
                    pipeline.execute(session, output_dir, backup_dir=backup_dir, is_overwriting=True)
                else:
                    output_folder = job.config.get("output_folder", "processed")
                    output_dir = os.path.join(source_dir, output_folder)
                    pipeline.execute(session, output_dir)

            errors = _job_errors(session)
            if errors:
                job.status = JobStatus.FAILED
                job.error = "\n".join(errors)
            else:
                job.status = JobStatus.COMPLETED
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)

        job.completed_at = datetime.now()

        if event_bus:
            event_bus.emit("job.complete", job_id=job.job_id,
                           status=job.status.value)

        return job

    def run_all(
        self,
        pipeline_factory: Callable[[dict[str, Any]], Pipeline],
        event_bus: EventBus | None = None,
        on_complete: Callable[[SessionJob], None] | None = None,
    ) -> list[SessionJob]:
        """Drain the queue. Callback fires after each job."""
        completed = []
        while self.pending():
            job = self.run_next(pipeline_factory, event_bus=event_bus)
            if job:
                completed.append(job)
                if on_complete:
                    on_complete(job)
        return completed


def _job_errors(session: SessionContext) -> list[str]:
    """Collect operational failures while keeping successful sibling results."""
    errors = list(session.config.get("_execute_errors", []))
    for track in session.tracks:
        if track.status != "OK":
            errors.append((track.filename, track.status))
        if message := track._cache.get("repair_error"):
            errors.append((track.filename, message))
        for result in (*track.detector_results.values(), *track.processor_results.values()):
            if result.error:
                errors.append((track.filename, result.error))
    for detector in session.detectors:
        for result in session.config.get(f"_session_det_{detector.id}", []):
            if result.error:
                errors.append((detector.id, result.error))
    return [f"{name}: {message}" for name, message in dict.fromkeys(errors)]
