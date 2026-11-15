from __future__ import annotations

from typing import Any

from ..daw_processor import DawProcessor
from .protools import ProToolsDawProcessor
from .dawproject import DawProjectDawProcessor


def default_daw_processors() -> list[DawProcessor]:
    """Returns all built-in DAW processors (for config schema / preferences)."""
    return [
        ProToolsDawProcessor(),
        DawProjectDawProcessor(),
    ]


def create_runtime_daw_processors(
    flat_config: dict[str, Any],
    *,
    include_disabled: bool = False,
) -> list[DawProcessor]:
    """Create configured processor instances for runtime use.

    Both Pro Tools and DAWProject expand into one instance per configured
    template. Processors that are disabled via their ``*_enabled`` config 
    key are excluded unless requested for reconciling template state.
    """
    processors: list[DawProcessor] = []

    for inst in ProToolsDawProcessor.create_instances(flat_config):
        inst.configure(flat_config)
        if include_disabled or inst.enabled:
            processors.append(inst)

    for inst in DawProjectDawProcessor.create_instances(flat_config):
        inst.configure(flat_config)
        if include_disabled or inst.enabled:
            processors.append(inst)

    return processors
