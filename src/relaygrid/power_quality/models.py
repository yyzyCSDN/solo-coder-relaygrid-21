from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MetricKind = Literal["voltage", "current", "unbalance", "flicker"]
QualityCode = Literal["good", "suspect", "bad"]


@dataclass(frozen=True)
class PowerQualitySample:
    """A timestamped three-phase PQ observation and its telemetry quality."""

    point_id: str
    section_id: str
    timestamp_s: float
    sequence: int
    voltages: tuple[float, float, float] | None = None
    currents: tuple[float, float, float] | None = None
    flicker: float | None = None
    telemetry_quality: QualityCode = "good"
    equipment_id: str | None = None
    source: str = "scada"


@dataclass(frozen=True)
class MetricEvidence:
    sample: PowerQualitySample
    kind: MetricKind
    phases: tuple[str, ...]
    severity: float


@dataclass(frozen=True)
class EventVersion:
    version: int
    revised_by_sequence: int
    reason: str
    evidence_keys: tuple[tuple[str, int, str], ...]
    raw_data: tuple[PowerQualitySample, ...]


@dataclass
class PowerQualityEvent:
    event_id: str
    section_id: str
    section_ids: set[str]
    start_s: float
    end_s: float
    evidence: dict[tuple[str, int, str], PowerQualitySample] = field(default_factory=dict)
    equipment_ids: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    version: int = 1
    version_history: list[EventVersion] = field(default_factory=list)
    frozen: bool = False
    status: str = "uncertain"
    probable_cause: str = "insufficient_evidence"
    confidence: float = 0.0
    reasons: tuple[str, ...] = ("insufficient_evidence",)
    metric_kinds: set[MetricKind] = field(default_factory=set)

    @property
    def affected_sections(self) -> tuple[str, ...]:
        return tuple(sorted(self.section_ids))

    @property
    def affected_equipment(self) -> tuple[str, ...]:
        return tuple(sorted(self.equipment_ids))

    @property
    def raw_data_versions(self) -> tuple[PowerQualitySample, ...]:
        return tuple(sorted(self.evidence.values(), key=lambda s: (s.timestamp_s, s.point_id, s.sequence)))

    def latest_version(self) -> EventVersion | None:
        return self.version_history[-1] if self.version_history else None


@dataclass(frozen=True)
class IngestResult:
    status: Literal["created", "updated", "merged", "ignored", "rejected", "unassigned"]
    event_id: str | None
    reason: str
