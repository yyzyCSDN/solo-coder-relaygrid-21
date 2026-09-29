from __future__ import annotations

import math

from .models import (
    EventVersion,
    IngestResult,
    MetricEvidence,
    MetricKind,
    PowerQualityEvent,
    PowerQualitySample,
)

SOURCE_WEIGHT = {"relay": 1.0, "scada": 0.95, "meter": 0.85, "manual": 0.6}
QUALITY_WEIGHT = {"good": 1.0, "suspect": 0.75, "bad": 0.35}
_PHASES = ("A", "B", "C")


class PowerQualityAttributor:
    """Build and revise one persistent record for each continuous disturbance."""

    def __init__(
        self,
        *,
        nominal_voltage: float = 1.0,
        voltage_low: float = 0.9,
        voltage_high: float = 1.1,
        unbalance_threshold_pct: float = 3.0,
        current_over_a: float = 100.0,
        flicker_threshold: float = 0.1,
        max_gap_s: float = 5.0,
        min_duration_s: float = 2.0,
        min_observations: int = 2,
        min_confidence: float = 0.6,
    ):
        values = [
            nominal_voltage,
            voltage_low,
            voltage_high,
            unbalance_threshold_pct,
            current_over_a,
            flicker_threshold,
            max_gap_s,
            min_duration_s,
            min_confidence,
        ]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError("numeric thresholds must be finite")
        if nominal_voltage <= 0 or current_over_a <= 0 or max_gap_s < 0 or min_duration_s < 0 or min_confidence < 0:
            raise ValueError("invalid power-quality threshold")
        if not 0.0 < voltage_low < voltage_high or unbalance_threshold_pct < 0 or flicker_threshold < 0 or min_confidence > 1.0:
            raise ValueError("invalid power-quality threshold")
        if min_observations < 1:
            raise ValueError("minimum observations must be positive")
        self.nominal_voltage = float(nominal_voltage)
        self.voltage_low = float(voltage_low)
        self.voltage_high = float(voltage_high)
        self.unbalance_threshold_pct = float(unbalance_threshold_pct)
        self.current_over_a = float(current_over_a)
        self.flicker_threshold = float(flicker_threshold)
        self.max_gap_s = float(max_gap_s)
        self.min_duration_s = float(min_duration_s)
        self.min_observations = int(min_observations)
        self.min_confidence = float(min_confidence)
        self._events: dict[str, PowerQualityEvent] = {}
        self._archived: list[PowerQualityEvent] = []
        self._latest: dict[str, tuple[int, float]] = {}
        self._raw_versions: dict[tuple[str, int], list[PowerQualitySample]] = {}
        self.unassigned: list[tuple[PowerQualitySample, str]] = []
        self.rejected: list[tuple[PowerQualitySample, str]] = []
        self._next_id = 1

    @property
    def events(self) -> list[PowerQualityEvent]:
        return list(self._events.values())

    def get_event(self, event_id: str) -> PowerQualityEvent:
        return self._events[event_id]

    def freeze(self, event_id: str) -> PowerQualityEvent:
        event = self.get_event(event_id)
        event.frozen = True
        return event

    def raw_versions(self, point_id: str, sequence: int) -> tuple[PowerQualitySample, ...]:
        return tuple(self._raw_versions.get((point_id, sequence), ()))

    def ingest(self, sample: PowerQualitySample) -> IngestResult:
        self._validate(sample)
        evidence = self._metric_evidence(sample)
        identity = (sample.point_id, sample.sequence)
        key = (sample.point_id, sample.sequence, "metric" if evidence else "quality")
        raw_versions = self._raw_versions.setdefault(identity, [])
        if sample not in raw_versions:
            raw_versions.append(sample)

        owner, existing_key = self._owner(identity)
        switched_category = owner is not None and existing_key != key
        if switched_category:
            if owner.frozen:
                self.rejected.append((sample, "frozen_event"))
                self._advance_watermark(sample)
                return IngestResult("rejected", owner.event_id, "frozen_event")
            del owner.evidence[existing_key]

        if owner is not None:
            old = owner.evidence.get(key)
            if old is not None:
                if old == sample:
                    return IngestResult("ignored", owner.event_id, "duplicate_sample")
                if owner.frozen:
                    self.rejected.append((sample, "frozen_event"))
                    self._advance_watermark(sample)
                    return IngestResult("rejected", owner.event_id, "frozen_event")
                if sample.timestamp_s < old.timestamp_s:
                    self.rejected.append((sample, "older_revision"))
                    self._advance_watermark(sample)
                    return IngestResult("rejected", owner.event_id, "older_revision")
                if sample.timestamp_s == old.timestamp_s:
                    self.rejected.append((sample, "conflicting_revision"))
                    self._advance_watermark(sample)
                    return IngestResult("rejected", owner.event_id, "conflicting_revision")
                owner.evidence[key] = sample
                owner.sources.add(sample.source)
                if sample.equipment_id:
                    owner.equipment_ids.add(sample.equipment_id)
                self._rebuild(owner)
                self._record_version(owner, sample.sequence, "corrected_sample")
                self._advance_watermark(sample)
                return IngestResult("updated", owner.event_id, "corrected_sample")

        if owner is not None:
            self._attach(owner, sample, evidence is not None)
            self._rebuild(owner)
            self._record_version(owner, sample.sequence, "corrected_evidence_type")
            self._advance_watermark(sample)
            return IngestResult("updated", owner.event_id, "corrected_evidence_type")

        latest = self._latest.get(sample.point_id)
        is_late = latest is not None and sample.sequence < latest[0]
        if evidence is None and sample.telemetry_quality == "good":
            self._advance_watermark(sample)
            return IngestResult("ignored", None, "normal_sample")

        candidates = self._candidates(sample)
        frozen = [e for e in candidates if e.frozen]
        if frozen:
            self.rejected.append((sample, "frozen_event"))
            self._advance_watermark(sample)
            return IngestResult("rejected", frozen[0].event_id, "frozen_event")

        if not candidates:
            if evidence is None or is_late:
                self.unassigned.append((sample, "late_without_open_event" if is_late else "quality_without_disturbance"))
                self._advance_watermark(sample)
                return IngestResult("unassigned", None, "insufficient_context")

            event = self._create(sample)
            self._advance_watermark(sample)
            return IngestResult("created", event.event_id, "new_disturbance")

        if len(candidates) > 1:
            event = self._merge(candidates, sample, evidence is not None)
            self._advance_watermark(sample)
            return IngestResult("merged", event.event_id, "duplicate_disturbance_merged")

        event = candidates[0]
        self._attach(event, sample, evidence is not None)
        self._rebuild(event)
        self._record_version(event, sample.sequence, "late_sample" if is_late else "new_evidence")
        self._advance_watermark(sample)
        return IngestResult("updated", event.event_id, "event_revised")

    def _create(self, sample: PowerQualitySample) -> PowerQualityEvent:
        event_id = f"PQ-{self._next_id:04d}"
        self._next_id += 1
        event = PowerQualityEvent(
            event_id=event_id,
            section_id=sample.section_id,
            start_s=sample.timestamp_s,
            end_s=sample.timestamp_s,
            section_ids={sample.section_id},
        )
        self._events[event_id] = event
        self._attach(event, sample, True)
        self._rebuild(event)
        self._record_version(event, sample.sequence, "created")
        return event

    def _merge(self, candidates: list[PowerQualityEvent], sample: PowerQualitySample, changes_interval: bool) -> PowerQualityEvent:
        target = min(candidates, key=lambda e: e.event_id)
        start = target.start_s
        end = target.end_s
        max_version = target.version
        for event in candidates:
            if event is target:
                continue
            start = min(start, event.start_s)
            end = max(end, event.end_s)
            max_version = max(max_version, event.version)
            target.evidence.update(event.evidence)
            target.equipment_ids.update(event.equipment_ids)
            target.sources.update(event.sources)
            target.section_ids.update(event.section_ids)
            self._archived.append(event)
            del self._events[event.event_id]
        if changes_interval:
            start = min(start, sample.timestamp_s)
            end = max(end, sample.timestamp_s)
        target.start_s, target.end_s = start, end
        self._attach(target, sample, changes_interval)
        target.version = max_version + 1
        self._rebuild(target)
        self._record_version(target, sample.sequence, "merged_duplicate_disturbance", target.version)
        return target

    def _attach(self, event: PowerQualityEvent, sample: PowerQualitySample, changes_interval: bool) -> None:
        evidence = self._metric_evidence(sample)
        key = (sample.point_id, sample.sequence, "metric" if evidence else "quality")
        event.evidence[key] = sample
        event.sources.add(sample.source)
        event.section_ids.add(sample.section_id)
        if sample.equipment_id:
            event.equipment_ids.add(sample.equipment_id)
        if changes_interval and evidence:
            event.start_s = min(event.start_s, sample.timestamp_s)
            event.end_s = max(event.end_s, sample.timestamp_s)

    def _candidates(self, sample: PowerQualitySample) -> list[PowerQualityEvent]:
        out = []
        for event in self._events.values():
            same_place = sample.section_id in event.section_ids
            if sample.equipment_id and sample.equipment_id in event.equipment_ids:
                same_place = True
            if not same_place:
                continue
            in_window = event.start_s - self.max_gap_s <= sample.timestamp_s <= event.end_s + self.max_gap_s
            if in_window:
                out.append(event)
        return out

    def _owner(self, identity: tuple[str, int]) -> tuple[PowerQualityEvent | None, tuple[str, int, str] | None]:
        for event in self._events.values():
            for point_id, sequence, category in event.evidence:
                if (point_id, sequence) == identity:
                    return event, (point_id, sequence, category)
        return None, None

    def _metric_evidence(self, sample: PowerQualitySample) -> list[MetricEvidence] | None:
        out: list[MetricEvidence] = []
        if sample.voltages is not None:
            low = self.nominal_voltage * self.voltage_low
            high = self.nominal_voltage * self.voltage_high
            low_phases = tuple(p for p, v in zip(_PHASES, sample.voltages) if v < low)
            high_phases = tuple(p for p, v in zip(_PHASES, sample.voltages) if v > high)
            if low_phases:
                severity = max((low - v) / self.nominal_voltage for v in sample.voltages if v < low)
                out.append(MetricEvidence(sample, "voltage", low_phases, severity))
            if high_phases:
                severity = max((v - high) / self.nominal_voltage for v in sample.voltages if v > high)
                out.append(MetricEvidence(sample, "voltage", high_phases, severity))
            avg = sum(sample.voltages) / 3.0
            if avg:
                deviation = max(abs(v - avg) for v in sample.voltages) / avg * 100.0
                if deviation > self.unbalance_threshold_pct:
                    out.append(MetricEvidence(sample, "unbalance", _PHASES, deviation / 100.0))
        if sample.currents is not None:
            phases = tuple(p for p, v in zip(_PHASES, sample.currents) if v > self.current_over_a)
            if phases:
                severity = max(max(sample.currents) - self.current_over_a, 0.0) / self.current_over_a
                out.append(MetricEvidence(sample, "current", phases, severity))
        if sample.flicker is not None and sample.flicker > self.flicker_threshold:
            out.append(MetricEvidence(sample, "flicker", _PHASES, sample.flicker))
        return out or None

    def _rebuild(self, event: PowerQualityEvent) -> None:
        metric_samples: dict[tuple[str, int], PowerQualitySample] = {}
        evidence: list[MetricEvidence] = []
        quality_only: list[PowerQualitySample] = []
        for sample in event.evidence.values():
            found = self._metric_evidence(sample)
            if found:
                metric_samples[(sample.point_id, sample.sequence)] = sample
                evidence.extend(found)
            elif sample.telemetry_quality != "good":
                quality_only.append(sample)

        event.sources = {s.source for s in event.evidence.values()}
        event.section_ids = {s.section_id for s in event.evidence.values()}
        event.section_id = min(event.section_ids)
        event.equipment_ids = {s.equipment_id for s in event.evidence.values() if s.equipment_id}
        if metric_samples:
            metric_times = list(metric_samples.values())
            event.start_s = min(s.timestamp_s for s in metric_times)
            event.end_s = max(s.timestamp_s for s in metric_times)
        event.metric_kinds = {e.kind for e in evidence}
        event.probable_cause = self._cause(evidence)
        if not metric_samples:
            event.confidence = 0.0
            event.status = "uncertain"
            event.reasons = ("no_electrical_evidence",)
            return

        total_weight = 0.0
        score = 0.0
        for sample in metric_samples.values():
            severity = max([e.severity for e in evidence if e.sample == sample] + [0.0])
            weight = max(1.0, min(3.0, 1.0 + severity))
            quality = QUALITY_WEIGHT.get(sample.telemetry_quality, 0.0)
            source = SOURCE_WEIGHT.get(sample.source, 0.5)
            total_weight += weight
            score += weight * quality * source
        confidence = score / total_weight if total_weight else 0.0

        kinds = {e.kind for e in evidence}
        if len(kinds) >= 2:
            confidence *= 1.1
        if event.sources and len(event.sources) >= 2:
            confidence *= 1.08
        if len({s.point_id for s in metric_samples.values()}) >= 2:
            confidence *= 1.05
        for sample in quality_only:
            if sample.telemetry_quality == "suspect":
                confidence *= 0.8
            elif sample.telemetry_quality == "bad":
                confidence = min(confidence, 0.49)
        if any(s.telemetry_quality == "bad" for s in metric_samples.values()):
            confidence = min(confidence, 0.49)

        duration = max(0.0, event.end_s - event.start_s)
        confidence *= min(1.0, duration / self.min_duration_s) if self.min_duration_s else 1.0
        confidence *= min(1.0, len(metric_samples) / self.min_observations)
        event.confidence = round(max(0.0, min(1.0, confidence)), 6)

        reasons = []
        if len(metric_samples) < self.min_observations:
            reasons.append("insufficient_observations")
        if duration < self.min_duration_s:
            reasons.append("insufficient_duration")
        if event.confidence < self.min_confidence:
            reasons.append("insufficient_confidence")
        if any(s.telemetry_quality == "bad" for s in metric_samples.values()) or any(
            s.telemetry_quality == "bad" for s in quality_only
        ):
            reasons.append("bad_telemetry_quality")
        event.reasons = tuple(reasons)
        event.status = "confirmed" if not reasons else "uncertain"

    def _cause(self, evidence: list[MetricEvidence]) -> str:
        kinds = {e.kind for e in evidence}
        voltage = [e for e in evidence if e.kind == "voltage"]
        sag = any(any(v < self.nominal_voltage * self.voltage_low for v in e.sample.voltages or ()) for e in voltage)
        swell = any(any(v > self.nominal_voltage * self.voltage_high for v in e.sample.voltages or ()) for e in voltage)
        if "current" in kinds and "flicker" in kinds:
            return "load_switching"
        if "flicker" in kinds and ("voltage" in kinds or "unbalance" in kinds):
            return "voltage_regulation"
        if "unbalance" in kinds and "current" in kinds:
            return "phase_unbalance"
        if "unbalance" in kinds:
            return "phase_unbalance"
        if sag and swell:
            return "voltage_fluctuation"
        if sag:
            return "voltage_sag"
        if swell:
            return "voltage_swell"
        if "flicker" in kinds:
            return "flicker"
        if "current" in kinds:
            return "overcurrent"
        return "insufficient_evidence"

    def _record_version(self, event: PowerQualityEvent, sequence: int, reason: str, version: int | None = None) -> None:
        if version is None:
            event.version += 1 if event.version_history else 0
            version = event.version
        event.version_history.append(
            EventVersion(
                version=version,
                revised_by_sequence=sequence,
                reason=reason,
                evidence_keys=tuple(sorted(event.evidence)),
                raw_data=tuple(sorted(event.evidence.values(), key=lambda s: (s.timestamp_s, s.point_id, s.sequence))),
            )
        )

    def _advance_watermark(self, sample: PowerQualitySample) -> None:
        current = self._latest.get(sample.point_id)
        if current is None or sample.sequence > current[0] or (
            sample.sequence == current[0] and sample.timestamp_s > current[1]
        ):
            self._latest[sample.point_id] = (sample.sequence, sample.timestamp_s)

    def _validate(self, sample: PowerQualitySample) -> None:
        if not sample.point_id or not sample.section_id:
            raise ValueError("point and section are required")
        values = [sample.timestamp_s, sample.flicker]
        values.extend(sample.voltages or ())
        values.extend(sample.currents or ())
        if not all(math.isfinite(v) for v in values if v is not None):
            raise ValueError("measurements must be finite")
        if sample.voltages is not None and len(sample.voltages) != 3:
            raise ValueError("three voltages are required")
        if sample.currents is not None and len(sample.currents) != 3:
            raise ValueError("three currents are required")
        if sample.telemetry_quality not in QUALITY_WEIGHT:
            raise ValueError("invalid telemetry quality")
