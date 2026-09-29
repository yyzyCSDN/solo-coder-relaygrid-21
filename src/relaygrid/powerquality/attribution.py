"""Power-quality event attribution.

Merges three-phase voltage/current, flicker indicators and telemetry quality
samples into *sustained disturbances*, links each disturbance to the affected
section and equipment, and guards the archive against out-of-order and
duplicate filings.

Rules implemented here:

* Out-of-order (late) samples can only revise *unfrozen* disturbances. A
  frozen event is immutable; the raw sample is still retained for audit.
* The same disturbance cannot be filed twice: re-delivered samples and
  neighbouring meters on the same section are merged into one archive entry.
* When evidence is insufficient the conclusion stays ``uncertain`` with an
  explicit reason, and every filing/revision keeps an immutable raw version.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_KIND_PRIORITY = {
    "interruption": 0,
    "sag": 1,
    "swell": 2,
    "flicker": 3,
    "telemetry": 4,
    "unknown": 5,
}


@dataclass(frozen=True)
class PQSample:
    """One power-quality measurement from a metering point.

    ``voltage_pu`` and ``current_a`` are per-phase mappings keyed by phase
    letter (``"A"``, ``"B"``, ``"C"``); missing phases are simply absent.
    """

    point_id: str
    timestamp_s: float
    sequence: int
    voltage_pu: dict[str, float] = field(default_factory=dict)
    current_a: dict[str, float] = field(default_factory=dict)
    flicker_pst: float | None = None
    flicker_plt: float | None = None
    frequency_hz: float = 60.0
    quality: str = "good"


@dataclass(frozen=True)
class RawVersion:
    """Immutable audit entry recorded every time a disturbance is filed,
    revised, merged or frozen."""

    version: int
    at_s: float
    reason: str
    sample_sequences: tuple[int, ...]
    note: str = ""


@dataclass
class Disturbance:
    """A sustained power-quality disturbance filed against one or more
    metering points and (when mapped) one section."""

    disturbance_id: str
    point_ids: list[str]
    section_id: str | None
    equipment_ids: list[str]
    start_s: float
    end_s: float
    kind: str
    phases: list[str]
    min_voltage_pu: float
    max_voltage_pu: float
    max_current_a: float
    max_flicker_pst: float
    sample_count: int
    last_sequence: int
    version: int
    frozen: bool
    conclusion: str  # "certain" | "uncertain"
    conclusion_reason: str
    versions: list[RawVersion]
    merged_into: str | None
    created_at_s: float


@dataclass(frozen=True)
class IngestResult:
    """Outcome of ingesting a single sample."""

    status: str  # created | revised | duplicate | rejected_frozen | stored
    disturbance: Disturbance | None
    sample: PQSample
    late: bool = False
    note: str = ""


def dominant_kind(kinds) -> str:
    """Pick the most severe disturbance kind from a set of observed kinds."""
    if not kinds:
        return "unknown"
    return min(kinds, key=lambda k: _KIND_PRIORITY.get(k, _KIND_PRIORITY["unknown"]))


class PQAttributionEngine:
    """Attributes sustained disturbances from power-quality samples.

    Parameters
    ----------
    gap_tolerance_s:
        Two disturbed samples more than this many seconds apart start a new
        disturbance.
    dedupe_tolerance_s:
        Re-delivered samples whose start lands within this window of an
        existing disturbance are treated as duplicates.
    sag/swell/interruption_threshold:
        Per-unit voltage thresholds used to classify a sample.
    flicker_threshold:
        Plt/Pst value above which flicker is flagged.
    point_section:
        Optional mapping ``point_id -> section_id`` used to link disturbances
        to the affected section.
    section_equipment:
        Optional mapping ``section_id -> [equipment_id, ...]`` used to link
        disturbances to affected equipment.
    """

    def __init__(
        self,
        *,
        gap_tolerance_s: float = 300.0,
        dedupe_tolerance_s: float = 2.0,
        sag_threshold: float = 0.90,
        swell_threshold: float = 1.10,
        interruption_threshold: float = 0.10,
        flicker_threshold: float = 1.0,
        point_section: dict[str, str] | None = None,
        section_equipment: dict[str, list[str]] | None = None,
    ):
        if gap_tolerance_s < 0 or dedupe_tolerance_s < 0:
            raise ValueError("tolerances cannot be negative")
        if not (0.0 < interruption_threshold < sag_threshold < swell_threshold):
            raise ValueError("require 0 < interruption < sag < swell")
        self.gap_tolerance_s = gap_tolerance_s
        self.dedupe_tolerance_s = dedupe_tolerance_s
        self.sag_threshold = sag_threshold
        self.swell_threshold = swell_threshold
        self.interruption_threshold = interruption_threshold
        self.flicker_threshold = flicker_threshold
        self.point_section = dict(point_section or {})
        self.section_equipment = {k: list(v) for k, v in (section_equipment or {}).items()}
        self._samples: dict[str, list[PQSample]] = {}
        self._disturbances: list[Disturbance] = []
        self._clock: float = 0.0
        self._counter = 0

    # ------------------------------------------------------------------ API

    def ingest(self, sample: PQSample) -> IngestResult:
        self._validate(sample)
        self._samples.setdefault(sample.point_id, []).append(sample)
        self._clock = max(self._clock, sample.timestamp_s)
        if not self._is_disturbed(sample):
            return IngestResult("stored", None, sample, note="sample within normal limits")
        section = self._section_of(sample.point_id)
        frozen = self._find_frozen(sample.point_id, section, sample.timestamp_s)
        if frozen is not None:
            return IngestResult(
                "rejected_frozen",
                frozen,
                sample,
                late=sample.sequence <= frozen.last_sequence,
                note="event is frozen; raw sample retained",
            )
        dup = self._duplicate_exists(sample.point_id, sample.timestamp_s)
        if dup is not None:
            return IngestResult(
                "duplicate", dup, sample, note="same disturbance already filed"
            )
        match = self._find_match(sample.point_id, section, sample.timestamp_s)
        if match is not None:
            late = sample.sequence <= match.last_sequence
            if sample.point_id not in match.point_ids:
                match.point_ids.append(sample.point_id)
            self._revise(match, sample, late)
            return IngestResult("revised", match, sample, late=late)
        disturbance = self._create(sample)
        return IngestResult("created", disturbance, sample)

    def freeze(self, disturbance_id: str) -> Disturbance:
        """Freeze an archived disturbance. Frozen events are immutable."""
        d = self._require(disturbance_id)
        if d.frozen:
            return d
        d.frozen = True
        d.versions.append(RawVersion(d.version, self._clock, "frozen", (), "archived"))
        return d

    def auto_freeze(self, now_s: float | None = None, freeze_after_s: float = 3600.0) -> list[str]:
        """Freeze unfrozen disturbances that have been quiet for longer than
        ``freeze_after_s``. Returns the ids that were frozen."""
        now_s = self._clock if now_s is None else now_s
        frozen = []
        for d in self._disturbances:
            if not d.frozen and d.end_s < now_s - freeze_after_s:
                d.frozen = True
                d.versions.append(RawVersion(d.version, self._clock, "frozen", (), "auto-archived"))
                frozen.append(d.disturbance_id)
        return frozen

    def disturbances(self, *, include_frozen: bool = True, include_merged: bool = False) -> list[Disturbance]:
        out = []
        for d in self._disturbances:
            if not include_frozen and d.frozen:
                continue
            if not include_merged and d.merged_into is not None:
                continue
            out.append(d)
        return out

    def get(self, disturbance_id: str) -> Disturbance | None:
        return next((d for d in self._disturbances if d.disturbance_id == disturbance_id), None)

    # ------------------------------------------------------------ internals

    def _validate(self, sample: PQSample) -> None:
        if sample.quality not in {"good", "suspect", "bad"}:
            raise ValueError(f"unknown quality {sample.quality!r}")
        if any(v < 0 for v in sample.voltage_pu.values()):
            raise ValueError("voltage cannot be negative")
        if any(i < 0 for i in sample.current_a.values()):
            raise ValueError("current cannot be negative")

    def _section_of(self, point_id: str) -> str | None:
        return self.point_section.get(point_id)

    def _in_window(self, d: Disturbance, ts: float) -> bool:
        return d.start_s - self.gap_tolerance_s <= ts <= d.end_s + self.gap_tolerance_s

    def _find_frozen(self, point_id: str, section: str | None, ts: float) -> Disturbance | None:
        for d in self._disturbances:
            if d.merged_into is not None or not d.frozen:
                continue
            if point_id in d.point_ids and self._in_window(d, ts):
                return d
            if section is not None and d.section_id == section and self._in_window(d, ts):
                return d
        return None

    def _find_match(self, point_id: str, section: str | None, ts: float) -> Disturbance | None:
        for d in self._disturbances:
            if d.merged_into is not None or d.frozen:
                continue
            if point_id in d.point_ids and self._in_window(d, ts):
                return d
            if section is not None and d.section_id == section and self._in_window(d, ts):
                return d
        return None

    def _duplicate_exists(self, point_id: str, ts: float) -> Disturbance | None:
        for d in self._disturbances:
            if d.merged_into is not None:
                continue
            if point_id in d.point_ids and abs(d.start_s - ts) <= self.dedupe_tolerance_s:
                return d
        return None

    def _create(self, sample: PQSample) -> Disturbance:
        section = self._section_of(sample.point_id)
        self._counter += 1
        d = Disturbance(
            disturbance_id=f"PQ-{self._counter:04d}",
            point_ids=[sample.point_id],
            section_id=section,
            equipment_ids=list(self.section_equipment.get(section, ())) if section else [],
            start_s=sample.timestamp_s,
            end_s=sample.timestamp_s,
            kind="unknown",
            phases=[],
            min_voltage_pu=float("inf"),
            max_voltage_pu=float("-inf"),
            max_current_a=0.0,
            max_flicker_pst=0.0,
            sample_count=0,
            last_sequence=sample.sequence,
            version=1,
            frozen=False,
            conclusion="uncertain",
            conclusion_reason="",
            versions=[],
            merged_into=None,
            created_at_s=self._clock,
        )
        self._disturbances.append(d)
        self._recompute(d)
        d.versions.append(RawVersion(1, self._clock, "created", (sample.sequence,), "initial filing"))
        return d

    def _revise(self, d: Disturbance, sample: PQSample, late: bool) -> None:
        d.start_s = min(d.start_s, sample.timestamp_s)
        d.end_s = max(d.end_s, sample.timestamp_s)
        self._recompute(d)
        d.version += 1
        d.versions.append(
            RawVersion(
                d.version,
                self._clock,
                "out_of_order" if late else "revision",
                (sample.sequence,),
                "late sample retained" if late else "window extended",
            )
        )
        self._merge_adjacent(d)

    def _merge_adjacent(self, d: Disturbance) -> int:
        """Absorb unfrozen disturbances that became contiguous with ``d``
        after a revision (e.g. a late sample bridged two runs)."""
        total = 0
        while True:
            absorbed: list[Disturbance] = []
            for other in self._disturbances:
                if other is d or other.frozen or other.merged_into is not None:
                    continue
                same_point = any(p in d.point_ids for p in other.point_ids)
                same_section = d.section_id is not None and d.section_id == other.section_id
                if not (same_point or same_section):
                    continue
                if d.start_s - self.gap_tolerance_s <= other.end_s and other.start_s <= d.end_s + self.gap_tolerance_s:
                    d.start_s = min(d.start_s, other.start_s)
                    d.end_s = max(d.end_s, other.end_s)
                    for p in other.point_ids:
                        if p not in d.point_ids:
                            d.point_ids.append(p)
                    other.merged_into = d.disturbance_id
                    absorbed.append(other)
            if not absorbed:
                break
            total += len(absorbed)
            self._recompute(d)
        if total:
            d.version += 1
            d.versions.append(
                RawVersion(d.version, self._clock, "merged", (), f"absorbed {total} adjacent disturbance(s)")
            )
        return total

    def _recompute(self, d: Disturbance) -> None:
        """Recompute window, severity and conclusion from raw samples.

        Only disturbed samples inside the current window are considered, so a
        later run on the same point cannot retroactively extend this event.
        """
        ts_list: list[float] = []
        min_v = float("inf")
        max_v = float("-inf")
        max_i = 0.0
        max_f = 0.0
        phases: set[str] = set()
        kinds: set[str] = set()
        count = 0
        last_seq = d.last_sequence
        for pid in d.point_ids:
            for s in self._samples.get(pid, ()):
                if not self._is_disturbed(s):
                    continue
                if s.timestamp_s < d.start_s or s.timestamp_s > d.end_s:
                    continue
                ts_list.append(s.timestamp_s)
                count += 1
                last_seq = max(last_seq, s.sequence)
                for ph, v in s.voltage_pu.items():
                    min_v = min(min_v, v)
                    max_v = max(max_v, v)
                    if v < self.sag_threshold or v > self.swell_threshold:
                        phases.add(ph)
                for i in s.current_a.values():
                    max_i = max(max_i, i)
                if s.flicker_pst is not None:
                    max_f = max(max_f, s.flicker_pst)
                if s.flicker_plt is not None:
                    max_f = max(max_f, s.flicker_plt)
                kinds.update(self._kinds(s))
        if ts_list:
            d.start_s = min(ts_list)
            d.end_s = max(ts_list)
        d.min_voltage_pu = min_v if min_v != float("inf") else float("nan")
        d.max_voltage_pu = max_v if max_v != float("-inf") else float("nan")
        d.max_current_a = max_i
        d.max_flicker_pst = max_f
        d.phases = sorted(phases)
        d.kind = dominant_kind(kinds)
        d.sample_count = count
        d.last_sequence = last_seq
        self._update_conclusion(d)

    def _is_disturbed(self, sample: PQSample) -> bool:
        if sample.quality == "bad":
            return True
        if any(v < self.sag_threshold or v > self.swell_threshold for v in sample.voltage_pu.values()):
            return True
        if sample.flicker_pst is not None and sample.flicker_pst > self.flicker_threshold:
            return True
        if sample.flicker_plt is not None and sample.flicker_plt > self.flicker_threshold:
            return True
        return False

    def _kinds(self, sample: PQSample) -> set[str]:
        kinds: set[str] = set()
        for v in sample.voltage_pu.values():
            if v < self.interruption_threshold:
                kinds.add("interruption")
            elif v < self.sag_threshold:
                kinds.add("sag")
            elif v > self.swell_threshold:
                kinds.add("swell")
        if sample.flicker_pst is not None and sample.flicker_pst > self.flicker_threshold:
            kinds.add("flicker")
        if sample.flicker_plt is not None and sample.flicker_plt > self.flicker_threshold:
            kinds.add("flicker")
        if sample.quality == "bad":
            kinds.add("telemetry")
        return kinds

    def _update_conclusion(self, d: Disturbance) -> None:
        reasons: list[str] = []
        if d.sample_count < 2:
            reasons.append("single_sample")
        if d.section_id is None:
            reasons.append("unmapped_section")
        if d.min_voltage_pu != d.min_voltage_pu:  # NaN -> no voltage data
            reasons.append("missing_voltage")
        if d.kind == "unknown":
            reasons.append("unclassified")
        if d.kind == "sag" and d.min_voltage_pu > self.sag_threshold - 0.05:
            reasons.append("marginal_sag")
        if d.kind == "swell" and d.max_voltage_pu < self.swell_threshold + 0.05:
            reasons.append("marginal_swell")
        if d.max_flicker_pst > 0.0 and d.max_flicker_pst < self.flicker_threshold * 1.2:
            reasons.append("marginal_flicker")
        if any(
            s.quality == "suspect"
            for pid in d.point_ids
            for s in self._samples.get(pid, ())
            if d.start_s <= s.timestamp_s <= d.end_s
        ):
            reasons.append("suspect_quality")
        if reasons:
            d.conclusion = "uncertain"
            d.conclusion_reason = ";".join(reasons)
        else:
            d.conclusion = "certain"
            d.conclusion_reason = ""

    def _require(self, disturbance_id: str) -> Disturbance:
        d = self.get(disturbance_id)
        if d is None:
            raise KeyError(f"unknown disturbance {disturbance_id!r}")
        return d
