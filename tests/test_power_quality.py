from relaygrid.power_quality import PowerQualityAttributor, PowerQualitySample


def sample(sequence, timestamp, **kwargs):
    defaults = {
        "point_id": "pq-1",
        "section_id": "feeder-a",
        "timestamp_s": timestamp,
        "sequence": sequence,
        "voltages": (0.82, 0.81, 0.83),
        "currents": (120.0, 118.0, 121.0),
        "flicker": 0.18,
        "equipment_id": "recloser-1",
    }
    defaults.update(kwargs)
    return PowerQualitySample(**defaults)


def test_evidence_groups_into_one_confirmed_disturbance():
    attributor = PowerQualityAttributor(min_duration_s=0.0)
    assert attributor.ingest(sample(1, 0.0)).status == "created"
    assert attributor.ingest(sample(2, 5.0)).status == "updated"

    event, = attributor.events
    assert event.status == "confirmed"
    assert event.probable_cause == "load_switching"
    assert event.affected_sections == ("feeder-a",)
    assert event.affected_equipment == ("recloser-1",)
    assert len(event.raw_data_versions) == 2
    assert [v.version for v in event.version_history] == [1, 2]


def test_late_sample_revises_only_unfrozen_event_and_never_duplicates():
    attributor = PowerQualityAttributor(min_duration_s=0.0)
    attributor.ingest(sample(1, 0.0))
    attributor.ingest(sample(2, 5.0))
    attributor.ingest(sample(3, 10.0))

    result = attributor.ingest(sample(0, 4.0))
    assert result.status == "updated"
    assert len(attributor.events) == 1
    event = attributor.events[0]
    assert event.start_s == 0.0
    assert event.end_s == 10.0
    assert len(event.raw_data_versions) == 4

    attributor.freeze(event.event_id)
    frozen = attributor.ingest(sample(4, 6.0))
    assert frozen.status == "rejected"
    assert frozen.reason == "frozen_event"
    assert attributor.rejected[0][1] == "frozen_event"


def test_insufficient_evidence_is_retained_as_uncertain():
    attributor = PowerQualityAttributor(min_duration_s=0.0)
    attributor.ingest(sample(1, 0.0))
    event, = attributor.events
    assert event.status == "uncertain"
    assert "insufficient_observations" in event.reasons
    assert "insufficient_confidence" in event.reasons
    assert len(event.raw_data_versions) == 1


def test_bad_telemetry_quality_keeps_conclusion_uncertain():
    attributor = PowerQualityAttributor(min_duration_s=0.0)
    attributor.ingest(sample(1, 0.0, telemetry_quality="bad"))
    attributor.ingest(sample(2, 5.0))
    event, = attributor.events
    assert event.status == "uncertain"
    assert event.confidence < 0.5
    assert "bad_telemetry_quality" in event.reasons


def test_quality_only_observation_without_disturbance_is_unassigned():
    attributor = PowerQualityAttributor()
    result = attributor.ingest(sample(1, 0.0, voltages=None, currents=None, flicker=None, telemetry_quality="bad"))
    assert result.status == "unassigned"
    assert attributor.events == []
    assert attributor.unassigned[0][1] == "quality_without_disturbance"


def test_bridge_sample_merges_duplicate_disturbance_records():
    attributor = PowerQualityAttributor(max_gap_s=5.0, min_duration_s=0.0)
    attributor.ingest(sample(1, 0.0, section_id="s1", equipment_id="r1"))
    attributor.ingest(sample(2, 10.0, section_id="s2", equipment_id="r2"))
    assert len(attributor.events) == 2

    result = attributor.ingest(sample(3, 5.0, section_id="s1", equipment_id="r2"))
    assert result.status == "merged"
    event, = attributor.events
    assert event.start_s == 0.0
    assert event.end_s == 10.0
    assert event.affected_sections == ("s1", "s2")
    assert event.affected_equipment == ("r1", "r2")


def test_revision_preserves_raw_data_versions():
    attributor = PowerQualityAttributor(min_duration_s=0.0)
    attributor.ingest(sample(1, 0.0, telemetry_quality="suspect"))
    attributor.ingest(sample(2, 5.0))
    attributor.ingest(sample(1, 1.0, telemetry_quality="good"))

    event, = attributor.events
    assert event.version == 3
    versions = attributor.raw_versions("pq-1", 1)
    assert [v.telemetry_quality for v in versions] == ["suspect", "good"]
    assert event.raw_data_versions[0].telemetry_quality == "good"
