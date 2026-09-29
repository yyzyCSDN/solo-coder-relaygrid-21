from relaygrid.powerquality import (
    PQAttributionEngine,
    PQSample,
    dominant_kind,
)


def sag(point, ts, seq, v=0.85, quality="good"):
    return PQSample(point, ts, seq, voltage_pu={"A": v, "B": v, "C": v}, quality=quality)


def normal(point, ts, seq):
    return PQSample(point, ts, seq, voltage_pu={"A": 1.0, "B": 1.0, "C": 1.0})


def flicker(point, ts, seq, pst=1.5):
    return PQSample(point, ts, seq, voltage_pu={"A": 1.0}, flicker_pst=pst)


def bad_telemetry(point, ts, seq):
    return PQSample(point, ts, seq, voltage_pu={"A": 1.0}, quality="bad")


def test_sag_filed_and_certain():
    e = PQAttributionEngine(
        point_section={"m1": "S1"}, section_equipment={"S1": ["F1", "F2"]}
    )
    r1 = e.ingest(sag("m1", 10, 1))
    r2 = e.ingest(sag("m1", 20, 2))
    assert r1.status == "created"
    assert r2.status == "revised"
    d = r2.disturbance
    assert d.kind == "sag"
    assert d.conclusion == "certain"
    assert d.section_id == "S1"
    assert d.equipment_ids == ["F1", "F2"]
    assert d.start_s == 10 and d.end_s == 20
    assert d.sample_count == 2
    assert d.phases == ["A", "B", "C"]
    assert d.version == 2
    assert [v.reason for v in d.versions] == ["created", "revision"]


def test_single_sample_is_uncertain():
    e = PQAttributionEngine(point_section={"m1": "S1"})
    r = e.ingest(sag("m1", 10, 1))
    assert r.status == "created"
    assert r.disturbance.conclusion == "uncertain"
    assert "single_sample" in r.disturbance.conclusion_reason


def test_unmapped_section_is_uncertain():
    e = PQAttributionEngine()
    e.ingest(sag("m1", 10, 1))
    e.ingest(sag("m1", 20, 2))
    d = e.disturbances()[0]
    assert d.section_id is None
    assert "unmapped_section" in d.conclusion_reason
    assert d.conclusion == "uncertain"


def test_normal_samples_are_stored_only():
    e = PQAttributionEngine()
    r = e.ingest(normal("m1", 10, 1))
    assert r.status == "stored"
    assert r.disturbance is None
    assert e.disturbances() == []


def test_out_of_order_revises_unfrozen():
    e = PQAttributionEngine(point_section={"m1": "S1"})
    e.ingest(sag("m1", 10, 1))
    e.ingest(sag("m1", 20, 3))  # high sequence first
    r = e.ingest(sag("m1", 15, 2))  # late sample, sequence 2 < 3
    assert r.status == "revised"
    assert r.late is True
    d = r.disturbance
    assert d.start_s == 10 and d.end_s == 20
    assert d.last_sequence == 3
    assert d.versions[-1].reason == "out_of_order"


def test_out_of_order_rejected_on_frozen():
    e = PQAttributionEngine(point_section={"m1": "S1"})
    e.ingest(sag("m1", 10, 1))
    e.ingest(sag("m1", 20, 2))
    d = e.disturbances()[0]
    e.freeze(d.disturbance_id)
    r = e.ingest(sag("m1", 15, 3))  # late sample inside frozen window
    assert r.status == "rejected_frozen"
    assert r.disturbance.frozen is True
    # frozen event unchanged
    assert d.start_s == 10 and d.end_s == 20 and d.sample_count == 2


def test_duplicate_redelivery_not_refiled():
    e = PQAttributionEngine(point_section={"m1": "S1"})
    e.ingest(sag("m1", 10, 1))
    # re-delivery: same point, nearly same start, different sequence
    r = e.ingest(sag("m1", 10.5, 99))
    assert r.status == "duplicate"
    assert len(e.disturbances()) == 1


def test_same_section_meters_merge_into_one_disturbance():
    e = PQAttributionEngine(
        point_section={"m1": "S1", "m2": "S1"}, section_equipment={"S1": ["F1"]}
    )
    r1 = e.ingest(sag("m1", 10, 1))
    r2 = e.ingest(sag("m2", 12, 2))  # neighbour on same section
    assert r1.status == "created"
    assert r2.status == "revised"
    d = r2.disturbance
    assert d.point_ids == ["m1", "m2"]
    assert d.section_id == "S1"
    assert len(e.disturbances()) == 1


def test_flicker_flagged():
    e = PQAttributionEngine(point_section={"m1": "S1"})
    r = e.ingest(flicker("m1", 10, 1))
    assert r.status == "created"
    assert r.disturbance.kind == "flicker"
    assert r.disturbance.max_flicker_pst == 1.5


def test_bad_telemetry_flagged():
    e = PQAttributionEngine(point_section={"m1": "S1"})
    r = e.ingest(bad_telemetry("m1", 10, 1))
    assert r.status == "created"
    assert r.disturbance.kind == "telemetry"


def test_interruption_dominates_sag():
    e = PQAttributionEngine(point_section={"m1": "S1"})
    e.ingest(PQSample("m1", 10, 1, voltage_pu={"A": 0.05, "B": 0.5, "C": 0.5}))
    d = e.disturbances()[0]
    assert d.kind == "interruption"
    assert d.min_voltage_pu == 0.05


def test_suspect_quality_uncertain():
    e = PQAttributionEngine(point_section={"m1": "S1"})
    e.ingest(sag("m1", 10, 1, quality="suspect"))
    e.ingest(sag("m1", 20, 2, quality="suspect"))
    d = e.disturbances()[0]
    assert "suspect_quality" in d.conclusion_reason


def test_auto_freeze():
    e = PQAttributionEngine(point_section={"m1": "S1"})
    e.ingest(sag("m1", 10, 1))
    e.ingest(sag("m1", 20, 2))
    frozen = e.auto_freeze(now_s=10000, freeze_after_s=3600)
    assert len(frozen) == 1
    assert e.disturbances()[0].frozen is True
    # already frozen -> not re-frozen
    assert e.auto_freeze(now_s=10000, freeze_after_s=3600) == []


def test_frozen_excluded_from_unfrozen_view():
    e = PQAttributionEngine(point_section={"m1": "S1"})
    e.ingest(sag("m1", 10, 1))
    e.freeze(e.disturbances()[0].disturbance_id)
    assert len(e.disturbances(include_frozen=False)) == 0
    assert len(e.disturbances(include_frozen=True)) == 1


def test_late_sample_bridges_runs_merges():
    e = PQAttributionEngine(gap_tolerance_s=300.0, point_section={"m1": "S1"})
    e.ingest(sag("m1", 100, 1))
    e.ingest(sag("m1", 500, 2))  # separate run (gap 400 > 300)
    assert len(e.disturbances()) == 2
    # late sample bridges the two runs
    r = e.ingest(sag("m1", 300, 3))
    assert r.status == "revised"
    active = e.disturbances()
    assert len(active) == 1
    d = active[0]
    assert d.start_s == 100 and d.end_s == 500
    assert any(v.reason == "merged" for v in d.versions)


def test_dominant_kind_priority():
    assert dominant_kind({"sag", "interruption"}) == "interruption"
    assert dominant_kind({"flicker", "sag"}) == "sag"
    assert dominant_kind(set()) == "unknown"


def test_invalid_quality_rejected():
    e = PQAttributionEngine()
    try:
        e.ingest(PQSample("m1", 10, 1, quality="bogus"))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_raw_versions_retained_on_revision():
    e = PQAttributionEngine(point_section={"m1": "S1"})
    e.ingest(sag("m1", 10, 1))
    e.ingest(sag("m1", 20, 2))
    e.ingest(sag("m1", 30, 3))
    d = e.disturbances()[0]
    assert d.version == 3
    assert [v.version for v in d.versions] == [1, 2, 3]
    # raw samples are still available for audit
    assert len(e._samples["m1"]) == 3
