from backend.rules import AlertEngine, Config, Detection, Level


def smoke(box=(100, 100, 200, 200), conf=0.8):
    return Detection("smoke", conf, box)


def fire(box=(120, 120, 160, 160), conf=0.8):
    return Detection("fire", conf, box)


def grow(i):
    s = 100 + 5 * i  # box grows 100 -> 120 wide and tall over 5 frames
    return (100, 100, 100 + s, 100 + s)


def test_clear_when_nothing_detected():
    e = AlertEngine()
    assert e.update([]).level == Level.CLEAR


def test_rule1_persistence_needs_3_frames():
    e = AlertEngine()
    assert e.update([smoke()]).level == Level.CLEAR
    assert e.update([smoke()]).level == Level.CLEAR
    s = e.update([smoke()])
    assert s.level == Level.POSSIBLE_FIRE and s.escalated


def test_rule1_single_frame_hit_is_ignored():
    e = AlertEngine()
    e.update([smoke()])
    assert e.update([]).level == Level.CLEAR
    assert e.update([]).level == Level.CLEAR


def test_rule2_growth_confirms_fire():
    e = AlertEngine()
    levels = [e.update([smoke(grow(i))]).level for i in range(5)]
    assert levels[2] == Level.POSSIBLE_FIRE
    assert levels[4] == Level.CONFIRMED_FIRE


def test_flat_smoke_never_confirms():
    e = AlertEngine()
    for _ in range(10):
        s = e.update([smoke()])
    assert s.level == Level.POSSIBLE_FIRE


def test_rule2_fire_inside_smoke_confirms():
    e = AlertEngine()
    for _ in range(3):
        e.update([smoke()])
    assert e.update([smoke(), fire()]).level == Level.CONFIRMED_FIRE


def test_rule3_exclusion_zone_never_alerts():
    e = AlertEngine(Config(exclusion_zones=[(90, 90, 210, 210)]))
    for _ in range(6):
        s = e.update([smoke()])
    assert s.level == Level.CLEAR
    assert e.log  # but it is logged


def test_low_confidence_is_watch_only():
    e = AlertEngine()
    for _ in range(6):
        s = e.update([smoke(conf=0.4)])
    assert s.level == Level.WATCH and not s.escalated


def test_below_watch_threshold_is_clear():
    e = AlertEngine()
    assert e.update([smoke(conf=0.2)]).level == Level.CLEAR


def test_track_survives_two_missed_frames():
    e = AlertEngine()
    e.update([smoke()])
    e.update([smoke()])
    e.update([])
    e.update([])
    assert e.update([smoke()]).level == Level.POSSIBLE_FIRE


def test_track_resets_after_three_missed_frames():
    e = AlertEngine()
    e.update([smoke()])
    e.update([smoke()])
    for _ in range(3):
        e.update([])
    assert e.update([smoke()]).level == Level.CLEAR


def test_bad_frame_is_feed_issue_not_clear():
    e = AlertEngine()
    assert e.update([], frame_ok=False).level == Level.FEED_ISSUE


def test_first_detection_timestamp_is_recorded():
    e = AlertEngine()
    for i in range(3):
        s = e.update([smoke()], timestamp=60.0 * i)
    assert s.first_detection_ts == 120.0


def test_escalation_fires_once_per_level():
    e = AlertEngine()
    flags = [e.update([smoke()]).escalated for _ in range(6)]
    assert flags.count(True) == 1
