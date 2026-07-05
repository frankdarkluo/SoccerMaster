from pipeline.stage2_events.verify import parse_verdict, normalize_outcome


def test_parse_verdict_extracts_json():
    raw = 'blah {"verdict": "confirm", "outcome": "success"} trailing'
    v = parse_verdict(raw)
    assert v.verdict == "confirm" and v.outcome == "success"


def test_parse_verdict_malformed_is_uncertain():
    v = parse_verdict("the model rambled with no json")
    assert v.verdict == "uncertain"


def test_parse_verdict_bad_json_is_uncertain():
    v = parse_verdict('{"verdict": "confirm", oops}')  # invalid json
    assert v.verdict == "uncertain"


def test_normalize_outcome_synonyms():
    assert normalize_outcome("succeeded") == "success"
    assert normalize_outcome("intercepted") is None  # unknown stays None
    assert normalize_outcome("failed") == "failure"
