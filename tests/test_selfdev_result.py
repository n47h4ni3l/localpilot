from localpilot.selfdev import classify_candidate_result


def test_no_changes_is_not_candidate_ready():
    assert classify_candidate_result(0, True) == "no_changes"


def test_changed_candidate_requires_passing_checks():
    assert classify_candidate_result(1, True) == "candidate_ready"
    assert classify_candidate_result(1, False) == "candidate_needs_work"
    assert classify_candidate_result(1, None) == "candidate_needs_work"
