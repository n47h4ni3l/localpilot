from localpilot.github_integration import classify_check_rollup


def test_check_rollup_is_conservative():
    assert classify_check_rollup([]) == "pending"
    assert classify_check_rollup([{"status": "IN_PROGRESS", "conclusion": None}]) == "pending"
    assert classify_check_rollup([{"status": "COMPLETED", "conclusion": "FAILURE"}]) == "failed"
    assert classify_check_rollup([{"status": "COMPLETED", "conclusion": "SUCCESS"}]) == "passed"

