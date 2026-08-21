from localpilot.safety import RiskLevel, SafetyPolicy


def test_default_policy_allows_readonly_and_reversible_but_not_destructive():
    policy = SafetyPolicy()
    assert policy.permits_without_confirmation(RiskLevel.READ_ONLY)
    assert policy.permits_without_confirmation(RiskLevel.REVERSIBLE)
    assert not policy.permits_without_confirmation(RiskLevel.DESTRUCTIVE)
