from localpilot.agent import LocalPilotAgent


def test_explicit_sources_are_detected_without_guessing_specific_tools():
    prompt = (
        "Inspect your actual local repository and your private GitHub repository yourself. "
        "Review PR #30 and revise the architecture from verified evidence."
    )
    assert LocalPilotAgent._evidence_requirements(prompt) == {
        "trusted repository",
        "private GitHub",
    }


def test_known_https_url_requires_public_read_attempt():
    assert LocalPilotAgent._evidence_requirements(
        "Read https://example.org/reference and summarize what it says."
    ) == {"public HTTPS"}


def test_ordinary_language_does_not_accidentally_trigger_repository_or_github_tools():
    for prompt in (
        "Review my report and make the wording clearer.",
        "What is the current branch of mathematics that studies this topic?",
        "Explain the issue with this argument.",
        "What is the latest commit people make in long-term relationships?",
    ):
        assert LocalPilotAgent._evidence_requirements(prompt) == set()


def test_explicit_pc_state_request_requires_pc_evidence():
    assert LocalPilotAgent._evidence_requirements(
        "Check this PC's current storage and Defender status."
    ) == {"Windows/PC state"}
