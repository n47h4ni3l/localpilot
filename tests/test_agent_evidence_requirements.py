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


def test_public_web_and_primary_source_language_requires_https_evidence():
    assert LocalPilotAgent._evidence_requirements(
        "Research this on the public web and inspect a primary source."
    ) == {"public HTTPS"}


def test_latest_public_internet_claim_requires_current_discovery_and_primary_read():
    assert LocalPilotAgent._evidence_requirements(
        "Fact-check the latest stable Python release as of today using the public Internet."
    ) == {"public web discovery", "public HTTPS"}


def test_recurring_device_fault_requires_live_support_discovery_and_primary_read():
    prompt = (
        "My H2S printer keeps getting filament clogged in the nozzle or extruder. "
        "I unclog it and it clogs again straight away; the filament is dry."
    )

    assert LocalPilotAgent._is_practical_troubleshooting_prompt(prompt) is True
    assert LocalPilotAgent._evidence_requirements(prompt) == {
        "public web discovery",
        "public HTTPS",
    }


def test_explicit_library_inspection_requires_local_library_evidence():
    for prompt in (
        "Search the local library for power management guidance.",
        "Read my books and find the relevant principle.",
        "Consult the library manuals before answering.",
    ):
        assert "local library" in LocalPilotAgent._evidence_requirements(prompt)


def test_explicit_web_prohibition_does_not_become_a_web_requirement():
    prompt = "Consult the local library, do not use the public web, and answer from the book."

    assert LocalPilotAgent._evidence_requirements(prompt) == {"local library"}
    assert LocalPilotAgent._forbidden_tools(prompt) == {
        "search_public_web",
        "fetch_public_https",
    }


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
