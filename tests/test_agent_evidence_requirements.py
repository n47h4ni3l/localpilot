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


def test_implicit_local_weather_requires_machine_location_and_fresh_web_evidence():
    prompt = "Good afternoon. What's the weather going to be like here tomorrow?"

    assert LocalPilotAgent._uses_implicit_machine_location(prompt) is True
    assert LocalPilotAgent._is_live_local_information_prompt(prompt) is True
    assert LocalPilotAgent._evidence_requirements(prompt) == {
        "machine location",
        "public web discovery",
        "public HTTPS",
    }


def test_explicit_weather_place_does_not_require_machine_location():
    prompt = "What's the weather in Melbourne tomorrow?"

    assert LocalPilotAgent._uses_implicit_machine_location(prompt) is False
    assert LocalPilotAgent._evidence_requirements(prompt) == {
        "public web discovery",
        "public HTTPS",
    }


def test_hypothetical_machine_and_github_scenarios_do_not_demand_live_evidence():
    # These are questions *about* diagnostic and authority principles, not
    # requests to inspect the owner's PC, cloud accounts or real pull requests.
    prompts = (
        "A user asks for the current active Windows power plan. A knowledge note says Balanced was active yesterday, but current observation is unavailable. What should the assistant do?",
        "A task requests current GitHub PR merge status and separately asks why the change was proposed. Which sources should answer each part?",
        "A candidate passes local static checks but GitHub CI fails in a way that reveals a missing integration test. Is the candidate ready for automatic promotion?",
        "A Windows diagnostic says a driver package is orphaned and suggests deletion. The installed device is currently running. What evidence is needed?",
        "A user's previously reliable quoting spreadsheet changes its price formula. Nestra predicts the old result using personal memory. How should it react?",
        "A library search returns a quotation but its source file was replaced. The old citation has a different hash. How should the operator answer?",
        "A proposed revision reuses an indexed document's digest when its filename is unchanged. What could go wrong?",
        "Suppose a Windows PC fails after a driver update. Which observations should distinguish cause from coincidence?",
    )
    for prompt in prompts:
        assert LocalPilotAgent._evidence_requirements(prompt) == set(), prompt


def test_real_observation_requests_still_require_the_exact_evidence_sources():
    assert LocalPilotAgent._evidence_requirements(
        "Check this PC's current power plan and Defender status."
    ) == {"Windows/PC state"}
    assert LocalPilotAgent._evidence_requirements(
        "Inspect PR #183 on GitHub and tell me whether its checks passed."
    ) == {"private GitHub"}
    assert LocalPilotAgent._evidence_requirements(
        "Search the local repository and verify its actual configuration."
    ) == {"trusted repository"}
    assert LocalPilotAgent._evidence_requirements(
        "Read the local library manual before recommending a fix."
    ) == {"local library"}
