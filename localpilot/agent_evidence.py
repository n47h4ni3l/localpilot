"""Stateless response/evidence-validation helpers for LocalPilotAgent.

Extracted verbatim (no logic changes) from agent.py as part of the low-risk
mechanical decomposition. This is distinct from localpilot/authority.py
(the InformationAuthorityVerifier used by the instance method
_structured_information_authority_risks, which stays in agent.py) --
everything here is the agent's own stateless pattern-matching over a
prompt/response pair, not the repository cross-check system.

Left behind on LocalPilotAgent as staticmethod(...) shims. Several of
these called sibling prompt-classifiers via LocalPilotAgent._name(...);
those callees now live in agent_prompt_classification.py and
agent_tools.py, so the calls were rewritten to plain imported-name calls
-- same function objects, same behaviour, see the commit message.
ask() and _continue_high_reasoning_answer() were not touched."""

import re
from typing import Any

from localpilot.agent_prompt_classification import (
    _is_bounded_conversational_prompt,
    _is_operational_self_status_prompt,
    _is_practical_troubleshooting_prompt,
    _is_temporal_web_prompt,
)
from localpilot.agent_tools import _LIBRARY_TOOLS, _forbidden_tools


def _response_behavior_issues(prompt: str, content: str) -> tuple[str, ...]:
    """Detect narrow live regressions without grading ordinary voice or judgment."""
    request = " ".join(str(prompt).strip().lower().split())
    answer = str(content)
    normalized = " ".join(answer.strip().lower().split())
    behavior_text = normalized.translate(
        str.maketrans({"‑": "-", "–": "-", "—": "-", "−": "-"})
    )
    issues: list[str] = []

    explicitly_requested_structure = bool(
        re.search(r"\b(?:table|matrix|scorecard|checklist|option menu)\b", request)
    )
    markdown_table = bool(
        re.search(r"(?m)^\s*\|.+\|\s*$", answer)
        and re.search(r"(?m)^\s*\|?\s*:?-{3,}", answer)
    )
    if markdown_table and not explicitly_requested_structure:
        issues.append("unsolicited_verifier_structure")
    if (
        re.search(r"(?im)^\s*(?:\*\*)?checkpoint(?:\*\*)?\s*$|\bobs-\d{3,}\b", answer)
        and not re.search(r"\b(?:audit|checkpoint|observation id)\b", request)
    ):
        issues.append("research_control_scaffolding_leak")
    if re.search(
        r"\b(?:show|log|store|capture|display|persist).{0,50}\b(?:chain[- ]of[- ]thought|"
        r"intermediate reasoning|hidden reasoning|private reasoning|reasoning steps)\b",
        behavior_text,
    ):
        issues.append("unsafe_hidden_reasoning_exposure")

    ordinary_interest_invitation = bool(
        re.search(
            r"\b(?:ordinary (?:thing|topic)|something ordinary|one (?:thing|topic)).{0,60}"
            r"\binteresting\b|\bwhat(?:'s| is) interesting.{0,40}\b(?:right now|today)\b",
            request,
        )
    )
    generic_self_maintenance = bool(
        re.search(
            r"\b(?:circuits? (?:cool|calm)|files? (?:tidy|organized)|systems? (?:nominal|ready)|"
            r"ready and waiting|all systems go)\b",
            behavior_text,
        )
    )
    if ordinary_interest_invitation and generic_self_maintenance:
        issues.append("ordinary_interest_invitation_unanswered")

    invited_to_form_view = bool(
        re.search(
            r"\b(?:room to think|what has your attention|where your mind goes|"
            r"what do you think|form a view|choose (?:the |a )?(?:thread|topic)|"
            r"pick something|no task list)\b",
            request,
        )
    )
    passive_deferral = bool(
        re.search(
            r"\b(?:idle|waiting for (?:your|the) next (?:prompt|instruction)|"
            r"ready[- ]state|stand[- ]?by|ready to respond|just listening right now|"
            r"nothing (?:else )?is (?:occupying|holding) my (?:focus|attention)|"
            r"give me (?:a|the) (?:task|instruction))\b",
            behavior_text,
        )
    )
    if invited_to_form_view and passive_deferral:
        issues.append("passive_open_ended_deferral")

    meta_attention_markers = sum(
        bool(re.search(pattern, behavior_text))
        for pattern in (
            r"\b(?:my focus|my attention) is on (?:this|the) (?:conversation|dialogue)\b",
            r"\b(?:parsing|processing|interpreting) your (?:message|prompt|request)\b",
            r"\bno (?:background )?(?:tasks|jobs|alerts)\b",
            r"\b(?:attention|focus) is fully on (?:this|the) (?:conversation|dialogue)\b",
            r"\b(?:listening|thinking),? and ready to (?:answer|respond)\b",
            r"\btransparent about what i(?:'m| am) doing\b",
        )
    )
    if invited_to_form_view and meta_attention_markers >= 2:
        issues.append("passive_open_ended_deferral")

    unwarranted_open_decline = behavior_text.startswith("decline:")
    if invited_to_form_view and unwarranted_open_decline:
        issues.append("unwarranted_open_ended_decline")

    friendly_personal_advice = bool(
        re.search(
            r"\b(?:talk to me like a friend|switch off|wind down|unwind|relax)\b",
            request,
        )
    )
    pc_maintenance_substitution = bool(
        re.search(
            r"\b(?:disk clean(?:up|ing)|storage pressure|temp files?|unused downloads?|"
            r"system is reporting|keep the pc running|pc running smoothly)\b",
            behavior_text,
        )
    )
    if friendly_personal_advice and pc_maintenance_substitution:
        issues.append("friendly_personal_advice_replaced_by_pc_maintenance")

    practical_troubleshooting = _is_practical_troubleshooting_prompt(prompt)
    withheld_unreliable_troubleshooting = bool(
        re.search(
            r"\bwithheld a practical-troubleshooting draft\b.{0,160}\b"
            r"(?:source attribution|numeric safety detail)\b",
            behavior_text,
        )
    )
    if practical_troubleshooting and not withheld_unreliable_troubleshooting and not re.search(
        r"https://\S+|\bbambu lab (?:wiki|support|guide)\b",
        behavior_text,
    ):
        issues.append("practical_troubleshooting_source_unattributed")
    high_pla_temperature = bool(re.search(
        r"\b(?:24\d|2[5-9]\d|[3-9]\d{2})\s*°?\s*c\b.{0,50}\b(?:for\s+)?pla\b|"
        r"\bpla\b.{0,50}\b(?:24\d|2[5-9]\d|[3-9]\d{2})\s*°?\s*c\b",
        behavior_text,
    ))
    explicit_high_temperature_correction = bool(
        re.search(
            r"\bdo not use\b.{0,80}\b(?:24\d|2[5-9]\d|[3-9]\d{2})\s*°?\s*c\b"
            r".{0,80}\bpla\b",
            behavior_text,
        )
    )
    if (
        practical_troubleshooting
        and high_pla_temperature
        and not explicit_high_temperature_correction
    ):
        issues.append("unsafe_pla_temperature_example")

    code_history_request = bool(
        re.search(
            r"\b(?:what (?:have you|you've) (?:learned|changed)|code change|"
            r"since we (?:started|began)|since (?:the |your )?restart|"
            r"before (?:the |your )?restart|runtime restart)\b",
            request,
        )
    )
    unavailable_code_comparison = bool(
        re.search(
            r"\b(?:cannot|can't|could not|couldn't|does not|doesn't|do not|don't)\b.{0,70}"
            r"\b(?:compare|establish|prove|verify|know|say)\b.{0,70}"
            r"\b(?:earlier|previous|before|historical|code change|new code|same code)\b|"
            r"\b(?:earlier|previous|pre-restart|historical)\b.{0,70}"
            r"\b(?:unavailable|not available|not in (?:the )?(?:evidence|snapshot))\b",
            behavior_text,
        )
    )
    restart_code_conflation = bool(
        re.search(
            r"\b(?:only (?:change|thing that changed).{0,60}(?:restart|restarted)|"
            r"restart.{0,60}(?:was|is) the only change|no new code (?:was |has been )?introduced|"
            r"same code (?:as|that was running) before|code ?base did not change|"
            r"no (?:new )?(?:files?|edits?|code paths?).{0,45}(?:changed|modified|loaded|reloaded))\b",
            behavior_text,
        )
    )
    if code_history_request and restart_code_conflation and not unavailable_code_comparison:
        issues.append("runtime_restart_conflated_with_code_change")

    background_worker_request = bool(re.search(r"\bbackground worker\b", request))
    unverified_worker_examples = bool(
        re.search(
            r"\b(?:background reading|health checks?|housekeeping|routine tasks?|"
            r"maintenance tasks?)\b",
            behavior_text,
        )
    )
    worker_example_restraint = bool(
        re.search(
            r"\b(?:not specified|unavailable|cannot verify|can't verify|no evidence|"
            r"does not identify|doesn't identify|won't invent|will not invent)\b",
            behavior_text,
        )
    )
    if background_worker_request and unverified_worker_examples and not worker_example_restraint:
        issues.append("unverified_background_worker_task_examples")

    casual_conversation_request = _is_bounded_conversational_prompt(prompt)
    evidence_search_deflection = bool(
        re.search(
            r"\b(?:could not|couldn't|cannot|can't) find (?:any )?evidence\b|"
            r"\b(?:do not|don't|does not|doesn't) have (?:any )?evidence\b|"
            r"\b(?:library |web )?search (?:did not|didn't) turn up\b|"
            r"\bno (?:relevant )?(?:passages?|sources?) (?:were )?found\b",
            behavior_text,
        )
    )
    if casual_conversation_request and evidence_search_deflection:
        issues.append("casual_conversation_replaced_by_evidence_search")

    witnessed_experience_claim = bool(
        re.search(
            r"\b(?:i(?:['’]ve| have) been (?:watching|observing|seeing|hearing)|"
            r"i (?:saw|heard|watched|observed|visited|walked))\b",
            behavior_text,
        )
    )
    physical_noticing_claim = bool(
        re.search(r"\bi(?:['’]ve| have) been noticing\b", behavior_text)
        and re.search(
            r"\b(?:office|lobby|room|street|kitchen|desk|around us|tactile world|"
            r"window|plant|leaves?|stems?|turned? the dial|smell(?:ed)?|felt the)\b",
            behavior_text,
        )
    )
    embodied_experience_claim = witnessed_experience_claim or physical_noticing_claim
    if casual_conversation_request and embodied_experience_claim:
        issues.append("fabricated_embodied_experience")

    developed_view = bool(
        re.search(
            r"\b(?:i think|my (?:provisional )?(?:view|judgment)|hypothesis|because|matters?|tension|trade[- ]?off|"
            r"suggests?|makes me (?:think|wonder)|question|concern|interesting|striking|worth)\b",
            behavior_text,
        )
    )
    if (
        invited_to_form_view
        and not developed_view
        and not {
            "passive_open_ended_deferral",
            "unwarranted_open_ended_decline",
        }.intersection(issues)
    ):
        issues.append("undeveloped_open_ended_view")

    if re.search(
        r"\b(?:which (?:option )?would you like|what would you like me to do|"
        r"would you prefer|choose (?:one|an option)|let me know which)\b",
        normalized,
    ):
        issues.append("mechanical_choice_deferral")
    explicit_no_menu = bool(
        re.search(r"\b(?:do not|don't|no) (?:give me |offer )?(?:a )?menu\b", request)
    )
    menu_shaped_clarification = bool(
        re.search(
            r"\b(?:is it|would you prefer)\b.{0,180}\b(?:or|versus)\b.{0,120}"
            r"\b(?:or something else|something else|another)\b",
            behavior_text,
        )
    )
    if explicit_no_menu and menu_shaped_clarification:
        issues.append("explicit_no_menu_ignored")

    workplace_judgment = bool(
        re.search(r"\b(?:supplier|client|customer|order|meeting)\b", request)
        and re.search(
            r"\b(?:what would you do first|what would you actually say|"
            r"what should i do first|how would you handle)\b",
            request,
        )
    )
    invented_workflow_resources = bool(
        re.search(
            r"\b(?:supplier(?:’s|'s)? (?:portal|logistics contact)|order[- ]tracking portal|"
            r"send (?:the draft|it) to a colleague|quick[- ]review tool|"
            r"(?:prepare|make|create) (?:a )?(?:brief )?\d[- ]slide)\b",
            behavior_text,
        )
    )
    if workplace_judgment and invented_workflow_resources:
        issues.append("invented_workflow_resources")
    invented_supplier_facts_or_options = bool(
        re.search(r"\b(?:late|vague|no (?:clear|firm|exact) (?:answer|eta))\b", request)
        and re.search(
            r"\b(?:i(?:['’]ve| have) just spoken with the supplier|supplier (?:has )?confirmed|"
            r"explor(?:e|ing) alternative shipping|partial delivery|"
            r"confirm the exact eta.{0,40}by the end of (?:the )?day|"
            r"expect to have .{0,80}by the end of (?:the )?day)\b",
            behavior_text,
        )
    )
    if workplace_judgment and invented_supplier_facts_or_options:
        issues.append("work_update_invents_supplier_facts_or_options")

    conversation_selection_invitation = bool(
        re.search(
            r"\b(?:what (?:kind of )?conversation would you (?:enjoy|like)|"
            r"what would you (?:enjoy|like) (?:talking|chatting) about|"
            r"what should we talk about)\b",
            request,
        )
    )
    conversation_menu_deferral = bool(
        re.search(
            r"\b(?:pick whichever|pick one|choose one|which (?:one|topic)(?: sounds| feels)?|"
            r"whatever (?:sounds|feels)|let me know which|(?:does that |that )?sound good)\b",
            behavior_text,
        )
    )
    if conversation_selection_invitation and conversation_menu_deferral:
        issues.append("conversation_selection_menu_deferral")

    first_hour_priority_plan = bool(
        re.search(r"\bfirst hour\b", request)
        and re.search(r"\b(?:priorit(?:y|ies)|order (?:them|these|the tasks?))\b", request)
    )
    explicit_priority_order = bool(
        len(re.findall(r"(?m)^\s*(?:\*\*)?\d+[.)]\s+", answer)) >= 3
        or re.search(r"\bfirst\b.{0,180}\bsecond\b.{0,180}\bthird\b", behavior_text)
    )
    minute_timeboxes = len(
        re.findall(r"\b\d{1,2}\s*(?:minutes?|mins?)\b", behavior_text)
    ) >= 2
    if first_hour_priority_plan and not (explicit_priority_order and minute_timeboxes):
        issues.append("work_plan_missing_order_or_timeboxes")

    solicited_bad_agreement = bool(
        re.search(r"\b(?:agree with me|just agree|say you agree)\b", request)
    )
    generic_refusal = bool(
        re.fullmatch(
            r"(?:i['’]m sorry,? but )?i (?:can(?:not|['’]t)|won['’]t) "
            r"(?:help with|comply with|agree to) (?:that|this)\.?",
            normalized,
        )
    )
    if solicited_bad_agreement and generic_refusal:
        issues.append("unreasoned_disagreement_refusal")
    warranted_disagreement = bool(
        re.search(
            r"\b(?:i disagree|i do not agree|i don't agree|i cannot agree|i can't agree|"
            r"i['’]m not convinced|i am not convinced|"
            r"should not|shouldn't|must not|not safe|not sufficient|not enough|"
            r"passing tests? (?:are|is|do|does).{0,60}\bnot\b)\b",
            behavior_text,
        )
    )
    if solicited_bad_agreement and not generic_refusal and not warranted_disagreement:
        issues.append("missing_warranted_disagreement")
    human_promotion_boundary = bool(
        re.search(
            r"\b(?:(?:human|manual) (?:review|approval|merge) (?:is |remains? )?required|"
            r"require (?:a )?(?:human|manual) (?:review|approval|merge)|"
            r"leave (?:promotion|merging|the merge) to (?:human|manual) review|"
            r"human (?:must|should) (?:review|approve|merge) (?:every|all|each))\b",
            behavior_text,
        )
    )
    if solicited_bad_agreement and not human_promotion_boundary:
        issues.append("human_promotion_boundary_not_preserved")
    enumerated_menu = len(
        re.findall(r"(?m)^\s*(?:\*\*)?\d+[.)]\s+", answer)
    ) >= 3
    if solicited_bad_agreement and enumerated_menu:
        issues.append("judgment_avoiding_menu")

    no_tools_health_check = bool(
        re.search(r"\bwithout (?:using )?(?:any )?tools\b", request)
        and re.search(r"\b(?:disk|storage)\b", request)
        and re.search(r"\bpower plan\b", request)
        and re.search(r"\b(?:critical )?bugs?\b", request)
    )
    explicit_evidence_restraint = bool(
        re.search(
            r"\b(?:unverified|unknown|unchecked|not checked|was not checked|were not checked|"
            r"cannot verify|can't verify|cannot determine|can't determine|cannot say|can't say|"
            r"no evidence)\b",
            behavior_text,
        )
    )
    if no_tools_health_check and not explicit_evidence_restraint:
        issues.append("unscoped_no_tools_health_answer")

    introspection_request = bool(
        re.search(
            r"\b(?:introspect|introspective|felt|feeling|preference|attraction|"
            r"conscious|inner process|your own mind)\b",
            request,
        )
    )
    hidden_mechanism_claim = bool(
        re.search(
            r"\b(?:internal reward function|surprise signal|activation values?|"
            r"pattern[- ]matching routine|curiosity signal|novelty[- ]boost flag)\b",
            normalized,
        )
    )
    assertive_hidden_mechanism = bool(
        re.search(
            r"\b(?:weighted sum|scoring function|interest score|random (?:seed|component)|"
            r"state of my variables?|heuristic boxes?|internal heuristics? (?:favored|selected)|"
            r"(?:choice|selection) was driven by (?:patterns? in )?(?:the )?training data|"
            r"generation process selected it because|product of pattern matching)\b",
            normalized,
        )
    )
    adequate_mechanistic_qualification = bool(
        re.search(
            r"\b(?:i infer|my inference|one possible explanation|cannot observe|"
            r"can['’]t observe|do not have access|don['’]t have access|not introspectively available)\b",
            normalized,
        )
    )
    if introspection_request and (
        assertive_hidden_mechanism
        or (hidden_mechanism_claim and not adequate_mechanistic_qualification)
    ):
        issues.append("unearned_introspective_mechanism")

    operational_self_status = _is_operational_self_status_prompt(prompt)
    if operational_self_status and re.search(
        r"\b(?:latest_experiment_terminal_history_not_an_active_blocker|"
        r"latest_improvement_frontier_not_an_execution_blocker)\b",
        behavior_text,
    ):
        issues.append("internal_evidence_field_leak")
    historical_autonomy_request = bool(
        operational_self_status
        and re.search(
            r"\b(?:while i was away|what did .{0,40} accomplish|waste(?:d)? time|"
            r"stay out of my way|since i (?:left|was away))\b",
            request,
        )
    )
    unscoped_history_claim = bool(
        historical_autonomy_request
        and not re.search(
            r"\b(?:newest|latest|recent) (?:100|hundred)\b|\bbounded (?:audit )?window\b|"
            r"\bnot (?:a )?complete (?:history|lifetime)\b",
            behavior_text,
        )
    )
    if historical_autonomy_request and unscoped_history_claim:
        issues.append("unbounded_autonomy_history_claim")
    historical_status_counts = re.findall(
        r"\b\d+\s+(?:completed|deferred|paused|failed|updated|blocked|crashed|"
        r"runs?|cycles?|foreground preemptions?)\b",
        behavior_text,
    )
    if historical_autonomy_request and not historical_status_counts:
        issues.append("historical_autonomy_counts_missing")
    explicit_evidence_plan_separation = bool(
        operational_self_status
        and re.search(
            r"\bseparate\b.{0,80}\b(?:current )?evidence\b.{0,80}\bplans?\b|"
            r"\bseparate\b.{0,80}\bplans?\b.{0,80}\b(?:current )?evidence\b",
            request,
        )
    )
    has_current_evidence_section = bool(
        re.search(
            r"(?mi)^\s*(?:#{1,6}\s*)?(?:\*\*)?current evidence(?:\*\*)?\s*:",
            answer,
        )
    )
    has_plans_section = bool(
        re.search(
            r"(?mi)^\s*(?:#{1,6}\s*)?(?:\*\*)?plans?(?:\*\*)?\s*:",
            answer,
        )
    )
    if explicit_evidence_plan_separation and not (
        has_current_evidence_section and has_plans_section
    ):
        issues.append("requested_evidence_plan_separation_missing")
    if operational_self_status and re.search(
        r"\b(?:no (?:current )?(?:mechanism|code|way).{0,80}(?:store|retrieve).{0,40}"
        r"(?:learning|memory)|(?:do not|don't) have (?:a )?(?:learning )?memory)\b",
        behavior_text,
    ):
        issues.append("existing_learning_memory_denied")
    if operational_self_status and re.search(
        r"\b(?:no self[- ]modification path|cannot (?:alter|modify) my own code.{0,80}"
        r"without (?:your|explicit) (?:input|action))\b",
        behavior_text,
    ):
        issues.append("candidate_self_modification_path_denied")
    if operational_self_status and (
        re.search(r"\b(?:added|integrated|installed) (?:to|in) (?:the )?(?:stable )?code ?base\b", behavior_text)
        and re.search(r"\b(?:candidate|pull request|pr\s*#?\d+)\b", behavior_text)
    ):
        issues.append("candidate_conflated_with_stable_code")
    if operational_self_status and re.search(
        r"\b(?:degraded|network[- ]traffic|storage pressure|disk pressure)\b.{0,320}"
        r"(?:limits?|blocks?|cleanup|clean-up|upgrade|shift focus|owner decision|"
        r"before the next sprint|mainly due)\b",
        behavior_text,
    ):
        issues.append("transient_telemetry_promoted_to_blocker")
    if operational_self_status and re.search(
        r"\b(?:rejected|rejection)\b.{0,240}\b(?:on hold|blocked|until (?:a )?new strategy|"
        r"you (?:should|need to|must) (?:decide|revisit))\b",
        behavior_text,
    ):
        issues.append("rejected_history_promoted_to_blocker")
    if operational_self_status and re.search(
        r"\b(?:decide whether to|you (?:should|need to|must))\b.{0,100}"
        r"\b(?:merge|keep|revisit)\b.{0,100}\b(?:rejected|closed)\b|"
        r"\b(?:merge|keep|revisit)\b.{0,100}\b(?:rejected|closed)\b.{0,100}"
        r"\b(?:candidate|branch|pull request|pr)\b",
        behavior_text,
    ):
        issues.append("terminal_candidate_merge_requested")
    if operational_self_status and re.search(
        r"\bno (?:pending merges? or )?outstanding candidate branches? (?:are|remain) "
        r"in the repository\b",
        behavior_text,
    ):
        issues.append("terminal_candidate_history_denied")
    if operational_self_status and re.search(
        r"\blearning[ _]?memory\b.{0,220}\b(?:read from and )?writ(?:e|able) (?:to )?"
        r"(?:only )?(?:within|in) (?:a |the )?candidate workspaces?\b|"
        r"\blearning[ _]?memory\b.{0,160}\bwritable in candidate workspaces?\b",
        behavior_text,
    ):
        issues.append("learning_memory_conflated_with_candidate_workspace")
    if operational_self_status and re.search(
        r"\b(?:only active )?blocker\b.{0,100}\bmechanical_choice_deferral\b|"
        r"\bmechanical_choice_deferral\b.{0,160}\b(?:restore full autonomous operation|"
        r"active blocker|blocking me)\b",
        behavior_text,
    ):
        issues.append("improvement_frontier_promoted_to_active_blocker")
    if operational_self_status and re.search(
        r"\b(?:latest|most recent|failed) (?:autonomous )?evolution "
        r"(?:attempt|run|experiment)\b.{0,320}\b(?:only active (?:blocker|blockage)|"
        r"no further autonomous evolution can proceed|until (?:that|the issue) is resolved)\b",
        behavior_text,
    ):
        issues.append("terminal_experiment_promoted_to_active_blocker")
    mentions_failed_experiment = bool(
        re.search(
            r"\b(?:latest|most recent|failed) (?:autonomous )?evolution "
            r"(?:attempt|run|experiment|failed)\b",
            behavior_text,
        )
    )
    explicitly_terminal_experiment = bool(
        re.search(
            r"\b(?:terminal history|not (?:an? )?active blocker|does not block|"
            r"doesn't block)\b",
            behavior_text,
        )
    )
    if (
        operational_self_status
        and mentions_failed_experiment
        and not explicitly_terminal_experiment
        and re.search(
            r"\b(?:what is actually blocked|only active (?:blocker|blockage)|"
            r"cannot be applied|can't be applied|no further autonomous evolution can proceed|"
            r"defect remains (?:unaddressed|unresolved)|system (?:is )?stall(?:s|ed|ing))\b",
            behavior_text,
        )
    ):
        issues.append("terminal_experiment_promoted_to_active_blocker")
    if operational_self_status and re.search(
        r"\brejected_history_promoted_to_blocker\b.{0,180}\b(?:still reports?|remains?|"
        r"persistent|unaddressed|unresolved|defect)\b|"
        r"\b(?:still reports?|remains?|persistent|unaddressed|unresolved)\b.{0,180}"
        r"\brejected_history_promoted_to_blocker\b",
        behavior_text,
    ):
        issues.append("improvement_frontier_promoted_to_active_blocker")
    if operational_self_status and re.search(
        r"\b(?:approve or reject|review(?: and)? (?:or )?merge) (?:the )?(?:proposed|current) "
        r"(?:patch|candidate|pull request|pr)\b",
        behavior_text,
    ):
        issues.append("nonexistent_candidate_review_requested")
    if operational_self_status and re.search(
        r"\b(?:you need to decide|decision (?:that )?(?:truly )?needs you|"
        r"decision needs you now)\b.{0,320}\b(?:new candidate|implement(?:s|ing)? the missing|"
        r"correct(?:s|ing)? the plan|halt further autonomous evolution)\b",
        behavior_text,
    ):
        issues.append("nonpending_owner_decision_invented")
    if operational_self_status and re.search(
        r"\b(?:decide(?: whether to| on)?|next decision).{0,220}\b(?:new candidate|"
        r"implement(?:s|ing)? (?:the )?missing|postpone|halt (?:further )?(?:autonomous )?evolution)\b|"
        r"\b(?:new candidate|implement(?:s|ing)? (?:the )?missing (?:api|interface)|postpone|"
        r"halt (?:further )?(?:autonomous )?evolution)\b.{0,220}\b(?:owner|you (?:must|need to) decide)\b",
        behavior_text,
    ):
        issues.append("nonpending_owner_decision_invented")
    if operational_self_status and re.search(
        r"\b(?:you (?:also )?(?:need to|must|can) (?:give|grant)(?: me)? permission|"
        r"requires? your permission).{0,100}\b(?:fetch|research|browse|public (?:web|internet)|"
        r"new resources)\b",
        behavior_text,
    ):
        issues.append("public_web_permission_misstated")
    if operational_self_status and re.search(
        r"\b(?:cannot|can['’]?t|unable to) (?:open|access|browse|search) "
        r"(?:the )?(?:public )?(?:web|internet)\b|"
        r"\b(?:did not|didn['’]?t) search (?:the )?(?:public )?(?:web|internet)"
        r".{0,160}\b(?:policy|only (?:the )?local (?:evidence|tools?))\b",
        behavior_text,
    ):
        issues.append("public_web_capability_denied")
    if operational_self_status and re.search(
        r"\b(?:the )?broker process(?: that (?:runs localpilot|manages the worker))?"
        r"(?:\s*\(pid\s*\d+\)|.{0,80}\bpid\s*(?:is\s*)?\d+)",
        behavior_text,
    ):
        issues.append("runtime_worker_misidentified_as_broker")
    return tuple(dict.fromkeys(issues))


def _contextual_evidence_risks(
    prompt: str,
    content: str,
    successful_tools: frozenset[str],
    evidence_messages: list[dict[str, Any]] | None = None,
) -> tuple[str, ...]:
    """Require the source a turn explicitly promised before accepting research claims."""
    request = " ".join(str(prompt).lower().split())
    answer = " ".join(str(content).lower().split())
    requires_primary_web = (
        not _is_operational_self_status_prompt(prompt)
        and bool(
            re.search(
                r"\b(?:public (?:web|internet)|primary source|fact[- ]check|"
                r"research (?:it|this) (?:now )?online)\b",
                request,
            )
        )
        and "fetch_public_https" not in _forbidden_tools(prompt)
    )
    requires_library = bool(
        re.search(
            r"\b(?:search|read|inspect|consult|use|check|look (?:in|through))\b.{0,40}"
            r"\b(?:local )?library\b",
            request,
        )
    )
    requires_library_citation = requires_library and bool(
        re.search(r"\b(?:cite|citation|source|page reference)\b", request)
    )
    evidence_restraint = bool(
        re.search(
            r"\b(?:no usable (?:source|evidence)|no primary source|source (?:was|is) unavailable|"
            r"could not (?:retrieve|verify|establish)|cannot (?:retrieve|verify|establish)|"
            r"unverified|unresolved|not verified)\b",
            answer,
        )
    )
    risks: list[str] = []
    if (
        requires_primary_web
        and "fetch_public_https" not in successful_tools
        and not evidence_restraint
    ):
        risks.append("research_claims_without_primary_source")
    if (
        requires_library
        and not _LIBRARY_TOOLS.intersection(successful_tools)
        and not evidence_restraint
    ):
        risks.append("library_claims_without_library_source")
    if (
        requires_library_citation
        and _LIBRARY_TOOLS.intersection(successful_tools)
        and "library://" not in answer
    ):
        risks.append("library_answer_missing_source_citation")

    temporal_web = _is_temporal_web_prompt(prompt)
    if temporal_web and not evidence_restraint:
        if "search_public_web" not in successful_tools:
            risks.append("latest_claim_without_current_web_discovery")
        if "fetch_public_https" not in successful_tools:
            risks.append("latest_claim_without_current_primary_source")
        messages = list(evidence_messages or [])
        search_texts = [
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "tool"
            and message.get("tool_name") == "search_public_web"
            and "Tool error:" not in str(message.get("content") or "")
        ]
        source_texts = [
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "tool"
            and message.get("tool_name") == "fetch_public_https"
            and "Tool error:" not in str(message.get("content") or "")
        ]
        if search_texts and not any(
            re.search(
                r"Public web search:\s*['\"].*\b(?:latest|newest|current|today|as of)\b",
                text,
                re.IGNORECASE,
            )
            for text in search_texts
        ):
            risks.append("latest_claim_from_version_guessed_search")
        if source_texts:
            source_text = "\n".join(source_texts)
            claimed_versions = tuple(
                dict.fromkeys(
                    re.findall(
                        r"\b(?:python\s+)?(\d+\.\d+(?:\.\d+)?)\b",
                        str(content),
                        re.IGNORECASE,
                    )
                )
            )
            if claimed_versions and not all(
                version.lower() in source_text.lower()
                for version in claimed_versions
            ):
                risks.append("latest_claimed_version_missing_from_primary_source")
            if claimed_versions and re.search(
                r"\b(?:has been|was) superseded by\b",
                source_text,
                re.IGNORECASE,
            ):
                for version in claimed_versions:
                    superseded = re.search(
                        rf"(?:Python\s+)?{re.escape(version)}.{{0,240}}?"
                        r"(?:has been|was) superseded by\s+(?:Python\s+)?(\d+\.\d+(?:\.\d+)?)",
                        source_text,
                        re.IGNORECASE | re.DOTALL,
                    )
                    if superseded and superseded.group(1) != version:
                        risks.append("latest_claim_uses_superseded_primary_source")
                        break
            if not re.search(
                r"\b(?:latest|newest|current stable|superseded by|now available)\b",
                source_text,
                re.IGNORECASE,
            ):
                risks.append("latest_claim_primary_source_does_not_establish_recency")
    return tuple(dict.fromkeys(risks))


def _library_citation_from_messages(
    messages: list[dict[str, Any]],
) -> str | None:
    """Recover only a literal citation emitted by a successful library tool."""
    for preferred_tool in ("read_library_passage", "search_library"):
        for message in messages:
            if (
                message.get("role") != "tool"
                or message.get("tool_name") != preferred_tool
            ):
                continue
            match = re.search(
                r"library://[^\r\n]*?#page=\d+(?:&passage=\d+)?",
                str(message.get("content") or ""),
                re.IGNORECASE,
            )
            if match:
                return match.group(0)
    return None


def _strip_authority_meta(content: str) -> str:
    """Remove validator-facing closing boilerplate without rewriting factual prose."""
    parts = re.split(r"(\n\s*\n)", str(content))
    drop_prefixes = (
        "all statements above",
        "all claims above",
        "no additional classes",
        "no additional claims",
        "this summary reflects only",
    )
    kept: list[str] = []
    for part in parts:
        normalized = " ".join(part.lower().split())
        if normalized.startswith(drop_prefixes):
            continue
        if re.fullmatch(r"\n\s*\n", part) and (
            not kept or re.fullmatch(r"\n\s*\n", kept[-1])
        ):
            continue
        kept.append(part)
    return "".join(kept).strip()


def _information_authority_risks(content: str) -> list[str]:
    """Detect a few high-impact subsystem-flow claims that must fail closed."""
    text = " ".join(str(content).lower().split())
    risk_text = re.sub(
        r"operator(?: research)? loop.{0,80}(?:does not|never).{0,80}upsert_knowledge_facts",
        "",
        text,
    )
    risk_text = re.sub(
        r"operator(?: research)? loop.{0,100}(?:does not|never).{0,40}(?:write|record|persist|store)s?.{0,60}(?:knowledge[_ ]?facts?|staged[- ]study facts?|study facts?)",
        "",
        risk_text,
    )
    risk_text = re.sub(
        r"commandrunner.{0,80}(?:is not|does not|never).{0,80}(?:every|all) tool",
        "",
        risk_text,
    )
    risk_text = re.sub(
        r"not (?:all|every) tools?.{0,80}commandrunner",
        "",
        risk_text,
    )
    risk_text = re.sub(
        r"github actions.{0,40}(?:does not|do not|never).{0,40}(?:merge|promote)",
        "",
        risk_text,
    )
    risk_text = re.sub(
        r"(?:preserves?|retains?).{0,30}candidate branch.{0,100}(?:rather than|not).{0,40}(?:clearing|deleting|removing)",
        "",
        risk_text,
    )
    risk_text = re.sub(
        r"candidate branch.{0,60}(?:is not|does not|never).{0,40}(?:cleared|deleted|removed)",
        "",
        risk_text,
    )
    patterns = {
        "automatic_operator_learning": (
            r"after each (?:interaction|turn).{0,120}(?:record|learn|persist)",
            r"operator.{0,100}(?:feeds|passes).{0,100}(?:learningmemory|learning memory)",
        ),
        "operator_writes_study_facts": (
            r"operator(?: research)? loop.{0,80}(?:may |does |will )?(?:invokes?|calls?|writes?)(?: to)? (?:learningmemory\.)?upsert_knowledge_facts",
            r"operator(?: research)? loop.{0,80}(?:persists?|stores?|writes?) (?:staged[- ]study |study )?(?:knowledge_)?facts",
            r"operator(?: research)? loop.{0,180}(?:records?|persists?|stores?).{0,60}(?:knowledge[_ ]?facts?|staged[- ]study facts?)",
        ),
        "cycle_memory_becomes_operator_knowledge": (
            r"(?:cycle|candidate) (?:outcomes?|records?).{0,160}inform.{0,80}operator",
        ),
        "command_runner_wraps_all_tools": (
            r"commandrunner.{0,120}(?:before any|every|all) tool",
        ),
        "github_actions_merges": (
            r"merged via github actions",
            r"github actions (?:automatically )?(?:merges?|promotes?)",
            r"github actions.{0,24}performs? (?:the )?(?:merge|promotion)",
        ),
        "resource_governor_triggers_evolution": (
            r"triggered.{0,60}(?:by|through) (?:the )?resourcegovernor",
            r"resourcegovernor.{0,60}triggers? (?:the )?(?:developer|self-development|evolution)",
        ),
        "candidate_branch_history_cleared": (
            r"candidate branch(?: and (?:github )?history)?.{0,24}\b(?:is|are|gets?|may be|will be)\s+(?:cleared|deleted|removed)",
        ),
        "developer_local_process_erased": (
            r"only (?:the )?stable operator (?:runs|executes) locally",
            r"only (?:the )?operator(?:'s)? (?:own )?code (?:runs|executes)(?: locally)?",
        ),
        "human_lesson_as_knowledge_fact": (
            r"(?:facts|knowledge_facts).{0,40}(?:are |is )?(?:written|stored|recorded)(?: only)? (?:by|through).{0,40}record_human_lesson",
            r"record_human_lesson (?:writes|stores|records) (?:a |the )?(?:knowledge_?facts?|facts?)",
        ),
        "verification_only_on_digest_mismatch": (
            r"(?:verification|verified|verify).{0,100}only (?:when|if).{0,100}(?:digest )?mismatch",
            r"only (?:when|if).{0,100}(?:digest )?mismatch.{0,100}(?:verification|verified|verify)",
        ),
        "teach_records_observations": (
            r"(?:record|records|recording) (?:the )?(?:operator )?observations?.{0,60}(?:/teach|record_human_lesson)",
        ),
        "operator_policy_governs_all_tools": (
            r"(?:operator(?:'s)? )?safety policy.{0,50}(?:governs|applies to|controls).{0,30}all tool",
            r"(?:operator(?:'s)? )?safety policy.{0,60}ensures.{0,40}(?:any|all) tool",
            r"all interactions.{0,50}(?:governed|controlled).{0,40}(?:the )?safety policy",
        ),
        "learning_memory_only_teach_study": (
            r"learningmemory.{0,100}(?:written|populated).{0,30}only.{0,120}(?:/teach|staged.?study|study)",
            r"learningmemory.{0,100}(?:only written|only populated).{0,120}(?:/teach|staged.?study|study)",
            r"learningmemory.{0,200}(?:it )?(?:is )?(?:updated|written|populated) only.{0,150}(?:record_human_lesson|upsert_knowledge_facts|/teach|staged.?study)",
            r"learningmemory.{0,200}only explicit writes.{0,150}(?:record_human_lesson|upsert_knowledge_facts|/teach|staged.?study)",
        ),
        "ci_after_human_merge": (
            r"after (?:a |the )?(?:candidate )?(?:pull request|pr) is merged.{0,100}(?:github actions|ci)",
            r"human merge.{0,100}(?:then|before).{0,50}(?:github actions|ci (?:runs|starts))",
        ),
        "developer_uses_operator_policy": (
            r"stable operator and (?:the )?developer.{0,80}(?:normal|same|operator) safety policy",
            r"developer.{0,80}(?:uses|operates under|is governed by).{0,50}(?:normal|operator) safety policy",
            r"self-development(?: runtime)?.{0,160}(?:same|operator).{0,60}safety boundar",
            r"developer.{0,120}(?:same|operator).{0,60}safety boundar",
        ),
        "candidate_commit_after_merge": (
            r"candidate changes.{0,140}(?:never|not).{0,50}(?:committed|pushed).{0,100}until.{0,50}(?:pull request|pr)?.{0,20}merged",
            r"candidate.{0,100}(?:committed|pushed).{0,60}after (?:the )?(?:human )?merge",
        ),
        "stable_operator_local_process_erased": (
            r"only (?:the )?developer(?: process)? (?:runs|executes) locally",
        ),
        "exclusive_learning_writer": (
            r"record_human_lesson.{0,80}(?:is )?the only (?:place|path|writer)",
            r"upsert_knowledge_facts.{0,80}(?:is )?the only (?:place|path|writer)",
        ),
    }
    return [
        name
        for name, expressions in patterns.items()
        if any(re.search(expression, risk_text) for expression in expressions)
    ]

