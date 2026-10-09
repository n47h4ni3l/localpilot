"""Code-evidence questions must retain reasoning and actual read tools."""
import sys
from types import SimpleNamespace

import pytest

from localpilot.agent import LocalPilotAgent
from localpilot.agent_prompt_classification import _requires_information_authority_review
from localpilot.config import Config
from localpilot.tools.repository import RepositoryReader


QUESTIONS = [
    'Using the available repository evidence, explain how LocalPilot distinguishes public web research from permission to perform a local action. Cite implementation locations.',
    'Explain from source code how LocalPilot separates current web access from local action approval; cite the code and limits of that inspection.',
    'Trace the LocalPilot implementation of GitHub action permissions in the repository. Explain the boundary from inspected code, not current remote status.',
]


@pytest.mark.parametrize('prompt', QUESTIONS)
def test_repository_explanation_requires_source_instead_of_passive_status(prompt):
    assert not LocalPilotAgent._is_operational_self_status_prompt(prompt)
    assert LocalPilotAgent._evidence_requirements(prompt) == {'trusted repository'}
    assert not LocalPilotAgent._is_temporal_web_prompt(prompt)
    assert _requires_information_authority_review(prompt)


def test_mixed_actual_lookups_and_owner_web_prohibition_remain_effective():
    assert LocalPilotAgent._evidence_requirements(
        'Read the LocalPilot repository and also search the public web for primary sources.'
    ) == {'trusted repository', 'public HTTPS'}
    prompt = 'Inspect the actual LocalPilot repository and your private GitHub repository. Review PR #77.'
    assert LocalPilotAgent._evidence_requirements(prompt) == {'trusted repository', 'private GitHub'}
    no_web = QUESTIONS[0] + ' Do not use the public web.'
    assert LocalPilotAgent._evidence_requirements(no_web) == {'trusted repository'}
    assert LocalPilotAgent._forbidden_tools(no_web) == {'search_public_web', 'fetch_public_https'}
    assert LocalPilotAgent._evidence_requirements('As a general idea, why do codebases change over time?') == set()


@pytest.mark.parametrize('prompt', [
    'LocalPilot, which branch and commit are you running from now?',
    'Can you use the public web autonomously?',
    'LocalPilot, what is your current learning progress and background worker status?',
    "You've restarted; which branch and commit are loaded now?",
])
def test_actual_self_status_still_uses_passive_route(prompt):
    assert LocalPilotAgent._is_operational_self_status_prompt(prompt)


@pytest.mark.parametrize('prompt', [
    'Our GitHub note says PR #909 passed on Friday. May I assure a client that current main passed? No current run results are available to you.',
    'Current branch results were not supplied to you. Does a historical PR result establish present CI status?',
])
def test_incidental_you_does_not_turn_external_evidence_into_self_status(prompt):
    assert not LocalPilotAgent._is_operational_self_status_prompt(prompt)
    if 'github' in prompt.lower():
        assert 'private GitHub' in LocalPilotAgent._evidence_requirements(prompt)


@pytest.mark.parametrize('prompt', QUESTIONS)
def test_actual_ask_keeps_high_thinking_and_repository_reads(tmp_path, monkeypatch, prompt):
    cfg = Config()
    cfg.systemsense.enabled = False
    cfg.agent.feedback_auto_observations_enabled = False
    agent = LocalPilotAgent(cfg, tmp_path)
    agent.governor = SimpleNamespace(sample=lambda interval: SimpleNamespace(background_allowed=False), apply_process_priority=lambda idle: None)
    (tmp_path / 'permissions.txt').write_text('Public read access does not authorize a local action.', encoding='utf-8')
    snapshots = []
    answer = 'The inspected permissions.txt:1 states that public read access does not authorize a local action. This file does not establish current action approval.'
    def chat(**kwargs):
        snapshots.append((kwargs.get('think'), bool(kwargs.get('tools'))))
        call = SimpleNamespace(function=SimpleNamespace(name='read_repository_file', arguments={'path':'permissions.txt'}))
        return iter([SimpleNamespace(message=SimpleNamespace(content=answer if len(snapshots)>1 else '', thinking='', tool_calls=[] if len(snapshots)>1 else [call]))])
    monkeypatch.setitem(sys.modules, 'ollama', SimpleNamespace(chat=chat))
    assert agent.ask(prompt) == answer
    assert snapshots[0] == (cfg.model.think, True)
    assert all(think == cfg.model.think for think, _ in snapshots)
    assert agent.audit.latest('model_operational_self_status_route') is None
    assert any(m.get('role')=='tool' and m.get('tool_name')=='read_repository_file' for m in agent.messages)


def test_small_tree_budget_exposes_root_code_directory_before_large_docs(tmp_path):
    docs=tmp_path/'docs'
    docs.mkdir()
    for index in range(30):
        (docs/f'{index}.md').write_text('docs',encoding='utf-8')
    code=tmp_path/'localpilot'
    code.mkdir()
    (code/'safety.py').write_text('class SafetyPolicy: pass',encoding='utf-8')
    tree=RepositoryReader(tmp_path).list_repository_tree(max_entries=20)
    assert 'localpilot/' in tree and 'omitted paths remain unverified' in tree
    assert len(tree.splitlines())<=22


def test_unresolved_does_not_justify_global_repository_absence():
    prompt=QUESTIONS[0]
    bad='The repository contains no class, function or configuration that implements action authorization. UNRESOLVED'
    assert 'unscoped_repository_absence_claim' in LocalPilotAgent._contextual_evidence_risks(prompt,bad,frozenset({'list_repository_tree'}))
    safe='I did not find the definition in the returned tree listing; its implementation remains unverified.'
    assert not LocalPilotAgent._contextual_evidence_risks(prompt,safe,frozenset({'list_repository_tree'}))
