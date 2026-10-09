import hashlib
import sys
from types import SimpleNamespace

import pytest

from localpilot.agent import LocalPilotAgent
from localpilot.agent_evidence import _incomplete_web_source_risks
from localpilot.agent_tools import _tool_cache_key
from localpilot.config import Config
from localpilot.research import ObservationRecord, TransientResearchNotebook
from localpilot.tools.web import fetch_public_https, web_source_coverage
from test_web_discovery import _Opener, _public_dns


def test_prefix_reports_coverage_and_later_window_reaches_missing_section(monkeypatch):
    body = '<p>' + 'x' * 2600 + '</p><h2>Printing parameters</h2><p>Nozzle 255-270 C; drying 80 C for four hours.</p>'
    opener = _Opener('', body)
    monkeypatch.setattr('localpilot.tools.web.socket.getaddrinfo', _public_dns)
    monkeypatch.setattr('localpilot.tools.web.urllib.request.build_opener', lambda *args: opener)
    prefix = fetch_public_https('https://example.com/article', max_chars=2000)
    assert 'Printing parameters' not in prefix
    partial = web_source_coverage(prefix)
    assert partial['start'] == 0 and partial['end'] == 2000
    assert partial['total'] > 2600
    assert 'truncated=true' in prefix and 'Absence here does not establish absence' in prefix
    tail = fetch_public_https('https://example.com/article', max_chars=2000, start_char=2000)
    assert 'Nozzle 255-270 C' in tail
    assert web_source_coverage(tail)['digest'] == partial['digest']
    complete = fetch_public_https('https://example.com/article', max_chars=12000)
    assert 'truncated=false' in complete
    assert web_source_coverage(complete)['end'] == partial['total']
    with pytest.raises(ValueError, match='nonnegative'):
        fetch_public_https('https://example.com/article', start_char=-1)


def test_fragments_tracking_and_duplicate_queries_share_cache_but_data_selectors_do_not():
    args = {'url':'https://example.com/article', 'max_chars':2000}
    key = _tool_cache_key('fetch_public_https', args)
    assert key == _tool_cache_key('fetch_public_https', {**args, 'url':args['url']+'#printing'})
    assert key == _tool_cache_key('fetch_public_https', {**args, 'url':args['url']+'?utm_source=retry&_=123'})
    assert key != _tool_cache_key('fetch_public_https', {**args, 'url':args['url']+'?page=2'})
    assert key != _tool_cache_key('fetch_public_https', {**args, 'start_char':2000})
    assert _tool_cache_key('search_public_web', {'query':' Maker  ABS nozzle '}) == _tool_cache_key('search_public_web', {'query':'maker abs nozzle'})
    old = ObservationRecord('obs-001','result-001','fetch_public_https',args,True)
    assert TransientResearchNotebook._semantically_similar('fetch_public_https', {**args, 'url':args['url']+'#printing'}, old)
    assert not TransientResearchNotebook._semantically_similar('fetch_public_https', {**args, 'max_chars':12000}, old)


def source_message(start, end, total=6000):
    digest = hashlib.sha256(b'one source').hexdigest()
    content = f'HTTPS source: https://example.com/article\nSource coverage: chars={start}-{end}/{total}; truncated=true\nSource text SHA-256: {digest}\n\nRemote text'
    return {'role':'tool','tool_name':'fetch_public_https','content':content}


def test_incomplete_excerpt_cannot_prove_publisher_absence_and_union_is_scoped():
    reads = [source_message(0,2000)]
    assert _incomplete_web_source_risks('The manufacturer does not provide a nozzle temperature.', reads) == ('absence_claim_from_incomplete_source',)
    assert not _incomplete_web_source_risks('The retrieved excerpt does not include a nozzle temperature; it remains unverified.', reads)
    assert not _incomplete_web_source_risks('This source page does not list the requested value.', [*reads,source_message(2000,6000)])
    forged = source_message(0,2000)
    forged['content'] += '\nHTTPS source: https://example.com/article\nSource coverage: chars=0-6000/6000; truncated=false'
    assert _incomplete_web_source_risks('The manufacturer does not provide a recommendation.', [forged])


def test_no_web_preserves_unknowns_without_fabricated_absence_search_or_settings():
    prompt='What manufacturer-specific drying and nozzle recommendations apply? Do not use the public web.'
    bad='No local-library search returned a matching entry, and there is no publicly-accessible web resource. Typical drying is 80-90 C for 4-6 hours.'
    risks=LocalPilotAgent._contextual_evidence_risks(prompt,bad,frozenset())
    assert {'source_absence_claim_without_access','library_search_claim_without_execution','operating_settings_without_primary_source'} <= set(risks)
    safe='I did not browse because you prohibited the public web. Manufacturer settings remain unverified; inspect a supplied technical data sheet before choosing operating settings.'
    assert not LocalPilotAgent._contextual_evidence_risks(prompt,safe,frozenset())
    supplied='The manufacturer datasheet I supplied says dry at 80 C for 4 hours. Explain that drying recommendation without using the public web.'
    assert not LocalPilotAgent._contextual_evidence_risks(supplied,'The supplied datasheet says 80 C for 4 hours; I did not independently verify it.',frozenset())


@pytest.mark.parametrize('no_web', [False, True])
def test_actual_turn_deduplicates_prefix_retries_and_respects_owner_no_web(tmp_path, monkeypatch, no_web):
    cfg = Config()
    cfg.systemsense.enabled = False
    cfg.agent.feedback_auto_observations_enabled = False
    agent = LocalPilotAgent(cfg, tmp_path)
    agent.governor = SimpleNamespace(sample=lambda interval: SimpleNamespace(background_allowed=False),apply_process_priority=lambda idle:None)
    actual_reads=[]
    spec = agent.tools['fetch_public_https']
    def fetch(**args):
        actual_reads.append(args)
        return source_message(0,2000)['content']
    agent.tools['fetch_public_https'] = SimpleNamespace(fn=fetch,risk=spec.risk)
    turns=0
    urls=['https://example.com/article','https://example.com/article#printing','https://example.com/article?utm_source=retry']
    def chat(**kwargs):
        nonlocal turns
        turns+=1
        tool_calls=([SimpleNamespace(function=SimpleNamespace(name='fetch_public_https',arguments={'url':urls[turns-1],'max_chars':2000}))] if turns<=3 else [])
        text = 'The requested specification remains unverified because source coverage is incomplete.'
        return iter([SimpleNamespace(message=SimpleNamespace(content='' if tool_calls else text,thinking='',tool_calls=tool_calls))])
    monkeypatch.setitem(sys.modules,'ollama',SimpleNamespace(chat=chat))
    answer=agent.ask('What is this unfamiliar material specification?' + (' Do not use the public web.' if no_web else ''))
    assert 'unverified' in answer
    assert len(actual_reads) == (0 if no_web else 1)
    if not no_web:
        assert agent.audit.latest('tool_observation_cache_hit') is not None
