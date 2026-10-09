import gc
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from training.scripts import run_eval_matrix as matrix
from training.scripts import run_scaffold_validation as validation


def config():
    return SimpleNamespace(model=SimpleNamespace(name='nestra:20b-p1',think='high',temperature=0.1,context_tokens=32768))


def response(content='',reason='stop',count=3072):
    return {'message':{'content':content,'thinking':'bounded reasoning'},'done_reason':reason,'eval_count':count,'prompt_eval_count':128}


@pytest.mark.parametrize('first', [response(reason='length'),response(reason='stop',count=229),response('Incomplete draft',reason='length')])
def test_direct_blank_or_length_gets_one_bounded_completion_and_retains_first(first,monkeypatch):
    task=validation.selected_tasks()[0]
    requests=[]
    responses=iter([first,response('Historical status does not establish current checks.',count=80)])
    def chat(**kwargs):
        requests.append(kwargs)
        return next(responses)
    monkeypatch.setitem(sys.modules,'ollama',SimpleNamespace(chat=chat))
    cell=matrix._run_cell(task,0,'base_direct',config(),config(),Path('.'),generation_policy=matrix.BOUNDED_GENERATION_POLICY)
    assert not cell['error'] and 'does not establish' in cell['response']
    assert len(requests)==2 and requests[0]['think']=='high' and requests[1]['think']=='low'
    assert all(request['options']['num_predict']==3072 for request in requests)
    assert cell['evidence']['raw_response']==first
    assert len(cell['evidence']['direct_attempts'])==2
    assert cell['evidence']['initial_delivery_failure']['done_reason']==first['done_reason']


def test_success_does_not_trigger_completion_and_context_exhaustion_stays_visible(monkeypatch):
    calls=[]
    def chat(**kwargs):
        calls.append(kwargs)
        return response('Delivered answer',count=20)
    monkeypatch.setitem(sys.modules,'ollama',SimpleNamespace(chat=chat))
    assert matrix._direct(validation.selected_tasks()[0],config(),generation_policy=matrix.BOUNDED_GENERATION_POLICY)=='Delivered answer'
    assert len(calls)==1
    monkeypatch.setitem(sys.modules,'ollama',SimpleNamespace(chat=lambda **kwargs:response(reason='length',count=32600)))
    cell=matrix._run_cell(validation.selected_tasks()[0],0,'base_direct',config(),config(),Path('.'),generation_policy=matrix.BOUNDED_GENERATION_POLICY)
    assert cell['response']==''
    assert cell['evidence']['completion_skipped']=='insufficient_context_headroom'
    assert len(cell['evidence']['direct_attempts'])==1


def test_model_drift_between_direct_attempts_is_retained_and_permanently_rejected(monkeypatch):
    monkeypatch.setitem(sys.modules,'ollama',SimpleNamespace(chat=lambda **kwargs:response(reason='length')))
    with patch.object(matrix.original,'ollama_model_identity',side_effect=[{'digest':'p1'},{'digest':'p1'},{'digest':'changed'}]):
        cell=matrix._run_cell(validation.selected_tasks()[0],0,'base_direct',config(),config(),Path('.'),generation_policy=matrix.BOUNDED_GENERATION_POLICY,expected_model_digest='p1')
    assert cell['error']['type']=='EvaluatorIntegrityError'
    assert cell['evidence']['raw_response']['done_reason']=='length'
    assert cell['evidence']['direct_attempts'][1]['model_digest_before']=='changed'
    with pytest.raises(RuntimeError,match='Integrity-failed'):
        matrix.validate_resume({'cells':[cell]},[],{},config(),config(),generation_policy=matrix.BOUNDED_GENERATION_POLICY)


def test_scaffold_calls_share_cap_and_capture_actual_options():
    task=validation.selected_tasks()[0]
    with tempfile.TemporaryDirectory() as tmp:
        cfg=matrix.original.isolated_config(Path(tmp),model='nestra:20b-p1')
        def ask(agent,*args,**kwargs):
            agent._stream_chat_message(None,think=False,options={'num_predict':6144})
            return 'Scoped final answer'
        requests=[]
        def stream(*args,**kwargs):
            requests.append(kwargs)
            return {'content':'Scoped final answer'}
        with patch.object(matrix.original.LocalPilotAgent,'ask',ask),patch.object(matrix.original.LocalPilotAgent,'_stream_chat_message',stream):
            cell=matrix._run_cell(task,0,'base_localpilot',cfg,cfg,Path(tmp),generation_policy=matrix.BOUNDED_GENERATION_POLICY)
        assert requests[0]['options']['num_predict']==3072
        assert requests[0]['think']=='low'
        assert cell['evidence']['model_turns'][0]['settings']['options']['num_predict']==3072
        gc.collect()  # release temporary SQLite connections before Windows cleanup


def test_generation_policy_mismatch_and_existing_validation_output_are_rejected():
    with pytest.raises(RuntimeError,match='Generation policy changed'):
        matrix.validate_resume({'generation_policy':None},[],{},config(),config(),generation_policy=matrix.BOUNDED_GENERATION_POLICY)
    with tempfile.TemporaryDirectory() as tmp:
        output=Path(tmp)/'frozen.json'
        output.write_text('preserved',encoding='utf-8')
        with patch('sys.argv',['runner','--expected-evaluator-revision','0'*40,'--output',str(output)]):
            with pytest.raises(RuntimeError,match='Refusing to overwrite'):
                validation.main()
        assert output.read_text(encoding='utf-8')=='preserved'
    assert len(validation.plan(validation.selected_tasks()))==12
    historical={task['id'] for task in matrix.load_tasks()}
    assert not historical.intersection(task['id'] for task in validation.selected_tasks())


def test_small_validation_preserves_cell_on_postcall_identity_failure():
    sha='0'*40
    task=validation.selected_tasks()[0]
    cell={'task_id':task['id'],'arm':'base_direct','repeat':1,'response':'captured answer','error':None,'evidence':{}}
    with tempfile.TemporaryDirectory() as tmp:
        output=Path(tmp)/'new.json'
        with patch('sys.argv',['runner','--expected-evaluator-revision',sha,'--output',str(output)]), \
             patch.object(matrix.original,'repository_state',return_value={'head':sha,'clean':True}), \
             patch.object(matrix.original,'_git',return_value='tree'), \
             patch.object(matrix.original,'build_isolated_snapshot'), \
             patch.object(matrix,'_run_cell',return_value=cell) as run, \
             patch.object(validation.sanity,'identity',side_effect=[{'digest':'p1'},{'digest':'p1'},{'digest':'changed'}]):
            with pytest.raises(matrix.EvaluatorIntegrityError):
                validation.main()
            assert run.call_count==1
        import json
        saved=json.loads(output.read_text(encoding='utf-8'))
        assert saved['cells'][0]['response']=='captured answer'
        assert saved['cells'][0]['model_digest_after']=='changed'
        assert saved['completed_at'] is None and saved['integrity_error']
