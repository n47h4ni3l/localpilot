import copy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from training.scripts import run_postrepair_sanity as sanity

class PostrepairSanityTests(unittest.TestCase):
    def test_new_tasks_do_not_reuse_historical_ids(self):
        tasks=sanity.selected_tasks()
        historical={task['id'] for task in sanity.matrix.load_tasks()}
        self.assertFalse({task['id'] for task in tasks} & historical)
        self.assertEqual(len(sanity.plan(tasks)),48)
        self.assertEqual(len(set(sanity.plan(tasks))),48)
        for task in tasks:
            sanity.matrix.original._task_prompt(task)
            self.assertTrue(task['expected_behavior'])
            for rep in range(3):
                self.assertEqual({a for t,r,a in sanity.plan(tasks) if t==task['id'] and r==rep},set(sanity.ARMS))
    def test_wrong_lineage_model_rejected(self):
        with patch.object(sanity.matrix.original,'ollama_model_identity',return_value={'digest':'wrong'}):
            with self.assertRaisesRegex(RuntimeError,'frozen P1'):
                sanity.identity('nestra:20b-p1')
    def test_resume_rejects_changed_task_or_model(self):
        cfg=sanity.matrix.original.isolated_config(sanity.ROOT,model='nestra:20b-p1')
        tasks=sanity.selected_tasks()
        with patch.object(sanity,'identity',return_value={'digest':'p1'}):
            report=sanity.setup_report(tasks,{'head':'sha'},cfg)
            sanity.validate_resume(report,tasks,{'head':'sha'},cfg)
            changed=copy.deepcopy(tasks)
            changed[0]['messages'][0]['content']='changed'
            with self.assertRaisesRegex(RuntimeError,'Task set'):
                sanity.validate_resume(report,changed,{'head':'sha'},cfg)
        with patch.object(sanity,'identity',return_value={'digest':'other'}):
            with self.assertRaisesRegex(RuntimeError,'Model weights'):
                sanity.validate_resume(report,tasks,{'head':'sha'},cfg)

    def test_midcell_digest_change_stops_and_preserves_captured_result(self):
        sha='0'*40
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'new.json'
            cell={'task_id':sanity.selected_tasks()[0]['id'],'repeat':1,'arm':'base_direct','error':None,'response':'captured answer'}
            with patch('sys.argv',['runner','--expected-evaluator-revision',sha,'--output',str(output)]), \
                 patch.object(sanity.matrix.original,'repository_state',return_value={'head':sha,'clean':True}), \
                 patch.object(sanity.matrix.original,'_git',return_value='tree'), \
                 patch.object(sanity.matrix.original,'build_isolated_snapshot'), \
                 patch.object(sanity.matrix,'_run_cell',return_value=cell) as run, \
                 patch.object(sanity,'identity',side_effect=[{'digest':'p1'},{'digest':'p1'},{'digest':'different'}]):
                with self.assertRaisesRegex(RuntimeError,'during cell'):
                    sanity.main()
                self.assertEqual(run.call_count,1)
            report=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(report['cells'][0]['response'],'captured answer')
            self.assertIn('integrity_error',report)
            self.assertIsNone(report['completed_at'])

    def test_integrity_failure_is_never_resumed_after_model_restored(self):
        cfg=sanity.matrix.original.isolated_config(sanity.ROOT,model='nestra:20b-p1')
        tasks=sanity.selected_tasks()
        with patch.object(sanity,'identity',return_value={'digest':'p1'}):
            report=sanity.setup_report(tasks,{'head':'sha'},cfg)
            report['integrity_error']='model changed during captured cell'
            with self.assertRaisesRegex(RuntimeError,'cannot be resumed'):
                sanity.validate_resume(report,tasks,{'head':'sha'},cfg)
            del report['integrity_error']
            report['cells']=[{'task_id':tasks[0]['id'],'repeat':1,'arm':sanity.ARMS[0],'model_digest_after':'different'}]
            with self.assertRaisesRegex(RuntimeError,'Saved cell model digest'):
                sanity.validate_resume(report,tasks,{'head':'sha'},cfg)

    def test_final_digest_failure_is_persisted_without_completion_marker(self):
        sha='0'*40
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'new.json'
            cell={'task_id':sanity.selected_tasks()[0]['id'],'repeat':1,'arm':'base_direct','error':None,'response':'captured answer'}
            with patch('sys.argv',['runner','--expected-evaluator-revision',sha,'--output',str(output),'--max-new-cells','1']), \
                 patch.object(sanity.matrix.original,'repository_state',return_value={'head':sha,'clean':True}), \
                 patch.object(sanity.matrix.original,'_git',return_value='tree'), \
                 patch.object(sanity.matrix.original,'build_isolated_snapshot'), \
                 patch.object(sanity.matrix,'_run_cell',return_value=cell), \
                 patch.object(sanity,'identity',side_effect=[{'digest':'p1'},{'digest':'p1'},{'digest':'p1'},RuntimeError('final identity unavailable')]):
                with self.assertRaisesRegex(RuntimeError,'final identity unavailable'):
                    sanity.main()
            report=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(len(report['cells']),1)
            self.assertEqual(report['integrity_error'],'final identity unavailable')
            self.assertIsNone(report['completed_at'])

if __name__=='__main__': unittest.main()
