"""Regression checks for source-derived self-improvement examples."""
import ast
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'training/scripts'))
import curate_p12_fresh as curator

def local_runner(tree):
    try:
        exec(compile(tree,'<authored-test-fixture>','exec'),{})
        return {'exit_code':0,'stdout':'','stderr':''}
    except AssertionError:
        return {'exit_code':1,'stdout':'AssertionError:','stderr':''}

class FreshCurationRegressionTests(unittest.TestCase):
    def test_test_setup_and_assertion_order_are_preserved(self):
        source='''INCREMENT=1
def offset(x):
    return x+INCREMENT
value=offset(1)
assert value==2
value=offset(2)
assert value==3
value=offset(3)
assert value==4
'''
        tree=ast.parse(source)
        assertions=[n for n in ast.walk(tree) if isinstance(n,ast.Assert)]
        with patch.object(curator,'run_code',local_runner):
            messages,evidence=curator.make_self_improvement({'instruction':'Add one to x.'},tree,assertions,{'exit_code':0})
        blocks=re.findall(r'```python\n(.*?)```',messages[0]['content'],re.S)
        checks=ast.parse(blocks[1])
        expected=ast.Module(body=tree.body[2:],type_ignores=[])
        self.assertEqual(ast.dump(checks,include_attributes=False),ast.dump(expected,include_attributes=False))
        implementation=ast.parse(blocks[0])
        self.assertIsInstance(implementation.body[0],ast.Assign)
        self.assertEqual(evidence['passed_execution']['exit_code'],0)

    def test_mutation_changes_one_expression_and_preserves_test_helpers(self):
        # CPython AST operator objects can be shared; changing every reference
        # would corrupt both additions and the test helper's comparisons.
        source='''def triangular(n):
    total=0
    for i in range(1,n+1):
        total+=i
    return total
def test_triangular():
    assert triangular(1)==1
    assert triangular(3)==6
    assert triangular(5)==15
test_triangular()
'''
        tree=ast.parse(source)
        assertions=[n for n in ast.walk(tree) if isinstance(n,ast.Assert)]
        with patch.object(curator,'run_code',local_runner):
            messages,evidence=curator.make_self_improvement({'instruction':'Compute the nth triangular number.'},tree,assertions,{'exit_code':0})
        blocks=re.findall(r'```python\n(.*?)```',messages[0]['content'],re.S)
        candidate=ast.parse(blocks[0])
        tests=ast.parse(blocks[1])
        corrected=ast.parse(re.search(r'```python\n(.*?)```',messages[1]['content'],re.S).group(1))
        self.assertFalse(any(isinstance(n,ast.Assert) for n in ast.walk(candidate)))
        original_checks=[ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Assert)]
        self.assertEqual(original_checks,[ast.dump(n,include_attributes=False) for n in ast.walk(tests) if isinstance(n,ast.Assert)])
        left=[type(n).__name__ for n in ast.walk(candidate)]
        right=[type(n).__name__ for n in ast.walk(corrected)]
        self.assertEqual(len(left),len(right))
        self.assertEqual(sum(a!=b for a,b in zip(left,right)),1)
        self.assertEqual(evidence['failed_execution']['exit_code'],1)
        self.assertEqual(evidence['passed_execution']['exit_code'],0)

    def test_source_snapshot_coverage_and_freshness(self):
        rows=[json.loads(x) for x in (ROOT/'training/datasets/p12_fresh_candidates.jsonl').read_bytes().splitlines()]
        sources=[json.loads(x) for x in (ROOT/'training/sources/p12_fresh_source_records.jsonl').read_bytes().splitlines()]
        self.assertEqual({r['id'] for r in rows},{r['id'] for r in sources})
        self.assertEqual(len(rows),1000)
        old=[json.loads(x) for x in (ROOT/'training/datasets/external_corpus_v1.jsonl').read_bytes().splitlines()]
        old_prompts={' '.join(curator.prompt(r).split()).casefold() for r in old}
        self.assertFalse(any(' '.join(curator.prompt(r).split()).casefold() in old_prompts for r in rows))
        for row in rows:
            if row['task_type'] in {'code_generation','self_development'}:
                run=row['metadata']['execution_validation']['result']
                self.assertEqual(run['exit_code'],0)
                self.assertTrue(run['all_assertions_executed'])
                self.assertGreaterEqual(run['executed_assertions'],3)

if __name__=='__main__':
    unittest.main()
