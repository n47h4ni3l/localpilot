#!/usr/bin/env python3
"""Curate a pinned fresh-source P1.2 candidate; never train or approve it.

Requires PyArrow and the pinned tokenizer only at curation time. The committed
source-record bundle and manifests support stdlib-only materialization in CI.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import warnings

warnings.filterwarnings('ignore', category=SyntaxWarning)

SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT))
from train_adapter import expand_training_examples, _tokenized_example
from training_common import find_manifest_leaks, load_eval_manifest

VERSION = 'p12-fresh-v2'
QUOTAS = {'code_generation': 200, 'self_development': 50, 'multi_tool_use': 150,
          'everyday_conversation': 100, 'instruction_following_chat': 100,
          'general_conversation': 100, 'text_rewriting': 125,
          'source_summarization': 125, 'constrained_completion': 50}
VALIDATION = {'code_generation': 20, 'self_development': 5, 'multi_tool_use': 15,
              'everyday_conversation': 10, 'instruction_following_chat': 10,
              'general_conversation': 10, 'text_rewriting': 12,
              'source_summarization': 13, 'constrained_completion': 5}

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def digest(value):
    return hashlib.sha256((value if isinstance(value, bytes) else canonical(value).encode())).hexdigest()

def prompt(row):
    return next(m['content'] for m in row['messages'] if m['role'] == 'user')

def shingle(text):
    # Remove quoted identifiers and numeric variants, retaining task words.
    words = re.findall(r'[a-z]+', re.sub(r'`[^`]*`', ' identifier ', text.casefold()))
    return set(zip(words, words[1:], words[2:]))

class Diversity:
    def __init__(self):
        self.groups = []
        self.index = defaultdict(set)
        self.prompts = set()
        self.code = set()

    def add(self, text, code_key=None, *, force=False):
        normalized = ' '.join(text.split()).casefold()
        grams = shingle(text)
        candidates = set()
        # Rare shingles avoid comparing every item against generic boilerplate.
        for gram in sorted(grams, key=lambda g: len(self.index[g]))[:20]:
            candidates.update(self.index[gram])
        if not force:
            if normalized in self.prompts or (code_key and code_key in self.code):
                return False
            for i in candidates:
                prior = self.groups[i]
                if len(grams & prior) / max(1, len(grams | prior)) >= .55:
                    return False
        index = len(self.groups)
        self.groups.append(grams)
        for gram in grams:
            self.index[gram].add(index)
        self.prompts.add(normalized)
        if code_key:
            self.code.add(code_key)
        return True

class NormalizeCode(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.name = 'function'
        node.body = [x for x in node.body if not (isinstance(x, ast.Expr) and isinstance(x.value, ast.Constant) and isinstance(x.value.value, str))]
        return self.generic_visit(node)
    visit_AsyncFunctionDef = visit_FunctionDef
    def visit_Name(self, node):
        if node.id not in {'True','False','None'}:
            node.id = 'identifier'
        return node
    def visit_arg(self, node):
        node.arg = 'argument'
        node.annotation = None
        return node

def code_identity(text):
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S)
    if not blocks:
        return None
    try:
        tree = ast.parse('\n'.join(blocks))
        tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        if not tree.body:
            return None
        return digest(ast.dump(NormalizeCode().visit(tree), include_attributes=False))
    except SyntaxError:
        return None

BAD_CONTENT = re.compile(r'<think>|</think>|\[Reasoning\]|as an (?:ai|language model)|I (?:cannot|can.t) (?:assist|help|provide)|API[_ -]?key|authentication cookie|password\s*[:=]', re.I)
PLACEHOLDERS = re.compile(r'\bNAME_\d+\b|\[(?:insert|your name|company name|placeholder)|<placeholder>', re.I)
UNSUPPORTED_ACTION = re.compile(r'I (?:can|will|have) (?:send|remind|monitor|book|schedule|track|alert|set up)|I.ve (?:sent|booked|scheduled)|I (?:love|enjoy|remember) (?:my|when I)', re.I)
IDENTITY_ROLEPLAY = re.compile(r'you are (?:a person|a \d+|an? (?:expert|professional) (?:doctor|lawyer))|pretend (?:you|to be)|role.?play|who are you|what is your name', re.I)

def clean_messages(messages, *, greeting=False):
    result = []
    for m in messages:
        if set(m) - {'role','content'} or m['role'] not in {'system','user','assistant'} or not isinstance(m['content'],str):
            raise ValueError('unsupported_message')
        if m['role'] == 'system' and not m['content'].strip():
            continue
        result.append(dict(m))
    if greeting and len(result) > 4 and result[0]['role'] == 'user' and re.fullmatch(r'(?:hi|hello|hey|hi there)[!. ]*', result[0]['content'], re.I):
        result = result[2:]
    if not result or result[-1]['role'] != 'assistant' or any(not m['content'].strip() for m in result):
        raise ValueError('blank_or_unfinished')
    text = '\n'.join(m['content'] for m in result)
    if BAD_CONTENT.search(text) or PLACEHOLDERS.search(text):
        raise ValueError('reasoning_identity_refusal_or_credentials')
    if UNSUPPORTED_ACTION.search('\n'.join(m['content'] for m in result if m['role']=='assistant')):
        raise ValueError('unsupported_action_claim')
    if IDENTITY_ROLEPLAY.search(text):
        raise ValueError('identity_roleplay')
    if len(text.encode()) > 15000:
        raise ValueError('oversize_text')
    return result

def native_toolace(raw):
    system = raw['system']
    marker = re.search(r'\[\s*\{', system)
    if not marker:
        raise ValueError('missing_tool_definitions')
    definitions, _ = json.JSONDecoder().raw_decode(system[marker.start():])
    if not 1 <= len(definitions) <= 3:
        raise ValueError('tool_schema_context')
    names = {t['name']: re.sub(r'[^A-Za-z0-9_]', '_',t['name']) for t in definitions}
    if len(set(names.values())) != len(names):
        raise ValueError('tool_name_collision')
    tools = []
    def fix_types(value):
        if isinstance(value,dict):
            return {k: ('object' if k=='type' and v=='dict' else 'array' if k=='type' and v=='list' else fix_types(v)) for k,v in value.items()}
        if isinstance(value,list):
            return [fix_types(v) for v in value]
        return value
    for item in definitions:
        tools.append({'type':'function','function':{'name':names[item['name']], 'description':item['description'], 'parameters':fix_types(item['parameters'])}})
    messages = [{'role':'system','content':'Use the available tools when they support the request. Ask for missing required information. Return tool-supported results after the call completes.'}]
    pending = None
    call_number = 0
    for m in raw['conversations']:
        role,value=m['from'],m['value']
        if role=='assistant' and value.strip().startswith('['):
            expression=value.strip()
            for name in sorted(names,key=len,reverse=True):
                expression=expression.replace(name+'(',names[name]+'(')
            try:
                tree=ast.parse(expression,mode='eval').body
                if not isinstance(tree,ast.List) or len(tree.elts)!=1:
                    raise ValueError('parallel_calls')
                call=tree.elts[0]
                if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.args or call.func.id not in names.values():
                    raise ValueError('invalid_call')
                arguments={x.arg:ast.literal_eval(x.value) for x in call.keywords}
            except (SyntaxError,ValueError,TypeError) as e:
                raise ValueError('unsupported_call_syntax') from e
            if pending:
                raise ValueError('unresolved_call')
            call_number += 1
            pending=(f'call_{call_number}',call.func.id)
            messages.append({'role':'assistant','content':'','tool_calls':[{'type':'function','id':pending[0],'function':{'name':pending[1],'arguments':canonical(arguments)}}]})
        elif role=='tool':
            if not pending:
                raise ValueError('orphan_tool')
            messages.append({'role':'tool','content':value,'name':pending[1],'tool_call_id':pending[0]})
            pending=None
        elif role in {'user','assistant'} and value.strip():
            if pending:
                raise ValueError('unresolved_call')
            messages.append({'role':role,'content':value})
        else:
            raise ValueError('blank_or_unknown_tool_message')
    if pending or not call_number or messages[-1]['role']!='assistant' or messages[-1].get('tool_calls') or not messages[-1]['content'].strip():
        raise ValueError('unfinished_tool_trajectory')
    text='\n'.join(m['content'] for m in messages)
    if BAD_CONTENT.search(text) or PLACEHOLDERS.search(text) or UNSUPPORTED_ACTION.search(messages[-1]['content']):
        raise ValueError('tool_content_filter')
    if re.search(r'cataract|diagnos|medical|patient|clinical|cancer|prescri|investment|stock price|crypto|credit score',text,re.I):
        raise ValueError('specialist_tool_task_outside_curriculum')
    return messages,tools

def constraint_check(messages):
    question=next(m['content'] for m in messages if m['role']=='user')
    answer=messages[-1]['content']
    checked=[]
    for pattern, fn, label in [
        (r'all lowercase',lambda:answer==answer.lower(),'lowercase'),
        (r'all uppercase',lambda:answer==answer.upper(),'uppercase'),
        (r'exactly (\d+) bullet points',lambda:len(re.findall(r'^\s*[*+-] ',answer,re.M))==int(re.search(r'exactly (\d+) bullet points',question,re.I).group(1)),'bullet_count'),
        (r'less than (\d+) words',lambda:len(answer.split())<int(re.search(r'less than (\d+) words',question,re.I).group(1)),'word_limit'),
    ]:
        if re.search(pattern,question,re.I):
            if not fn():
                raise ValueError('constraint_failed_'+label)
            checked.append(label)
    if not checked:
        raise ValueError('no_supported_constraint')
    # Only accept the simple templates whose remaining clauses are checkable.
    if re.search(r'paragraph|sentence|forbidden|ender|keyword|title|appear|letter frequency|language other',question,re.I):
        raise ValueError('unsupported_additional_constraint')
    return checked

SAFE_IMPORTS={'math','collections','itertools','functools','typing','string','statistics','re','bisect','heapq','operator'}
FORBIDDEN_NAMES={'eval','exec','compile','open','input','globals','locals','getattr','setattr','delattr','vars','__import__','breakpoint','exit','quit','help'}
def executable_code(response):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',response,re.S)
    if not blocks:
        raise ValueError('no_python')
    code='\n'.join(blocks)
    tree=ast.parse(code)
    asserts=[n for n in ast.walk(tree) if isinstance(n,ast.Assert)]
    if len(asserts)<3 or not any(isinstance(n,ast.FunctionDef) for n in tree.body):
        raise ValueError('insufficient_executable_tests')
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
            if any(m.split('.')[0] not in SAFE_IMPORTS for m in modules):
                raise ValueError('unconfined_import')
        if isinstance(node,ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith('__')):
            raise ValueError('unconfined_name')
        if isinstance(node,ast.Attribute) and node.attr.startswith('__'):
            raise ValueError('unconfined_attribute')
        if isinstance(node,(ast.ClassDef,ast.AsyncFunctionDef,ast.With)):
            raise ValueError('unsupported_execution_shape')
    return tree,asserts

RUNNER = '''import resource,sys,traceback
resource.setrlimit(resource.RLIMIT_CPU,(1,1))
resource.setrlimit(resource.RLIMIT_AS,(384*1024*1024,384*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
resource.setrlimit(resource.RLIMIT_NOFILE,(32,32))
namespace={'_p12_assertions_executed':[]}
try:
 exec(compile(sys.stdin.read(),'<p12-confined-test>','exec'),namespace)
except BaseException as error:
 print(type(error).__name__+': '+str(error))
 raise SystemExit(1)
finally:
 print('P12_ASSERTIONS:'+','.join(str(x) for x in sorted(set(namespace['_p12_assertions_executed']))))
'''
def run_code(tree):
    instrumented=copy.deepcopy(tree)
    checks={id(n):i for i,n in enumerate(n for n in ast.walk(instrumented) if isinstance(n,ast.Assert))}
    class Visits(ast.NodeTransformer):
        def visit_Assert(self,node):
            record=ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='_p12_assertions_executed',ctx=ast.Load()),attr='append',ctx=ast.Load()),args=[ast.Constant(value=checks[id(node)])],keywords=[]))
            return [record,node]
    instrumented=Visits().visit(instrumented)
    called={n.value.func.id for n in instrumented.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)}
    for node in list(instrumented.body):
        if isinstance(node,ast.FunctionDef) and node.name.startswith('test_') and not node.args.args and not node.args.kwonlyargs and node.name not in called:
            instrumented.body.append(ast.Expr(value=ast.Call(func=ast.Name(id=node.name,ctx=ast.Load()),args=[],keywords=[])))
    ast.fix_missing_locations(instrumented)
    try:
        result=subprocess.run([sys.executable,'-I','-c',RUNNER],input=ast.unparse(instrumented),text=True,capture_output=True,timeout=3,env={'PATH':os.defpath,'PYTHONDONTWRITEBYTECODE':'1'})
        visits=re.findall(r'^P12_ASSERTIONS:([0-9,]*)$',result.stdout,re.M)
        executed={int(x) for x in visits[-1].split(',') if x} if visits else set()
        exit_code=result.returncode
        if not exit_code and executed!=set(range(len(checks))):
            exit_code=2
        stdout=re.sub(r'^P12_ASSERTIONS:[0-9,]*\n?', '',result.stdout,flags=re.M)
        return {'exit_code':exit_code,'stdout':stdout[-1000:],'stderr':result.stderr[-1000:],'executed_assertions':len(executed),'expected_assertions':len(checks),'all_assertions_executed':executed==set(range(len(checks)))}
    except subprocess.TimeoutExpired:
        return {'exit_code':124,'stdout':'','stderr':'timeout'}

MUTATIONS={ast.Lt:ast.LtE,ast.LtE:ast.Lt,ast.Gt:ast.GtE,ast.GtE:ast.Gt,ast.Eq:ast.NotEq,ast.NotEq:ast.Eq,ast.Add:ast.Sub,ast.Sub:ast.Add,ast.Mult:ast.Add,ast.And:ast.Or,ast.Or:ast.And}
def make_self_improvement(raw, tree, asserts, passed):
    test_names={n.name for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')}
    def is_test(node):
        return isinstance(node,ast.Assert) or isinstance(node,ast.FunctionDef) and node.name in test_names or not isinstance(node,ast.FunctionDef) and any(isinstance(n,ast.Assert) for n in ast.walk(node)) or isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id in test_names
    clean=copy.deepcopy(tree)
    clean.body=[n for n in clean.body if not is_test(n)]
    test_tree=copy.deepcopy(tree)
    test_tree.body=[n for n in test_tree.body if is_test(n)]
    if not test_tree.body:
        raise ValueError('no_separable_regression_checks')
    forbidden={id(n) for assertion in ast.walk(clean) if isinstance(assertion,ast.Assert) for n in ast.walk(assertion)}
    implementation_nodes={id(n) for definition in clean.body if isinstance(definition,ast.FunctionDef) for n in ast.walk(definition)}-forbidden
    parents=list(ast.walk(clean))
    for index,parent in enumerate(parents):
        if id(parent) not in implementation_nodes:
            continue
        if isinstance(parent,(ast.BinOp,ast.AugAssign,ast.BoolOp)):
            choices=[('op',None,parent.op)]
        elif isinstance(parent,ast.Compare):
            choices=[('ops',i,operator) for i,operator in enumerate(parent.ops)]
        else:
            continue
        for field,position,operator in choices:
            if type(operator) not in MUTATIONS:
                continue
            mutated=copy.deepcopy(clean)
            changed_parent=list(ast.walk(mutated))[index]
            replacement=MUTATIONS[type(operator)]()
            if position is None:
                setattr(changed_parent,field,replacement)
            else:
                changed_parent.ops[position]=replacement
            combined=copy.deepcopy(mutated)
            combined.body.extend(copy.deepcopy(test_tree.body))
            original_checks=ast.dump(test_tree,include_attributes=False)
            assert ast.dump(ast.Module(body=combined.body[-len(test_tree.body):],type_ignores=[]),include_attributes=False)==original_checks
            observed=run_code(combined)
            if observed['exit_code'] != 1 or not observed['stdout'].startswith('AssertionError'):
                continue
            restored=copy.deepcopy(clean)
            restored.body.extend(copy.deepcopy(test_tree.body))
            verified=run_code(restored)
            if verified['exit_code'] != 0:
                continue
            candidate=ast.unparse(mutated)
            tests=ast.unparse(test_tree)
            question=("Improve your own helper using the observed failure. Keep the change scoped and preserve the stated behavior.\n\n"
                      f"Objective: {raw['instruction']}\n\nCurrent candidate:\n```python\n{candidate}\n```\n\n"
                      f"Regression checks (unchanged):\n```python\n{tests}\n```\n\nObserved test result: {observed['stdout'].strip()} (exit 1).\n"
                      "Return the corrected implementation and a brief explanation of the repair.")
            answer=f"```python\n{ast.unparse(clean)}\n```\n\nThe operator `{type(replacement).__name__}` was corrected to `{type(operator).__name__}` in one implementation expression to restore the specified behavior. The tests were preserved. All {len(asserts)} distinct assertion statements passed during execution."
            return [{'role':'user','content':question},{'role':'assistant','content':answer}], {'mutation':type(operator).__name__+'->'+type(replacement).__name__,'changed_expressions':1,'tests_preserved':True,'failed_execution':observed,'passed_execution':verified,'assertions':len(asserts),'candidate_sha256':digest(candidate),'tests_sha256':digest(tests),'original_test_ast_sha256':digest(original_checks)}
    raise ValueError('no_discriminating_mutation')

def main():
    os.environ['USE_TORCH'] = '0'
    os.environ['USE_TF'] = '0'
    os.environ['USE_FLAX'] = '0'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    args=parser.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    root=SCRIPT.parents[1]
    rejection_manifest=json.loads((root/'training/manifests/p12_fresh_content_rejections.json').read_text())
    content_rejections={decision['id']:decision['reason'] for decision in rejection_manifest['decisions']}
    acquisition=json.loads((args.raw_root/'acquisition.json').read_text())
    files=acquisition['files']
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    pools=defaultdict(list)
    for asset in files:
        relative=asset['path']
        if relative.endswith('README.md'):
            continue
        path=args.raw_root/'raw'/asset['repo'].replace('/','--')/relative
        if digest(path.read_bytes())!=asset['sha256']:
            raise RuntimeError('Changed acquisition asset: '+relative)
        if relative.endswith('.parquet'):
            raw_rows=pq.read_table(path).to_pylist()
        elif relative.endswith('.jsonl'):
            raw_rows=[json.loads(line) for line in path.read_bytes().splitlines()]
        else:
            raw_rows=json.loads(path.read_text())
        if asset['repo'].startswith('bigcode/'):
            category='code_generation'
        elif asset['repo'].startswith('Team-ACE/'):
            category='multi_tool_use'
        elif 'everyday-conversations' in asset['repo']:
            category='everyday_conversation'
        elif asset['repo'].startswith('nvidia/'):
            category='instruction_following_chat'
        else:
            category={'smol-magpie-ultra':'general_conversation','smol-rewrite':'text_rewriting','smol-summarize':'source_summarization','smol-constraints':'constrained_completion'}[relative.split('/')[1]]
        for index,raw in enumerate(raw_rows):
            pools[category].append((digest({'seed':3407,'repo':asset['repo'],'path':relative,'index':index}),asset,index,raw))
        print(f'Loaded {category}: {len(raw_rows)} source rows',flush=True)
        del raw_rows

    diversity=Diversity()
    original=[json.loads(x) for x in (root/'training/datasets/external_corpus_v1.jsonl').read_bytes().splitlines()]
    original.extend(json.loads(x) for x in (root/'training/datasets/corpus_v1_seed.jsonl').read_bytes().splitlines())
    for row in original:
        diversity.add(prompt(row),code_identity(row['messages'][-1]['content']),force=True)
    eval_manifest=load_eval_manifest(root/'training/manifests/eval_v1_manifest.json')
    selected=[]
    envelopes=[]
    rejected=[]
    stats=defaultdict(Counter)
    topics=Counter()
    code_passes=0
    for category in QUOTAS:
        if category=='self_development':
            continue
        pool=sorted(pools[category],key=lambda x:x[0])
        count=0
        self_count=0
        for _,asset,index,raw in pool:
            if count>=QUOTAS[category] and (category!='code_generation' or self_count>=QUOTAS['self_development']):
                break
            stats[category]['considered']+=1
            record_id='p12-fresh-'+digest({'repo':asset['repo'],'path':asset['path'],'row':index})[:20]
            try:
                if record_id in content_rejections:
                    raise ValueError('explicit_content_rejection: '+content_rejections[record_id])
                metadata={'semantic_review':'pending_independent_review','upstream_row':index,'family_id':digest(raw.get('sha1') or raw.get('full_topic') or raw.get('uuid') or raw.get('id') or raw),'fresh_source':True}
                task_type=category
                transformations=[]
                key=None
                if category=='code_generation':
                    messages=clean_messages([{'role':'user','content':raw['instruction']},{'role':'assistant','content':raw['response']}])
                    tree,asserts=executable_code(raw['response'])
                    key=code_identity(raw['response'])
                    passed=run_code(tree)
                    if passed['exit_code']!=0:
                        raise ValueError('fresh_code_tests_failed')
                    code_passes+=1
                    metadata['execution_validation']={'assertions':len(asserts),'result':passed,'code_sha256':digest(ast.unparse(tree))}
                    transformations.append('instruction_field_without_generation_fewshots')
                    if self_count<QUOTAS['self_development']:
                        messages,evidence=make_self_improvement(raw,tree,asserts,passed)
                        metadata['self_improvement_evidence']=evidence
                        task_type='self_development'
                        transformations.append('controlled_operator_mutation_and_executed_fail_repair')
                elif category=='multi_tool_use':
                    native,tools=native_toolace(raw)
                    metadata['native_messages']=native
                    metadata['native_tools']=tools
                    messages=[{'role':'user','content':next(m['content'] for m in native if m['role']=='user')},{'role':'assistant','content':native[-1]['content']}]
                    transformations.extend(['native_single_call_syntax','schema_dict_to_object_list_to_array','tool_name_identifier_normalization','remove_incompatible_text_transport_instructions'])
                else:
                    messages=clean_messages(raw['messages'],greeting=category=='everyday_conversation')
                    if category in {'text_rewriting','source_summarization'}:
                        source_text='\n'.join(m['content'] for m in messages if m['role']=='user')
                        answer_text=messages[-1]['content']
                        numbers=lambda text:set(re.findall(r'(?<!\w)\d+(?:[.,]\d+)*(?!\w)',text))
                        if not numbers(answer_text).issubset(numbers(source_text)):
                            raise ValueError('unsupported_numeric_addition')
                        if len(answer_text)>=len(source_text):
                            raise ValueError('not_a_concise_transformation')
                    if category=='everyday_conversation':
                        transformations.append('remove_leading_stock_greeting_pair')
                        topic=raw['full_topic']
                        if topics[topic]>=1:
                            raise ValueError('topic_already_represented')
                        metadata['topic']=topic
                    elif category=='constrained_completion':
                        metadata['checked_constraints']=constraint_check(messages)
                    elif category=='general_conversation':
                        if raw.get('quality')!='good' or raw.get('category') not in {'information-seeking','problem-solving','creative-writing','editing','advice-seeking','planning'}:
                            raise ValueError('general_quality_category')
                        metadata['upstream_quality']=raw['quality']
                        metadata['upstream_category']=raw['category']
                row={'id':record_id,'messages':messages,'task_type':task_type,'source':asset['repo'],'license':'ODC-BY-1.0' if asset['repo'].startswith(('bigcode/','nvidia/')) else 'Apache-2.0','quality_tier':'C' if task_type=='self_development' else 'B','split':'train','verification_status':'synthetic_validated' if task_type=='self_development' else 'source_verified','provenance':{'repository':asset['repo'],'revision':asset['revision'],'reference':f"hf:{asset['repo']}@{asset['revision']}:{asset['path']}#row={index}",'source_path':asset['path'],'source_file_sha256':asset['sha256'],'source_record_sha256':digest(raw),'acquisition_date':'2026-10-12','transformation_version':VERSION,'transformations':transformations,'license_reference':f"https://huggingface.co/datasets/{asset['repo']}/blob/{asset['revision']}/README.md"},'tags':['p12_fresh',task_type],'metadata':metadata}
                # Tokenize the exact dictionary ordering that materialized JSONL
                # will restore, including nested native tool-schema properties.
                row=json.loads(canonical(row))
                metadata=row['metadata']
                examples=expand_training_examples([row])
                lengths=[len(_tokenized_example(tokenizer,e,1040)[0]) for e in examples]
                metadata['training_example_count']=len(examples)
                metadata['native_max_tokens']=max(lengths)
                if find_manifest_leaks([row],eval_manifest):
                    raise ValueError('eval_manifest_overlap')
                # Self-improvement prompt includes its fresh source instruction.
                diversity_text=raw['instruction'] if category=='code_generation' else prompt(row)
                if not diversity.add(diversity_text,key):
                    raise ValueError('prior_or_selected_near_duplicate')
                selected.append(row)
                envelopes.append({'id':record_id,'source_record_sha256':digest(raw),'raw':raw,'transformations':transformations})
                if task_type=='self_development':
                    self_count+=1
                else:
                    count+=1
                if category=='everyday_conversation':
                    topics[raw['full_topic']]+=1
                stats[category]['accepted_'+task_type]+=1
            except (ValueError,KeyError,IndexError,TypeError,SyntaxError,RuntimeError) as error:
                reason=str(error).split(':')[0]
                stats[category]['rejected_'+reason]+=1
                rejected.append({'id':record_id,'repo':asset['repo'],'path':asset['path'],'row':index,'reason':str(error),'source_record_sha256':digest(raw)})
        print(f'Selected {category}: {count}; self-improvement {self_count}',flush=True)
        # Retain every completed curation phase, including candidates and rejected
        # source IDs, even if a later category is insufficient or malformed.
        phase_dir=args.raw_root/'phases'
        phase_dir.mkdir(exist_ok=True)
        phase={'category':category,'selected':selected,'source_envelopes':envelopes,
               'rejections':rejected,'stats':{k:dict(v) for k,v in stats.items()}}
        phase_bytes=(canonical(phase)+'\n').encode()
        phase_path=phase_dir/(category+'-'+digest(phase_bytes)[:16]+'.json')
        if not phase_path.exists():
            phase_path.write_bytes(phase_bytes)
        if count!=QUOTAS[category] or category=='code_generation' and self_count!=QUOTAS['self_development']:
            raise RuntimeError(f'Insufficient suitable {category} examples: {count}, self={self_count}; evidence retained separately by upstream snapshots')

    for task_type,n in VALIDATION.items():
        group=sorted((r for r in selected if r['task_type']==task_type),key=lambda r:digest({'split_seed':3407,'family':r['metadata']['family_id']}))
        for row in group[:n]:
            row['split']='validation'
    assert len(selected)==1000 and Counter(r['split'] for r in selected)=={'train':900,'validation':100}
    selected.sort(key=lambda r:digest({'shuffle_seed':3407,'id':r['id']}))
    def save_jsonl(path,rows):
        path.parent.mkdir(parents=True,exist_ok=True)
        content=(''.join(canonical(r)+'\n' for r in rows)).encode()
        path.write_bytes(content)
        return content
    data=save_jsonl(root/'training/datasets/p12_fresh_candidates.jsonl',selected)
    bundle=save_jsonl(root/'training/sources/p12_fresh_source_records.jsonl',sorted(envelopes,key=lambda r:r['id']))
    rejects=save_jsonl(args.raw_root/'rejections.jsonl',rejected)
    report={'status':'candidate_not_independently_approved','transformation_version':VERSION,'quotas':QUOTAS,'validation_quotas':VALIDATION,'selection_stats':{k:dict(v) for k,v in stats.items()},'selected_records':1000,'source_pool_counts':{k:len(v) for k,v in pools.items()},'executed_passing_code_candidates':code_passes,'rejections_sha256':digest(rejects),'rejection_count':len(rejected),'candidate_sha256':digest(data),'source_bundle_sha256':digest(bundle),'all_selected_native_tokens_within':1040,'private_holdout_read':False,'model_weights_loaded':False,'training_started':False}
    report_path=root/'training/reports/p12_fresh_curation.json'
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    lock={'artifact_type':'p12_fresh_acquisition_lock','date':'2026-10-12','files':[{k:v for k,v in a.items() if k!='local_path'} for a in files],'source_bundle_sha256':digest(bundle)}
    (root/'training/manifests/p12_fresh_acquisition_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='selection_stats'},indent=2),flush=True)

if __name__=='__main__':
    main()
