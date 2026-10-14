from pathlib import Path
import json
import shutil
import subprocess

import pytest

NODE = shutil.which("node")
ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not NODE, reason="Node required for frontend interaction")


@pytest.mark.parametrize("choice", ["apply", "later"])
def test_notice_choice_uses_exact_target_and_preserves_off_preference(choice):
    script = r"""
const fs = require('fs'), vm = require('vm');
const assert = require('assert');
const ids = ['toggle-auto-updates','update-last-checked','update-current-version','settings-toggle',
             'update-notice','update-notice-text','update-notice-apply','update-notice-later'];
const elements = Object.fromEntries(ids.map(id => [id, {
  hidden: true, listeners: {}, classList: {toggle(){}, contains(){return false;}},
  setAttribute(){}, addEventListener(name, fn){this.listeners[name] = fn;}
}]));
const calls = [];
let status = {ok: true, enabled: false, noticeAvailable: true, targetSha: 'b'.repeat(40),
              applyRequested: false, currentVersion: 'aaaaaaa', lastChecked: null};
const handlers = {};
const window = {addEventListener(name, fn){handlers[name] = fn;}, pywebview: {api: {
  async get_update_settings(){return status;},
  async request_update(sha){calls.push(['apply', sha]); status = {...status, applyRequested: true}; return {ok:true};},
  async dismiss_update(sha){calls.push(['later', sha]); status = {...status, noticeAvailable: false}; return {ok:true};},
  async check_for_updates(){throw Error('OFF chat must not own the periodic fetch');}
}}};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
  window, document: {getElementById(id){return elements[id];}}, console, Date,
  setInterval(){}, setTimeout(fn){fn();}
});
(async () => {
  handlers.pywebviewready();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(elements['update-notice'].hidden, false);
  elements['update-notice-'+process.argv[2]].listeners.click();
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(calls, [[process.argv[2], 'b'.repeat(40)]]);
  assert.equal(status.enabled, false);
  if(process.argv[2] === 'apply') {
    assert.equal(elements['update-notice-apply'].disabled, true);
    assert.match(elements['update-notice-text'].textContent, /Save any draft and close chat/);
  } else { assert.equal(elements['update-notice'].hidden, true); }
  console.log(JSON.stringify({choice:process.argv[2], verified:true}));
})().catch(error => {console.error(error); process.exitCode=1;});
"""
    result = subprocess.run([NODE, "-e", script, str(ROOT / "localpilot/webview/settings-updates.js"), choice],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["verified"] is True
