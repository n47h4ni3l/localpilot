const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Execute the production history handlers. This fixture does not launch a UI.
class Element {
  constructor() {
    this.children = []; this.dataset = {}; this.listeners = {}; this.value = '';
    this.className = ''; this.classes = new Set(); this.hidden = false;
    this.classList = {
      add: value => this.classes.add(value), remove: value => this.classes.delete(value),
      contains: value => this.classes.has(value),
      toggle: (value, enabled) => enabled ? this.classes.add(value) : this.classes.delete(value),
    };
  }
  appendChild(child) { this.children.push(child); }
  replaceChildren() { this.children = []; }
  addEventListener(type, fn) { this.listeners[type] = fn; }
  setAttribute() {}
  focus() { document.activeElement = this; }
  select() { this.selected = true; }
  contains(child) { return child === this || this.children.some(c => c.contains(child)); }
  querySelectorAll(selector) {
    return this.children.flatMap(c => [
      ...(c.className.split(' ').includes(selector.slice(1)) ? [c] : []), ...c.querySelectorAll(selector),
    ]);
  }
}
const document = {createElement: () => new Element(), activeElement: null};
const elements = {};
for (const name of ['historySheet','historyToggle','historyNew','historyClose','historyList','historyEditor',
  'historyEditorHeading','historyEditorDescription','historyTitleLabel','historyTitle','historySave',
  'historyCancel','historyFeedback','composerInput']) elements[name] = new Element();
const sessions = new Map([['a', {id:'a', title:'Alpha', updated_at:'2026-09-01'}],
  ['b', {id:'b', title:'Beta', updated_at:'2026-09-02'}]]);
const calls = [];
let failure = null;
const context = vm.createContext({document, console, ...elements, activeSessionId:'a', closeSettings() {},
  api: async (method, route, body) => {
    calls.push({method, route, body});
    if (failure) throw Object.assign(new Error('Request failed'), {status:failure});
    const id = route.split('/').pop();
    if (method === 'GET') return {sessions:[...sessions.values()]};
    if (method === 'PATCH') { const s = sessions.get(id); s.title = body.title; return {session:s}; }
    if (method === 'DELETE') { sessions.delete(id); return {deleted:id}; }
    if (method === 'POST') { const s = {id:'new',title:'New conversation',updated_at:'2026-09-03'}; sessions.set(s.id,s); return {session:s}; }
    throw Error(method);
  },
  switchSession: async id => { context.activeSessionId = id; },
});
const source = fs.readFileSync(path.join(__dirname, '../localpilot/webview/app.js'), 'utf8');
const start = source.indexOf('  function openHistory()');
const end = source.indexOf('  function openSettings()', start);
assert(start > 0 && end > start);
vm.runInContext(source.slice(start, end), context);
const submit = () => elements.historySave.listeners.click({preventDefault(){}});

(async () => {
  await context.loadSessionList();
  assert.equal(elements.historyList.children.length, 2);
  const entry = elements.historyList.children[0];
  assert.equal(entry.children[0].dataset.sessionId, 'b');
  assert.deepEqual(entry.children[1].children.map(b => b.textContent), ['Rename', 'Delete']);
  context.editHistorySession(sessions.get('b'), 'rename');
  elements.historyTitle.value = '  Renamed beta  ';
  await submit();
  assert.equal(sessions.get('b').title, 'Renamed beta');
  assert.equal(context.activeSessionId, 'a');
  assert.equal(elements.historyEditor.hidden, true);

  context.editHistorySession(sessions.get('b'), 'delete');
  assert.equal(document.activeElement, elements.historyCancel);
  assert.equal(elements.historyTitle.disabled, true);
  const beforeCancel = calls.length;
  await elements.historyCancel.listeners.click();
  assert.equal(calls.length, beforeCancel);
  assert(sessions.has('b'));

  context.editHistorySession(sessions.get('b'), 'delete');
  failure = 409;
  await submit();
  assert(sessions.has('b'));
  assert.equal(elements.historyEditor.hidden, false);
  assert.match(elements.historyFeedback.textContent, /finishes responding/);
  assert.equal(elements.historySave.disabled, false);
  failure = null;
  await submit();
  assert(!sessions.has('b'));
  assert.equal(context.activeSessionId, 'a');

  context.editHistorySession(sessions.get('a'), 'delete');
  await Promise.all([submit(), submit()]);
  assert.equal(calls.filter(c => c.method === 'DELETE' && c.route.endsWith('/a')).length, 1);
  assert.equal(context.activeSessionId, 'new');
  assert.equal(sessions.size, 1);
  assert.equal(calls.filter(c => c.method === 'POST').length, 1);

  sessions.set('other', {id:'other',title:'Other',updated_at:'2026-09-04'});
  sessions.delete('new');
  await context.handleSessionDeleted('new');
  assert.equal(context.activeSessionId, 'other');
  context.editHistorySession(sessions.get('other'), 'rename');
  elements.historyTitle.value = '   ';
  const beforeEmpty = calls.length;
  await submit();
  assert.equal(calls.length, beforeEmpty);
  assert.match(elements.historyFeedback.textContent, /1 and 120/);
  // A successful broker reload must clear a stale Offline label without
  // replacing an active reply's state on subsequent healthy polls.
  context.checkHealth = async () => ({reachable:true, runtime:'running'});
  context.settingsStatusText = new Element();
  context.currentState = 'offline';
  context.setGlobalState = state => { context.currentState = state; };
  context.setTimeout = () => {};
  const healthStart = source.indexOf('  let lastHealthState = null;');
  const healthEnd = source.indexOf('  composerInput.addEventListener(', healthStart);
  vm.runInContext(source.slice(healthStart, healthEnd), context);
  await context.healthLoop();
  assert.equal(context.currentState, 'idle');
  context.currentState = 'working';
  await context.healthLoop();
  assert.equal(context.currentState, 'working');
  console.log('Production history: rename, cancel, active/background/last delete, conflict, and duplicate-submit checks passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
