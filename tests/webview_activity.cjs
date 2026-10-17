const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// A small DOM fixture executes the shipped rendering and event handlers. Tool
// output is intentionally hostile text, never parsed as HTML by this fixture.
class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {};
    this.attributes = {}; this.className = ''; this.listeners = {}; this.open = false;
    this.ownText = ''; this.ownerDocument = document; this.parentNode = null;
    this.classList = {
      add: name => { this.className = [...new Set([...this.className.split(' '), name])].join(' ').trim(); },
      remove: name => { this.className = this.className.split(' ').filter(n => n !== name).join(' '); },
      contains: name => this.className.split(' ').includes(name),
    };
  }
  set textContent(value) { this.ownText = String(value); this.children = []; }
  get textContent() { return this.ownText + this.children.map(c => c.textContent).join(''); }
  get firstElementChild() { return this.children[0]; }
  appendChild(child) { this.children.push(child); child.parentNode = this; return child; }
  replaceChildren() { this.children.forEach(c => {c.parentNode = null;}); this.children = []; this.ownText = ''; }
  remove() { this.parentNode.children = this.parentNode.children.filter(c => c !== this); this.parentNode = null; }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(type, fn) { this.listeners[type] = fn; }
  focus() { document.activeElement = this; }
  contains(child) { return child === this || this.children.some(c => c.contains(child)); }
  querySelectorAll(selector) {
    return this.children.flatMap(c => [
      ...(selector[0] === '.' ? (c.className.split(' ').includes(selector.slice(1)) ? [c] : []) :
        (c.tagName.toLowerCase() === selector ? [c] : [])), ...c.querySelectorAll(selector),
    ]);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  set innerHTML(_) { throw new Error('Tool data must never become HTML'); }
}
const document = {createElement: tag => new Element(tag), activeElement: null};
const window = {document};
const timers = new Map();
let timerId = 0;
const context = vm.createContext({
  window, document, console, Set, Map, Date, Number, Object, Array,
  messageElementsById: new Map(), knownMessageIds: new Set(),
  messageStream: new Element(), activityStrip: new Element(), previewLine: new Element(),
  activityText: new Element(), activitySteps: new Element(),
  composerInput: new Element(), sendBtn: new Element(), historySheet: new Element(),
  activeSessionId: 'a', currentState: 'idle', lastUserContent: 'Question',
  chatFiles: null, sendDrafts: new Map(),
  followMessages: true, lastEventId: 0, avatars: [],
  scrollToBottom() {}, renderSafeMarkdown: (element, content) => {element.textContent = content;},
  renderUserTurn() {}, loadSessionList: async () => [], handleSessionDeleted() {},
  setGlobalState: state => {context.currentState = state;},
  setTimeout: fn => {timers.set(++timerId, fn); return timerId;},
  clearTimeout: id => timers.delete(id),
});
const root = path.join(__dirname, '../localpilot/webview');
vm.runInContext(fs.readFileSync(path.join(root, 'chat-activity.js'), 'utf8'), context);
const source = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
const renderStart = source.indexOf('  function renderAssistantTurn(');
const sessionStart = source.indexOf('  async function loadMessages(');
vm.runInContext(source.slice(renderStart, sessionStart), context);
const sessionEnd = source.indexOf('  async function bootstrapSession(', sessionStart);
vm.runInContext(source.slice(sessionStart, sessionEnd), context);
const pollStart = source.indexOf('  async function pollEvents(');
const pollEnd = source.indexOf('  let lastHealthState = null;', pollStart);
vm.runInContext(source.slice(pollStart, pollEnd), context);

const activity = window.LocalPilotActivity;
const step = {id:'call-1', tool:'get_library_summary', status:'running', arguments_preview:'<img src=x onerror=bad()>'};
let steps = activity.upsert([], step);
steps = activity.upsert(steps, step);
steps = activity.upsert(steps, {...step, status:'succeeded', duration_ms:1432, result_preview:'60 documents'});
steps = activity.upsert(steps, step);
assert.equal(steps.length, 1);
assert.equal(steps[0].status, 'succeeded', 'Replayed start cannot downgrade a completed call');
steps = activity.upsert(steps, {...step, id:'call-2', status:'failed', error:'<script>not executable</script>'});
assert.equal(steps.length, 2, 'Two calls to one tool stay distinct');

const container = new Element();
activity.render(container, steps);
let details = container.firstElementChild;
assert.equal(details.tagName, 'DETAILS');
assert.equal(details.open, false);
assert.match(details.firstElementChild.textContent, /2 steps.*1 unsuccessful/);
const rows = details.querySelectorAll('.chat-activity__step');
assert.match(rows[0].firstElementChild.textContent, /Checked the library.*Succeeded.*1.4 s/);
assert.match(rows[1].textContent, /Failed.*<script>not executable<\/script>/);
assert.equal(rows[0].querySelector('pre').textContent, '<img src=x onerror=bad()>');
details.open = true; rows[1].open = true; rows[1].firstElementChild.focus();
activity.render(container, [...steps, {...step, id:'call-3', tool:'new_future_tool', status:'succeeded', result_preview:'a'.repeat(5000)}]);
details = container.firstElementChild;
assert.equal(details.open, true);
assert.equal(details.querySelectorAll('.chat-activity__step')[1].open, true);
assert.equal(document.activeElement, details.querySelectorAll('.chat-activity__step')[1].firstElementChild);
assert.match(details.textContent, /new_future_tool/);
assert.match(details.textContent, /Preview truncated/);

const streaming = {id:1, role:'assistant', session_id:'a', status:'streaming', content:'', steps:[]};
context.renderAssistantTurn(streaming);
const entry = context.messageElementsById.get(1);
const emit = (type, payload, session_id = 'a') => context.handleEvent({type, payload, session_id});
emit('tool.started', {message_id:1, step});
emit('tool.started', {message_id:1, step});
emit('tool.completed', {message_id:1, step:{...step, status:'succeeded', result_preview:'60 documents'}});
assert.equal(entry.steps.length, 1);
assert.match(entry.activityEl.textContent, /60 documents/);

emit('assistant.delta', {message_id:1, content:'Part of a reply'});
assert(entry.revealTimer, 'Fixture has an active reveal timer before the error');
emit('runtime.error', {message_record:{...streaming, status:'error', content:'ConnectionError: Ollama is unavailable', steps:entry.steps}});
assert.equal(entry.revealEl.textContent, 'ConnectionError: Ollama is unavailable');
assert.equal(entry.caretEl.parentNode, null);
assert.equal(entry.revealTimer, null);
assert.equal(context.currentState, 'error');
assert(entry.bubbleEl.classList.contains('bubble-plain--error'));
emit('assistant.delta', {message_id:1, content:'Stale partial content'});
emit('tool.started', {message_id:1, step:{...step, id:'late-start'}});
assert.equal(entry.steps.length, 1);
assert.equal(entry.revealEl.textContent, 'ConnectionError: Ollama is unavailable');

emit('runtime.error', {message_record:{...streaming, id:2, session_id:'b', status:'error', content:'Other session error'}}, 'b');
emit('runtime.state', {state:'working'}, 'b');
assert(!context.messageElementsById.has(2));
assert.equal(context.currentState, 'error');

(async () => {
  context.api = async () => ({messages:[]});
  await context.switchSession('b');
  assert.equal(context.currentState, 'idle', 'New conversation clears an old turn error');
  context.api = async () => ({messages:[{...streaming, id:3, session_id:'a', steps:[step]}]});
  await context.switchSession('a');
  assert.equal(context.currentState, 'working', 'Reopened streaming turn keeps its real tool state');
  assert.equal(context.activityStrip.classList.contains('is-visible'), true);

  context.api = async () => {
    // A live completion can arrive while the history response is in flight.
    context.renderAssistantTurn({...streaming, id:5, status:'complete', content:'Already finished'});
    return {messages:[{...streaming, id:5}]};
  };
  await context.switchSession('a');
  assert.equal(context.currentState, 'idle', 'An old history response cannot revive a completed turn');

  const saved = {...streaming, id:4, status:'complete', content:'Complete answer', steps};
  context.renderAssistantTurn(saved);
  context.handleEvent({type:'message.created', session_id:'a', payload:{message:saved}});
  emit('tool.started', {message_id:4, step});
  const savedEntry = context.messageElementsById.get(4);
  assert.equal(savedEntry.steps.length, 2, 'Saved activity survives message replay without duplication');
  assert.equal(savedEntry.steps[0].status, 'succeeded');

  const pollOnce = context.pollEvents;
  let nextPolls = 0;
  context.pollEvents = () => {nextPolls += 1;};
  context.api = async () => {
    context.activeSessionId = 'b';
    return {events:[{id:20, type:'runtime.error', session_id:'a', payload:{message_record:saved}}]};
  };
  context.currentState = 'idle';
  await pollOnce();
  assert.equal(context.lastEventId, 0, 'Discarded old-session poll must not skip new-session events');
  assert.equal(context.currentState, 'idle');
  assert.equal(nextPolls, 1);
  console.log('Production tool activity: expandable safe previews, durable replay, error display, session state, and late polls passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
