const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Exercise the actual file controller with a small DOM fixture. No app, model,
// clipboard, native file opener, or screenshot service is launched by this test.
class Element {
  constructor(tag = 'div') {
    this.tagName = tag; this.children = []; this.listeners = {}; this.attributes = {};
    this.className = ''; this.classes = new Set(); this.textContent = '';
    this.hidden = false; this.disabled = false; this.files = []; this.value = '';
    this.classList = {
      add: value => this.classes.add(value), remove: value => this.classes.delete(value),
      toggle: (value, enabled) => enabled ? this.classes.add(value) : this.classes.delete(value),
    };
  }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  setAttribute(key, value) { this.attributes[key] = value; }
  removeAttribute(key) { delete this.attributes[key]; }
  focus() { this.focused = true; }
  click() { this.clicks = (this.clicks || 0) + 1; return this.listeners.click?.({}); }
  contains(target) { return target === this || this.children.some(c => c.contains(target)); }
  showModal() { this.open = true; }
  close() { this.open = false; this.listeners.close?.(); }
}
const source = fs.readFileSync(path.join(__dirname, '../localpilot/webview/chat-files.js'), 'utf8');
function fixture() {
  const ids = Object.fromEntries(['chat-file-input', 'chat-attach-btn', 'chat-screenshot-btn',
    'chat-pending-files', 'chat-file-feedback', 'composer-input', 'panel'].map(id => [id, new Element()]));
  const document = {getElementById: id => ids[id], createElement: tag => new Element(tag), body: new Element('body')};
  const calls = [], bridges = [], reads = [], created = [], revoked = [];
  let nextId = 0, sessionId = 'a', handler = null, bridgeHandler = null;
  const window = {
    document, Blob, TextDecoder,
    atob: value => Buffer.from(value, 'base64').toString('binary'),
    URL: {createObjectURL: () => { const url = 'blob:test-' + created.length; created.push(url); return url; }, revokeObjectURL: url => revoked.push(url)},
    FileReader: class {
      readAsDataURL(file) {
        reads.push(file.name);
        this.result = 'data:' + file.type + ';base64,' + Buffer.from(file.data || '').toString('base64');
        queueMicrotask(() => this.onload());
      }
    },
  };
  const api = async (method, route, body) => {
    calls.push({method, route, body});
    if (handler) return handler(method, route, body);
    return {file: {id: 'file-' + (++nextId), session_id: route.split('/')[3], name: body.name, mime: 'text/plain', size: 5, text_status: 'extracted'}};
  };
  const bridge = async (...args) => { bridges.push(args); return bridgeHandler ? bridgeHandler(...args) : {ok: true}; };
  vm.runInNewContext(source, {window, Uint8Array, Map, Set, console});
  let notifications = 0;
  const controller = window.LocalPilotFiles.init({api, bridge, getSessionId: () => sessionId, onPendingChange: () => { notifications++; }});
  return {
    ids, document, window, controller, calls, bridges, reads, created, revoked,
    handler: fn => { handler = fn; }, bridgeHandler: fn => { bridgeHandler = fn; },
    switchSession: id => { sessionId = id; controller.sessionChanged(id); },
    attach: files => { ids['chat-file-input'].files = files; return ids['chat-file-input'].listeners.change(); },
    notifications: () => notifications,
  };
}
const file = (name = 'notes.txt', size = 5) => ({name, size, type: 'text/plain', data: 'hello'});
const descendants = element => element.children.flatMap(child => [child, ...descendants(child)]);
const findButton = (element, label) => descendants(element).find(child => child.tagName === 'button' && child.textContent === label);
const tick = () => new Promise(resolve => setImmediate(resolve));

(async () => {
  const f = fixture(), c = f.controller;
  assert.equal(f.calls.length, 0, 'initializing must not capture or upload');
  assert.equal(c.hasDraft(), false);
  await f.ids['chat-attach-btn'].click();
  assert.equal(f.ids['chat-file-input'].clicks, 1);
  await f.attach([file()]);
  assert.equal(c.pendingIds().join(','), 'file-1');
  assert.equal(c.hasDraft(), true);
  assert.equal(c.isBusy(), false);
  assert.equal(f.ids['chat-file-input'].value, '');
  assert.deepEqual(JSON.parse(JSON.stringify(f.calls[0])), {method: 'POST', route: '/v1/sessions/a/files', body: {name: 'notes.txt', data_base64: 'aGVsbG8='}});
  const pendingCard = f.ids['chat-pending-files'].children[0];
  assert.match(pendingCard.className, /chat-file-card--pending/);
  assert.match(pendingCard.children[0].children[1].textContent, /Ready/);
  assert.match(pendingCard.children[0].children[1].attributes.title, /Text extracted/);
  assert.match(pendingCard.children[0].children[1].attributes['aria-label'], /Text extracted/);
  assert.equal(findButton(pendingCard, 'Open'), undefined);
  assert.equal(findButton(pendingCard, 'Save as…'), undefined);
  const removePending = findButton(pendingCard, '×');
  assert.match(removePending.className, /chat-file-action--remove/);
  assert.equal(removePending.attributes['aria-label'], 'Remove notes.txt from this message');
  assert.equal(removePending.attributes.title, removePending.attributes['aria-label']);
  assert.equal(f.ids['chat-file-feedback'].hidden, true, 'ready file cards do not need a separate success banner');
  const sent = c.pendingItems();
  c.clearPending();
  assert.equal(c.hasPending(), false);
  c.restorePending(sent); c.restorePending(sent);
  assert.equal(c.pendingIds().length, 1, 'send failure restore cannot duplicate files');
  f.switchSession('b');
  assert.equal(c.hasPending(), false);
  assert.equal(c.hasDraft(), true, 'window close must preserve attachments in other conversations');
  await f.attach([file('other.txt')]);
  assert.equal(c.pendingIds().join(','), 'file-2');
  f.switchSession('a');
  assert.equal(c.pendingIds().join(','), 'file-1');
  c.clearPending('b', ['file-2']);
  assert.equal(c.pendingIds().join(','), 'file-1', 'send completion clears its original session only');
  assert.equal(c.pendingIds('b').length, 0);

  // New uploads retain their original session, even when completion arrives
  // after a conversation change; the send/close guard sees in-flight work.
  c.clearPending();
  let resolveUpload;
  f.handler(() => new Promise(resolve => { resolveUpload = resolve; }));
  const upload = f.attach([file('in-flight.txt')]);
  await tick();
  assert.equal(c.isBusy(), true);
  assert.equal(c.hasDraft(), true);
  assert.equal(f.ids['chat-file-feedback'].hidden, false);
  assert.match(f.ids['chat-file-feedback'].textContent, /Attaching files/);
  c.clearPending();
  assert.equal(c.hasPending(), true, 'cannot clear an in-flight upload');
  f.switchSession('c');
  resolveUpload({file: {id: 'raced', name: 'in-flight.txt', size: 5, mime: 'text/plain', session_id: 'a'}});
  await upload;
  assert.equal(c.hasPending(), false, 'late upload cannot attach to another conversation');
  f.switchSession('a');
  assert.equal(c.pendingIds().join(','), 'raced');

  const compact = fixture();
  compact.handler(async () => ({file: {id: 'image', name: 'screen.png', mime: 'image/png', size: 20, extraction: {status: 'unavailable'}}}));
  await compact.attach([file('screen.png')]);
  let compactMeta = compact.ids['chat-pending-files'].children[0].children[0].children[1];
  assert.match(compactMeta.textContent, /Image attached/);
  assert.match(compactMeta.attributes.title, /visual understanding unavailable/);
  compact.controller.clearPending();
  compact.handler(async () => ({file: {id: 'ocr', name: 'scan.png', mime: 'image/png', size: 20, extraction: {status: 'ocr', truncated: true}}}));
  await compact.attach([file('scan.png')]);
  compactMeta = compact.ids['chat-pending-files'].children[0].children[0].children[1];
  assert.match(compactMeta.textContent, /Ready/);
  assert.match(compactMeta.attributes.title, /OCR.*Text extraction limited/);
  // Removing a final attachment clears a lingering native action success,
  // while transport/extraction errors remain visible for troubleshooting.
  const successMessage = new Element();
  compact.controller.renderMessage(successMessage, {files: [{id: 'saved', name: 'notes.txt'}]});
  await findButton(successMessage, 'Save as…').click();
  assert.equal(compact.ids['chat-file-feedback'].hidden, false);
  await findButton(compact.ids['chat-pending-files'], '×').click();
  assert.equal(compact.controller.hasPending(), false);
  assert.equal(compact.ids['chat-file-feedback'].hidden, true);
  compact.controller.restorePending([{id: 'remaining', name: 'notes.txt', text_status: 'no_text', size: 2}]);
  compactMeta = compact.ids['chat-pending-files'].children[0].children[0].children[1];
  assert.match(compactMeta.textContent, /No readable text/);
  compact.handler(async () => { throw Error('Upload unavailable'); });
  await compact.attach([file('failed.txt')]);
  await findButton(compact.ids['chat-pending-files'], '×').click();
  assert.equal(compact.controller.hasPending(), false);
  assert.equal(compact.ids['chat-file-feedback'].hidden, false, 'removing the last card must retain a real upload error');
  assert.match(compact.ids['chat-file-feedback'].textContent, /Upload unavailable/);

  // Bounds reject the whole selection before reading or posting anything.
  const bounds = fixture();
  await bounds.attach([file('large.txt', 10 * 1024 * 1024 + 1)]);
  assert.equal(bounds.reads.length, 0); assert.equal(bounds.calls.length, 0);
  assert.match(bounds.ids['chat-file-feedback'].textContent, /10 MB/);
  await bounds.attach(Array.from({length: 5}, () => file()));
  assert.equal(bounds.calls.length, 0);
  assert.match(bounds.ids['chat-file-feedback'].textContent, /4 files/);
  await bounds.attach([file(), file('two.txt'), file('three.txt'), file('four.txt')]);
  assert.equal(bounds.controller.pendingIds().length, 4);
  assert.equal(bounds.ids['chat-attach-btn'].disabled, true);
  assert.equal(bounds.ids['chat-screenshot-btn'].disabled, true);

  // A transport failure remains visible without leaving a stuck placeholder.
  const failed = fixture();
  failed.handler(async () => { throw new Error('HTTP 413'); });
  await failed.attach([file('failed.txt')]);
  assert.equal(failed.controller.hasDraft(), false);
  assert.equal(failed.controller.isBusy(), false);
  assert.match(failed.ids['chat-file-feedback'].textContent, /failed.txt.*HTTP 413/);
  assert.equal(failed.ids['chat-file-feedback'].hidden, false);

  const io = fixture();
  let prevented = 0;
  await io.ids['composer-input'].listeners.paste({clipboardData: {files: [file('paste.png')]}, preventDefault() { prevented++; }});
  assert.equal(prevented, 1); assert.equal(io.calls.length, 1);
  io.ids['composer-input'].listeners.paste({clipboardData: {files: []}, preventDefault() { throw Error('ordinary text paste must be preserved'); }});
  await io.ids.panel.listeners.drop({dataTransfer: {types: ['Files'], files: [file('drop.txt')]}, preventDefault() { prevented++; }});
  assert.equal(io.calls.length, 2);
  assert.equal(prevented, 2);
  let finishCapture;
  io.handler(async (method, route) => {
    assert.equal(method, 'POST'); assert.equal(route, '/v1/sessions/a/screenshot');
    return new Promise(resolve => { finishCapture = resolve; });
  });
  const capture = io.ids['chat-screenshot-btn'].click();
  await tick();
  assert.equal(io.ids['chat-file-feedback'].hidden, false, 'actual capture progress stays visible');
  assert.match(io.ids['chat-file-feedback'].textContent, /Taking a screenshot/);
  finishCapture({file: {id: 'capture', name: 'screen.png', mime: 'image/png', size: 10, session_id: 'a'}});
  await capture;
  assert.equal(io.calls.at(-1).route, '/v1/sessions/a/screenshot');
  assert.equal(io.ids['chat-file-feedback'].hidden, true, 'successful capture uses its card without a redundant banner');

  // Artifact actions use only broker file identity, never a supplied path.
  const artifacts = fixture();
  const message = new Element();
  const output = {id: '../unsafe', name: '<img src=x onerror=alert(1)>.html', mime: 'text/html', size: 30, session_id: 'a'};
  artifacts.controller.renderMessage(message, {role: 'assistant', files: [output]});
  assert.equal(message.children[0].attributes['aria-label'], 'Returned files');
  assert.equal(message.children[0].children[0].children[0].children[0].textContent, output.name);
  await findButton(message, 'Open').click();
  await findButton(message, 'Save as…').click();
  assert.equal(artifacts.bridges[0][0], 'open_chat_file');
  assert.equal(artifacts.bridges[0][1].file_id, output.id);
  assert.equal(artifacts.bridges[1][0], 'save_chat_file');
  artifacts.bridgeHandler(async () => null);
  await findButton(message, 'Open').click();
  assert.match(artifacts.ids['chat-file-feedback'].textContent, /unavailable/);

  artifacts.handler(async () => ({mime: 'text/html', data_base64: Buffer.from('<script>window.hacked=true</script>').toString('base64')}));
  await findButton(message, 'Preview').click();
  let dialog = artifacts.document.body.children[0];
  assert.equal(dialog.open, true);
  let body = dialog.children[1];
  assert.equal(body.children[0].tagName, 'pre');
  assert.equal(body.children[0].textContent, '<script>window.hacked=true</script>');
  assert.equal(artifacts.created.length, 0, 'HTML is inert text, never a Blob URL document');
  assert.equal(artifacts.calls.at(-1).route, '/v1/sessions/a/files/..%2Funsafe/content');

  // Only allowed raster MIME types become image elements/Blob URLs, and
  // closing or switching conversation revokes their buffers.
  artifacts.handler(async () => ({mime: 'image/png', data_base64: 'aGVsbG8='}));
  await findButton(message, 'Preview').click();
  assert.equal(dialog.children[1].children[0].tagName, 'img');
  assert.equal(artifacts.created.length, 1);
  artifacts.switchSession('b');
  assert.equal(dialog.open, false);
  assert.deepEqual(artifacts.revoked, artifacts.created);
  artifacts.handler(async () => ({mime: 'image/svg+xml', data_base64: Buffer.from('<svg onload=alert(1) />').toString('base64')}));
  const svg = new Element();
  artifacts.controller.renderMessage(svg, {files: [{id: 'svg', name: 'test.svg', mime: 'image/svg+xml', session_id: 'b'}]});
  await findButton(svg, 'Preview').click();
  assert.equal(dialog.children[1].children[0].tagName, 'p');
  assert.match(dialog.children[1].children[0].textContent, /unavailable/);
  assert.equal(artifacts.created.length, 1);

  console.log('Chat files: bounded uploads, pending/session races, paste/drop/capture, native actions, and inert previews passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
