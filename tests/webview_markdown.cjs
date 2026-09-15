const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Minimal DOM test fixture; no alternate UI or simulated broker is launched.
class Node {
  constructor(tag = '#text', text = '') {
    this.tagName = tag.toUpperCase(); this.children = []; this.attrs = {}; this.value = text;
  }
  appendChild(node) { this.children.push(node); return node; }
  replaceChildren() { this.children = []; this.value = ''; }
  setAttribute(key, value) { this.attrs[key] = value; }
  set textContent(value) { this.value = value; this.children = []; }
  get textContent() { return this.value + this.children.map(n => n.textContent).join(''); }
}
const document = {createElement: tag => new Node(tag), createTextNode: text => new Node('#text', text)};
const source = fs.readFileSync(path.join(__dirname, '../localpilot/webview/app.js'), 'utf8');
const start = source.indexOf('  function appendInlineMarkdown(');
const end = source.indexOf('  function renderUserTurn(', start);
assert(start >= 0 && end > start);
const context = vm.createContext({document});
vm.runInContext(source.slice(start, end), context);
const render = content => { const root = new Node('div'); context.renderSafeMarkdown(root, content); return root; };
const find = (root, tag) => root.children.flatMap(n => [...(n.tagName === tag ? [n] : []), ...find(n, tag)]);

let root = render('Before\n```python\ndef greet():\n    print("hello")\n\n    return 1\n```\nAfter');
assert.equal(find(root, 'PRE').length, 1);
assert.equal(find(root, 'CODE')[0].textContent, 'def greet():\n    print("hello")\n\n    return 1');
assert.equal(find(root, 'PRE')[0].tabIndex, 0);
assert(root.textContent.endsWith('After'));
root = render('````text\n```\n  inner\n````\n~~~\nsecond\n~~~');
assert.equal(find(root, 'PRE').length, 2);
assert.equal(find(root, 'CODE')[0].textContent, '```\n  inner');
assert.equal(find(render('```\n  unfinished\n'), 'CODE')[0].textContent, '  unfinished\n');
assert.equal(find(render('1. Example\n     ```text\n     indented\n     ```'), 'CODE')[0].textContent, '     indented');
root = render('# Heading\n- first\n- second\n\n3. third\n4. fourth');
assert.equal(root.children[0].attrs['aria-level'], '1');
assert.equal(find(root, 'UL').length, 1);
assert.equal(find(root, 'OL')[0].start, 3);
assert.equal(find(root, 'LI').length, 4);
root = render('[Docs](https://example.com/docs) and https://example.com/page. **bold** `code`');
const links = find(root, 'A');
assert.equal(links.length, 2);
assert.equal(links[0].href, 'https://example.com/docs');
assert.equal(links[0].textContent, 'Docs');
assert.equal(links[1].href, 'https://example.com/page');
assert.equal(links[0].rel, 'noopener noreferrer');
assert.equal(links[0].target, '_blank');
assert(root.textContent.includes('page.'));
root = render('<script>alert(1)</script> [bad](javascript:alert(1)) [file](file:///secret)');
assert.equal(find(root, 'SCRIPT').length, 0);
assert.equal(find(root, 'A').length, 0);
assert(root.textContent.includes('<script>'));
assert.equal(find(render('`https://example.com`'), 'A').length, 0);
console.log('Production markdown renderer: all behavior checks passed');
