const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {TextDecoder, TextEncoder} = require('node:util');
const source = fs.readFileSync(path.join(__dirname, '../localpilot/webview/app.js'), 'utf8');
const start = source.indexOf('  async function api(');
const end = source.indexOf('  async function checkHealth(', start);
let reply;
const requests = [];
const context = vm.createContext({
  BASE_URL:'http://127.0.0.1:1234', TOKEN:'local-token', Uint8Array, TextDecoder,
  fetch: async (url, options) => {requests.push({url, options}); return reply;},
});
vm.runInContext(source.slice(start, end), context);
const encoder = new TextEncoder();
function failure(chunks, status = 400) {
  const observation = {read:0, cancelled:false, released:false};
  reply = {ok:false, status, body:{getReader() {return {
    async read() {
      observation.read += 1;
      return chunks.length ? {done:false, value:encoder.encode(chunks.shift())} : {done:true};
    },
    async cancel() {observation.cancelled = true;},
    releaseLock() {observation.released = true;},
  };}}, text() {throw Error('Error bodies must not be read without a byte limit');}};
  return observation;
}
async function expectError(message, status = 400) {
  await assert.rejects(context.api('POST', '/v1/sessions/a/files', {name:'test.png'}), error => {
    assert.equal(error.status, status);
    assert.equal(error.message, message);
    return true;
  });
}
(async () => {
  let observed = failure(['{"error":"Unsupported ', 'file type. Choose text or an image."}']);
  await expectError('Unsupported file type. Choose text or an image.');
  assert(observed.released);
  assert.equal(requests[0].options.headers.Authorization, 'Bearer local-token');
  assert.equal(JSON.parse(requests[0].options.body).name, 'test.png');

  failure(['<html>Bad gateway</html>'], 502);
  await expectError('HTTP 502 for POST /v1/sessions/a/files', 502);
  failure(['{"error":{"unexpected":"object"}}']);
  await expectError('HTTP 400 for POST /v1/sessions/a/files');
  failure(['{"error":"' + 'x'.repeat(1000) + '"}']);
  await expectError('x'.repeat(800));
  observed = failure(['x'.repeat(8193), 'must not be read']);
  await expectError('HTTP 400 for POST /v1/sessions/a/files');
  assert(observed.cancelled);
  assert(observed.released);
  assert.equal(observed.read, 1, 'Stops consuming immediately at the byte bound');

  reply = {ok:false, status:503, body:null};
  await expectError('HTTP 503 for POST /v1/sessions/a/files', 503);
  reply = {ok:true, status:201, async text() {return '{"file":{"id":"saved"}}';}};
  const result = await context.api('POST', '/v1/sessions/a/files', {});
  assert.equal(result.file.id, 'saved', 'Successful API response behavior is unchanged');
  console.log('Production API errors: bounded JSON reasons, fallback/status, cancellation, and success passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
