import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import {
  decodeRecordingText, fetchRecording, inflateGzip, isGzipBytes, parseRecordingBytes, readRecordingFile
} from '../src/RecordingCodec.js';
import { BattleLoader } from '../src/BattleLoader.js';

// A real recording written by src/recording_io.py in both formats (14 KB
// plain, 1.5 KB gzip: header mtime 0, no file name).
const FIX = new URL('./fixtures/', import.meta.url);
const plainBytes = new Uint8Array(readFileSync(new URL('mini_recording.json', FIX)));
const gzBytes = new Uint8Array(readFileSync(new URL('mini_recording.json.gz', FIX)));
const expected = JSON.parse(new TextDecoder().decode(plainBytes));

test('gzip magic is sniffed from the bytes, not the name', () => {
  assert.equal(isGzipBytes(gzBytes), true);
  assert.equal(isGzipBytes(plainBytes), false);
  assert.equal(isGzipBytes(new Uint8Array([0x1f])), false, 'too short');
  assert.equal(isGzipBytes(gzBytes.buffer.slice(gzBytes.byteOffset, gzBytes.byteOffset + 4)), true,
    'ArrayBuffer input');
  assert.deepEqual([...gzBytes.slice(0, 8)], [0x1f, 0x8b, 8, 0, 0, 0, 0, 0],
    'deterministic header from the Python writer: no FNAME flag, mtime 0');
});

test('plain text bytes parse as they are', async () => {
  assert.deepEqual(await parseRecordingBytes(plainBytes), expected);
  assert.deepEqual(await parseRecordingBytes(plainBytes.buffer), expected);
});

test('raw gzip bytes (no Content-Encoding) are inflated in the browser', async () => {
  const data = await parseRecordingBytes(gzBytes);
  assert.deepEqual(data, expected);
  assert.equal(data.battle_name, 'gzip codec fixture');
  // Node's own gzip of the plain text decodes the same (a different writer).
  assert.deepEqual(await parseRecordingBytes(gzipSync(plainBytes)), expected);
});

test('already-inflated text (dev server sent Content-Encoding: gzip) is not inflated twice', async () => {
  const inflated = await inflateGzip(gzBytes);
  assert.equal(await decodeRecordingText(inflated), new TextDecoder().decode(inflated));
  assert.deepEqual(await parseRecordingBytes(inflated), expected);
});

test('fetchRecording handles .json, .json.gz and HTTP errors', async () => {
  const realFetch = globalThis.fetch;
  const bodies = { '/r.json': plainBytes, '/r.json.gz': gzBytes, '/misnamed.json': gzBytes };
  globalThis.fetch = async (url) => bodies[url]
    ? new Response(bodies[url], { status: 200 })
    : new Response('nope', { status: 404, statusText: 'Not Found' });
  try {
    for (const url of Object.keys(bodies)) assert.deepEqual(await fetchRecording(url), expected, url);
    await assert.rejects(fetchRecording('/missing.json.gz'), /HTTP 404/);
    const loader = new BattleLoader();
    await loader.load('/r.json.gz');
    assert.equal(loader.metadata.battleName, 'gzip codec fixture');
    assert.ok(loader.simTrace.length > 0);
  } finally {
    globalThis.fetch = realFetch;
  }
});

test('a picked/dropped file loads in either format', async () => {
  assert.deepEqual(await readRecordingFile(new Blob([gzBytes])), expected);
  assert.deepEqual(await readRecordingFile(new Blob([plainBytes])), expected);
});

test('no DecompressionStream: a clear error for gzip, plain still loads', async () => {
  const real = globalThis.DecompressionStream;
  try {
    globalThis.DecompressionStream = undefined;
    await assert.rejects(parseRecordingBytes(gzBytes), /DecompressionStream/);
    assert.deepEqual(await parseRecordingBytes(plainBytes), expected);
  } finally {
    globalThis.DecompressionStream = real;
  }
});
