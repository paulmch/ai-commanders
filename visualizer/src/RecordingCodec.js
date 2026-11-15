/**
 * Recording files: plain JSON (`*.json`) or gzip-compressed JSON (`*.json.gz`,
 * the default for new recordings; src/recording_io.py writes them).
 *
 * The format is decided by the first two bytes (gzip magic 1f 8b), never by
 * the name or the HTTP headers: a dev server may send `x.json.gz` with
 * `Content-Encoding: gzip` (the browser has already inflated it: plain text
 * arrives) or as raw `application/gzip` bytes (we inflate them here), and a
 * misnamed file still loads. Inflation uses the browser's
 * DecompressionStream('gzip'); without it a gzip recording fails with a clear
 * message (plain recordings still load).
 */

export const GZIP_MAGIC_0 = 0x1f;
export const GZIP_MAGIC_1 = 0x8b;

/** True when the bytes start with the gzip magic number. */
export function isGzipBytes(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  return u8.length >= 2 && u8[0] === GZIP_MAGIC_0 && u8[1] === GZIP_MAGIC_1;
}

/** Inflate gzip bytes to a Uint8Array (browser DecompressionStream). */
export async function inflateGzip(bytes) {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('This browser cannot open compressed (.json.gz) recordings: no '
      + 'DecompressionStream. Use a current Chrome/Firefox/Safari, or gunzip the file '
      + 'and open the .json');
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Recording bytes (plain or gzip) -> JSON text. */
export async function decodeRecordingText(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const plain = isGzipBytes(u8) ? await inflateGzip(u8) : u8;
  return new TextDecoder('utf-8').decode(plain);
}

/** Recording bytes (ArrayBuffer / Uint8Array, plain or gzip) -> parsed object. */
export async function parseRecordingBytes(bytes) {
  return JSON.parse(await decodeRecordingText(bytes));
}

/** Fetch and parse a recording URL (`.json` or `.json.gz`). */
export async function fetchRecording(url, init = undefined) {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`Failed to load recording: HTTP ${response.status} ${response.statusText || ''}`.trim());
  }
  return parseRecordingBytes(await response.arrayBuffer());
}

/** Parse a recording File / Blob picked or dropped by the user. */
export async function readRecordingFile(file) {
  return parseRecordingBytes(await file.arrayBuffer());
}
