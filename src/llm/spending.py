"""Conservative, thread-safe reservations for explicitly budgeted battles.

Every request reserves an upper bound on its cost before it is sent and is
settled against OpenRouter's reported cost afterwards. Text is priced by its
JSON byte count (bytes always exceed tokens); image parts are priced as a
generous prompt-token allowance derived from their pixel size, never from
their base64 length (a 200 KB PNG would otherwise reserve ~270k "tokens").
"""
import base64
import binascii
import json
import math
import struct
import threading

# Safety factor applied to every reservation.
RESERVE_FACTOR = 1.25
# Anthropic bills an image at about width * height / 750 tokens.
PIXELS_PER_IMAGE_TOKEN = 750
# Anthropic downscales server-side: high-resolution models (Sonnet 5,
# Opus 4.7+) cap an image at 2576 px on the long edge, ~4784 tokens; older
# models at ~1600. The larger cap covers both.
ANTHROPIC_MAX_IMAGE_TOKENS = 4784
# Any other provider (image tokenizer unknown), or an image whose size cannot
# be read: a fixed generous floor, raised for images large enough to need
# more at the Anthropic rate.
DEFAULT_IMAGE_TOKENS = 3000
# Content-part types this estimator cannot price.
_UNPRICED_PARTS = {'input_audio', 'audio', 'audio_url', 'video_url', 'file', 'input_file'}


def image_dimensions(url):
    """(width, height) of a base64 data-URL PNG, GIF or JPEG, else None."""
    if not isinstance(url, str) or not url.startswith('data:'):
        return None
    header, _, data = url.partition(',')
    if ';base64' not in header:
        return None
    try:
        # Headers live at the front; decoding 48 KB is plenty.
        raw = base64.b64decode(data[:65536])
    except (binascii.Error, ValueError):
        return None
    if raw.startswith(b'\x89PNG\r\n\x1a\n') and raw[12:16] == b'IHDR' and len(raw) >= 24:
        return struct.unpack('>II', raw[16:24])
    if raw[:6] in (b'GIF87a', b'GIF89a') and len(raw) >= 10:
        return struct.unpack('<HH', raw[6:10])
    if raw.startswith(b'\xff\xd8'):
        i = 2
        while i + 9 < len(raw) and raw[i] == 0xFF:
            marker = raw[i + 1]
            # SOF0-SOF15 except DHT (C4), JPG (C8) and DAC (CC)
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                height, width = struct.unpack('>HH', raw[i + 5:i + 9])
                return width, height
            i += 2 + struct.unpack('>H', raw[i + 2:i + 4])[0]
    return None


def image_prompt_tokens(url, model):
    """Generous prompt-token allowance for one image part sent to `model`."""
    anthropic = 'anthropic/' in str(model).lower()
    size = image_dimensions(url)
    if size is None:
        return ANTHROPIC_MAX_IMAGE_TOKENS if anthropic else DEFAULT_IMAGE_TOKENS
    width, height = size
    tokens = math.ceil(max(width, 1) * max(height, 1) / PIXELS_PER_IMAGE_TOKEN)
    if anthropic:
        return min(tokens, ANTHROPIC_MAX_IMAGE_TOKENS)
    return max(tokens, DEFAULT_IMAGE_TOKENS)


def _strip_images(value, images):
    """Copy of `value` without image data; each image URL is appended to `images`."""
    if isinstance(value, dict):
        part_type = value.get('type')
        if part_type == 'image_url' or (part_type is None and 'image_url' in value):
            image = value.get('image_url')
            images.append(image.get('url') if isinstance(image, dict) else image)
            return {'type': 'image_url'}
        if part_type in _UNPRICED_PARTS or 'input_audio' in value:
            raise ValueError('Budget reservations price text and image input only, '
                             f'not {part_type or "input_audio"}')
        return {key: _strip_images(item, images) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_images(item, images) for item in value]
    return value


class RequestBudget:
    def __init__(self, limit_usd, pricing):
        if not math.isfinite(limit_usd) or limit_usd <= 0:
            raise ValueError('Budget must be positive and finite')
        self.limit_usd = limit_usd
        self.pricing = pricing
        self.committed_usd = 0.0
        self.reported_usd = 0.0
        self.exhausted = False
        self.stopped = None          # reason, once stop() refuses every later request
        self.images_reserved = 0
        self._lock = threading.Lock()

    def stop(self, reason='stopped'):
        """Refuse every later reservation (an interrupted run: Ctrl-C). Calls
        already in flight settle as usual; nothing new is sent."""
        with self._lock:
            self.stopped = str(reason)
            self.exhausted = True

    def reserve(self, payload):
        pricing = self.pricing[payload['model']]
        tiers = [pricing, *pricing.get('overrides', [])]
        input_rate = max(float(t.get(k, 0)) for t in tiers for k in ('prompt', 'input_cache_write'))
        output_rate = max(float(t.get(k, 0)) for t in tiers for k in ('completion', 'internal_reasoning'))
        # OpenRouter's `image` field is a per-image-token rate (it equals the
        # prompt rate wherever the catalog advertises it); take the higher.
        image_rate = max([input_rate, *(float(t.get('image', 0)) for t in tiers)])
        if not all(math.isfinite(rate) and rate >= 0 for rate in (input_rate, output_rate, image_rate)):
            raise ValueError('Invalid model pricing')
        # Enforce the priced envelope at routing time as well as locally.
        # OpenRouter specifies prompt/completion ceilings in dollars per million.
        provider = payload.setdefault('provider', {})
        ceilings = provider.setdefault('max_price', {})
        for key, limit in [('prompt', input_rate * 1e6), ('completion', output_rate * 1e6),
                           ('request', float(pricing.get('request', 0)))]:
            ceilings[key] = min(ceilings.get(key, limit), limit)
        # Byte count plus framing allowance exceeds ordinary text token counts.
        # Image parts are priced by pixel size instead (image_prompt_tokens).
        images = []
        encoded = json.dumps(_strip_images(payload, images), ensure_ascii=True)
        image_tokens = sum(image_prompt_tokens(url, payload['model']) for url in images)
        reserved = ((len(encoded.encode()) + 4096) * input_rate
                    + image_tokens * image_rate
                    + payload['max_tokens'] * output_rate
                    + float(pricing.get('request', 0))) * RESERVE_FACTOR
        return self.reserve_usd(reserved, images=len(images))

    def reserve_usd(self, reserved, images=0):
        """Reserve a precomputed upper bound (providers that price requests
        themselves, e.g. the Anthropic client's token-priced bound)."""
        reserved = float(reserved)
        if not math.isfinite(reserved) or reserved < 0:
            raise ValueError('Invalid reservation')
        with self._lock:
            if self.stopped is not None:
                raise RuntimeError(f'Battle API budget closed before request ({self.stopped})')
            if self.committed_usd + reserved > self.limit_usd:
                self.exhausted = True
                raise RuntimeError('Battle API budget exhausted before request')
            self.committed_usd += reserved
            self.images_reserved += images
        return reserved

    def settle(self, reserved, cost):
        # Uncertain requests (timeouts, missing usage) retain their reservation.
        if cost is None:
            return
        cost = float(cost)
        if not math.isfinite(cost) or cost < 0:
            return
        with self._lock:
            self.committed_usd += cost - reserved
            self.reported_usd += cost


# -----------------------------------------------------------------------------
# Cumulative ledger of paid runs
# -----------------------------------------------------------------------------

class LedgerCapExceeded(RuntimeError):
    """A run's cap would take the ledger past --ledger-cap: the run must not start."""


class SpendLedger:
    """
    Append-only JSONL ledger of paid runs (default
    data/recordings/spend_ledger.jsonl, git-ignored), shared by every paid
    script (run_live_draft_battle, run_llm_battle, replay_turn).

    Each run writes two lines under an exclusive file lock:
      {"event": "open", run_id, ts, script, name, models, cap_usd, mode}
        before the first request - it RESERVES the run's cap, so a crashed
        run (no close line) still counts at its full cap;
      {"event": "close", run_id, ts, script, name, models, calls, tokens,
       usd, committed_usd, cap_usd, status, mode}
        when it ends - the run then counts at its reported dollars (or its
        committed dollars, if larger: calls whose cost never came back).
    An optional free-text `label` (which experiment the run belongs to) rides
    on both lines.
    total_usd() = sum of closed runs' usd + caps of runs still open. Runs with
    mode "mock" are written (when a mock run is pointed at a ledger
    explicitly) but never counted; the ledger cap is checked for them too.
    """

    def __init__(self, path):
        from pathlib import Path
        self.path = Path(path)

    def _lock(self):
        import contextlib
        import fcntl

        @contextlib.contextmanager
        def held():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path.with_suffix(self.path.suffix + '.lock'), 'a') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
        return held()

    def entries(self):
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def _total(self, entries):
        runs = {}
        for entry in entries:
            if entry.get('mode') == 'mock' or not entry.get('run_id'):
                continue
            run = runs.setdefault(entry['run_id'], {})
            run[entry.get('event')] = entry
        total = 0.0
        for run in runs.values():
            if 'close' in run:
                # A close counts at its reported dollars, or at what was still
                # committed when it closed if that is more: reservations of
                # calls whose cost never came back (timeouts) and of calls
                # still in flight when an interrupted run closed.
                close = run['close']
                total += max(float(close.get('usd') or 0.0), float(close.get('committed_usd') or 0.0))
            elif 'open' in run:
                total += float(run['open'].get('cap_usd') or 0.0)
        return total

    def total_usd(self):
        with self._lock():
            return self._total(self.entries())

    def _append(self, entry):
        with open(self.path, 'a') as handle:
            handle.write(json.dumps(entry, sort_keys=True) + '\n')

    def check(self, cap_usd, ledger_cap_usd=None):
        """Raise LedgerCapExceeded if this cap would pass ledger_cap_usd (writes nothing).
        Scripts call it before any pre-flight work, then open_run (which checks again,
        atomically) right before the first request."""
        if ledger_cap_usd is None:
            return
        total = self.total_usd()
        if total + float(cap_usd) > float(ledger_cap_usd):
            raise LedgerCapExceeded(
                f'Ledger {self.path}: ${total:.2f} spent or reserved + this run\'s cap '
                f'${float(cap_usd):.2f} = ${total + float(cap_usd):.2f} > --ledger-cap '
                f'${float(ledger_cap_usd):.2f}; not starting')

    def open_run(self, *, script, name, models, cap_usd, ledger_cap_usd=None, mock=False, label=None):
        """Reserve this run's cap; raises LedgerCapExceeded (nothing written) past ledger_cap_usd."""
        import time
        import uuid
        cap_usd = float(cap_usd)
        with self._lock():
            total = self._total(self.entries())
            # Checked for mock runs too (they never count, but the refusal path
            # is then testable offline).
            if ledger_cap_usd is not None and total + cap_usd > float(ledger_cap_usd):
                raise LedgerCapExceeded(
                    f'Ledger {self.path}: ${total:.2f} spent or reserved + this run\'s cap '
                    f'${cap_usd:.2f} = ${total + cap_usd:.2f} > --ledger-cap ${float(ledger_cap_usd):.2f}; '
                    f'not starting')
            run_id = uuid.uuid4().hex[:12]
            self._append({'event': 'open', 'run_id': run_id, 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          'script': script, 'name': name, 'models': list(models), 'cap_usd': cap_usd,
                          'ledger_total_before_usd': round(total, 4), 'mode': 'mock' if mock else 'paid',
                          **({'label': str(label)} if label else {})})
        self._open = getattr(self, '_open', {})
        self._open[run_id] = {'script': script, 'name': name, 'models': list(models),
                              'cap_usd': cap_usd, 'mode': 'mock' if mock else 'paid',
                              **({'label': str(label)} if label else {})}
        return run_id

    def close_run(self, run_id, *, calls, tokens, usd, committed_usd=None, status='done'):
        import time
        meta = getattr(self, '_open', {}).get(run_id, {})
        with self._lock():
            self._append({'event': 'close', 'run_id': run_id,
                          'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          **meta, 'calls': int(calls), 'tokens': tokens,
                          'usd': round(float(usd), 6),
                          'committed_usd': None if committed_usd is None else round(float(committed_usd), 6),
                          'status': status})
