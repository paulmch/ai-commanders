"""Conservative, thread-safe reservations for explicitly budgeted text battles."""
import json
import math
import threading


class RequestBudget:
    def __init__(self, limit_usd, pricing):
        if not math.isfinite(limit_usd) or limit_usd <= 0:
            raise ValueError('Budget must be positive and finite')
        self.limit_usd = limit_usd
        self.pricing = pricing
        self.committed_usd = 0.0
        self.reported_usd = 0.0
        self.exhausted = False
        self._lock = threading.Lock()

    def reserve(self, payload):
        pricing = self.pricing[payload['model']]
        tiers = [pricing, *pricing.get('overrides', [])]
        input_rate = max(float(t.get(k, 0)) for t in tiers for k in ('prompt', 'input_cache_write'))
        output_rate = max(float(t.get('completion', 0)) for t in tiers)
        if not all(math.isfinite(rate) and rate >= 0 for rate in (input_rate, output_rate)):
            raise ValueError('Invalid model pricing')
        # Enforce the priced envelope at routing time as well as locally.
        # OpenRouter specifies prompt/completion ceilings in dollars per million.
        provider = payload.setdefault('provider', {})
        ceilings = provider.setdefault('max_price', {})
        for key, limit in [('prompt', input_rate * 1e6), ('completion', output_rate * 1e6),
                           ('request', float(pricing.get('request', 0)))]:
            ceilings[key] = min(ceilings.get(key, limit), limit)
        # Byte count plus framing allowance exceeds ordinary text token counts.
        # Reject multimodal input: image billing needs a separate estimator.
        encoded = json.dumps(payload, ensure_ascii=True)
        if 'image_url' in encoded or 'input_audio' in encoded:
            raise ValueError('Budget reservations currently support text battles only')
        reserved = ((len(encoded.encode()) + 4096) * input_rate
                    + payload['max_tokens'] * output_rate
                    + float(pricing.get('request', 0))) * 1.25
        with self._lock:
            if self.committed_usd + reserved > self.limit_usd:
                self.exhausted = True
                raise RuntimeError('Battle API budget exhausted before request')
            self.committed_usd += reserved
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
