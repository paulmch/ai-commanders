"""RequestBudget prices image parts by pixel size instead of rejecting them.

Everything is offline: an in-process transport stands in for OpenRouter.
"""
import base64
import json
import math
import struct
from unittest.mock import Mock

import httpx
import pytest

from src.llm.client import CaptainClient, forced_tool_use_conflicts
from src.llm.fleet_config import AdmiralConfig
from src.llm.mock_openrouter import mock_transport
from src.llm.spending import (
    ANTHROPIC_MAX_IMAGE_TOKENS, DEFAULT_IMAGE_TOKENS, RESERVE_FACTOR,
    RequestBudget, image_dimensions, image_prompt_tokens,
)

SONNET = 'anthropic/claude-sonnet-5'
DEEPSEEK = 'deepseek/deepseek-v4.1-flash'
PRICING = {
    SONNET: {'prompt': '0.000002', 'completion': '0.00001', 'input_cache_read': '0.0000002',
             'input_cache_write': '0.0000025', 'input_cache_write_1h': '0.000004'},
    DEEPSEEK: {'prompt': '0.0000003', 'completion': '0.0000012', 'input_cache_read': '0.000000006'},
}


def png_url(width, height, padding=0):
    """A data URL whose PNG header declares width x height (body is filler)."""
    ihdr = struct.pack('>II', width, height) + b'\x08\x06\x00\x00\x00'
    raw = b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + ihdr + b'\x00' * (4 + padding)
    return 'data:image/png;base64,' + base64.b64encode(raw).decode()


def jpeg_url(width, height):
    app0 = b'\xff\xe0' + struct.pack('>H', 16) + b'JFIF\x00' + b'\x00' * 9
    sof0 = b'\xff\xc0' + struct.pack('>HBHH', 11, 8, height, width) + b'\x01\x01\x11\x00'
    return 'data:image/jpeg;base64,' + base64.b64encode(b'\xff\xd8' + app0 + sof0 + b'\xff\xd9').decode()


def vision_payload(model, url, text='Checkpoint 3. Set your directive.'):
    return {'model': model, 'max_tokens': 8192, 'messages': [
        {'role': 'system', 'content': 'You are an admiral.'},
        {'role': 'user', 'content': [{'type': 'text', 'text': text},
                                     {'type': 'image_url', 'image_url': {'url': url}}]}]}


def test_image_dimensions_read_from_png_jpeg_and_gif_headers():
    assert image_dimensions(png_url(1428, 798)) == (1428, 798)
    assert image_dimensions(jpeg_url(1920, 1080)) == (1920, 1080)
    gif = 'data:image/gif;base64,' + base64.b64encode(b'GIF89a' + struct.pack('<HH', 64, 32) + b'\x00' * 8).decode()
    assert image_dimensions(gif) == (64, 32)
    assert image_dimensions('https://example.com/plot.png') is None
    assert image_dimensions('data:image/png;base64,@@not-base64@@') is None


def test_anthropic_images_cost_pixels_over_750_capped_at_high_res_limit():
    assert image_prompt_tokens(png_url(1428, 798), SONNET) == math.ceil(1428 * 798 / 750)
    assert image_prompt_tokens(png_url(8000, 8000), SONNET) == ANTHROPIC_MAX_IMAGE_TOKENS
    assert image_prompt_tokens('https://example.com/plot.png', SONNET) == ANTHROPIC_MAX_IMAGE_TOKENS


def test_unknown_tokenizers_get_a_generous_floor_that_grows_with_size():
    assert image_prompt_tokens(png_url(1428, 798), DEEPSEEK) == DEFAULT_IMAGE_TOKENS
    assert image_prompt_tokens(png_url(4000, 3000), DEEPSEEK) == math.ceil(4000 * 3000 / 750)
    assert image_prompt_tokens('https://example.com/plot.png', DEEPSEEK) == DEFAULT_IMAGE_TOKENS


def test_image_bytes_are_priced_by_pixels_not_base64_length():
    """A 1.5 MB image must not reserve 1.5M 'tokens' of prompt."""
    budget = RequestBudget(100, PRICING)
    small = budget.reserve(vision_payload(SONNET, png_url(1428, 798)))
    huge = budget.reserve(vision_payload(SONNET, png_url(1428, 798, padding=1_500_000)))
    assert huge == pytest.approx(small)
    # Versus the text-only request: exactly the image allowance (plus the
    # few bytes of the stripped part), at the cache-write rate, x1.25.
    text_only = vision_payload(SONNET, '')
    text_only['messages'][1]['content'].pop()
    rate = 0.0000025
    extra = small - budget.reserve(text_only)
    expected = math.ceil(1428 * 798 / 750) * rate * RESERVE_FACTOR
    assert extra == pytest.approx(expected, abs=64 * rate * RESERVE_FACTOR)
    assert budget.images_reserved == 2


def test_advertised_image_rate_applies_when_above_prompt_rate():
    pricing = {'g/vision': {'prompt': '0.000001', 'completion': '0.000002', 'image': '0.00001'}}
    budget = RequestBudget(100, pricing)
    with_image = budget.reserve(vision_payload('g/vision', png_url(750, 1000)))
    payload = vision_payload('g/vision', '')
    payload['messages'][1]['content'].pop()
    extra = with_image - budget.reserve(payload)
    assert extra == pytest.approx(DEFAULT_IMAGE_TOKENS * 0.00001 * RESERVE_FACTOR, rel=0.01)


def test_ceilings_and_limit_still_enforced_for_vision_requests():
    budget = RequestBudget(0.2, PRICING)
    payload = vision_payload(SONNET, png_url(1428, 798))
    first = budget.reserve(payload)
    assert payload['provider']['max_price'] == {'prompt': 2.5, 'completion': 10.0, 'request': 0.0}
    # The payload sent upstream still carries the image itself.
    assert payload['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/png')
    assert 0.1 < first < 0.2
    with pytest.raises(RuntimeError, match='exhausted'):
        budget.reserve(vision_payload(SONNET, png_url(1428, 798)))
    assert budget.exhausted


def test_audio_is_still_refused_and_image_words_in_text_are_just_text():
    budget = RequestBudget(10, PRICING)
    audio = {'model': DEEPSEEK, 'max_tokens': 10, 'messages': [{'role': 'user', 'content': [
        {'type': 'input_audio', 'input_audio': {'data': 'AAAA', 'format': 'wav'}}]}]}
    with pytest.raises(ValueError, match='text and image input only'):
        budget.reserve(audio)
    budget.reserve({'model': DEEPSEEK, 'max_tokens': 10, 'messages': [
        {'role': 'user', 'content': 'the tool takes an image_url field'}]})
    assert budget.images_reserved == 0


def test_client_with_mock_openrouter_reserves_and_settles_an_image_call():
    client = CaptainClient(model=DEEPSEEK, api_key='test', temperature=None, max_tokens=8192,
                           reasoning_effort='high')
    client._client = httpx.Client(transport=mock_transport(PRICING, points=150, max_ships=8))
    client.budget = RequestBudget(1, PRICING)
    tools = [{'type': 'function', 'function': {'name': 'set_fleet_directive', 'parameters': {}}}]
    calls = client.decide_with_tools(vision_payload(DEEPSEEK, png_url(1428, 798))['messages'], tools)
    assert [c.name for c in calls] == ['set_fleet_directive']
    assert client.budget.images_reserved == 1
    assert 0 < client.budget.reported_usd < 0.01
    # Settled: committed spend equals the reported cost once nothing is in flight.
    assert client.budget.committed_usd == pytest.approx(client.budget.reported_usd)


def test_forced_tool_use_is_relaxed_only_for_claude_with_reasoning():
    sent = []
    def respond(request):
        sent.append(json.loads(request.content)['tool_choice'])
        return httpx.Response(200, json={'choices': [{'message': {'content': 'ok'}}], 'usage': {}})
    for model, effort in [(SONNET, 'high'), (SONNET, None), (DEEPSEEK, 'high')]:
        client = CaptainClient(model=model, api_key='test', temperature=None, reasoning_effort=effort)
        client._client = httpx.Client(transport=httpx.MockTransport(respond))
        client.decide_with_tools([{'role': 'user', 'content': 'buy'}], [], tool_choice='required')
    assert sent == ['auto', 'required', 'required']
    assert forced_tool_use_conflicts('openrouter/anthropic/claude-opus-5', 'low')
    assert not forced_tool_use_conflicts(SONNET, 'none')


def test_catalog_verified_image_input_overrides_family_allowlist():
    from src.llm.admiral import LLMAdmiral
    def admiral(**kwargs):
        return LLMAdmiral(AdmiralConfig(model=DEEPSEEK, name='A', **kwargs),
                          faction='beta', client=Mock(), fleet_data={'ships': {}})
    assert not admiral(vision=True).wants_vision            # allowlist: text-only family
    assert admiral(vision=True, image_input=True).wants_vision
    assert not admiral(vision=False, image_input=True).wants_vision


def test_draft_retries_a_rejected_forced_tool_call_with_auto():
    from src.llm.battle_runner import load_fleet_data
    from src.llm.client import LLMCallError, ToolCall
    from src.llm.fleet_draft import run_admiral_draft

    class RejectsRequired:
        def __init__(self):
            self.choices = []

        def decide_with_tools(self, messages, tools, model=None, temperature=None, tool_choice='auto'):
            self.choices.append(tool_choice)
            if tool_choice == 'required':
                raise LLMCallError('HTTP 400: tool_choice required unsupported', status_code=400)
            if tools[0]['function']['name'] == 'select_fleet':
                return [ToolCall(id='s', name='select_fleet',
                                 arguments={'ships': [{'ship_type': 'frigate', 'count': 2}]})]
            return [ToolCall(id='f', name='set_formation', arguments={'placements': []})]

    client = RejectsRequired()
    draft = run_admiral_draft(client, DEEPSEEK, 'Admiral', 'beta', load_fleet_data(),
                              budget=150, max_ships=8, allow_auto_fallback=False, verbose=False)
    assert [s.ship_type for s in draft.ships] == ['frigate', 'frigate']
    assert client.choices == ['required', 'auto', 'auto']
