"""
Regression tests for prompt structure and the information handed to AI commanders.

The captain prompt used to be a single string that interleaved per-checkpoint
state with the static doctrine that followed it, which made a cached prefix
impossible and re-charged full input price on every 30 s checkpoint.
"""

import json

import pytest

from src.llm.prompts import build_captain_messages, build_captain_prompt


def _kwargs(sim_time: float, hull: float, heat: float, nose_armor: float):
    """Two checkpoints of the same battle differing only in volatile state."""
    return dict(
        captain_name="Vance",
        ship_name="TIS Resolute",
        ship_type="destroyer",
        ship_status={
            "hull_integrity": hull,
            "heat_percent": heat,
            "delta_v_remaining": 480,
            "nose_armor": nose_armor,
            "lateral_armor": 26.0,
            "tail_armor": 30.3,
            "heatsink_capacity": 525,
            "radiators_extended": False,
        },
        tactical_status={
            "sim_time": sim_time,
            "angle_to_enemy_deg": 12.0,
            "ship_forward": {"x": 1, "y": 0, "z": 0},
            "enemies": [],
            "friendlies": [],
            "our_shots": int(sim_time), "our_hits": 0,
            "our_damage_dealt": 0, "our_damage_taken": 0,
        },
    )


class TestPromptCacheability:
    def test_system_prompt_is_stable_across_checkpoints(self):
        """
        The doctrine half must be byte-identical between checkpoints so it can be
        served from the provider's prompt cache for the whole battle.
        """
        early = build_captain_messages(**_kwargs(30.0, 100.0, 5.0, 151.2))
        late = build_captain_messages(**_kwargs(600.0, 61.0, 74.0, 88.4))

        assert early[0]["role"] == "system" and late[0]["role"] == "system"
        assert early[0]["content"] == late[0]["content"], (
            "System prompt changed between checkpoints - the cached prefix is "
            "invalidated and every checkpoint re-pays full input price."
        )

    def test_volatile_state_lives_in_the_user_turn(self):
        """Per-checkpoint state must appear in the user turn, not the system prompt."""
        early = build_captain_messages(**_kwargs(30.0, 100.0, 5.0, 151.2))
        late = build_captain_messages(**_kwargs(600.0, 61.0, 74.0, 88.4))

        assert len(early) == 2 and early[1]["role"] == "user"
        assert early[1]["content"] != late[1]["content"], "user turn did not change"

        # Sim time is volatile and must not be baked into the cached prefix.
        assert "T+30" in early[1]["content"]
        assert "T+30" not in early[0]["content"]

    def test_no_state_is_lost_by_the_split(self):
        """Splitting must not drop content that used to reach the model."""
        kw = _kwargs(120.0, 80.0, 40.0, 120.0)
        combined = "\n\n".join(m["content"] for m in build_captain_messages(**kw))

        for required in ("MY SHIP", "=== CONTROLS ===", "TACTICAL DATA",
                         "weapons:", "hull ", "THREATS TO ME", "ENEMIES", "FRIENDLIES"):
            assert required in combined, f"{required!r} vanished from the prompt"

    def test_live_armor_values_reach_the_captain(self):
        """
        Regression: armor was read with string keys against an enum-keyed dict, so
        captains saw a frozen 10/5/3 cm fallback regardless of real armor.
        """
        prompt = build_captain_prompt(**_kwargs(60.0, 100.0, 10.0, 151.2))
        assert "151.2cm" in prompt or "151.2" in prompt, (
            "real nose armor not present in the prompt"
        )
        assert "Nose 10cm | Lateral 5cm | Tail 3cm" not in prompt, (
            "captain is still being shown the hardcoded armor fallback"
        )


class TestPromptMatchesSimulation:
    """Numbers quoted to commanders must match what the simulation computes."""

    def test_documented_hit_probabilities_match_the_model(self):
        """
        Regression: the doctrine quoted a hardcoded accuracy table that had drifted
        from the fire-control model - first ~50% at 200 km where the model said
        ~39%, then a model that itself ignored where the target goes during the
        flight (captains were shown 12-19% at 435-490 km while 0/609 rounds
        beyond 200 km landed on dodging targets). Every number in the table is
        the live CombatSimulation.gun_firing_solution - the FIRE_WHEN_OPTIMAL
        gate - for a destroyer holding course, flying PADLOCK with evade
        'dodge' (between jinks, or caught mid-jink: burning across the line of
        fire), or flying the EVADE maneuver (ManeuverType.EVASIVE; the 2026-10
        recalibrations moved them; see scripts/fire_control_calibration.py).
        """
        from src.llm.prompts import ADMIRAL_GUN_DOCTRINE, PROJECTILE_PHYSICS_REFERENCE

        def flat(text):
            return " ".join(text.split())

        doctrine = flat(PROJECTILE_PHYSICS_REFERENCE)
        ranges = (50, 100, 150, 200, 300, 400)
        table = {
            "holding course": ((82, 79), (64, 57), (48, 37), (36, 20), (18, 4), (7, 1)),
            "evade dodge": ((82, 73), (55, 26), (27, 6), (10, 1), (2, 0), (0, 0)),
            "dodge, mid-jink": ((74, 37), (21, 3), (4, 1), (1, 0), (0, 0), (0, 0)),
            "EVADE maneuver": ((49, 46), (36, 24), (22, 7), (11, 2), (2, 0), (1, 0)),
        }
        for mode, cells in table.items():
            row = f"- {mode}: " + " ".join(f"{s}/{t}%" for s, t in cells)
            assert row in doctrine, f"doctrine row drifted: {row!r}"
            for range_km, (spinal, turret) in zip(ranges, cells):
                for muzzle, claimed in ((9.9, spinal), (6.0, turret)):
                    actual = _doctrine_p_hit("destroyer", mode, range_km, muzzle) * 100
                    assert abs(actual - claimed) <= 1.0, (
                        f"Doctrine tells captains {claimed}% ({mode}, {range_km} km, "
                        f"{muzzle} km/s) but the gate computes {actual:.1f}%. Update "
                        f"GUN_PHYSICS_REFERENCE or the fire-control model.")
        examples = (
            ("frigate", 100, 42, "a dodging frigate: spinal 42% at 100km"),
            ("frigate", 150, 14, "14% at 150km"),
            ("battleship", 200, 31, "a dodging battleship: 31% at 200km"),
            ("battleship", 300, 8, "8% at 300km"),
        )
        for hull, range_km, claimed, text in examples:
            assert text in doctrine
            actual = _doctrine_p_hit(hull, "evade dodge", range_km, 9.9) * 100
            assert abs(actual - claimed) <= 1.0, (hull, range_km, actual)
        assert "x0.6" in doctrine  # EVADE's conversion penalty (firecontrol)

        admiral = flat(ADMIRAL_GUN_DOCTRINE)
        assert "spinal 55% at 100 km, 10% at 200 km, 2% at 300 km" in admiral
        for range_km, claimed in ((100, 55), (200, 10), (300, 2)):
            actual = _doctrine_p_hit("destroyer", "evade dodge", range_km, 9.9) * 100
            assert abs(actual - claimed) <= 1.0, (range_km, actual)
        assert "far harder (21% at 100 km)" in admiral
        assert abs(_doctrine_p_hit("destroyer", "dodge, mid-jink", 100, 9.9) * 100 - 21) <= 1.0
        assert "broadside is ~1.5x the hit chance of nose-on at 200 km (~1.2x at 100 km)" in admiral
        for range_km, claimed in ((200, 1.5), (100, 1.2)):
            ratio = (_doctrine_p_hit("destroyer", "holding course", range_km, 9.9, broadside=True)
                     / _doctrine_p_hit("destroyer", "holding course", range_km, 9.9))
            assert abs(ratio - claimed) <= 0.06, (range_km, ratio)


    def test_closing_raises_and_crossing_lowers_hit_probability(self):
        """
        The doctrine used to say "high closing velocity = harder to hit ...
        sweet spot 1-3 km/s relative". The gate says the opposite: closure
        shortens the flight (P(hit) up), crossing and opening speed lower it.
        Every quoted number is the live gate's.
        """
        from src.llm.prompts import ADMIRAL_GUN_DOCTRINE, GUN_PHYSICS_REFERENCE, GUN_TACTICS_BLOCK

        def flat(text):
            return " ".join(text.split())

        cases = {"at rest": (0.0, 0.0), "closing": (-3000.0, 0.0),
                 "crossing": (0.0, 3000.0), "opening": (3000.0, 0.0)}
        p = {k: _doctrine_p_hit("destroyer", "holding course", 200, 9.9, velocity=v) * 100
             for k, v in cases.items()}
        assert p["closing"] > p["at rest"] > p["crossing"] > p["opening"], p
        for mode in ("holding course", "evade dodge"):
            for muzzle in (9.9, 6.0):
                for rng in (100, 200, 300):
                    at_rest = _doctrine_p_hit("destroyer", mode, rng, muzzle)
                    closing = _doctrine_p_hit("destroyer", mode, rng, muzzle, velocity=(-3000.0, 0.0))
                    crossing = _doctrine_p_hit("destroyer", mode, rng, muzzle, velocity=(0.0, 3000.0))
                    assert closing > at_rest >= crossing, (mode, muzzle, rng)

        claims = {"at rest": 36, "closing": 47, "crossing": 28, "opening": 20}
        for key, claimed in claims.items():
            assert abs(p[key] - claimed) <= 1.0, (key, p[key])
        physics = flat(GUN_PHYSICS_REFERENCE)
        assert ("Spinal vs a destroyer holding course at 200km: 36% at rest, 47% closing at "
                "3 km/s, 28% crossing at 3 km/s, 20% opening at 3 km/s") in physics
        admiral = flat(ADMIRAL_GUN_DOCTRINE)
        assert ("spinal vs a destroyer holding course at 200 km: 36% at rest, 47% closing at "
                "3 km/s, 28% crossing, 20% opening") in admiral
        for text in (physics, flat(GUN_TACTICS_BLOCK), admiral):
            assert "Closing speed RAISES" in text
            assert "harder to hit" not in text and "Sweet spot" not in text


def _doctrine_p_hit(hull, mode, range_km, muzzle_kps, broadside=False, velocity=(0.0, 0.0)):
    """The gate's P(hit) for a battleship's gun with this muzzle velocity at a
    target `range_km` out down +x, nose-on unless broadside, moving at
    `velocity` (vx, vy) m/s - negative vx closes on the shooter at rest."""
    from src.physics import Vector3D
    from src.simulation import CombatSimulation, Maneuver, ManeuverType, create_ship_from_fleet_data

    with open("data/fleet_ships.json") as f:
        fleet = json.load(f)
    sim = CombatSimulation(seed=0)
    shooter = create_ship_from_fleet_data("a", "battleship", "alpha", fleet,
                                          position=Vector3D(0, 0, 0), forward=Vector3D(1, 0, 0))
    target = create_ship_from_fleet_data(
        "b", hull, "beta", fleet, position=Vector3D(range_km * 1000.0, 0, 0),
        velocity=Vector3D(velocity[0], velocity[1], 0.0),
        forward=Vector3D(0, 1, 0) if broadside else Vector3D(-1, 0, 0))
    sim.add_ship(shooter)
    sim.add_ship(target)
    if mode == "evade dodge":
        target.current_maneuver = Maneuver(ManeuverType.PADLOCK, 0.0, target_id="a", evade="dodge")
    elif mode == "dodge, mid-jink":
        # Its evasion is flying it: full burn across the line of fire (nose
        # broadside on), which a PADLOCK dodger's lead does not extrapolate.
        target.current_maneuver = Maneuver(ManeuverType.PADLOCK, 0.0, target_id="a", evade="dodge")
        target.kinematic_state.forward = Vector3D(0, 1, 0)
        target.execution_mode = "EVADE"
        target.applied_throttle = 1.0
    elif mode == "EVADE maneuver":
        target.current_maneuver = Maneuver(ManeuverType.EVASIVE, 0.0)
    weapon = next(w for w in shooter.weapons.values()
                  if abs(w.weapon.muzzle_velocity_kps - muzzle_kps) < 1e-9)
    return sim.gun_firing_solution(shooter, target, weapon).hit_probability


class TestCacheRequestShape:
    """
    OpenRouter/Anthropic only caches when the request carries an explicit
    breakpoint. Verified live: without it 0% of prompt tokens were served from
    cache; with it, 91.8%.
    """

    def test_anthropic_system_prompt_gets_a_cache_breakpoint(self):
        from src.llm.client import apply_cache_breakpoint

        msgs = [
            {"role": "system", "content": "stable doctrine"},
            {"role": "user", "content": "volatile turn"},
        ]
        out = apply_cache_breakpoint(msgs, "anthropic/claude-sonnet-5")

        assert isinstance(out[0]["content"], list), "system content must be block form"
        block = out[0]["content"][0]
        # 1-hour lifetime: checkpoints are 7-17 min apart in wall-clock time,
        # past the 5-minute default (audit CP-05).
        assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert block["text"] == "stable doctrine"
        # The volatile turn must NOT be marked, or the breakpoint moves every turn.
        assert out[1]["content"] == "volatile turn"

    def test_cache_lifetime_per_provider_family(self):
        from src.llm.client import apply_cache_breakpoint

        msgs = [{"role": "system", "content": "doctrine"}, {"role": "user", "content": "turn"}]
        google = apply_cache_breakpoint(msgs, "google/gemini-3.5-flash")[0]["content"][0]
        assert google["cache_control"] == {"type": "ephemeral"}   # provider-managed TTL
        forced = apply_cache_breakpoint(msgs, "anthropic/claude-sonnet-5", ttl=None)
        assert forced[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_block_form_system_message_is_marked_on_its_last_text_block(self):
        from src.llm.client import apply_cache_breakpoint

        msgs = [{"role": "system", "content": [{"type": "text", "text": "a"},
                                               {"type": "text", "text": "b"}]},
                {"role": "user", "content": "turn"}]
        out = apply_cache_breakpoint(msgs, "anthropic/claude-sonnet-5")
        assert "cache_control" not in out[0]["content"][0]
        assert out[0]["content"][1]["cache_control"]["ttl"] == "1h"
        assert "cache_control" not in msgs[0]["content"][1], "caller's message mutated"

    def test_cache_outcomes_are_counted_per_call(self):
        from src.llm.client import CallStats

        stats = CallStats()
        stats.record_usage({"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 900}})
        stats.record_usage({"prompt_tokens": 1000,
                            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 950}})
        stats.record_usage({"prompt_tokens": 500, "prompt_tokens_details": {}})
        assert (stats.cache_hits, stats.cache_writes, stats.cache_cold) == (1, 1, 1)
        assert stats.cache_write_tokens == 950
        assert stats.cache_summary().startswith("cache: 1 hit / 1 write / 1 cold calls, 0.9k read")

    def test_per_call_overrides_and_finish_reason_reach_the_caller(self):
        import httpx
        from src.llm.client import CaptainClient

        sent = []

        def respond(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{"message": {"content": None, "tool_calls": []},
                             "finish_reason": "length"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8192,
                          "completion_tokens_details": {"reasoning_tokens": 7005},
                          "prompt_tokens_details": {"cached_tokens": 0}}})

        client = CaptainClient(model="anthropic/claude-sonnet-5", api_key="test",
                               max_tokens=8192, reasoning_effort="medium")
        client._client = httpx.Client(transport=httpx.MockTransport(respond))
        assert client.decide_with_tools([{"role": "user", "content": "x"}], []) == []
        meta = client.last_call_meta()
        assert meta["finish_reason"] == "length" and meta["reasoning_tokens"] == 7005
        assert meta["max_tokens"] == 8192
        client.decide_with_tools([{"role": "user", "content": "x"}], [],
                                 max_tokens=12288, reasoning_effort="low")
        assert sent[0]["reasoning"] == {"effort": "medium"} and sent[0]["max_tokens"] == 8192
        assert sent[1]["reasoning"] == {"effort": "low"} and sent[1]["max_tokens"] == 12288
        assert client.stats.truncated == 2

    def test_auto_caching_providers_are_left_alone(self):
        from src.llm.client import apply_cache_breakpoint

        msgs = [{"role": "system", "content": "doctrine"}]
        for model in ("openai/gpt-5.6-terra", "x-ai/grok-4.20", "deepseek/deepseek-v4-pro"):
            out = apply_cache_breakpoint(msgs, model)
            assert out[0]["content"] == "doctrine", (
                f"{model} caches automatically; cache_control markup is unnecessary"
            )

    def test_doctrine_clears_the_model_cache_minimum(self):
        """A prefix below the model's minimum silently will not cache."""
        from src.llm.client import cache_minimum_tokens

        system = build_captain_messages(**_kwargs(30.0, 100.0, 5.0, 151.2))[0]["content"]
        approx_tokens = len(system) / 4

        for model in ("anthropic/claude-sonnet-5", "anthropic/claude-opus-5"):
            assert approx_tokens >= cache_minimum_tokens(model), (
                f"doctrine is ~{approx_tokens:.0f} tokens, below the "
                f"{cache_minimum_tokens(model)}-token cache minimum for {model}"
            )


class TestToolSchemaMatchesExecutor:
    """A tool schema that lies about its coordinate frame is an info-fidelity bug."""

    def test_set_heading_schema_declares_the_world_frame(self):
        """
        Regression: the schema advertised "ship-relative coordinates" with
        +forward/+starboard/+up axes, but the executor passes the vector straight
        through to the simulation as a world-frame heading. A captain asking to
        fly "forward" flew along world +X regardless of where its nose pointed -
        and beta-fleet ships start facing -X, so "forward" meant backwards.
        """
        from src.llm.tools import get_captain_tools

        tools = get_captain_tools(has_torpedoes=False)
        heading = next(
            t for t in tools if t["function"]["name"] == "set_heading"
        )
        direction = heading["function"]["parameters"]["properties"]["direction"]
        text = (direction.get("description", "") + " ".join(
            p.get("description", "") for p in direction.get("properties", {}).values()
        )).lower()

        assert "world" in text, "set_heading must declare that it takes world-frame axes"
        assert "ship-relative coordinates" not in text, (
            "schema still claims a ship-relative frame the executor does not implement"
        )

    def test_prompt_and_schema_agree_on_the_heading_frame(self):
        """Doctrine and tool schema must not describe different frames."""
        from src.llm.prompts import CAPTAIN_DOCTRINE
        from src.llm.tools import get_captain_tools

        doctrine = CAPTAIN_DOCTRINE.lower()
        assert "world-frame direction vector" in doctrine or "world" in doctrine

        tools = get_captain_tools(has_torpedoes=False)
        heading = next(t for t in tools if t["function"]["name"] == "set_heading")
        schema_text = json.dumps(heading).lower()
        assert "world" in schema_text


class TestPersonalityPromptShipClass:
    """A captain must form its doctrine around the ship it actually commands."""

    def test_personality_prompt_uses_the_real_ship_class(self):
        """
        Regression: build_personality_selection_prompt defaults ship_class to
        "Destroyer" and captain.py never passed the real one, so every captain in
        every battle believed it commanded a destroyer. In a corvette duel that
        is badly wrong - a corvette carries torpedoes and point defense, no guns
        - and captains reasoned about gunnery they did not have.
        """
        from unittest.mock import Mock

        from src.llm.captain import LLMCaptain, LLMCaptainConfig
        from src.llm.prompts import build_personality_selection_prompt

        text = build_personality_selection_prompt(
            400.0, model_name="X", ship_class="Corvette", enemy_ship_class="Corvette"
        )
        assert "Corvette" in text
        assert "Destroyer" not in text

        # And the captain must actually supply it rather than taking the default.
        captain = LLMCaptain(
            LLMCaptainConfig(name="C", ship_name="S", ship_type="corvette"),
            client=Mock(),
        )
        assert captain.config.ship_type == "corvette"
        assert hasattr(captain, "enemy_ship_class")


class TestWeaponToolsMatchTheShip:
    """A captain must be offered the weapons its hull actually mounts."""

    def test_corvette_captain_is_offered_launch_torpedo(self):
        """
        Regression: LLMCaptainConfig.has_torpedoes defaulted to False and nothing
        ever derived it from fleet data, so launch_torpedo was never offered to
        any captain. A corvette's only weapon is the torpedo launcher, so a
        corvette captain had no usable weapon at all - a corvette-vs-corvette
        battle could not be won by either side.
        """
        import json
        from unittest.mock import Mock

        from src.llm.captain import LLMCaptain, LLMCaptainConfig, ship_has_torpedoes

        with open("data/fleet_ships.json") as f:
            fleet = json.load(f)

        assert ship_has_torpedoes("corvette", fleet) is True
        assert ship_has_torpedoes("destroyer", fleet) is False

        captain = LLMCaptain(
            LLMCaptainConfig(name="C", ship_name="S", ship_type="corvette", fleet_data=fleet),
            client=Mock(),
        )
        names = [t["function"]["name"] for t in captain.tools]
        assert "launch_torpedo" in names, (
            "corvette captain cannot launch torpedoes - it has no other weapon"
        )

    def test_gun_ships_are_not_offered_torpedoes(self):
        """The converse: a destroyer must not be handed a launcher it lacks."""
        import json
        from unittest.mock import Mock

        from src.llm.captain import LLMCaptain, LLMCaptainConfig

        with open("data/fleet_ships.json") as f:
            fleet = json.load(f)

        captain = LLMCaptain(
            LLMCaptainConfig(name="C", ship_name="S", ship_type="destroyer", fleet_data=fleet),
            client=Mock(),
        )
        names = [t["function"]["name"] for t in captain.tools]
        assert "launch_torpedo" not in names


class TestTorpedoCommandAndControl:
    """Captains must be able to launch salvos, and must see inbound ordnance."""

    def _fleet(self):
        with open("data/fleet_ships.json") as f:
            return json.load(f)

    def test_corvette_doctrine_states_torpedo_statistics(self):
        from src.llm.prompts import build_ship_capabilities_from_fleet

        text = build_ship_capabilities_from_fleet(
            ship_name="TIS Wasp", ship_type="corvette", fleet_data=self._fleet(),
            hull_integrity=100, heat_percent=0, delta_v_remaining=500,
            nose_armor=212, lateral_armor=36, tail_armor=42,
            heatsink_capacity=525, radiators_extended=False,
        )
        assert "TORPEDOES" in text
        for fact in ("Magazine", "delta-v", "SQUARE OF CLOSING SPEED", "point defense"):
            assert fact in text, f"doctrine omits {fact!r}"

    def test_gun_ships_are_not_told_about_torpedoes(self):
        from src.llm.prompts import build_ship_capabilities_from_fleet

        text = build_ship_capabilities_from_fleet(
            ship_name="TIS Line", ship_type="destroyer", fleet_data=self._fleet(),
            hull_integrity=100, heat_percent=0, delta_v_remaining=500,
            nose_armor=151, lateral_armor=26, tail_armor=30,
            heatsink_capacity=525, radiators_extended=False,
        )
        assert "TORPEDOES (" not in text

    def test_inbound_ordnance_is_reported(self):
        """
        Regression: torpedo_threats was computed by the captain every checkpoint
        and never rendered anywhere, so captains were never told ordnance was
        inbound at all.
        """
        from src.llm.prompts import format_torpedo_threats

        assert "none detected" in format_torpedo_threats([])

        text = format_torpedo_threats([
            {"distance_km": 180.0, "closing_kps": 22.4, "eta_seconds": 8.0,
             "source": "HFS Sonnet5"},
        ])
        assert "INBOUND ORDNANCE" in text
        assert "HFS Sonnet5" in text
        assert "180 km" in text
        assert "8s" in text

    def test_launch_tool_exposes_salvo_and_target(self):
        from src.llm.tools import get_captain_tools

        tool = next(
            t for t in get_captain_tools(has_torpedoes=True)
            if t["function"]["name"] == "launch_torpedo"
        )
        props = tool["function"]["parameters"]["properties"]
        assert "count" in props and "target_id" in props
        assert props["count"]["maximum"] >= 2

    def test_admiral_can_order_a_coordinated_salvo(self):
        from src.llm.admiral import AdmiralOrder
        from src.llm.admiral_tools import ADMIRAL_TOOLS
        from src.llm.prompts import format_admiral_orders_for_captain

        order_tool = next(
            t for t in ADMIRAL_TOOLS if t["function"]["name"] == "issue_order"
        )
        props = order_tool["function"]["parameters"]["properties"]
        assert "torpedo_salvo" in props and "torpedo_target" in props

        rendered = format_admiral_orders_for_captain(
            [AdmiralOrder(
                target_ship_id="a1", target_ship_name="TIS Wasp",
                order_text="Close and strike.", priority="CRITICAL",
                torpedo_salvo=2, torpedo_target="OCS Leviathan",
            )],
            fleet_directive="Concentrate on the dreadnought.",
        )
        assert "COORDINATED SALVO" in rendered
        assert "2 torpedo(es)" in rendered
        assert "OCS Leviathan" in rendered


class TestToolSurfaceMatchesArmament:
    """Captains must only be offered controls for weapons they actually mount."""

    def _fleet(self):
        with open("data/fleet_ships.json") as f:
            return json.load(f)

    def _tools_for(self, ship_type):
        from unittest.mock import Mock

        from src.llm.captain import LLMCaptain, LLMCaptainConfig

        captain = LLMCaptain(
            LLMCaptainConfig(name="C", ship_name="S", ship_type=ship_type,
                             fleet_data=self._fleet()),
            client=Mock(),
        )
        return {t["function"]["name"]: t for t in captain.tools}

    def test_gunless_hulls_get_no_fire_control(self):
        """
        Regression: every captain was handed set_weapons_order with
        spinal_mode/turret_mode regardless of armament. A corvette carries
        neither, so its captain spent its weapon order on guns it did not have
        and never reached for launch_torpedo.
        """
        for ship_type in ("corvette", "cruiser_torpedo"):
            tools = self._tools_for(ship_type)
            assert "set_weapons_order" not in tools, ship_type
            assert "launch_torpedo" in tools, ship_type

    def test_gun_hulls_get_fire_control_but_no_torpedoes(self):
        tools = self._tools_for("destroyer")
        assert "set_weapons_order" in tools
        assert "launch_torpedo" not in tools

    def test_every_advertised_mode_matches_a_weapon_group(self):
        """
        The executor resolves orders by weapon-group key, so a property named
        'coilgun_mode' against a 'coilguns' group silently drops every turret
        order - the exact defect that made turret fire a no-op.
        """
        from src.llm.tools import build_weapon_tool_for_ship, get_weapon_groups_for_ship

        fleet = self._fleet()
        for ship_type in ("destroyer", "cruiser", "battleship", "dreadnought"):
            groups = set(get_weapon_groups_for_ship(ship_type, fleet))
            if not groups:
                continue
            props = build_weapon_tool_for_ship(
                ship_type, fleet
            )["function"]["parameters"]["properties"]
            modes = {k[: -len("_mode")] for k in props if k.endswith("_mode")}
            assert modes == groups, (
                f"{ship_type}: advertised {sorted(modes)} but weapon groups are "
                f"{sorted(groups)} - orders for the mismatched groups are dropped"
            )
