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

        for required in ("CURRENT STATUS", "=== CONTROLS ===", "TACTICAL DATA",
                         "WEAPON STATUS", "Hull:"):
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
        from firecontrol.calculate_hit_probability - it promised ~50% at 200 km
        where the model yields ~39%, biasing captains toward standing off at ranges
        that are less effective than advertised.
        """
        from src.firecontrol import calculate_hit_probability
        from src.physics import Vector3D
        from src.simulation import create_ship_from_fleet_data
        from src.llm.prompts import PROJECTILE_PHYSICS_REFERENCE

        with open("data/fleet_ships.json") as f:
            fleet = json.load(f)
        target = create_ship_from_fleet_data("t", "destroyer", "beta", fleet)

        documented = {50: 82, 100: 64, 200: 39, 400: 17, 600: 10}
        for range_km, claimed in documented.items():
            solution = calculate_hit_probability(
                shooter_position=Vector3D(0, 0, 0),
                shooter_velocity=Vector3D(0, 0, 0),
                target_position=Vector3D(range_km * 1000, 0, 0),
                target_velocity=Vector3D(0, 0, 0),
                target_geometry=target.geometry,
                target_forward=Vector3D(-1, 0, 0),
                muzzle_velocity_kps=9.9,
                target_is_evading=False,
            )
            actual = solution.hit_probability * 100
            assert abs(actual - claimed) <= 3.0, (
                f"Doctrine tells captains ~{claimed}% at {range_km} km but the "
                f"simulation computes {actual:.1f}%. Update the table in "
                f"PROJECTILE_PHYSICS_REFERENCE or the accuracy model."
            )
            assert f"{claimed}%" in PROJECTILE_PHYSICS_REFERENCE


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
        assert block["cache_control"] == {"type": "ephemeral"}
        assert block["text"] == "stable doctrine"
        # The volatile turn must NOT be marked, or the breakpoint moves every turn.
        assert out[1]["content"] == "volatile turn"

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
