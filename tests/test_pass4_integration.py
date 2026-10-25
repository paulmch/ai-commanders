"""
Cross-workstream seams of pass 4 (comms x admiral intel x captain intel),
fixed while merging the parallel branches.
"""
import types

from src.llm.battle_runner import LLMBattleRunner
from src.llm.battle_runner import load_fleet_data
from src.llm.communication import FleetCommunicationChannel
from src.llm.prompts import _format_captain_reports


def _channel():
    channel = FleetCommunicationChannel()
    channel.register_ship("alpha_1", "Cpt A", "TIS Dart-1", "alpha")
    channel.register_ship("alpha_2", "Cpt B", "TIS Lancer-1", "alpha")
    channel.register_ship("beta_2", "Cpt D", "OCS Lancer-1", "beta")
    return channel


class TestLabelReferences:
    """The admiral turn state labels every ship 'NAME [id]'; references written
    that way by commanders must resolve to that ship, the id winning."""

    def test_label_resolves_by_id(self):
        assert _channel().match_ship("OCS Lancer-1 [beta_2]") == ["beta_2"]

    def test_label_id_beats_a_colliding_name(self):
        # "Lancer-1" alone is ambiguous across the factions; the id is not.
        assert _channel().match_ship("Lancer-1 [alpha_2]") == ["alpha_2"]

    def test_unknown_label_id_falls_back_to_the_name(self):
        assert _channel().match_ship("TIS Dart-1 [nope]") == ["alpha_1"]

    def test_excluded_id_is_not_returned(self):
        assert _channel().match_ship("TIS Dart-1 [alpha_1]", exclude="alpha_1") == []


class TestReportsCarryTheDiscussion:
    """discuss_with_admiral Q&A rides on the asking ship's report; the admiral
    must see it next checkpoint, so such an ACK is never collapsed."""

    @staticmethod
    def _snapshot():
        ship = types.SimpleNamespace(ship_name="TIS Dart-1", ship_id="alpha_1")
        return types.SimpleNamespace(friendly_ships=[ship], checkpoint=3,
                                     labels={"alpha_1": "TIS Dart-1 [alpha_1]"})

    def test_ack_with_discussion_is_listed_with_the_exchange(self):
        report = {"ship_name": "TIS Dart-1", "ship_id": "alpha_1", "status": "ACK",
                  "checkpoint": 2, "compliance": None,
                  "discussion": [{"question": "Hold torpedoes until 200 km?",
                                  "answer": "Yes, hold until the salvo order."}]}
        text = _format_captain_reports([report], self._snapshot())
        assert "ACK, nothing to flag" not in text
        assert "asked you: \"Hold torpedoes until 200 km?\"" in text
        assert "you said: \"Yes, hold until the salvo order.\"" in text

    def test_plain_ack_still_collapses(self):
        report = {"ship_name": "TIS Dart-1", "ship_id": "alpha_1", "status": "ACK",
                  "checkpoint": 2, "compliance": None}
        text = _format_captain_reports([report], self._snapshot())
        assert "ACK, nothing to flag (1): TIS Dart-1 [alpha_1]" in text


class TestForceRetryNamesOnlyRealTools:
    @staticmethod
    def _captain(ship_type, has_torpedoes):
        cfg = types.SimpleNamespace(ship_type=ship_type, fleet_data=load_fleet_data(),
                                    has_torpedoes=has_torpedoes)
        return types.SimpleNamespace(config=cfg)

    def test_gunless_corvette_is_not_told_to_set_weapons_order(self):
        text = LLMBattleRunner._force_retry_text(self._captain("corvette", True))
        assert "set_weapons_order" not in text
        assert "launch_torpedo" in text and "set_maneuver" in text

    def test_gun_hull_keeps_set_weapons_order(self):
        text = LLMBattleRunner._force_retry_text(self._captain("destroyer", False))
        assert "set_weapons_order" in text and "launch_torpedo" not in text
