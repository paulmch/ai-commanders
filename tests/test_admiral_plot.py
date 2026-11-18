"""
The admiral's tactical plot (src/llm/admiral_view.py): layout rules, names,
framing, aggregation and determinism. Pure layout is tested without
rendering; rendering tests skip without matplotlib.
"""

import math
import random
import struct

import pytest

from src.llm import admiral_view as av
from src.llm import fleet_intel as intel
from src.simulation import CombatSimulation
from tests.test_admiral_intel import add_torpedo, ship

CLASSES = ["corvette", "frigate", "destroyer", "cruiser", "battlecruiser", "battleship",
           "dreadnought", "cruiser_torpedo"]


def random_battle(seed, n_alpha=None, n_beta=None, gap_km=None, spread_km=None, z_km=20.0):
    """A frame + meta of two random formations (km inputs, trace units out)."""
    rng = random.Random(seed)
    n_alpha = n_alpha or rng.randint(1, 10)
    n_beta = n_beta or rng.randint(1, 10)
    gap = gap_km if gap_km is not None else rng.choice([20.0, 60.0, 150.0, 400.0, 900.0])
    spread = spread_km if spread_km is not None else rng.choice([10.0, 40.0, 120.0, 300.0])
    frame = {"t": 90.0, "ships": {}, "torpedoes": [], "projectiles": []}
    meta = {}
    names = {"alpha": "TIS", "beta": "OCS"}
    for faction, n, x0, vx in (("alpha", n_alpha, -gap / 2, 1.5), ("beta", n_beta, gap / 2, -1.5)):
        for i in range(1, n + 1):
            sid = f"{faction}_{i}"
            ang = rng.uniform(0, 2 * math.pi)
            fwd = (math.cos(ang), math.sin(ang), rng.uniform(-0.4, 0.4))
            frame["ships"][sid] = {
                "pos": [(x0 + rng.uniform(-spread, spread) / 3) * 1000.0,
                        rng.uniform(-spread, spread) * 1000.0, rng.uniform(-z_km, z_km) * 1000.0],
                "vel": [(vx + rng.uniform(-1, 1)) * 1000.0, rng.uniform(-1, 1) * 1000.0, 0.0],
                "fwd": list(fwd), "thrust": rng.choice([0.0, 0.05, 1.0]),
                "destroyed": rng.random() < 0.1, "hull": rng.uniform(10, 100)}
            meta[sid] = {"name": f"{names[faction]} {rng.choice(['Dart', 'Halberd', 'Lancer'])}-{i}",
                         "type": rng.choice(CLASSES), "faction": faction}
    return frame, meta


def check_layout(lay):
    """Hard label rules: inside the panel, never on a glyph, never on each other."""
    view = lay.view
    rects = [p.rect for p in lay.labels]
    for p in lay.labels:
        r = p.rect
        assert view.rect[0] <= r[0] and r[2] <= view.rect[2], p
        assert view.rect[1] <= r[1] and r[3] <= view.rect[3], p
        for cx, cy, rad in lay.glyph_circles:
            assert not av.rect_circle_overlap(r, cx, cy, rad), (p.text, (cx, cy, rad))
        if lay.inset:
            assert not av.rects_overlap(r, av.inset_frame(lay.inset)), p.text
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not av.rects_overlap(rects[i], rects[j]), (lay.labels[i].text, lay.labels[j].text)
    irects = [p.rect for p in lay.inset_labels]
    for i in range(len(irects)):
        for j in range(i + 1, len(irects)):
            assert not av.rects_overlap(irects[i], irects[j])


class TestLabels:
    @pytest.mark.parametrize("seed", range(40))
    def test_labels_never_overlap_glyphs_or_each_other(self, seed):
        frame, meta = random_battle(seed)
        scene = av.build_scene(av.annotate_torpedoes(frame), meta, faction="alpha")
        lay = av.layout_scene(scene)
        check_layout(lay)
        ids = {it.sid for it in scene.ships}
        assert all(p.key[2:] in ids for p in lay.labels + lay.inset_labels if p.key.startswith("s:"))

    @pytest.mark.parametrize("seed", range(20))
    def test_every_ship_gets_a_label_in_ordinary_formations(self, seed):
        frame, meta = random_battle(seed, n_alpha=8, n_beta=8, gap_km=400.0, spread_km=120.0)
        scene = av.build_scene(av.annotate_torpedoes(frame), meta, faction="alpha")
        lay = av.layout_scene(scene)
        shown = {p.key for p in lay.labels} | {p.key for p in lay.inset_labels}
        wanted = {"s:" + it.sid for it in scene.ships if it.alive or it.dead_new}
        assert wanted <= shown, wanted - shown

    @pytest.mark.parametrize("seed", range(30))
    def test_every_ship_gets_a_label_at_merge_range(self, seed):
        """The merge is where the admiral most needs a name for issue_order."""
        frame, meta = random_battle(seed, n_alpha=8, n_beta=8, gap_km=[4.0, 15.0, 40.0][seed % 3],
                                    spread_km=[12.0, 30.0][seed % 2])
        for faction in ("alpha", "beta"):
            scene = av.build_scene(av.annotate_torpedoes(frame), meta, faction=faction)
            lay = av.layout_scene(scene)
            check_layout(lay)
            shown = {p.key for p in lay.labels} | {p.key for p in lay.inset_labels}
            wanted = {"s:" + it.sid for it in scene.ships if (it.alive or it.dead_new)
                      and lay.view.contains_px(*lay.view.to_px(*it.pos[:2]))}
            assert wanted <= shown, wanted - shown

    def test_merge_triggers_a_zoom_that_covers_no_ship(self):
        frame, meta = random_battle(3, n_alpha=6, n_beta=6, gap_km=4.0, spread_km=12.0)
        # one straggler far away keeps the plan wide, so the merge cannot be separated
        frame["ships"]["alpha_6"]["pos"] = [-600e3, 0.0, 0.0]
        frame["ships"]["alpha_6"]["destroyed"] = False
        for s in frame["ships"].values():
            s["destroyed"] = False
        scene = av.build_scene(frame, meta, faction="alpha")
        lay = av.layout_scene(scene)
        assert lay.inset is not None
        assert lay.inset.view.px_per_km >= av.INSET_MIN_ZOOM * lay.view.px_per_km * 0.99
        frame_rect = av.inset_frame(lay.inset)
        for cx, cy, rad in lay.glyph_circles:
            assert not av.rect_circle_overlap(frame_rect, cx, cy, rad)
        check_layout(lay)
        assert {"s:" + sid for sid in lay.inset.ships} <= {p.key for p in lay.inset_labels} | \
            {"s:" + it.sid for it in scene.ships if not it.alive}

    def test_no_zoom_when_the_plan_separates_the_fleets(self):
        frame, meta = random_battle(5, n_alpha=4, n_beta=4, gap_km=500.0, spread_km=80.0)
        scene = av.build_scene(frame, meta, faction="alpha")
        assert av.layout_scene(scene).inset is None


class TestNames:
    def test_labels_are_the_names_the_text_and_issue_order_use(self):
        frame, meta = random_battle(11, n_alpha=6, n_beta=6, gap_km=300.0, spread_km=80.0)
        scene = av.build_scene(frame, meta, faction="beta")
        names = {sid: m["name"] for sid, m in meta.items()}
        lay = av.layout_scene(scene)
        for p in lay.labels + lay.inset_labels:
            if not p.key.startswith("s:"):
                continue
            sid = p.key[2:]
            assert p.text.startswith(names[sid])
            assert intel.resolve_ship_ref(names[sid], names) == sid
            rest = p.text[len(names[sid]):].split()
            assert all(w.endswith("%") or w in ("SIDE", "TAIL", "DESTROYED") or w.startswith("Z")
                       for w in rest), p.text

    def test_old_one_on_one_ids_resolve(self):
        names = {"alpha": "TIS Resolute", "beta": "OCS Vigilant"}
        frame = {"t": 0.0, "torpedoes": [], "ships": {
            "alpha": {"pos": [0, 0, 0], "vel": [0, 0, 0], "fwd": [1, 0, 0], "hull": 100.0},
            "beta": {"pos": [300e3, 0, 0], "vel": [0, 0, 0], "fwd": [-1, 0, 0], "hull": 100.0}}}
        meta = {sid: {"name": n, "type": "destroyer", "faction": sid} for sid, n in names.items()}
        scene = av.build_scene(frame, meta, faction="alpha")
        for it in scene.ships:
            assert it.name == names[it.sid]
            assert intel.resolve_ship_ref(it.label(), names) == it.sid

    def test_wave_labels_name_the_target(self):
        sim = CombatSimulation(seed=1)
        ship(sim, "alpha_1", "corvette", "alpha", (0, 0, 0))
        ship(sim, "beta_1", "destroyer", "beta", (300, 0, 0), fwd=(-1, 0, 0))
        add_torpedo(sim, "torp_alpha_1_0001", "alpha_1", "beta_1", (60, 0, 0), (6, 0, 0))
        meta = {"alpha_1": {"name": "TIS alpha_1", "type": "corvette", "faction": "alpha"},
                "beta_1": {"name": "OCS beta_1", "type": "destroyer", "faction": "beta"}}
        scene = av.build_scene(av.build_live_frame(sim), meta, faction="alpha")
        assert [w.label().split(" -> ")[1].rsplit(" ", 1)[0] for w in scene.waves] == ["OCS beta_1"]


class TestTorpedoes:
    def _sim(self):
        sim = CombatSimulation(seed=1)
        ship(sim, "alpha_1", "corvette", "alpha", (0, 0, 0))
        ship(sim, "alpha_2", "corvette", "alpha", (0, 30, 0))
        ship(sim, "beta_1", "destroyer", "beta", (300, 0, 0), fwd=(-1, 0, 0))
        ship(sim, "beta_2", "destroyer", "beta", (300, 60, 0), fwd=(-1, 0, 0))
        return sim

    def meta(self, sim):
        return {sid: {"name": s.name, "type": s.ship_type, "faction": s.faction}
                for sid, s in sim.ships.items()}

    def test_one_wave_per_launching_group_and_target_with_first_impact(self):
        sim = self._sim()
        for k in range(4):
            add_torpedo(sim, f"torp_alpha_1_{k}", "alpha_1", "beta_1", (40 + 30 * k, 0, 0), (8, 0, 0))
        add_torpedo(sim, "torp_alpha_2_9", "alpha_2", "beta_1", (90, 10, 0), (8, 0, 0))
        add_torpedo(sim, "torp_alpha_2_8", "alpha_2", "beta_2", (90, 40, 0), (8, 0, 0))
        frame = av.build_live_frame(sim)
        scene = av.build_scene(frame, self.meta(sim), faction="alpha")
        by_target = {w.target: w for w in scene.waves}
        assert sorted(by_target) == ["beta_1", "beta_2"]
        w = by_target["beta_1"]
        assert (w.n_live, w.n_blind) == (5, 0) and w.own
        etas = [tp["eta"] for tp in frame["torpedoes"] if tp["target"] == "beta_1"]
        assert w.eta == pytest.approx(min(etas))
        assert scene.torps_own == (6, 0) and scene.torps_enemy == (0, 0)

    def test_eta_is_the_text_turn_states_number(self):
        sim = self._sim()
        add_torpedo(sim, "torp_alpha_1_0001", "alpha_1", "beta_1", (100, 0, 0), (6, 0, 0), dv_kps=9.0)
        frame = av.build_live_frame(sim)
        rounds = {r.round_id: r for r in intel.gather_rounds(sim, "alpha")}
        assert frame["torpedoes"][0]["eta"] == pytest.approx(rounds["torp_alpha_1_0001"].eta_s)
        # the same from a trace (mass from the delta-v left, as recorded)
        trace = {k: v for k, v in frame.items()}
        trace["torpedoes"] = [{k: v for k, v in tp.items() if k not in ("state", "threat", "eta")}
                              for tp in frame["torpedoes"]]
        again = av.frame_from_trace(trace)
        assert again["torpedoes"][0]["eta"] == pytest.approx(frame["torpedoes"][0]["eta"], rel=1e-6)

    def test_blind_rounds_are_told_apart_and_spent_ones_only_counted(self):
        sim = self._sim()
        # blinded, still on beta_1's line
        add_torpedo(sim, "torp_alpha_1_a", "alpha_1", "beta_1", (290, 0, 0), (8, 0, 0), disabled=True)
        # blinded, passing wide of everything
        add_torpedo(sim, "torp_alpha_1_b", "alpha_1", "beta_1", (290, -200, 0), (8, 0, 0), disabled=True)
        scene = av.build_scene(av.build_live_frame(sim), self.meta(sim), faction="alpha")
        assert [(w.n_live, w.n_blind, w.target) for w in scene.waves] == [(0, 1, "beta_1")]
        assert "1 unguided -> " in scene.waves[0].label()
        assert scene.spent_own == 1 and scene.torps_own == (0, 2)

    def test_new_rounds_since_the_last_checkpoint(self):
        sim = self._sim()
        add_torpedo(sim, "torp_alpha_1_old", "alpha_1", "beta_1", (100, 0, 0), (8, 0, 0))
        prev = av.build_live_frame(sim)
        add_torpedo(sim, "torp_alpha_1_new", "alpha_1", "beta_1", (20, 0, 0), (8, 0, 0))
        scene = av.build_scene(av.build_live_frame(sim), self.meta(sim), faction="alpha", prev_frame=prev)
        assert scene.waves[0].new and scene.waves[0].label().endswith("NEW")
        scene = av.build_scene(prev, self.meta(sim), faction="alpha", prev_frame=prev)
        assert not scene.waves[0].new


class TestScene:
    def test_face_tags_follow_the_engine_hit_cones(self):
        sim = CombatSimulation(seed=1)
        me = ship(sim, "alpha_1", "destroyer", "alpha", (0, 0, 0))
        ship(sim, "beta_1", "battlecruiser", "beta", (200, 0, 0), fwd=(0, 1, 0))
        ship(sim, "beta_2", "frigate", "beta", (200, 50, 0), fwd=(-1, -0.25, 0))
        ship(sim, "beta_3", "destroyer", "beta", (300, 0, 0), fwd=(1, 0, 0))
        meta = {sid: {"name": s.name, "type": s.ship_type, "faction": s.faction}
                for sid, s in sim.ships.items()}
        scene = av.build_scene(av.build_live_frame(sim), meta, faction="alpha")
        tags = {it.sid: it.face for it in scene.ships}
        for sid in ("beta_1", "beta_2", "beta_3"):
            assert tags[sid] == intel.observed_facing(sim.ships[sid], me.position)["face"]
        assert [tags["beta_1"], tags["beta_2"], tags["beta_3"]] == ["SIDE", "NOSE", "TAIL"]

    def test_ghost_is_the_coasting_position_and_flags_the_gun_band(self):
        frame, meta = random_battle(2, n_alpha=1, n_beta=1, gap_km=200.0, spread_km=0.0, z_km=0.0)
        frame["ships"]["alpha_1"].update(pos=[0, 0, 0], vel=[2000.0, 0, 0], destroyed=False)
        frame["ships"]["beta_1"].update(pos=[250e3, 0, 0], vel=[-1000.0, 0, 0], destroyed=False)
        meta["alpha_1"]["type"] = meta["beta_1"]["type"] = "destroyer"
        scene = av.build_scene(frame, meta, faction="alpha", interval_s=30.0)
        a = next(it for it in scene.ships if it.sid == "alpha_1")
        assert a.ghost == pytest.approx((60.0, 0.0, 0.0))
        assert not a.ghost_in_band        # ghosts at 60 and 220 km: 160 km apart
        frame["ships"]["beta_1"]["pos"] = [230e3, 0, 0]       # ghosts 60 and 200: 140 km
        scene = av.build_scene(frame, meta, faction="alpha", interval_s=30.0)
        assert next(it for it in scene.ships if it.sid == "alpha_1").ghost_in_band

    def test_deaths_since_the_last_checkpoint(self):
        frame, meta = random_battle(4, n_alpha=2, n_beta=2)
        for s in frame["ships"].values():
            s["destroyed"] = False
        prev = {k: (dict(v) if k == "ships" else v) for k, v in frame.items()}
        prev["ships"] = {sid: dict(s) for sid, s in frame["ships"].items()}
        frame["ships"]["beta_1"]["destroyed"] = True
        scene = av.build_scene(frame, meta, faction="alpha", prev_frame=prev)
        b1 = next(it for it in scene.ships if it.sid == "beta_1")
        assert b1.dead_new and b1.label().endswith("DESTROYED")
        scene = av.build_scene(frame, meta, faction="alpha", prev_frame=frame)
        assert not next(it for it in scene.ships if it.sid == "beta_1").dead_new


class TestFraming:
    def test_framing_holds_every_point_and_is_the_smallest_step(self):
        rng = random.Random(7)
        for _ in range(200):
            cx, cy = rng.uniform(-3000, 3000), rng.uniform(-3000, 3000)
            spread = rng.choice([5.0, 50.0, 300.0, 1500.0])
            pts = [(cx + rng.uniform(-spread, spread), cy + rng.uniform(-spread / 2, spread / 2))
                   for _ in range(rng.randint(1, 20))]
            view = av.choose_view(pts)
            for x, y in pts:
                px, py = view.to_px(x, y)
                assert view.contains_px(px, py, pad=min(av.FRAME_MARGIN_PX) - 1e-6)
            assert view.width_km in av.WIDTH_LADDER_KM
            k = view.cx / (view.width_km / 20.0)
            assert k == pytest.approx(round(k), abs=1e-6)
            idx = av.WIDTH_LADDER_KM.index(view.width_km)
            assert view == av.frame_at(pts, view.width_km)
            if idx > 0:   # the smallest step that holds the content
                assert av.frame_at(pts, av.WIDTH_LADDER_KM[idx - 1]) is None

    def test_framing_is_stable_under_small_motion(self):
        pts = [(-200.0, 10.0), (190.0, -40.0), (0.0, 60.0)]
        v1 = av.choose_view(pts)
        v2 = av.choose_view([(x + 3.0, y - 2.0) for x, y in pts])
        assert (v1.cx, v1.cy, v1.width_km) == (v2.cx, v2.cy, v2.width_km)
        assert av.choose_view(list(reversed(pts))) == v1

    def test_strip_never_exaggerates_height(self):
        frame, meta = random_battle(9, z_km=400.0)
        scene = av.build_scene(frame, meta, faction="alpha")
        lay = av.layout_scene(scene)
        assert lay.strip_kmpp >= 1.0 / lay.view.px_per_km - 1e-12


@pytest.fixture
def mpl():
    return pytest.importorskip("matplotlib")


class TestRender:
    def test_fixed_size_and_deterministic_bytes(self, mpl):
        frame, meta = random_battle(21, n_alpha=8, n_beta=8)
        a = av.render_frame_png(frame, meta, faction="alpha", checkpoint=3, total_checkpoints=20)
        b = av.render_frame_png(frame, meta, faction="alpha", checkpoint=3, total_checkpoints=20)
        assert a == b
        assert struct.unpack(">II", a[16:24]) == (av.CANVAS_W, av.CANVAS_H)
        # the image budget: width x height / 750 tokens, within +10 % of the old ~1.23k
        assert av.CANVAS_W * av.CANVAS_H / 750 <= 1.1 * 1230
        assert b"matplotlib" not in a     # no version stamp in the PNG

    def test_live_state_renders_and_the_header_carries_the_clock_and_score(self, mpl):
        sim = CombatSimulation(seed=1)
        ship(sim, "alpha_1", "destroyer", "alpha", (0, 0, 0))
        ship(sim, "beta_1", "destroyer", "beta", (300, 0, 0), fwd=(-1, 0, 0))
        meta = {sid: {"name": s.name, "type": s.ship_type, "faction": s.faction}
                for sid, s in sim.ships.items()}
        view = av.AdmiralViewState()
        png = view.render(sim, meta, faction="beta", checkpoint=4, total_checkpoints=20,
                          score_text=av.score_header(sim, "beta"))
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        scene = av.build_scene(av.build_live_frame(sim), meta, faction="beta", checkpoint=4,
                               total_checkpoints=20, score_text=av.score_header(sim, "beta"))
        assert "checkpoint 4 of 20" in scene.header and "YOU BETA = cyan" in scene.header
        assert scene.header.endswith(av.score_header(sim, "beta"))
        assert av.score_header(sim, "beta").startswith("score you ")


def test_height_tag_from_fifty_km_off_the_viewers_plane():
    frame, meta = random_battle(12, n_alpha=3, n_beta=2, gap_km=300.0, spread_km=40.0, z_km=0.0)
    for s in frame["ships"].values():
        s["destroyed"] = False
    frame["ships"]["beta_1"]["pos"][2] = 120e3
    frame["ships"]["beta_2"]["pos"][2] = -30e3
    scene = av.build_scene(frame, meta, faction="alpha")
    tags = {it.sid: it.label() for it in scene.ships}
    assert tags["beta_1"].endswith(" Z+120")
    assert " Z" not in tags["beta_2"].replace(meta["beta_2"]["name"], "")


def test_legend_numbers_are_the_plots():
    """The cached system-prompt legend quotes the plot's own constants."""
    from src.llm.prompts import ADMIRAL_VISION_LEGEND as legend
    flat = " ".join(legend.split())
    assert f"within {av.GUN_BAND_KM:.0f} km of an enemy ship's" in flat
    assert f"{av.GUN_BAND_KM:.0f} km from its gun ships" in flat
    assert f"PD umbrella ({av.class_kit('destroyer')['pd_km']:.0f} km)" in flat
    assert f"(from {av.HEIGHT_TAG_KM:.0f} km on)" in flat
    view = av.choose_view([(0.0, 0.0), (333.0, 50.0)])
    assert "snaps to 1/20 of the width" in flat
    k = view.cx / (view.width_km / 20.0)
    assert k == pytest.approx(round(k))
    for word in ("cyan", "orange", "SIDE", "TAIL", "Z+N", "NEW", "unguided", "ZOOM", "SIDE VIEW",
                 "DESTROYED"):
        assert word in flat, word
