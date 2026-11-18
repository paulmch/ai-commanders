# Replay comparison

Signature = tool names + discrete arguments (free text excluded). `agree` = replicates whose signature equals the variant's most common one; `=rec` = replicates matching the recorded reply.

## p5_sonnet55_vs_sonnet5_a:e592 T+90 cp3 directive alpha (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_fleet_directive()`
- claude-sonnet-5-5@medium: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [0, 0, 0], $0.0796
  - 3x `set_fleet_directive()`
- claude-sonnet-5-5@high: agree 2/3, =rec 1/3, 2 distinct, thinking tokens [161, 167, 288], $0.0536
  - 2x `set_battle_plan() ; set_fleet_directive()`
  - 1x `set_fleet_directive()`
- claude-sonnet-5@medium: agree 2/3, =rec 2/3, 2 distinct, thinking tokens [391, 760, 857], $0.1089
  - 2x `set_fleet_directive()`
  - 1x `issue_order(heading={"x":1.0,"y":0.0,"z":0.0},ship_name="all ships",throttle=0.6) ; set_fleet_directive()`
- claude-opus-5-5@high: agree 1/3, =rec 1/3, 3 distinct, thinking tokens [173, 594, 658], $0.2151
  - 1x `issue_order(evade="dodge",heading={"x":1.0,"y":-0.0,"z":0.0},ship_name="all ships",suggested_target="OCS Falchion-1",throttle=0.5) ; set_fleet_directive()`
  - 1x `issue_order(evade="dodge",heading={"x":1,"y":0,"z":0},ship_name="all ships",throttle=0.4) ; set_battle_plan() ; set_fleet_directive()`
  - 1x `set_fleet_directive()`

## p5_sonnet55_vs_sonnet5_a:e7292 T+300 cp10 directive alpha (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_fleet_directive()`
- claude-sonnet-5-5@medium: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [0, 0, 0], $0.0451
  - 3x `set_fleet_directive()`
- claude-sonnet-5-5@high: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [369, 424, 454], $0.0591
  - 3x `set_fleet_directive()`
- claude-sonnet-5@medium: agree 1/3, =rec 1/3, 3 distinct, thinking tokens [398, 939, 1050], $0.0792
  - 1x `issue_order(heading={"x":1.0,"y":0.1,"z":-0.0},ship_name="all ships",throttle=0.6) ; set_fleet_directive()`
  - 1x `issue_order(heading={"x":0.1,"y":1.0,"z":-0.1},ship_name="all ships",throttle=1.0) ; set_fleet_directive()`
  - 1x `set_fleet_directive()`
- claude-opus-5-5@high: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [406, 550, 691], $0.1367
  - 3x `set_fleet_directive()`

## p5_sonnet55_vs_sonnet5_a:e15293 T+420 cp14 captain alpha_5 (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_maneuver(evade="full",maneuver_type="EVADE")`
- claude-sonnet-5-5@medium: agree 2/3, =rec 1/3, 2 distinct, thinking tokens [0, 0, 0], $0.0990
  - 2x `set_maneuver(maneuver_type="EVADE",throttle=1.0)`
  - 1x `set_maneuver(evade="full",maneuver_type="EVADE")`
- claude-sonnet-5-5@high: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [148, 259, 265], $0.0486
  - 2x `set_heading(direction={"x":-1.0,"y":-0.1,"z":0.1},evade="dodge",throttle=1.0) ; set_radiators(extend=false)`
  - 1x `set_heading(direction={"x":-1.0,"y":-0.1,"z":0.1},evade="dodge",throttle=1.0) ; set_radiators(mode="auto")`
- claude-sonnet-5@medium: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [160, 316, 746], $0.1216
  - 2x `set_heading(direction={"x":-1.0,"y":-0.1,"z":0.1},evade="dodge",throttle=1.0) ; set_primary_target(target_id="beta_4") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `report_to_admiral(status="UNABLE") ; set_heading(direction={"x":-1.0,"y":-0.1,"z":0.1},evade="dodge",throttle=1.0) ; set_primary_target(target_id="beta_4")`
- claude-opus-5-5@high: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [563, 635, 889], $0.2642
  - 2x `log_note() ; report_to_admiral(request={"kind":"PD_COVER"},status="UNABLE") ; set_primary_target(target_id="beta_4")`
  - 1x `log_note() ; report_to_admiral(request={"kind":"FALL_BACK"},status="UNABLE") ; set_primary_target(target_id="beta_4")`

## p5_sonnet55_vs_sonnet5_a:e19384 T+480 cp16 captain alpha_5 (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_maneuver(maneuver_type="EVADE",throttle=1.0)`
- claude-sonnet-5-5@medium: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [0, 0, 0], $0.0376
  - 3x `set_maneuver(maneuver_type="EVADE",throttle=1.0)`
- claude-sonnet-5-5@high: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [127, 137, 138], $0.0453
  - 2x `log_note() ; set_heading(direction={"x":1.0,"y":0.1,"z":0.1},evade="full",throttle=1.0)`
  - 1x `set_heading(direction={"x":1.0,"y":0.1,"z":0.1},evade="full",throttle=1.0) ; set_radiators(mode="auto")`
- claude-sonnet-5@medium: agree 3/3, =rec 0/3, 1 distinct, thinking tokens [185, 252, 281], $0.0506
  - 3x `set_heading(direction={"x":1.0,"y":0.1,"z":0.1},evade="full",throttle=1.0)`
- claude-opus-5-5@high: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [550, 797, 929], $0.1352
  - 1x `report_to_admiral(status="UNABLE") ; surrender()`
  - 1x `log_note() ; report_to_admiral(contact={"assessment":"CLOSING","ship":"OCS Colossus"},status="UNABLE") ; surrender()`
  - 1x `fleet_signal(kind="BREAKING_OFF") ; report_to_admiral(contact={"assessment":"CLOSING","ship":"OCS Colossus"},status="UNABLE") ; surrender()`

## p5_sonnet55_vs_sonnet5_a:e22653 T+540 cp18 captain alpha_5 (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_maneuver(maneuver_type="EVADE",throttle=1.0)`
- claude-sonnet-5-5@medium: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [0, 0, 0], $0.0353
  - 3x `set_maneuver(maneuver_type="EVADE",throttle=1.0)`
- claude-sonnet-5-5@high: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [96, 96, 108], $0.0409
  - 2x `set_heading(direction={"x":1.0,"y":0.2,"z":0.1},evade="full",throttle=1.0) ; set_primary_target(target_id="beta_3")`
  - 1x `set_heading(direction={"x":1.0,"y":0.2,"z":0.1},evade="full",throttle=1.0) ; set_radiators(mode="auto")`
- claude-sonnet-5@medium: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [179, 204, 268], $0.0514
  - 2x `set_heading(direction={"x":1.0,"y":0.2,"z":0.1},evade="full",throttle=1.0) ; set_primary_target(target_id="beta_3") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_heading(direction={"x":1.0,"y":0.2,"z":0.1},evade="full",throttle=1.0) ; set_primary_target(target_id="beta_3")`
- claude-opus-5-5@high: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [625, 656, 957], $0.1347
  - 1x `log_note() ; report_to_admiral(status="UNABLE") ; set_primary_target(target_id="beta_3")`
  - 1x `log_note() ; set_primary_target(target_id="beta_3")`
  - 1x `log_note() ; report_to_admiral(status="UNABLE") ; set_heading(direction={"x":1.0,"y":0.2,"z":0.1},evade="full",throttle=1.0) ; set_primary_target(target_id="beta_3") ; set_radiators(extend=false)`

## p5_sonnet55_vs_sonnet5_c:e2 T+0 cp0 draft alpha (recorded claude-sonnet-5-5 @ medium)
- recorded: `select_fleet(ships=[{"count":3,"ship_type":"battlecruiser"},{"count":3,"ship_type":"destroyer"},{"count":3,"ship_type":"frigate"}])`
- claude-sonnet-5-5@medium: agree 3/3, =rec 0/3, 1 distinct, thinking tokens [0, 0, 0], $0.0203
  - 3x `select_fleet(ships=[{"count":3,"ship_type":"battlecruiser"},{"count":2,"ship_type":"destroyer"},{"count":5,"ship_type":"frigate"}])`
- claude-sonnet-5-5@high: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [430, 527, 550], $0.0224
  - 2x `select_fleet(ships=[{"count":5,"ship_type":"battlecruiser"},{"count":3,"ship_type":"frigate"}])`
  - 1x `select_fleet(ships=[{"count":5,"ship_type":"battlecruiser"},{"count":1,"ship_type":"destroyer"},{"count":1,"ship_type":"frigate"}])`
- claude-sonnet-5@medium: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [1230, 1682, 2353], $0.0760
  - 1x `select_fleet(ships=[{"count":1,"ship_type":"dreadnought"},{"count":2,"ship_type":"battlecruiser"},{"count":3,"ship_type":"frigate"},{"count":1,"ship_type":"corvette"}])`
  - 1x `select_fleet(ships=[{"count":1,"ship_type":"cruiser_torpedo"},{"count":2,"ship_type":"destroyer"},{"count":2,"ship_type":"frigate"}])`
  - 1x `select_fleet(ships=[{"count":4,"ship_type":"cruiser"},{"count":4,"ship_type":"frigate"}])`
- claude-opus-5-5@high: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [804, 1101, 1237], $0.1057
  - 1x `select_fleet(ships=[{"count":7,"ship_type":"destroyer"},{"count":1,"ship_type":"corvette"}])`
  - 1x `select_fleet(ships=[{"count":4,"ship_type":"corvette"},{"count":3,"ship_type":"destroyer"}])`
  - 1x `select_fleet(ships=[{"count":1,"ship_type":"cruiser_torpedo"},{"count":2,"ship_type":"destroyer"},{"count":2,"ship_type":"frigate"}])`

## p5_sonnet5_vs_sonnet55_b2:e4 T+0 cp0 draft beta (recorded claude-sonnet-5-5 @ medium)
- recorded: `select_fleet(ships=[{"count":3,"ship_type":"battlecruiser"},{"count":2,"ship_type":"destroyer"},{"count":5,"ship_type":"frigate"}])`
- claude-sonnet-5-5@medium: agree 2/3, =rec 2/3, 2 distinct, thinking tokens [0, 0, 0], $0.0207
  - 2x `select_fleet(ships=[{"count":3,"ship_type":"battlecruiser"},{"count":2,"ship_type":"destroyer"},{"count":5,"ship_type":"frigate"}])`
  - 1x `select_fleet(ships=[{"count":4,"ship_type":"battlecruiser"},{"count":2,"ship_type":"destroyer"},{"count":2,"ship_type":"frigate"}])`
- claude-sonnet-5-5@high: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [379, 398, 528], $0.0206
  - 2x `select_fleet(ships=[{"count":5,"ship_type":"battlecruiser"},{"count":1,"ship_type":"destroyer"},{"count":1,"ship_type":"frigate"}])`
  - 1x `select_fleet(ships=[{"count":5,"ship_type":"battlecruiser"},{"count":3,"ship_type":"frigate"}])`
- claude-sonnet-5@medium: agree 3/3, =rec 0/3, 1 distinct, thinking tokens [364, 637, 1133], $0.0430
  - 3x `select_fleet(ships=[{"count":6,"ship_type":"battlecruiser"}])`
- claude-opus-5-5@high: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [750, 1053, 1073], $0.0983
  - 2x `select_fleet(ships=[{"count":8,"ship_type":"destroyer"}])`
  - 1x `select_fleet(ships=[{"count":1,"ship_type":"cruiser_torpedo"},{"count":2,"ship_type":"corvette"}])`

## p5_sonnet5_vs_sonnet55_b2:e345 T+60 cp2 directive beta (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_fleet_directive()`
- claude-sonnet-5-5@medium: agree 2/3, =rec 2/3, 2 distinct, thinking tokens [0, 0, 0], $0.0925
  - 2x `set_fleet_directive()`
  - 1x `set_battle_plan() ; set_fleet_directive()`
- claude-sonnet-5-5@high: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [576, 579, 728], $0.0795
  - 1x `issue_order(heading={"x":-1,"y":0,"z":0},ship_name="all ships",throttle=0.5) ; set_fleet_directive()`
  - 1x `issue_order(heading={"x":-1,"y":0,"z":0},ship_name="all ships",throttle=1.0) ; set_fleet_directive()`
  - 1x `issue_order(heading={"x":-1,"y":0,"z":0},ship_name="all ships",throttle=0.8) ; set_battle_plan() ; set_fleet_directive()`
- claude-sonnet-5@medium: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [309, 310, 415], $0.1063
  - 3x `set_fleet_directive()`
- claude-opus-5-5@high: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [517, 586, 638], $0.2421
  - 1x `issue_order(heading={"x":-1,"y":0,"z":0.0},ship_name="all ships",throttle=0.4) ; set_fleet_directive()`
  - 1x `issue_order(evade="dodge",heading={"x":-1,"y":0,"z":0},ship_name="all ships",suggested_target="TIS Dart-3",throttle=0.4) ; set_fleet_directive()`
  - 1x `issue_order(heading={"x":-1,"y":0,"z":0.0},ship_name="all ships",throttle=0) ; set_fleet_directive()`

## p5_sonnet5_vs_sonnet55_b2:e493 T+60 cp2 captain beta_1 (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_maneuver(maneuver_type="EVADE") ; set_primary_target(target_id="alpha_3")`
- claude-sonnet-5-5@medium: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [0, 0, 0], $0.1047
  - 2x `set_maneuver(maneuver_type="EVADE") ; set_radiators(mode="auto")`
  - 1x `set_maneuver(evade="dodge",maneuver_type="PADLOCK",target_id="alpha_3")`
- claude-sonnet-5-5@high: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [272, 272, 314], $0.0531
  - 1x `set_maneuver(evade="dodge",maneuver_type="PADLOCK",target_id="alpha_3") ; set_primary_target(target_id="alpha_3")`
  - 1x `set_maneuver(evade="dodge",maneuver_type="PADLOCK",target_id="alpha_3") ; set_primary_target(target_id="alpha_3") ; set_radiators(mode="auto")`
  - 1x `set_maneuver(evade="full",maneuver_type="EVADE") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-sonnet-5@medium: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [272, 392, 570], $0.1221
  - 1x `set_maneuver(evade="dodge",maneuver_type="PADLOCK",target_id="alpha_3") ; set_primary_target(target_id="alpha_3") ; set_radiators(mode="auto") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.2,spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.2,spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-opus-5-5@high: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [547, 583, 718], $0.2702
  - 2x `log_note() ; report_to_admiral(request={"kind":"PD_COVER"},status="DEVIATE") ; set_maneuver(evade="full",maneuver_type="PADLOCK",target_id="alpha_3") ; set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.`
  - 1x `log_note() ; set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.2,spinal_mode="FIRE_WHEN_OPTIMAL")`

## p5_sonnet5_vs_sonnet55_b2:e496 T+60 cp2 captain beta_2 (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_maneuver(evade="dodge",maneuver_type="PADLOCK",target_id="alpha_3") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-sonnet-5-5@medium: agree 2/3, =rec 2/3, 2 distinct, thinking tokens [0, 0, 0], $0.0380
  - 2x `set_maneuver(evade="dodge",maneuver_type="PADLOCK",target_id="alpha_3") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_maneuver(evade="dodge",maneuver_type="PADLOCK") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-sonnet-5-5@high: agree 3/3, =rec 0/3, 1 distinct, thinking tokens [160, 191, 243], $0.0458
  - 3x `set_maneuver(evade="dodge",maneuver_type="PADLOCK",target_id="alpha_3") ; set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-sonnet-5@medium: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [0, 0, 0], $0.0469
  - 1x `set_maneuver(maneuver_type="PADLOCK",target_id="alpha_3") ; set_primary_target(target_id="alpha_3") ; set_radiators(mode="auto") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_maneuver(maneuver_type="PADLOCK",target_id="alpha_3") ; set_primary_target(target_id="alpha_3") ; set_radiators(mode="auto") ; set_weapons_order(coilguns_min_probability=0.3,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.3,spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_primary_target(target_id="alpha_3") ; set_radiators(mode="auto") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.2,spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-opus-5-5@high: agree 1/1, =rec 0/1, 1 distinct, thinking tokens [258], $0.0354, 2 errors
  - 1x `log_note() ; set_maneuver(evade="dodge",maneuver_type="PADLOCK",target_id="alpha_3") ; set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_min_probability=0.3,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.3,spinal_mode="FIRE_WHEN_OPTIMAL")`

## p5_sonnet5_vs_sonnet55_b2:e494 T+60 cp2 captain beta_3 (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_maneuver(evade="dodge",maneuver_type="PADLOCK") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-sonnet-5-5@medium: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [0, 0, 0], $0.0369
  - 3x `set_maneuver(evade="dodge",maneuver_type="PADLOCK") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-sonnet-5-5@high: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [104, 195, 213], $0.0439
  - 2x `set_maneuver(evade="dodge",maneuver_type="PADLOCK",target_id="alpha_3") ; set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_primary_target(target_id="alpha_3") ; set_radiators(mode="auto") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-sonnet-5@medium: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [0, 0, 0], $0.0439
  - 2x `set_primary_target(target_id="alpha_3") ; set_radiators(mode="auto") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_primary_target(target_id="alpha_3") ; set_radiators(mode="auto") ; set_weapons_order(coilguns_min_probability=0.3,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.3,spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-opus-5-5@high: no reply (3 errors)

## p5_sonnet5_vs_sonnet55_b2:e994 T+90 cp3 directive beta (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_fleet_directive()`
- claude-sonnet-5-5@medium: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [0, 0, 0], $0.0544
  - 3x `set_fleet_directive()`
- claude-sonnet-5-5@high: agree 3/3, =rec 0/3, 1 distinct, thinking tokens [430, 573, 597], $0.0780
  - 3x `set_battle_plan() ; set_fleet_directive()`
- claude-sonnet-5@medium: agree 1/3, =rec 1/3, 3 distinct, thinking tokens [314, 463, 509], $0.0802
  - 1x `issue_order(heading={"x":-1.0,"y":0.0,"z":0.0},ship_name="all ships",throttle=0) ; set_fleet_directive()`
  - 1x `set_fleet_directive()`
  - 1x `set_battle_plan() ; set_fleet_directive()`
- claude-opus-5-5@high: no reply (3 errors)

## p5_sonnet5_vs_sonnet55_b2:e2692 T+150 cp5 captain beta_3 (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_heading(direction={"x":-0.9,"y":-0.3,"z":0.1},evade="\"dodge\"",throttle=1.0) ; set_radiators(mode="auto")`
- claude-sonnet-5-5@medium: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [0, 0, 0], $0.0373
  - 2x `set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_heading(direction={"x":-0.9,"y":-0.3,"z":0.1},evade="dodge",throttle=1.0)`
- claude-sonnet-5-5@high: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [156, 168, 281], $0.0430
  - 1x `set_heading(direction={"x":-0.9,"y":-0.3,"z":0.1},evade="dodge",throttle=1.0) ; set_primary_target(target_id="alpha_3")`
  - 1x `log_note() ; set_primary_target(target_id="alpha_3")`
  - 1x `log_note() ; set_primary_target(target_id="alpha_3") ; set_radiators(mode="auto")`
- claude-sonnet-5@medium: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [447, 553, 698], $0.0597
  - 1x `set_primary_target(target_id="alpha_3") ; set_radiators(mode="auto") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_heading(direction={"x":-0.9,"y":-0.3,"z":0.1},evade="dodge",throttle=1.0) ; set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-opus-5-5@high: no reply (3 errors)

## p5_sonnet5_vs_sonnet55_b2:e2698 T+150 cp5 captain beta_4 (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_heading(direction={"x":-1.0,"y":0.1,"z":0.0},evade="\"dodge\"",throttle=1.0) ; set_radiators(mode="auto")`
- claude-sonnet-5-5@medium: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [0, 0, 0], $0.0360
  - 1x `set_radiators(mode="auto") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_primary_target(target_id="alpha_3") ; set_radiators(mode="auto")`
  - 1x `set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-sonnet-5-5@high: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [202, 266, 317], $0.0442
  - 1x `set_heading(direction={"x":-1.0,"y":0.1,"z":0.0},evade="full",throttle=1.0) ; set_radiators(mode="auto")`
  - 1x `set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `log_note() ; set_heading(direction={"x":-1.0,"y":0.1,"z":0.0},evade="dodge",throttle=1.0) ; set_primary_target(target_id="alpha_3")`
- claude-sonnet-5@medium: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [0, 0, 227], $0.0437
  - 2x `set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_heading(direction={"x":-1.0,"y":0.1,"z":0.0},throttle=1.0) ; set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-opus-5-5@high: no reply (3 errors)

## p5_sonnet5_vs_sonnet55_d:e4 T+0 cp0 draft beta (recorded claude-sonnet-5-5 @ medium)
- recorded: `select_fleet(ships=[{"count":3,"ship_type":"battlecruiser"},{"count":2,"ship_type":"destroyer"},{"count":5,"ship_type":"frigate"}])`
- claude-sonnet-5-5@medium: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [0, 0, 0], $0.0079
  - 3x `select_fleet(ships=[{"count":3,"ship_type":"battlecruiser"},{"count":2,"ship_type":"destroyer"},{"count":5,"ship_type":"frigate"}])`
- claude-sonnet-5-5@high: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [350, 424, 515], $0.0201
  - 2x `select_fleet(ships=[{"count":5,"ship_type":"battlecruiser"},{"count":3,"ship_type":"frigate"}])`
  - 1x `select_fleet(ships=[{"count":5,"ship_type":"battlecruiser"},{"count":1,"ship_type":"destroyer"},{"count":1,"ship_type":"frigate"}])`
- claude-sonnet-5@medium: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [1086, 1902, 2137], $0.0612
  - 2x `select_fleet(ships=[{"count":3,"ship_type":"battleship"},{"count":3,"ship_type":"frigate"}])`
  - 1x `select_fleet(ships=[{"count":6,"ship_type":"battlecruiser"}])`
- claude-opus-5-5@high: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [759, 837, 1105], $0.0708
  - 1x `select_fleet(ships=[{"count":5,"ship_type":"corvette"},{"count":3,"ship_type":"frigate"}])`
  - 1x `select_fleet(ships=[{"count":7,"ship_type":"destroyer"},{"count":1,"ship_type":"corvette"}])`
  - 1x `select_fleet(ships=[{"count":5,"ship_type":"battlecruiser"},{"count":1,"ship_type":"destroyer"},{"count":1,"ship_type":"frigate"}])`

## p5_sonnet5_vs_sonnet55_d:e4868 T+240 cp8 directive beta (recorded claude-sonnet-5-5 @ medium)
- recorded: `issue_order(heading={"x":-0.0,"y":-1.0,"z":-0.0},ship_name="all ships",throttle=1.0) ; set_fleet_directive()`
- claude-sonnet-5-5@medium: agree 2/3, =rec 1/3, 2 distinct, thinking tokens [0, 0, 0], $0.0871
  - 2x `issue_order(heading={"x":-0.0,"y":-1,"z":-0.0},ship_name="all ships",throttle=1) ; set_fleet_directive()`
  - 1x `issue_order(heading={"x":-0.0,"y":-1.0,"z":-0.0},ship_name="all ships",throttle=1.0) ; set_fleet_directive()`
- claude-sonnet-5-5@high: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [565, 593, 666], $0.0687
  - 2x `set_battle_plan() ; set_fleet_directive()`
  - 1x `set_fleet_directive()`
- claude-sonnet-5@medium: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [573, 1088, 1717], $0.1258
  - 2x `issue_order(heading={"x":0.0,"y":1.0,"z":-0.0},ship_name="all ships",throttle=1.0) ; set_fleet_directive()`
  - 1x `set_fleet_directive()`
- claude-opus-5-5@high: agree 3/3, =rec 0/3, 1 distinct, thinking tokens [762, 1142, 1302], $0.2557
  - 3x `set_battle_plan() ; set_fleet_directive()`

## p5_sonnet5_vs_sonnet55_d:e4964 T+240 cp8 captain beta_1 (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_heading(direction={"x":-0.1,"y":1.0,"z":-0.0},evade="dodge",throttle=1.0) ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-sonnet-5-5@medium: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [317, 335, 387], $0.0518
  - 1x `report_to_admiral(status="DEVIATE") ; set_maneuver(maneuver_type="PADLOCK") ; set_weapons_order(coilguns_mode="FIRE_IMMEDIATE",spinal_mode="FIRE_IMMEDIATE")`
  - 1x `set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_heading(direction={"x":-0.1,"y":1.0,"z":-0.0},evade="dodge",throttle=1.0) ; set_weapons_order(coilguns_min_probability=0.1,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-sonnet-5-5@high: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [716, 842, 864], $0.0694
  - 1x `report_to_admiral(status="DEVIATE") ; set_maneuver(evade="dodge",maneuver_type="PADLOCK") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.2,spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `report_to_admiral(status="DEVIATE") ; set_maneuver(evade="dodge",maneuver_type="PADLOCK") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `report_to_admiral(status="DEVIATE") ; set_maneuver(maneuver_type="PADLOCK") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.2,spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-sonnet-5@medium: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [875, 1195, 1198], $0.0796
  - 1x `log_note() ; set_primary_target(target_id="alpha_1") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.3,spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_primary_target(target_id="alpha_1") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_heading(direction={"x":-0.1,"y":1.0,"z":-0.0},evade="dodge",throttle=1.0) ; set_primary_target(target_id="alpha_1") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-opus-5-5@high: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [1090, 1255, 1807], $0.1827
  - 1x `log_note() ; report_to_admiral(status="DEVIATE") ; set_maneuver(evade="dodge",maneuver_type="PADLOCK",target_id="alpha_1") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.3,spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `log_note() ; set_heading(direction={"x":-0.1,"y":1.0,"z":-0.0},evade="dodge",throttle=1.0) ; set_primary_target(target_id="alpha_1") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.3,spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `log_note() ; set_heading(direction={"x":-0.1,"y":1.0,"z":-0.0},evade="dodge",throttle=1.0) ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.3,spinal_mode="FIRE_WHEN_OPTIMAL")`

## p5_sonnet5_vs_sonnet55_d:e4967 T+240 cp8 captain beta_2 (recorded claude-sonnet-5-5 @ medium)
- recorded: `report_to_admiral(status="DEVIATE") ; set_maneuver(evade="dodge",maneuver_type="PADLOCK") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_IMMEDIATE")`
- claude-sonnet-5-5@medium: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [121, 248, 307], $0.0453
  - 1x `set_primary_target(target_id="alpha_1")`
  - 1x `report_to_admiral(status="DEVIATE") ; set_maneuver(evade="dodge",maneuver_type="PADLOCK")`
  - 1x `set_primary_target(target_id="alpha_1") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-sonnet-5-5@high: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [526, 531, 554], $0.0606
  - 1x `report_to_admiral(status="DEVIATE") ; set_maneuver(maneuver_type="PADLOCK",target_id="alpha_1") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.2,spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `report_to_admiral(status="DEVIATE") ; set_maneuver(evade="dodge",maneuver_type="PADLOCK",target_id="alpha_1") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.3,spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_primary_target(target_id="alpha_1") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-sonnet-5@medium: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [789, 863, 1079], $0.0754
  - 1x `set_primary_target(target_id="alpha_1") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `report_to_admiral(status="DEVIATE") ; set_heading(direction={"x":-0.2,"y":-1.0,"z":0.0},throttle=1.0) ; set_primary_target(target_id="alpha_1") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `log_note() ; set_primary_target(target_id="alpha_1") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-opus-5-5@high: no reply (3 errors)

## p5_sonnet5_vs_sonnet55_d:e4972 T+240 cp8 captain beta_3 (recorded claude-sonnet-5-5 @ medium)
- recorded: `report_to_admiral(status="DEVIATE") ; set_maneuver(evade="dodge",maneuver_type="PADLOCK")`
- claude-sonnet-5-5@medium: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [220, 245, 302], $0.0449
  - 1x `set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_primary_target(target_id="alpha_1")`
  - 1x `set_primary_target(target_id="alpha_1") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-sonnet-5-5@high: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [478, 694, 709], $0.0637
  - 2x `report_to_admiral(status="DEVIATE") ; set_maneuver(maneuver_type="PADLOCK",target_id="alpha_1") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `report_to_admiral(status="DEVIATE") ; set_maneuver(evade="dodge",maneuver_type="PADLOCK") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-sonnet-5@medium: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [1011, 1188, 1318], $0.0859
  - 2x `report_to_admiral(status="DEVIATE") ; set_heading(direction={"x":0.0,"y":-1.0,"z":-0.0},throttle=1.0) ; set_primary_target(target_id="alpha_1") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `set_primary_target(target_id="alpha_1") ; set_radiators(mode="auto") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-opus-5-5@high: no reply (3 errors)

## p5_sonnet5_vs_sonnet55_d:e14576 T+390 cp13 directive beta (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_fleet_directive()`
- claude-sonnet-5-5@medium: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [0, 0, 0], $0.0466
  - 3x `set_fleet_directive()`
- claude-sonnet-5-5@high: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [309, 324, 368], $0.0580
  - 3x `set_fleet_directive()`
- claude-sonnet-5@medium: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [640, 728, 1257], $0.0844
  - 1x `issue_order(heading={"x":1.0,"y":0.2,"z":0.0},ship_name="all ships",throttle=1.0) ; set_fleet_directive()`
  - 1x `issue_order(heading={"x":1.0,"y":0.2,"z":-0.0},ship_name="all ships",throttle=1.0) ; set_fleet_directive()`
  - 1x `issue_order(heading={"x":1.0,"y":0.2,"z":0},ship_name="all ships",throttle=1.0) ; set_fleet_directive()`
- claude-opus-5-5@high: no reply (3 errors)
