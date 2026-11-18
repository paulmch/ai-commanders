# Replay comparison

Signature = tool names + discrete arguments (free text excluded). `agree` = replicates whose signature equals the variant's most common one; `=rec` = replicates matching the recorded reply.

## p5_sonnet5_vs_sonnet55_b2:e494 T+60 cp2 captain beta_3 (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_maneuver(evade="dodge",maneuver_type="PADLOCK") ; set_weapons_order(coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_WHEN_OPTIMAL")`
- claude-opus-5-5@high: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [217, 287, 287], $0.1057
  - 1x `log_note() ; set_maneuver(evade="dodge",maneuver_type="PADLOCK",target_id="alpha_3") ; set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_min_probability=0.3,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.2,spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `log_note() ; set_maneuver(evade="dodge",maneuver_type="PADLOCK",target_id="alpha_3") ; set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_min_probability=0.3,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.3,spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `log_note() ; set_maneuver(evade="dodge",maneuver_type="PADLOCK",target_id="alpha_3") ; set_weapons_order(coilguns_min_probability=0.3,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.2,spinal_mode="FIRE_WHEN_OPTIMAL")`

## p5_sonnet5_vs_sonnet55_b2:e994 T+90 cp3 directive beta (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_fleet_directive()`
- claude-opus-5-5@high: agree 2/3, =rec 1/3, 2 distinct, thinking tokens [197, 316, 355], $0.1509
  - 2x `set_battle_plan() ; set_fleet_directive()`
  - 1x `set_fleet_directive()`

## p5_sonnet5_vs_sonnet55_b2:e2692 T+150 cp5 captain beta_3 (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_heading(direction={"x":-0.9,"y":-0.3,"z":0.1},evade="\"dodge\"",throttle=1.0) ; set_radiators(mode="auto")`
- claude-opus-5-5@high: agree 3/3, =rec 0/3, 1 distinct, thinking tokens [166, 187, 190], $0.0954
  - 3x `log_note() ; set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_min_probability=0.3,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.3,spinal_mode="FIRE_WHEN_OPTIMAL")`

## p5_sonnet5_vs_sonnet55_b2:e2698 T+150 cp5 captain beta_4 (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_heading(direction={"x":-1.0,"y":0.1,"z":0.0},evade="\"dodge\"",throttle=1.0) ; set_radiators(mode="auto")`
- claude-opus-5-5@high: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [169, 200, 371], $0.0986
  - 2x `log_note() ; set_primary_target(target_id="alpha_3") ; set_weapons_order(coilguns_min_probability=0.3,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.3,spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `log_note() ; set_heading(direction={"x":-1.0,"y":0.1,"z":0.0},evade="dodge",throttle=1.0) ; set_weapons_order(coilguns_min_probability=0.3,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.3,spinal_mode="FIRE_WHEN_OPTIMAL")`

## p5_sonnet5_vs_sonnet55_d:e4967 T+240 cp8 captain beta_2 (recorded claude-sonnet-5-5 @ medium)
- recorded: `report_to_admiral(status="DEVIATE") ; set_maneuver(evade="dodge",maneuver_type="PADLOCK") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_mode="FIRE_IMMEDIATE")`
- claude-opus-5-5@high: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [790, 801, 984], $0.1530
  - 1x `log_note() ; report_to_admiral(status="DEVIATE") ; set_maneuver(evade="dodge",maneuver_type="PADLOCK",target_id="alpha_1") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.2,spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `log_note() ; report_to_admiral(status="DEVIATE") ; set_maneuver(evade="dodge",maneuver_type="PADLOCK",target_id="alpha_1") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.3,spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `log_note() ; set_primary_target(target_id="alpha_1") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.3,spinal_mode="FIRE_WHEN_OPTIMAL")`

## p5_sonnet5_vs_sonnet55_d:e4972 T+240 cp8 captain beta_3 (recorded claude-sonnet-5-5 @ medium)
- recorded: `report_to_admiral(status="DEVIATE") ; set_maneuver(evade="dodge",maneuver_type="PADLOCK")`
- claude-opus-5-5@high: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [567, 685, 732], $0.1350
  - 1x `log_note() ; set_primary_target(target_id="alpha_1") ; set_radiators(auto_extend_pct=65,mode="auto") ; set_weapons_order(coilguns_min_probability=0.1,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.2,spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `log_note() ; set_primary_target(target_id="alpha_1") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.3,spinal_mode="FIRE_WHEN_OPTIMAL")`
  - 1x `log_note() ; set_heading(direction={"x":-0.0,"y":-1.0,"z":-0.0},evade="dodge",throttle=1.0) ; set_primary_target(target_id="alpha_1") ; set_weapons_order(coilguns_min_probability=0.2,coilguns_mode="FIRE_WHEN_OPTIMAL",spinal_min_probability=0.2,spinal_mode="FIRE_WHEN_OPTIMAL")`

## p5_sonnet5_vs_sonnet55_d:e14576 T+390 cp13 directive beta (recorded claude-sonnet-5-5 @ medium)
- recorded: `set_fleet_directive()`
- claude-opus-5-5@high: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [402, 419, 561], $0.1554
  - 1x `issue_order(heading={"x":1.0,"y":0.2,"z":0},ship_name="all ships",throttle=1.0) ; set_fleet_directive()`
  - 1x `issue_order(evade="dodge",heading={"x":1.0,"y":0.2,"z":0},ship_name="all ships",suggested_target="TIS Lancer-2",throttle=0.6) ; set_battle_plan() ; set_fleet_directive()`
  - 1x `issue_order(heading={"x":1.0,"y":0.2,"z":0},ship_name="all ships",throttle=1.0) ; set_battle_plan() ; set_fleet_directive()`
