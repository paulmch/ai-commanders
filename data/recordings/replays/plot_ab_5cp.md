# Replay comparison

Signature = tool names + discrete arguments (free text excluded). `agree` = replicates whose signature equals the variant's most common one; `=rec` = replicates matching the recorded reply.

## p5w5_b2_same_fleets_score_shown:e11723 T+330 cp11 directive alpha (recorded claude-sonnet-5-5 @ high)
- recorded: `set_battle_plan() ; set_fleet_directive()`
- claude-sonnet-5-5@high: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [152, 238, 260], $0.1004
  - 3x `set_battle_plan() ; set_fleet_directive()`
- claude-sonnet-5-5@high+replot: agree 2/3, =rec 1/3, 2 distinct, thinking tokens [182, 212, 264], $0.0978
  - 2x `set_fleet_directive()`
  - 1x `set_battle_plan() ; set_fleet_directive()`
- claude-sonnet-5-5@high+noimage: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [97, 152, 218], $0.0886
  - 3x `set_battle_plan() ; set_fleet_directive()`
- claude-opus-5-5@high: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [191, 212, 261], $0.2121
  - 3x `set_battle_plan() ; set_fleet_directive()`
- claude-opus-5-5@high+replot: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [180, 355, 388], $0.2216
  - 3x `set_battle_plan() ; set_fleet_directive()`
- claude-opus-5-5@high+noimage: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [146, 202, 216], $0.1926
  - 3x `set_battle_plan() ; set_fleet_directive()`

## p5w5_b2_same_fleets_score_shown:e13795 T+360 cp12 directive beta (recorded claude-sonnet-5 @ medium)
- recorded: `set_battle_plan() ; set_fleet_directive()`
- claude-sonnet-5-5@high: agree 2/3, =rec 2/3, 2 distinct, thinking tokens [555, 592, 1056], $0.1280
  - 2x `set_battle_plan() ; set_fleet_directive()`
  - 1x `issue_order(evade="dodge",heading={"x":1.0,"y":-0.0,"z":-0.1},ship_name="all ships",throttle=1.0) ; set_battle_plan() ; set_fleet_directive()`
- claude-sonnet-5-5@high+replot: agree 2/3, =rec 2/3, 2 distinct, thinking tokens [553, 654, 842], $0.1303
  - 2x `set_battle_plan() ; set_fleet_directive()`
  - 1x `issue_order(evade="dodge",heading={"x":-0.4,"y":-0.5,"z":0.8},ship_name="all ships",throttle=1.0) ; set_battle_plan() ; set_fleet_directive()`
- claude-sonnet-5-5@high+noimage: agree 1/3, =rec 1/3, 3 distinct, thinking tokens [492, 584, 651], $0.1129
  - 1x `issue_order(heading={"x":-0.4,"y":-0.5,"z":0.8},ship_name="all ships",throttle=1.0) ; set_battle_plan() ; set_fleet_directive()`
  - 1x `set_battle_plan() ; set_fleet_directive()`
  - 1x `set_fleet_directive()`
- claude-opus-5-5@high: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [484, 573, 709], $0.2543
  - 3x `set_battle_plan() ; set_fleet_directive()`
- claude-opus-5-5@high+replot: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [704, 748, 793], $0.2722
  - 3x `set_battle_plan() ; set_fleet_directive()`
- claude-opus-5-5@high+noimage: agree 1/3, =rec 1/3, 3 distinct, thinking tokens [1115, 1164, 1391], $0.2898
  - 1x `issue_order(evade="dodge",heading={"x":-0.4,"y":-0.5,"z":0.8},ship_name="all ships",suggested_target="TIS Ward-2",throttle=1.0) ; set_battle_plan() ; set_fleet_directive()`
  - 1x `set_battle_plan() ; set_fleet_directive()`
  - 1x `issue_order(evade="dodge",heading={"x":1.0,"y":-0.0,"z":-0.1},ship_name="all ships",suggested_target="TIS Ward-2",throttle=1.0) ; set_battle_plan() ; set_fleet_directive()`

## p5w5_b2_same_fleets_score_shown:e22543 T+540 cp18 directive alpha (recorded claude-sonnet-5-5 @ high)
- recorded: `set_fleet_directive()`
- claude-sonnet-5-5@high: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [0, 0, 0], $0.0471
  - 3x `set_fleet_directive()`
- claude-sonnet-5-5@high+replot: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [0, 0, 0], $0.0477
  - 3x `set_fleet_directive()`
- claude-sonnet-5-5@high+noimage: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [0, 0, 0], $0.0393
  - 3x `set_fleet_directive()`
- claude-opus-5-5@high: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [0, 68, 71], $0.1035
  - 3x `set_fleet_directive()`
- claude-opus-5-5@high+replot: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [117, 129, 146], $0.1101
  - 3x `set_fleet_directive()`
- claude-opus-5-5@high+noimage: agree 3/3, =rec 3/3, 1 distinct, thinking tokens [65, 75, 83], $0.0915
  - 3x `set_fleet_directive()`

## p5w5_b3_opus_admiral_haiku_captains_vs_sonnet55:e909 T+90 cp3 directive alpha (recorded claude-opus-5-5 @ high)
- recorded: `set_fleet_directive(salvo_plan=[{"rounds":0,"ship_name":"TIS Dart-1"},{"rounds":0,"ship_name":"TIS Dart-2"},{"rounds":0,"ship_name":"TIS Dart-3"},{"rounds":0,"ship_name":"TIS Dart-4"},{"rounds":0,"ship_name":"TIS Dart-5"}])`
- claude-sonnet-5-5@high: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [173, 208, 208], $0.1142
  - 2x `set_battle_plan() ; set_fleet_directive()`
  - 1x `set_fleet_directive()`
- claude-sonnet-5-5@high+replot: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [144, 178, 205], $0.1190
  - 2x `set_battle_plan() ; set_fleet_directive(salvo_plan=[])`
  - 1x `set_battle_plan() ; set_fleet_directive()`
- claude-sonnet-5-5@high+noimage: agree 3/3, =rec 0/3, 1 distinct, thinking tokens [282, 285, 361], $0.1047
  - 3x `set_fleet_directive()`
- claude-opus-5-5@high: agree 2/3, =rec 2/3, 2 distinct, thinking tokens [271, 279, 288], $0.2379
  - 2x `set_fleet_directive(salvo_plan=[{"rounds":0,"ship_name":"TIS Dart-1"},{"rounds":0,"ship_name":"TIS Dart-2"},{"rounds":0,"ship_name":"TIS Dart-3"},{"rounds":0,"ship_name":"TIS Dart-4"},{"rounds":0,"ship_name":"TIS Dart-5"}])`
  - 1x `set_fleet_directive(salvo_plan=[])`
- claude-opus-5-5@high+replot: agree 2/3, =rec 2/3, 2 distinct, thinking tokens [309, 315, 418], $0.2472
  - 2x `set_fleet_directive(salvo_plan=[{"rounds":0,"ship_name":"TIS Dart-1"},{"rounds":0,"ship_name":"TIS Dart-2"},{"rounds":0,"ship_name":"TIS Dart-3"},{"rounds":0,"ship_name":"TIS Dart-4"},{"rounds":0,"ship_name":"TIS Dart-5"}])`
  - 1x `set_fleet_directive(salvo_plan=[])`
- claude-opus-5-5@high+noimage: agree 1/3, =rec 1/3, 3 distinct, thinking tokens [312, 331, 696], $0.2446
  - 1x `set_fleet_directive(salvo_plan=[{"rounds":0,"ship_name":"TIS Dart-1"},{"rounds":0,"ship_name":"TIS Dart-2"},{"rounds":0,"ship_name":"TIS Dart-3"},{"rounds":0,"ship_name":"TIS Dart-4"},{"rounds":0,"ship_name":"TIS Dart-5"}])`
  - 1x `set_battle_plan() ; set_fleet_directive(salvo_plan=[{"rounds":0,"ship_name":"TIS Dart-1"},{"rounds":0,"ship_name":"TIS Dart-2"},{"rounds":0,"ship_name":"TIS Dart-3"},{"rounds":0,"ship_name":"TIS Dart-4"},{"rounds":0,"ship_name":"TIS Dart-5"}])`
  - 1x `set_battle_plan() ; set_fleet_directive(salvo_plan=[{"rounds":3,"ship_name":"TIS Dart-3","target":"OCS Halberd-3"},{"rounds":3,"ship_name":"TIS Dart-4","target":"OCS Halberd-3"},{"rounds":0,"ship_name":"TIS Dart-1"},{"rounds":0,"ship_name":"TIS Dart-2"},{"rounds":0,"ship_name":"TIS Dart-5"}])`

## p5w5_b3_opus_admiral_haiku_captains_vs_sonnet55:e1776 T+120 cp4 directive beta (recorded claude-sonnet-5-5 @ high)
- recorded: `issue_order(evade="full",heading={"x":1.0,"y":0.0,"z":-0.0},ship_name="all ships",throttle=1.0) ; set_battle_plan() ; set_fleet_directive()`
- claude-sonnet-5-5@high: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [627, 676, 1084], $0.1427
  - 1x `issue_order(heading={"x":1.0,"y":0.0,"z":-0.0},ship_name="all ships",throttle=1.0) ; set_battle_plan() ; set_fleet_directive()`
  - 1x `issue_order(heading={"x":-1.0,"y":0.0,"z":0.0},ship_name="all ships",throttle=1.0) ; set_battle_plan() ; set_fleet_directive()`
  - 1x `issue_order(evade="dodge",heading={"x":1.0,"y":0.0,"z":-0.0},ship_name="all ships",throttle=1.0) ; set_battle_plan() ; set_fleet_directive()`
- claude-sonnet-5-5@high+replot: agree 1/3, =rec 1/3, 3 distinct, thinking tokens [280, 361, 525], $0.1332
  - 1x `issue_order(heading={"x":1,"y":0,"z":-0.0},ship_name="all ships",throttle=1.0) ; set_battle_plan() ; set_fleet_directive()`
  - 1x `issue_order(evade="full",heading={"x":1.0,"y":0.0,"z":-0.0},ship_name="all ships",throttle=1.0) ; set_battle_plan() ; set_fleet_directive()`
  - 1x `issue_order(evade="full",heading={"x":1.0,"y":-0.1,"z":0.0},priority="HIGH",ship_name="all ships",throttle=1.0) ; set_battle_plan() ; set_fleet_directive()`
- claude-sonnet-5-5@high+noimage: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [839, 854, 865], $0.1335
  - 2x `set_battle_plan() ; set_fleet_directive()`
  - 1x `issue_order(evade="dodge",heading={"x":-1,"y":0,"z":0},ship_name="all ships",throttle=1.0) ; set_battle_plan() ; set_fleet_directive()`
- claude-opus-5-5@high: agree 1/3, =rec 0/3, 3 distinct, thinking tokens [934, 1069, 1076], $0.3174
  - 1x `issue_order(evade="dodge",heading={"x":-1.0,"y":-0.1,"z":-0.1},ship_name="all ships",suggested_target="TIS Dart-4",throttle=1.0) ; set_battle_plan() ; set_fleet_directive()`
  - 1x `issue_order(evade="dodge",heading={"x":-1,"y":0,"z":0},ship_name="all ships",suggested_target="TIS Dart-4",throttle=1.0) ; set_battle_plan() ; set_fleet_directive()`
  - 1x `issue_order(evade="dodge",heading={"x":-1,"y":0,"z":0},ship_name="all ships",throttle=0.3) ; set_battle_plan() ; set_fleet_directive()`
- claude-opus-5-5@high+replot: agree 2/3, =rec 0/3, 2 distinct, thinking tokens [918, 1300, 1465], $0.3315
  - 2x `issue_order(evade="dodge",heading={"x":-1,"y":0,"z":0},ship_name="all ships",throttle=1.0) ; set_battle_plan() ; set_fleet_directive()`
  - 1x `issue_order(evade="dodge",heading={"x":-1,"y":0,"z":0},ship_name="all ships",suggested_target="TIS Dart-4",throttle=1) ; set_battle_plan() ; set_fleet_directive()`
- claude-opus-5-5@high+noimage: agree 3/3, =rec 0/3, 1 distinct, thinking tokens [781, 827, 1081], $0.2879
  - 3x `issue_order(evade="dodge",heading={"x":-1,"y":0,"z":0},ship_name="all ships",throttle=1.0) ; set_battle_plan() ; set_fleet_directive()`
