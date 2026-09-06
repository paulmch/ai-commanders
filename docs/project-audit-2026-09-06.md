# Project audit — September 6, 2026

This pass checked battle configuration, the information and tools given to
commanders, projectile targeting and collision detection, thrust vectoring,
torpedo launch execution, and battle outcomes. Regression cases are in
`tests/test_project_audit.py`.

## Findings and fixes

| Area | Problem | Resulting behavior |
| --- | --- | --- |
| Hit detection | Sweeping a projectile against the target's final world position could turn a receding miss into a hit after adding a common velocity. | Both coarse and fine collision sweeps use target-relative motion and restore the impact coordinates to world space. |
| Torpedo salvos | Captain commands all executed immediately; rounds requested during reload were silently lost. Capacity also consulted only one tube's magazine. | Ready tubes launch together, remaining rounds wait for reload, and reservations expire at the next decision or when the target is lost. Capacity accounts for every tube and its magazine/cooldown. A ready corvette can launch at 0/12/24 seconds; a torpedo cruiser can launch four at each of those times. |
| Targeting | Fixed-point lead iteration failed to converge at high relative velocities. Fire control could recommend shots at targets moving sideways faster than the projectile. | A shared solver finds the earliest positive intercept, including constant target acceleration. Fire control rejects impossible constant-velocity intercepts. Turret lead directions respect their arcs. |
| Weapon resources | Failed launches could still consume ammunition, capacitor charge and cooldown. | Resources are committed only after a successful launch. Recorded gun ETAs use the firing solution. |
| Target changes | Captain gun orders captured the old primary target indefinitely. | Persistent orders follow the ship's current primary target; changing it redirects the guns without requiring new firing policies. |
| Battle outcomes | Healthy torpedo-only hulls were considered disabled because they had no guns. Simultaneous surrender favored Beta. | Loaded torpedo launchers count as offensive capability, and simultaneous surrender produces a draw. |
| Thrust vectoring | Fuel depletion limited linear impulse but allowed torque for the entire timestep. Torque also bypassed the linear thrust gimbal limits. | Angular impulse uses actual burn duration and the same gimbal limits as thrust. |
| Fleet options | The regular launcher ignored fleet JSON flags for unlimited mode, personality selection and recording. | A shared runtime-config factory carries fleet settings through; explicit CLI values override them. JSON also accepts a checkpoint limit and combat seed. |
| Configuration validation | Duplicate IDs could overwrite ships; nonfinite or invalid timing values were accepted. Partial positions replaced unspecified formation axes with zero. | Invalid settings and duplicate IDs are rejected, and unspecified position axes retain formation defaults. |
| Checkpoint timing | Fractional intervals were truncated; captain and MCP maneuvers expired at 30 seconds regardless of the configured interval. | All runner loops reach fractional checkpoints/time limits exactly. Maneuver duration, prompt timing and salvo capacity use the actual decision window. |
| Prompt accuracy | Captains were told drifting lost momentum and that all orders reset. Admiral doctrine said deployed radiators had no damage penalty. Some firing-range defaults disagreed with execution. | Prompts describe coasting, persistent gun policies, radiator exposure and the actual defaults. |
| Duel options | Explicit captain/ship names were ignored. Heuristic captains and notebooks worked through the fleet factory but not the duel setup. | Names are honored; duel setup uses the same captain factory. Fully heuristic CLI battles need no model API key. |

## Verification

- The original suite passed all 1,632 tests before changes.
- Added 45 regressions covering real command execution, physical invariants,
  model input construction, CLI option precedence, launch cancellation, and
  fractional checkpoint timing.
- The final Python suite passed all 1,677 tests. All 11 viewer tests and the
  production viewer build also passed; the build retains its existing bundle
  size advisory.
- Ran gun duels, torpedo duels, and mixed four-versus-four fleets for 240 seconds
  at 30-, 60-, and 45-second decision intervals. Repeating each with the same
  seed produced identical event logs. Ship positions, velocities, heat and mass
  remained finite; ammunition and propellant stayed nonnegative.
- The mixed-fleet recording contained 240 simulation frames and parsed
  successfully. Existing combat, doctrine, damage, PD, MCP and recording tests
  are included in the full regression suite.

## Scope of the result

The existing probabilistic dispersion envelope remains part of gun combat:
passing within the 500 m tolerance is an opportunity for the launch-time hit
probability to resolve, not a literal hull intersection. Ship motion still uses
the existing Euler integrator and normally one-second simulation steps.

Commander requests were tested with local stubs and heuristic captains. This
pass did not call paid models or evaluate the quality of a particular model's
tactical decisions.
