# Pass 5: recording additions

## `torpedo_miss.reason` (lane `hunt`)

- **Event:** `torpedo_miss` (existing event type).
- **New field:** `reason`, a string, present only when set. Its one value so far is
  `"left_battle"`.
- **When it is emitted:** the engine culls a torpedo once it is more than 10,000 km from
  its target or has flown for more than an hour. Before pass 5 the cull removed the round
  without any event. A long battle could lose a tenth of its torpedoes this way: in seed
  8, budget 200, 900 s, 9 of 78 rounds disappeared, all of them seeker-killed and coasting
  away. A culled round now records a `torpedo_miss` with `reason: "left_battle"`. Its
  `closest_approach_km` (km) is the closest the round came to its target, and its
  `hit_probability` is 0.
- **Consumers:** `launched = impact + miss + intercepted + in_flight_at_end` now balances
  per side. Viewers that ignore the field are unaffected.
