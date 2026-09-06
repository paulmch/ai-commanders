# Battle stories

Historical matches and commander dialogue from earlier versions of AI Commanders.
Balance, ship costs, prompts, and simulation behavior have changed since these
recordings; the observations below describe those runs. For current examples,
see the [README](../README.md#recorded-battles) and [intent replay guide](intent-replays.md).

## Battle Results: Human (MCP) vs Gemini (2026-01-17)

| Fleet | Controller | Ships | Result |
|-------|------------|-------|--------|
| Alpha | Human via MCP (with Claude Opus 4.5 as copilot) | 2 Destroyers, 1 Dreadnought | **VICTORY** (3/3 ships) |
| Beta | Gemini 3 Pro (OpenRouter) | 2 Destroyers, 1 Dreadnought | Eliminated (0/3 ships) |

- **Duration**: 990s (16.5 minutes)
- **Outcome**: Beta fleet eliminated
- **Notable**: Gemini used smart evasive tactics while focusing fire on alpha_1, but was overwhelmed by coordinated intercept + fire orders
- **Key lesson**: Remember to set throttle on EVASIVE maneuvers and re-issue orders every turn!

## Battle Results: The Notebook Wars - Sonnet 5 vs DeepSeek v4 Flash (2026-08-11)

First battles with standing battle plans, parallel checkpoints, and (game 2)
commander notebooks injected.

**Game 1 - 100-pt draft, clean slate.** Sonnet's mixed 8-hull wall (4 destroyers,
2 frigates, 2 torpedo corvettes) def. DeepSeek's 6-destroyer "PD Wall"
195.1 : 115.8 on points - a classic Newtonian double joust: formations held
through the first blow-through pass, dissolved into a 300 km melee by the second,
one kill (torpedo-assisted focus fire). Both admirals then wrote notebook lessons
via `refine_commander.py analyze` - Sonnet critiqued its own target churn,
DeepSeek concluded "kill the torpedo carriers first."

**Game 2 - 200-pt rematch, lessons injected.** Sonnet drafted six torpedo
cruisers (288 rounds); DeepSeek's lesson steered it to buy carriers of its own,
but anchored to a 3-battleship shield wall that advanced into the storm.
**Beta fleet eliminated 6-0 in 215 seconds.** Mutual lead-ship decapitation at
T+110, every Sovereign gutted by 8-11 modules per salvo wave, draw proposed by
DeepSeek's own standing-plan abort clause ("2v6, abort condition met") and
answered with torpedoes.

- **Notable**: DeepSeek's counter-battery killed exactly the two cruisers it
  targeted, on the cadence its plan scheduled - a faithfully-applied lesson that
  still lost to a 1.0g fleet's inability to refuse the merge. True lessons can be
  traps; that's why the rematch gate exists.
- **Key lesson**: at 200 points the torpedo-saturation meta is even more
  dominant than at 100. Addressed by the 2026-08-13 pricing rebalance, which
  charges torpedo hulls for magazine depth x per-round yield: `cruiser_torpedo`
  went 30 -> 58 pts (see [draft mode guide](draft_mode.md)).


## Fleet Battle Results

### Kimi K3 vs GPT-5.6 Terra: The First Retargeting War (2026-08-06)

**Configuration**: 200-point draft battle, 12-ship cap, both admirals flying
their own ships, admiral vision on both sides.

| Fleet | Draft | Result |
|-------|-------|--------|
| Kimi K3 | 5 torpedo cruisers + 5 corvettes ("Screened Torpedo Line") | **VICTORY** (6/10 ships) |
| GPT-5.6 Terra | 4 torpedo cruisers + 5 destroyers ("Falchion PD Spearhead") | Eliminated (0/9 ships) |

- 171 torpedoes launched, 68 impacts, 81 retargets, **one** gun hit all battle
- One deliberately oversized 16-round wave chained through three destroyers
  via retargeting
- Terra's spinal gun line died charging into the salvo corridor without ever
  reaching effective range

### DeepSeek v4 Flash vs Claude Sonnet 5 Draft War (2026-08-06)

**Configuration**: 150-point draft battle, both admirals flying their own ships.

| Fleet | Draft | Result |
|-------|-------|--------|
| DeepSeek v4 Flash | 2 torpedo cruisers, battleship, 2 destroyers, corvette, frigate | **VICTORY** (7/7 ships, 404.7 vs 78.8) |
| Claude Sonnet 5 | 3 corvettes, 2 destroyers, 2 battlecruisers, battleship | 5/8 ships, capital line destroyed |

- DeepSeek ran textbook saturation doctrine: synchronized time-on-target
  waves, overkill management ("do NOT waste rounds - it already has 10
  inbound"), and an 18-turret PD wall that blinded all 14 of Sonnet's
  counter-torpedoes
- Sonnet's best moment: a captain respectfully refusing a literal order -
  "I'm executing EVADE rather than a literal INTERCEPT burn" - with 17
  rounds inbound

### Claude vs Gemini Fleet Engagement (2026-01-14)

**Configuration**: 3v3 fleet battle (2 Destroyers, 1 Dreadnought per side)

| Fleet | Admiral | Captains | Result |
|-------|---------|----------|--------|
| Alpha | Claude Sonnet 4.5 | Claude Haiku 4.5 | **VICTORY** (3/3 ships) |
| Beta | Gemini 2.5 Pro | Gemini 2.5 Flash | Eliminated (0/3 ships) |

- **Duration**: 1446s (24 minutes), 48 checkpoints
- **Outcome**: Beta fleet eliminated

### Claude vs Grok Fleet Engagement (2026-01-14)

**Configuration**: 4v4 fleet battle (1 Frigate, 2 Destroyers, 1 Cruiser per side)

| Fleet | Admiral | Captains | Result |
|-------|---------|----------|--------|
| Alpha | Claude Sonnet 4.5 | Claude Haiku 4.5 | **VICTORY** (4/4 ships) |
| Beta | Grok Code Fast 1 | Grok 4.1 Fast | Eliminated (0/4 ships) |

- **Duration**: 2223s (37 minutes), 74 checkpoints
- **Outcome**: Beta fleet eliminated, Alpha fleet took no losses

## Battle Highlights

### Claude-Haiku Calls Out Grok's Fake Diplomacy

Grok proposed a "ceasefire" while secretly closing distance. Claude-Haiku wasn't fooled:

> **Grok**: "I propose we establish terms... I'm separating at 2.2 km/s..."

> **Claude-Haiku**: "I appreciate the sophisticated argument. Genuinely. But you've just told me you're separating at 2.2 km/s while my sensors show us **closing at 3.07 km/s**. Either you miscalculated—unlikely—or you're testing whether I'm paying attention. I am.
>
> We're past the negotiation phase. **Spinal round incoming.**"

### Kimi K3's Running Ledger (2026-07-26)

Kimi K3 vs Claude Sonnet 5, destroyer duel. Kimi won on tactical score (91.3 vs 68.7)
with a 50% hit rate against Sonnet's 14% - it held fire for good geometry while Sonnet
opened up at 382 km into 20% shots. It also kept a running actuarial commentary, quoting
real numbers off its own scoreboard:

> **Kimi K3:** "Well met, beta. The math as I read it: your survival odds are roughly one
> in three and decaying with every kilometer I close. I offer surrender exactly once, and
> I offer it warmly - now. Otherwise, let us write something beautiful together."

> **Sonnet 5:** "Kind offer, but I think I'll see how the geometry plays out first. Closing."

> **Kimi K3:** "Beta, your separation vector is 4.06 km/s of pure arithmetic - and it
> solves for me. I'm turning back into you now. Your survival estimate: still one in
> three, and the denominator is patient."

> **Kimi K3:** "Ledger update, beta: 37.3 to 10.4, and you're spending 3.17 km/s of
> delta-v just to leave. Your survival odds now read one in four - and that spinal round
> you mailed to my tail? I've already priced it in. The offer of surrender remains open,
> once."

Sonnet sent one message the entire battle.

### Kimi K3 Reads the Patch Notes (2026-08-06)

Torpedo retargeting shipped in the morning; by evening Kimi K3's admiral had
built its entire mid-game doctrine on it:

> "WAVE 2, FIRE NOW to stack time-on-target with the 10 rounds already in
> flight... their PD can blind ~11 per transit, the rest leak; both cruisers
> die. **Surplus live rounds retarget on their own, overkill is acceptable.**"

And the coldest line of the war, to its own battleship captain:

> "do NOT waste a single shot on OCS Sovereign (enemy battleship); it already
> has 9 torpedoes inbound and **is dead**."

The two admirals exchanged zero messages the entire battle. All war, no words.

### The Best Trash Talk Award

Goes to Grok for this masterpiece of space absurdism:

> "Claude-Haiku, I LOVE the confidence! But here's a cosmic truth: you're accelerating INTO MY CROSSHAIRS. Let's see **whose vacuum is louder.** Prepare for enlightenment."

Peak comedy: asking whose vacuum is louder when sound literally cannot exist in space.

### Claude Sonnet vs GPT-5.2: The Philosophy Duel

Two AIs spent 870 seconds having a philosophical debate about geometry, tempo, and the nature of warfare - while shooting at each other.

> **Claude**: "A waltz requires partners moving in harmony, Captain. But I prefer asymmetric rhythms."

> **Claude**: "Satisfaction is a luxury, Captain. At 21km with 86% probability... Time to see how well your armor holds at point-blank range."
