# Open Questions for Discussion

## Physics & Realism

### Q1: Delta-V Budget
The spec mentions 700+ km/s delta-v with 4g acceleration. This is extremely high for realistic physics.

**Options:**
1. Keep as specified (game-y, allows lots of maneuvering)
2. Reduce to ~100-200 km/s (more realistic)
3. Make fuel a critical resource (tactical fuel management)

**Implications:**
- High delta-v = engagement can last longer
- High delta-v = ships can reposition many times
- Realistic = more commitment to maneuvers

### Q2: Combat Range
At what range should combat typically occur?

**Options:**
1. Close range (10-50 km) - Intense, quick decisions
2. Medium range (50-200 km) - Balanced gameplay
3. Long range (200-1000 km) - Strategic, slower paced
4. Variable based on weapons (spinal at range, turrets close)

### Q3: Time Scale
How long should a typical engagement last?

**Options:**
1. 5-15 minutes (quick battles)
2. 30-60 minutes (tactical depth)
3. Hours (realistic approach phases)
4. Variable with time compression

## Damage Model

### Q4: Module Destruction Consequences
When a module hits 0 HP, what happens?

**Options:**
1. Module destroyed, attached systems disabled
2. Module destroyed, chance of catastrophic failure
3. Module heavily damaged, systems degraded but usable
4. Immediate ship destruction if central module

### Q5: Critical Hit Severity
How impactful should critical hits be?

**Options:**
1. Minor inconvenience (slightly reduced effectiveness)
2. Significant (system offline until repaired)
3. Severe (permanent damage, no repair possible)
4. Variable by system criticality

### Q6: Heat Death
At 100% heat, the ship is destroyed. How should the approach to 100% feel?

**Options:**
1. Gradual penalties leading to failure
2. Sudden death at threshold
3. Emergency shutdown options (survive but disabled)
4. Chance-based destruction above 90%

## Agent Design

### Q7: Agent Autonomy
How much should agents decide vs follow orders?

**Options:**
1. Strict obedience (crew does exactly what captain says)
2. Smart interpretation (crew optimizes within orders)
3. Full autonomy (crew can override bad orders)
4. Personality-based (varies by agent)

### Q8: Communication Verbosity
How much should agents "talk" to each other?

**Options:**
1. Minimal (just actions)
2. Brief status updates
3. Full tactical discussion
4. Configurable per match

### Q9: Error Handling
What happens when an agent gives an invalid action?

**Options:**
1. Action fails, turn continues
2. Action converted to "hold" (do nothing)
3. Agent re-prompted for valid action
4. Fallback to default behavior

## Gameplay Balance

### Q10: Attacker vs Defender
The attacker has two win conditions. How to balance?

**Options:**
1. Attacker starts further from station (must commit)
2. Defender has slight ship advantage
3. Station provides defender support (sensors, etc.)
4. Escape requires specific conditions (angle, distance)

### Q11: Point Defense Effectiveness
How good should PD be at stopping missiles/torpedoes?

**Options:**
1. Very effective (most missiles stopped, torpedo salvos needed)
2. Moderate (some get through, requires PD management)
3. Weak (PD is supplementary, not reliable)
4. Based on quantity vs quality

### Q12: Torpedo Balance
Torpedoes are powerful but limited. How to balance?

**Options:**
1. Few torpedoes, high damage
2. More torpedoes, moderate damage
3. Torpedoes as area denial (force evasion)
4. Smart torpedoes that coordinate

## Technical

### Q13: Visualization Priority
How important is visualization?

**Options:**
1. Essential (needed for debugging and enjoyment)
2. Nice to have (can run headless primarily)
3. Post-hoc only (replay viewer)
4. Real-time 3D required

### Q14: LLM Call Frequency
How often should agents make decisions?

**Options:**
1. Every physics tick (expensive, responsive)
2. Every N seconds (balanced)
3. Only when situation changes significantly
4. Captain decides turn length

### Q15: Model Comparison Methodology
How should we compare different LLM models?

**Options:**
1. Same scenario, multiple runs, statistics
2. Tournament format (round robin)
3. Elo rating system
4. Specific test scenarios (defensive, offensive, etc.)

## Feature Priorities

### Q16: MVP Features
What's the minimum viable product?

**Suggested MVP:**
1. Basic physics (position, velocity, thrust)
2. One weapon type working (turrets)
3. Simple damage model
4. Single LLM per ship (no crew separation)
5. Text-based status output

### Q17: Future Features
What should be added after MVP?

**Potential additions:**
- Full crew (3 agents per ship)
- All weapon types
- Heat system
- Visualization
- Tournament mode
- Custom ship configurations
- Multiple ship battles

## Your Input Requested

Please consider these questions and let me know your preferences! This will help shape the implementation to match your vision.

### Response Format

For each question, you can answer:
- **Option number** (e.g., "Q1: Option 2")
- **Custom answer** (your own idea)
- **Defer** (let me decide)
- **More discussion needed** (we should talk more)
