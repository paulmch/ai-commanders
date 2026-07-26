# Ship Specifications & Combat Analysis

This document details all ship classes in AI Commanders, including their module layouts, armor configurations, and combat survivability tables.

## Table of Contents

1. [Ship Overview](#ship-overview)
2. [Weapon Systems](#weapon-systems)
3. [Armor System](#armor-system)
4. [Ship Details](#ship-details)
   - [Corvette](#corvette)
   - [Frigate](#frigate)
   - [Destroyer](#destroyer)
   - [Cruiser](#cruiser)
   - [Battlecruiser](#battlecruiser)
   - [Battleship](#battleship)
   - [Dreadnought](#dreadnought)
5. [Combat Tables](#combat-tables)
   - [Shots to First Penetration](#shots-to-first-penetration)
   - [Shots to Destruction](#shots-to-destruction)

---

## Ship Overview

| Ship Class | Mass (tons) | Length (m) | Accel (g) | Crew | Hardpoints | Role |
|------------|-------------|------------|-----------|------|------------|------|
| Corvette | 400 | 65 | 3.0 | 8 | 2 | Torpedo Boat |
| Frigate | 600 | 100 | 3.0 | 20 | 3 | Escort |
| Destroyer | 825 | 125 | 2.0 | 40 | 4 | Main Combat |
| Cruiser | 1,000 | 175 | 1.5 | 60 | 5 | Heavy Combatant |
| Battlecruiser | 1,200 | 175 | 1.5 | 70 | 5 | Fast Capital |
| Battleship | 1,600 | 200 | 1.0 | 80 | 8 | Line of Battle |
| Dreadnought | 2,400 | 275 | 0.75 | 120 | 11 | Mobile Fortress |

### Rotation Times (90°)

| Ship Class | Thrust Vectoring | RCS Only |
|------------|-----------------|----------|
| Corvette | 12.1s | 54.2s |
| Frigate | 15.1s | 83.3s |
| Destroyer | 20.6s | 127.6s |
| Cruiser | 28.2s | 206.3s |
| Battlecruiser | 28.2s | 206.3s |
| Battleship | 36.9s | 288.7s |
| Dreadnought | 49.9s | 458.4s |

---

## Weapon Systems

All ships use Adamantane armor. Weapons deal damage through kinetic energy ablation.

### Coilgun Weapons

| Weapon | Mass (kg) | Velocity (km/s) | Energy (GJ) | Flat Chip | Range (km) | Cooldown |
|--------|-----------|-----------------|-------------|-----------|------------|----------|
| Spinal Coiler Mk3 | 88 | 9.9 | 4.29 | 0.35 | 900 | 15s |
| Heavy Siege Coiler Mk3 | 656 | 4.7 | 7.25 | 0.25 | 900 | 24s |
| Heavy Coilgun Battery Mk3 | 50 | 7.0 | 1.22 | 0.35 | 600 | 20s |
| Coilgun Battery Mk3 | 40 | 6.0 | 0.72 | 0.35 | 500 | 20s |
| Light Coilgun Battery Mk3 | 10 | 5.0 | 0.125 | 0.35 | 400 | 20s |

### Torpedoes

| Weapon | Penetrator Mass (kg) | Delta-V (km/s) | Accel (g) | Magazine |
|--------|---------------------|----------------|-----------|----------|
| Trident Torpedo | 250 | 14.0 | 12.0 | 8 |

**Torpedo at 5 km/s impact velocity:**
- Kinetic Energy = ½ × 250 kg × (5,000 m/s)² = **3.125 GJ**

---

## Armor System

All ships use **Adamantane** armor with the following properties:

| Property | Value |
|----------|-------|
| Density | 1,800 kg/m³ |
| Baryonic Half-Value | 115.8 cm |
| Chip Resistance | 75% |
| Kinetics Resistance | 50% |
| Heat of Vaporization | 59.534 MJ/kg |

### Armor Distribution

Ships allocate armor mass as:
- **Nose**: 50% of armor mass, 15% of surface area (heaviest protection)
- **Lateral**: 40% of armor mass, 70% of surface area (thinnest protection)
- **Tail**: 10% of armor mass, 15% of surface area

### Damage Calculation

Ablation per hit is calculated as:
1. Kinetic damage = 50% of kinetic energy
2. Effective damage = kinetic damage × (1 - kinetics_resist) = 25% of KE
3. Ablation (cm) = effective_damage × 93.3 cm/GJ

**Ablation per weapon against Adamantane:**

| Weapon | Kinetic Energy | Effective Ablation |
|--------|---------------|-------------------|
| Spinal Coiler Mk3 | 4.29 GJ | ~100 cm |
| Heavy Siege Coiler Mk3 | 7.25 GJ | ~169 cm |
| Heavy Coilgun Battery Mk3 | 1.22 GJ | ~28 cm |
| Coilgun Battery Mk3 | 0.72 GJ | ~17 cm |
| Light Coilgun Battery Mk3 | 0.125 GJ | ~3 cm |
| Torpedo @ 5 km/s | 3.125 GJ | ~73 cm |

---

## Ship Details

### Corvette

**Specifications:**
- Hull Mass: 400 tons | Total Wet Mass: 1,990 tons
- Length: 65m | Crew: 8
- Combat Acceleration: 3.0g
- Structural Integrity: 8

**Armor Sections:**

| Section | Thickness | Protection | Area |
|---------|-----------|------------|------|
| Nose | 212.0 cm | 73.6% | 81 m² |
| Lateral | 36.4 cm | 19.4% | 378 m² |
| Tail | 42.3 cm | 22.0% | 81 m² |

**Weapons:**
- 1× Torpedo Launcher (8 rounds)
- 1× PD Laser Turret

**Module Layout (6 layers, nose to tail):**

```
Layer 0 (Nose): Primary Sensor Array
Layer 1: Forward Hull Section
Layer 2: Command Bridge [CRITICAL], Crew Quarters
Layer 3: Main Reactor [CRITICAL]
Layer 4: Aft Hull Section
Layer 5 (Tail): Main Engine Assembly, Fuel Tank
```

---

### Frigate

**Specifications:**
- Hull Mass: 600 tons | Total Wet Mass: 1,990 tons
- Length: 100m | Crew: 20
- Combat Acceleration: 3.0g
- Structural Integrity: 12

**Armor Sections:**

| Section | Thickness | Protection | Area |
|---------|-----------|------------|------|
| Nose | 71.3 cm | 39.8% | 106 m² |
| Lateral | 12.3 cm | 8.6% | 496 m² |
| Tail | 14.2 cm | 9.8% | 106 m² |

**Weapons:**
- 1× Coilgun Battery Mk3
- 2× PD Laser Turrets

**Module Layout (7 layers, nose to tail):**

```
Layer 0 (Nose): Primary Sensor Array
Layer 1: Forward Hull Section
Layer 2: Command Bridge [CRITICAL], Crew Quarters
Layer 3: Central Hull Section
Layer 4: Main Reactor [CRITICAL], Coilgun Battery
Layer 5: Aft Hull Section, Fuel Tank
Layer 6 (Tail): Main Engine Assembly
```

---

### Destroyer

**Specifications:**
- Hull Mass: 825 tons | Total Wet Mass: 2,985 tons
- Length: 125m | Crew: 40
- Combat Acceleration: 2.0g
- Structural Integrity: 18

**Armor Sections:**

| Section | Thickness | Protection | Area |
|---------|-----------|------------|------|
| Nose | 151.2 cm | 58.3% | 132 m² |
| Lateral | 26.0 cm | 15.8% | 614 m² |
| Tail | 30.3 cm | 17.8% | 132 m² |

**Weapons:**
- 1× Spinal Coiler Mk3 (450 rounds)
- 1× Coilgun Battery Mk3 (1,800 rounds)
- 2× PD Laser Turrets

**Module Layout (8 layers, nose to tail):**

```
Layer 0 (Nose): Spinal Coiler Mount
Layer 1: Forward Hull Section, Main Magazine, Dorsal Turret Mount
Layer 2: Command Bridge [CRITICAL], Primary Sensor Array, Targeting Computer,
         Bridge Armored Bulkheads (Port/Starboard)
Layer 3: Central Hull Section, Crew Quarters
Layer 4: Main Reactor [CRITICAL], Reactor Armored Bulkheads (Port/Starboard)
Layer 5: Aft Hull Section, Secondary Reactor
Layer 6: Main Fuel Tank, Reserve Fuel Tank, PD Lasers (Dorsal/Ventral)
Layer 7 (Tail): Main Engine Assembly
```

---

### Cruiser

**Specifications:**
- Hull Mass: 1,000 tons | Total Wet Mass: 3,980 tons
- Length: 175m | Crew: 60
- Combat Acceleration: 1.5g
- Structural Integrity: 20

**Armor Sections:**

| Section | Thickness | Protection | Area |
|---------|-----------|------------|------|
| Nose | 240.7 cm | 80.1% | 150 m² |
| Lateral | 41.3 cm | 23.4% | 698 m² |
| Tail | 48.2 cm | 26.4% | 150 m² |

**Weapons:**
- 1× Spinal Coiler Mk3 (450 rounds)
- 2× Coilgun Battery Mk3 (1,800 rounds each)
- 2× PD Laser Turrets

**Module Layout (10 layers, nose to tail):**

```
Layer 0 (Nose): Spinal Coiler Mount
Layer 1: Forward Hull Section
Layer 2: Main Magazine
Layer 3: Command Bridge [CRITICAL], Primary Sensor Array, Targeting Sensors,
         Bridge Armored Bulkheads (Port/Starboard)
Layer 4: Central Hull Section
Layer 5: Main Reactor [CRITICAL], Reactor Armored Bulkheads (Port/Starboard)
Layer 6: Aft Hull Section
Layer 7: Coilgun Battery A, Coilgun Battery B, Cargo Bay
Layer 8: Crew Quarters, Main Fuel Tank
Layer 9 (Tail): Main Engine Assembly
```

---

### Battlecruiser

**Specifications:**
- Hull Mass: 1,200 tons | Total Wet Mass: 3,980 tons
- Length: 175m | Crew: 70
- Combat Acceleration: 1.5g
- Structural Integrity: 24

**Armor Sections:**

| Section | Thickness | Protection | Area |
|---------|-----------|------------|------|
| Nose | 176.9 cm | 65.9% | 169 m² |
| Lateral | 30.3 cm | 17.8% | 789 m² |
| Tail | 35.4 cm | 20.1% | 169 m² |

**Weapons:**
- 1× Spinal Coiler Mk3 (450 rounds)
- 2× Coilgun Battery Mk3 (1,800 rounds each)
- 2× PD Laser Turrets

**Module Layout:** Same as Cruiser (10 layers)

---

### Battleship

**Specifications:**
- Hull Mass: 1,600 tons | Total Wet Mass: 5,969 tons
- Length: 200m | Crew: 80
- Combat Acceleration: 1.0g
- Structural Integrity: 40

**Armor Sections:**

| Section | Thickness | Protection | Area |
|---------|-----------|------------|------|
| Nose | 262.1 cm | 84.3% | 205 m² |
| Lateral | 45.0 cm | 25.1% | 957 m² |
| Tail | 52.5 cm | 28.4% | 205 m² |

**Weapons:**
- 1× Spinal Coiler Mk3 (450 rounds)
- 3× Heavy Coilgun Battery Mk3 (1,800 rounds each)
- 1× Coilgun Battery Mk3 (1,800 rounds)
- 3× PD Laser Turrets

**Module Layout (11 layers, nose to tail):**

```
Layer 0 (Nose): Spinal Coiler Mount
Layer 1: Forward Hull Section
Layer 2: Forward Magazine, Heavy Coilgun Battery A, Heavy Coilgun Battery B
Layer 3: Command Bridge [CRITICAL], Long Range Sensor Array, Fire Control Radar,
         Bridge Armored Bulkheads (Port/Starboard)
Layer 4: Central Hull Section
Layer 5: Main Fusion Reactor [CRITICAL], Reactor Armored Bulkheads (Port/Starboard)
Layer 6: Aft Hull Section
Layer 7: Heavy Coilgun Battery C, Point Defense Array, Main Magazine
Layer 8: Crew Quarters
Layer 9: Main Fuel Tank, Reserve Fuel Tank
Layer 10 (Tail): Main Engine Assembly
```

---

### Dreadnought

**Specifications:**
- Hull Mass: 2,400 tons | Total Wet Mass: 7,959 tons
- Length: 275m | Crew: 120
- Combat Acceleration: 0.75g
- Structural Integrity: 48

**Armor Sections:**

| Section | Thickness | Protection | Area |
|---------|-----------|------------|------|
| Nose | 250.8 cm | 81.7% | 269 m² |
| Lateral | 42.9 cm | 24.1% | 1,256 m² |
| Tail | 50.1 cm | 27.3% | 269 m² |

**Weapons:**
- 1× Spinal Coiler Mk3 (450 rounds)
- 5× Heavy Coilgun Battery Mk3 (1,800 rounds each)
- 1× Coilgun Battery Mk3 (1,800 rounds)
- 4× PD Laser Turrets

**Module Layout (12 layers, nose to tail):**

```
Layer 0 (Nose): Spinal Coiler Mount
Layer 1: Forward Hull Section
Layer 2: Magazine Forward, Heavy Coilgun Battery A, Heavy Coilgun Battery B
Layer 3: Command Bridge [CRITICAL], Long Range Sensor Array, Fire Control Radar,
         Combat Information Center, Bridge Armored Bulkheads (Port/Starboard)
Layer 4: Central Hull Section
Layer 5: Main Fusion Reactor [CRITICAL], Secondary Reactor,
         Reactor Armored Bulkheads (Port/Starboard)
Layer 6: Aft Hull Section
Layer 7: Main Magazine, Missile Magazine, Main Cargo Bay
Layer 8: Crew Quarters A, Crew Quarters B
Layer 9: Heavy Coilgun Battery C, Heavy Coilgun Battery D, Point Defense Array
Layer 10: Main Fuel Tank, Reserve Fuel Tanks (A/B)
Layer 11 (Tail): Main Engine Assembly, Maneuvering Thrusters
```

---

## Combat Tables

### Shots to Kill (Simulated)

Number of hits required to destroy each ship class, starting from full armor.
Regenerate with `uv run python scripts/calculate_shots_to_kill.py` after any change
to the armor or damage model - these tables are derived from the live engine and
were previously an order of magnitude too optimistic (a corvette was listed as
dying to 4 nose hits; it takes ~41).

Ships are destroyed when:
1. Main Reactor is destroyed (critical), OR
2. Command Bridge is destroyed (critical), OR
3. Hull integrity falls below 25%

#### vs Corvette (Armor: 212/36/42 cm)

| Weapon | Nose | Lateral | Tail |
|--------|------|---------|------|
| Spinal Coiler Mk3 | 38 | 10 | 12 |
| Heavy Siege Coiler Mk3 | 36 | 9 | 10 |
| Heavy Coilgun Battery Mk3 | 89 | 29 | 36 |
| Coilgun Battery Mk3 | 126 | 46 | 57 |
| Light Coilgun Battery Mk3 | 347 | 200 | 249 |
| Torpedo @ 14.0 km/s | 4 | 1 | 2 |
| Torpedo @ 26.0 km/s | 2 | 1 | 1 |

#### vs Frigate (Armor: 71/12/14 cm)

| Weapon | Nose | Lateral | Tail |
|--------|------|---------|------|
| Spinal Coiler Mk3 | 16 | 8 | 7 |
| Heavy Siege Coiler Mk3 | 14 | 6 | 5 |
| Heavy Coilgun Battery Mk3 | 44 | 24 | 23 |
| Coilgun Battery Mk3 | 67 | 40 | 38 |
| Light Coilgun Battery Mk3 | 248 | 213 | 193 |
| Torpedo @ 14.0 km/s | 2 | 1 | 1 |
| Torpedo @ 26.0 km/s | 1 | 1 | 1 |

#### vs Destroyer (Armor: 151/26/30 cm)

| Weapon | Nose | Lateral | Tail |
|--------|------|---------|------|
| Spinal Coiler Mk3 | 31 | 14 | 14 |
| Heavy Siege Coiler Mk3 | 28 | 10 | 11 |
| Heavy Coilgun Battery Mk3 | 79 | 44 | 44 |
| Coilgun Battery Mk3 | 116 | 73 | 72 |
| Light Coilgun Battery Mk3 | 381 | 376 | 361 |
| Torpedo @ 14.0 km/s | 4 | 2 | 2 |
| Torpedo @ 26.0 km/s | 1 | 1 | 1 |

#### vs Cruiser (Armor: 241/41/48 cm)

| Weapon | Nose | Lateral | Tail |
|--------|------|---------|------|
| Spinal Coiler Mk3 | 42 | 18 | 17 |
| Heavy Siege Coiler Mk3 | 40 | 13 | 14 |
| Heavy Coilgun Battery Mk3 | 100 | 55 | 52 |
| Coilgun Battery Mk3 | 142 | 89 | 85 |
| Light Coilgun Battery Mk3 | 399 | 439 | 397 |
| Torpedo @ 14.0 km/s | 5 | 3 | 2 |
| Torpedo @ 26.0 km/s | 2 | 1 | 1 |

#### vs Battlecruiser (Armor: 177/30/35 cm)

| Weapon | Nose | Lateral | Tail |
|--------|------|---------|------|
| Spinal Coiler Mk3 | 34 | 16 | 15 |
| Heavy Siege Coiler Mk3 | 31 | 12 | 12 |
| Heavy Coilgun Battery Mk3 | 83 | 50 | 47 |
| Coilgun Battery Mk3 | 121 | 82 | 77 |
| Light Coilgun Battery Mk3 | 368 | 420 | 377 |
| Torpedo @ 14.0 km/s | 4 | 3 | 2 |
| Torpedo @ 26.0 km/s | 1 | 1 | 1 |

#### vs Battleship (Armor: 262/45/52 cm)

| Weapon | Nose | Lateral | Tail |
|--------|------|---------|------|
| Spinal Coiler Mk3 | 47 | 21 | 19 |
| Heavy Siege Coiler Mk3 | 45 | 15 | 15 |
| Heavy Coilgun Battery Mk3 | 113 | 64 | 58 |
| Coilgun Battery Mk3 | 160 | 104 | 93 |
| Light Coilgun Battery Mk3 | 480 | >500 | 437 |
| Torpedo @ 14.0 km/s | 6 | 3 | 3 |
| Torpedo @ 26.0 km/s | 2 | 1 | 1 |

#### vs Dreadnought (Armor: 251/43/50 cm)

| Weapon | Nose | Lateral | Tail |
|--------|------|---------|------|
| Spinal Coiler Mk3 | 46 | 21 | 24 |
| Heavy Siege Coiler Mk3 | 43 | 15 | 17 |
| Heavy Coilgun Battery Mk3 | 111 | 67 | 74 |
| Coilgun Battery Mk3 | 158 | 109 | 120 |
| Light Coilgun Battery Mk3 | 478 | >500 | >500 |
| Torpedo @ 14.0 km/s | 6 | 3 | 3 |
| Torpedo @ 26.0 km/s | 2 | 1 | 1 |

### Summary: Shots to Kill by Ship Class

Using Spinal Coiler Mk3 (4.29 GJ per shot):

| Ship | Armor (N/L/T cm) | Nose | Lateral | Tail |
|------|-----------------|------|---------|------|
| Corvette | 212/36/42 | 38 | 10 | 12 |
| Frigate | 71/12/14 | 16 | 8 | 7 |
| Destroyer | 151/26/30 | 31 | 14 | 14 |
| Cruiser | 241/41/48 | 42 | 18 | 17 |
| Battlecruiser | 177/30/35 | 34 | 16 | 15 |
| Battleship | 262/45/52 | 47 | 21 | 19 |
| Dreadnought | 251/43/50 | 46 | 21 | 24 |

**Torpedoes vs guns:** impact energy scales with the SQUARE of closing speed, and a
Trident carries 14 km/s of its own delta-v before either ship's velocity is added. A
head-on pass at 26 km/s closure delivers ~85 GJ - roughly 20 spinal rounds in one hit -
which is why a torpedo can kill a capital ship in 1-2 hits while a coilgun needs dozens.
Conversely a torpedo lobbed at low closing speed is nearly worthless. Torpedo boats live
or die by the geometry of the run.

**Damage Path Notes:**
- **Nose hits**: Spinal weapon → Forward hull → Bridge [CRITICAL] → Reactor [CRITICAL]
- **Lateral hits**: Only affect middle ~50% of ship (cannot reach nose/tail extremes), armored bulkheads protect criticals
- **Tail hits**: Engine → Fuel → Aft hull → Reactor [CRITICAL]

---

## Tactical Recommendations

### Best Approach Angles

1. **Nose-first attack run** - Enemy must present nose (strongest) armor
2. **Tail chase** - Target thinnest armor, but hard to maintain angle
3. **Lateral passes** - Highest hit probability (70%), thinnest armor

### Weapon Selection

| Scenario | Best Weapon |
|----------|-------------|
| Opening long-range salvos | Spinal Coiler (900 km range) |
| Sustained broadside fire | Heavy Coilgun Battery |
| Finishing damaged ships | Coilgun Battery (conserve heavy ammo) |
| Light ships/corvettes | Light Coilgun or Torpedo |

### Key Vulnerabilities

- **Frigates** are extremely fragile - any weapon penetrates in 1-5 hits
- **Lateral armor** is thin on all ships - broadside attacks are devastating
- **Corvettes** have heavy nose armor but paper-thin sides
- **Dreadnoughts** are surprisingly vulnerable from the side (43 cm)

---

*Data extracted from Terra Invicta game mechanics. Calculations assume perpendicular impacts and standard 0.01 m² impact area.*
