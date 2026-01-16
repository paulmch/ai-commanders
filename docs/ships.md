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

### Shots to First Penetration

Number of hits required to deplete armor at each facing. Lower is worse for the defender.

#### vs Corvette (212/36/42 cm)

| Weapon | Ablation/Shot | Nose | Lateral | Tail |
|--------|--------------|------|---------|------|
| Spinal Coiler Mk3 | 100 cm | 3 | 1 | 1 |
| Heavy Siege Coiler | 169 cm | 2 | 1 | 1 |
| Heavy Coilgun Mk3 | 28 cm | 8 | 2 | 2 |
| Coilgun Mk3 | 17 cm | 13 | 3 | 3 |
| Light Coilgun Mk3 | 3 cm | 71 | 13 | 15 |
| Torpedo @ 5 km/s | 73 cm | 3 | 1 | 1 |

#### vs Frigate (71/12/14 cm)

| Weapon | Ablation/Shot | Nose | Lateral | Tail |
|--------|--------------|------|---------|------|
| Spinal Coiler Mk3 | 100 cm | 1 | 1 | 1 |
| Heavy Siege Coiler | 169 cm | 1 | 1 | 1 |
| Heavy Coilgun Mk3 | 28 cm | 3 | 1 | 1 |
| Coilgun Mk3 | 17 cm | 5 | 1 | 1 |
| Light Coilgun Mk3 | 3 cm | 24 | 5 | 5 |
| Torpedo @ 5 km/s | 73 cm | 1 | 1 | 1 |

#### vs Destroyer (151/26/30 cm)

| Weapon | Ablation/Shot | Nose | Lateral | Tail |
|--------|--------------|------|---------|------|
| Spinal Coiler Mk3 | 100 cm | 2 | 1 | 1 |
| Heavy Siege Coiler | 169 cm | 1 | 1 | 1 |
| Heavy Coilgun Mk3 | 28 cm | 6 | 1 | 2 |
| Coilgun Mk3 | 17 cm | 9 | 2 | 2 |
| Light Coilgun Mk3 | 3 cm | 51 | 9 | 11 |
| Torpedo @ 5 km/s | 73 cm | 3 | 1 | 1 |

#### vs Cruiser (241/41/48 cm)

| Weapon | Ablation/Shot | Nose | Lateral | Tail |
|--------|--------------|------|---------|------|
| Spinal Coiler Mk3 | 100 cm | 3 | 1 | 1 |
| Heavy Siege Coiler | 169 cm | 2 | 1 | 1 |
| Heavy Coilgun Mk3 | 28 cm | 9 | 2 | 2 |
| Coilgun Mk3 | 17 cm | 15 | 3 | 3 |
| Light Coilgun Mk3 | 3 cm | 81 | 14 | 17 |
| Torpedo @ 5 km/s | 73 cm | 4 | 1 | 1 |

#### vs Battlecruiser (177/30/35 cm)

| Weapon | Ablation/Shot | Nose | Lateral | Tail |
|--------|--------------|------|---------|------|
| Spinal Coiler Mk3 | 100 cm | 2 | 1 | 1 |
| Heavy Siege Coiler | 169 cm | 2 | 1 | 1 |
| Heavy Coilgun Mk3 | 28 cm | 7 | 2 | 2 |
| Coilgun Mk3 | 17 cm | 11 | 2 | 3 |
| Light Coilgun Mk3 | 3 cm | 60 | 11 | 12 |
| Torpedo @ 5 km/s | 73 cm | 3 | 1 | 1 |

#### vs Battleship (262/45/53 cm)

| Weapon | Ablation/Shot | Nose | Lateral | Tail |
|--------|--------------|------|---------|------|
| Spinal Coiler Mk3 | 100 cm | 3 | 1 | 1 |
| Heavy Siege Coiler | 169 cm | 2 | 1 | 1 |
| Heavy Coilgun Mk3 | 28 cm | 10 | 2 | 2 |
| Coilgun Mk3 | 17 cm | 16 | 3 | 4 |
| Light Coilgun Mk3 | 3 cm | 88 | 15 | 18 |
| Torpedo @ 5 km/s | 73 cm | 4 | 1 | 1 |

#### vs Dreadnought (251/43/50 cm)

| Weapon | Ablation/Shot | Nose | Lateral | Tail |
|--------|--------------|------|---------|------|
| Spinal Coiler Mk3 | 100 cm | 3 | 1 | 1 |
| Heavy Siege Coiler | 169 cm | 2 | 1 | 1 |
| Heavy Coilgun Mk3 | 28 cm | 9 | 2 | 2 |
| Coilgun Mk3 | 17 cm | 15 | 3 | 3 |
| Light Coilgun Mk3 | 3 cm | 84 | 15 | 17 |
| Torpedo @ 5 km/s | 73 cm | 4 | 1 | 1 |

### Summary: Shots to Penetrate by Facing

| Ship | Nose (Spinal) | Nose (Heavy) | Lateral (Coilgun) | Tail (Torpedo) |
|------|---------------|--------------|-------------------|----------------|
| Corvette | 3 | 8 | 3 | 1 |
| Frigate | 1 | 3 | 1 | 1 |
| Destroyer | 2 | 6 | 2 | 1 |
| Cruiser | 3 | 9 | 3 | 1 |
| Battlecruiser | 2 | 7 | 2 | 1 |
| Battleship | 3 | 10 | 3 | 1 |
| Dreadnought | 3 | 9 | 3 | 1 |

---

### Shots to Destruction

After penetration, damage must destroy internal modules. Ships are destroyed when:
1. Main Reactor is destroyed (critical), OR
2. Command Bridge is destroyed (critical), OR
3. Hull integrity falls below threshold

**Module destruction:** ~2 GJ destroys a module completely (50% health per GJ)

#### Estimated Shots to Kill (penetrating shots after armor breach)

Based on structural integrity and critical module protection:

| Ship | Structural Integrity | Extra Shots (Nose) | Extra Shots (Lateral) | Extra Shots (Tail) |
|------|---------------------|-------------------|----------------------|-------------------|
| Corvette | 8 | 2-3 | 3-4 | 2-3 |
| Frigate | 12 | 2-3 | 3-4 | 2-3 |
| Destroyer | 18 | 3-4 | 4-5 | 3-4 |
| Cruiser | 20 | 3-4 | 5-6 | 3-4 |
| Battlecruiser | 24 | 3-4 | 5-6 | 3-4 |
| Battleship | 40 | 4-5 | 6-8 | 4-5 |
| Dreadnought | 48 | 5-6 | 7-9 | 5-6 |

**Notes:**
- Nose hits travel through spinal weapon → hull → bridge → reactor
- Lateral hits only affect middle ~50% of ship (cannot reach nose/tail extremes)
- Tail hits travel through engine → fuel → hull → reactor
- Armored bulkheads protect critical modules from lateral penetration

#### Total Shots to Kill (including armor penetration)

| Ship | Spinal (Nose) | Spinal (Lat) | Spinal (Tail) | Coilgun (Lat) |
|------|---------------|--------------|---------------|---------------|
| Corvette | 5-6 | 4-5 | 3-4 | 6-7 |
| Frigate | 3-4 | 4-5 | 3-4 | 4-5 |
| Destroyer | 5-6 | 5-6 | 4-5 | 6-7 |
| Cruiser | 6-7 | 6-7 | 4-5 | 8-9 |
| Battlecruiser | 5-6 | 6-7 | 4-5 | 7-8 |
| Battleship | 7-8 | 7-8 | 5-6 | 9-11 |
| Dreadnought | 8-9 | 8-9 | 6-7 | 10-12 |

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
