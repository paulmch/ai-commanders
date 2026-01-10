# Terra Invicta Reference Data (Human Tech)

Data extracted directly from Terra Invicta game files for realistic near-future values.

## Human Fusion Drives

### High-Thrust Fusion Drives (Combat-Capable)

| Drive | Thrust (MN) | Exhaust Vel (km/s) | Efficiency | Power Req (GW) | Required Reactor |
|-------|-------------|-------------------|------------|----------------|------------------|
| **Helion Torus Lantern x1** | 0.56 | 690 | 92.5% | 0.21 | Tokamak |
| **Helion Torus Lantern x4** | 2.23 | 690 | 92.5% | 0.83 | Tokamak |
| **Helion Torus Lantern x6** | 3.34 | 690 | 92.5% | 1.25 | Tokamak |
| **Neutron Flux Lantern x1** | 12.9 | 66 | 80% | 0 (self-powered) | Any |
| **Neutron Flux Lantern x6** | 77.4 | 66 | 80% | 0 (self-powered) | Any |
| **Neutron Flux Torch x1** | 13.0 | 1700 | 80% | 0 (self-powered) | Any |
| **Neutron Flux Torch x6** | 78.0 | 1700 | 80% | 0 (self-powered) | Any |
| **Firestar Fission Lantern x1** | 5.0 | 50 | 85% | 0.15 | Gas Core Fission |
| **Firestar Fission Lantern x6** | 30.0 | 50 | 85% | 0.88 | Gas Core Fission |
| **Lodestar Fission Lantern x1** | 11.0 | 31.4 | 92.5% | 0.19 | Gas Core Fission |
| **Lodestar Fission Lantern x6** | 66.0 | 31.4 | 92.5% | 1.12 | Gas Core Fission |

### Mid-Tier Electromagnetic Drives

| Drive | Thrust (kN) | Exhaust Vel (km/s) | Efficiency | Power Req (GW) | Notes |
|-------|-------------|-------------------|------------|----------------|-------|
| **VASIMR x1** | 1 | 147 | 60% | 0.12 | Variable Isp |
| **VASIMR x6** | 6 | 147 | 60% | 0.74 | Variable Isp |
| **Ponderomotive VASIMR x6** | 13.5 | 147 | 72% | 1.38 | Improved |
| **Pulsed Plasmoid x6** | 13.2 | 425 | 72% | 3.90 | High Isp |

## Human Fusion Reactors

### Recommended for Combat Ships

| Reactor | Output (GW) | Specific Power (t/GW) | Efficiency | Crew | Mass (tons) |
|---------|-------------|----------------------|------------|------|-------------|
| **Z-Pinch Fusion I** | 260 | 3.0 | 95% | 12 | 780 |
| **Z-Pinch Fusion II** | 610 | 2.0 | 95% | 12 | 1220 |
| **Z-Pinch Fusion III** | 2510 | 1.4 | 96% | 12 | 3514 |
| **Z-Pinch Fusion IV** | 3970 | 0.4 | 98% | 12 | 1588 |
| **Flow Stabilized Z-Pinch** | 7590 | 0.0068 | 99.5% | 12 | 52 |
| **Fusion Tokamak III** | 624 | 1.0 | 96% | 8 | 624 |
| **Fusion Tokamak IV** | 1260 | 0.5 | 98.5% | 8 | 630 |
| **Fusion Tokamak V** | 5060 | 0.1 | 99% | 8 | 506 |
| **Inertial Confinement IV** | 5500 | 0.5 | 95% | 12 | 2750 |
| **Inertial Confinement V** | 19090 | 0.25 | 97.5% | 10 | 4773 |
| **Inertial Confinement VI** | 20420 | 0.068 | 99% | 10 | 1389 |
| **Hybrid Confinement III** | 1900 | 0.5 | 99% | 12 | 950 |
| **Hybrid Confinement IV** | 11370 | 0.05 | 99% | 12 | 569 |

## Magnetic Guns (Coilguns & Railguns)

### Coilgun Turrets (Burst Fire, Higher Velocity)

| Weapon | Mass (t) | Magazine | Ammo (kg) | Velocity (km/s) | Range (km) | Cooldown | Salvo |
|--------|----------|----------|-----------|-----------------|------------|----------|-------|
| **Light Coilgun Mk2** | 45 | 1500 | 10 | 5.4 | 550 | 30s | 4 shots |
| **Light Coilgun Mk3** | 40 | 1500 | 10 | 6.3 | 600 | 20s | 5 shots |
| **Coilgun Battery Mk2** | 90 | 1800 | 20 | 6.3 | 700 | 30s | 4 shots |
| **Coilgun Battery Mk3** | 80 | 1800 | 20 | 7.2 | 750 | 20s | 5 shots |
| **Heavy Coilgun Mk3** | 160 | 2160 | 40 | 8.1 | 900 | 20s | 5 shots |

### Railgun Turrets (Single Shot, Simpler)

| Weapon | Mass (t) | Magazine | Ammo (kg) | Velocity (km/s) | Range (km) | Cooldown |
|--------|----------|----------|-----------|-----------------|------------|----------|
| **Light Railgun Mk3** | 40 | 1000 | 12 | 3.6 | 600 | 10s |
| **Railgun Battery Mk3** | 80 | 1200 | 24 | 4.32 | 750 | 10s |
| **Heavy Railgun Mk3** | 160 | 1440 | 48 | 5.18 | 900 | 10s |

### Spinal Weapons (Fixed Forward, High Damage)

| Weapon | Mass (t) | Magazine | Ammo (kg) | Velocity (km/s) | Range (km) | Cooldown | Pivot |
|--------|----------|----------|-----------|-----------------|------------|----------|-------|
| **Spinal Railgun Mk3** | 200 | 300 | 120 | 7.77 | 1000 | 15s | 30° |
| **Spinal Coilgun Mk3** | 200 | 450 | 100 | 10.8 | 1000 | 24s (5 shots) | 30° |

### Ammo Mass Budget

For a ship with 2x Coilgun Battery Mk3:
- Magazine: 1800 rounds each = 3600 total
- Ammo mass: 20 kg each = **72 tons total ammo**
- Each shot reduces ship mass by 20 kg

---

## Armor System

Terra Invicta uses a sophisticated damage model with different damage types:

### Damage Types

| Damage Type | Source | Countered By |
|-------------|--------|--------------|
| **Baryonic (Kinetic)** | Coilguns, Railguns, Missiles | High baryonicHalfValue |
| **X-Ray** | Lasers, Particle Beams | High xRayHalfValue |
| **Chipping** | Sustained fire | ChippingResistance |

### Armor Materials (Non-Exotic)

| Armor | Density (kg/m³) | Baryonic Resist | X-Ray Resist | Heat Vapor (MJ/kg) | Best Against |
|-------|-----------------|-----------------|--------------|-------------------|--------------|
| **Steel** | 7850 | 1.0 | 0.27 | 6.8 | Baseline |
| **Titanium** | 4820 | 1.11 | 0.44 | 8.77 | Balanced, lighter |
| **Silicon Carbide** | 3210 | 4.61 | 0.72 | 9.9 | Kinetics |
| **Boron Carbide** | 2520 | 0.62 | 1.0 | 7.14 | Lasers |
| **Composite** | 1930 | 5.21 | 1.11 | 15.0 | Both |
| **Foamed Metal** | 920 | 7.45 | 1.62 | 24.0 | Both (very light) |
| **Nanotube** | 1720 | 19.78 | 2.53 | 29.6 | Kinetics (best) |
| **Adamantane** | 1800 | 31.02 | 4.82 | 59.5 | Both (top tier) |

### Armor Half-Value Thickness

The "half-value" is how much armor thickness (cm) halves the damage:

| Armor | X-Ray Half (cm) | Baryonic Half (cm) |
|-------|-----------------|-------------------|
| Steel | 2.0 | 7.5 |
| Titanium | 4.2 | 10.5 |
| Silicon Carbide | 9.1 | 58 |
| Composite | 15.3 | 72 |
| Foamed Metal | 29.4 | 134.9 |
| Nanotube | 19.9 | 155.4 |
| Adamantane | 18.0 | 115.8 |

### Armor Points System

Terra Invicta uses **armor points** (armorValue) to specify thickness:
- Ships have 3 armor zones: Nose, Lateral, Tail
- Each zone has a material and a point value (0-50+)
- Points translate to thickness based on area coverage

**Armor Point to Thickness/Mass:**

```python
# Armor zones have different surface areas based on ship geometry
# For a 100m × 20m cylindrical frigate:
ARMOR_AREAS = {
    'nose': 314,      # π * r² (front cap) ~314 m²
    'lateral': 6280,  # 2 * π * r * length (cylinder sides) ~6280 m²
    'tail': 314,      # π * r² (rear cap) ~314 m²
}

def calculate_armor_mass(zone: str, points: int, material_density: float) -> float:
    """
    Calculate armor mass from points.

    1 armor point = 1 cm of thickness
    Mass = area × thickness × density
    """
    area_m2 = ARMOR_AREAS[zone]
    thickness_m = points * 0.01  # 1 point = 1 cm
    volume_m3 = area_m2 * thickness_m
    mass_kg = volume_m3 * material_density
    return mass_kg / 1000  # tons

# Example: 10 points of Composite armor on nose
# area = 314 m², thickness = 0.1m, density = 1930 kg/m³
# mass = 314 * 0.1 * 1930 = 60,602 kg = 60.6 tons
```

**Typical Armor Configurations:**

| Ship Type | Nose | Lateral | Tail | Total Mass (Composite) |
|-----------|------|---------|------|------------------------|
| Light (Gunship) | 3 pts | 1 pt | 1 pt | ~25 tons |
| Medium (Corvette) | 5 pts | 2 pts | 2 pt | ~75 tons |
| Heavy (Frigate) | 8 pts | 3 pts | 3 pts | ~130 tons |
| Assault | 15 pts | 5 pts | 3 pts | ~250 tons |

### Recommended Armor Configuration

For a balanced 600t frigate facing both lasers and kinetics:

| Zone | Material | Points | Thickness | Mass | Notes |
|------|----------|--------|-----------|------|-------|
| **Nose** | Adamantane | 10 | 10 cm | ~56 tons | Best protection, faces enemy |
| **Lateral** | Composite | 3 | 3 cm | ~36 tons | Balance weight/protection |
| **Tail** | Titanium | 2 | 2 cm | ~30 tons | Light, protect drive |
| **Total** | - | - | - | **~122 tons** | ~20% of ship mass |

### Armor Damage Formula

```python
def calculate_armor_reduction(damage_mj: float, damage_type: str,
                               armor_thickness_cm: float, armor_material: dict) -> float:
    """
    Calculate damage after armor reduction using half-value system.

    damage_remaining = damage * 0.5^(thickness / half_value)
    """
    if damage_type == 'baryonic':
        half_value = armor_material['baryonic_half_cm']
    else:  # xray
        half_value = armor_material['xray_half_cm']

    reduction_factor = 0.5 ** (armor_thickness_cm / half_value)
    return damage_mj * reduction_factor

# Example: 518 MJ coilgun hit vs 10cm Adamantane
# half_value = 115.8 cm (baryonic)
# factor = 0.5^(10/115.8) = 0.942
# penetrating damage = 518 * 0.942 = 488 MJ
# Adamantane barely slows kinetics - need LOTS of it!

# Example: 50 MJ laser hit vs 10cm Adamantane
# half_value = 18.0 cm (xray)
# factor = 0.5^(10/18) = 0.68
# penetrating damage = 50 * 0.68 = 34 MJ
# Adamantane is better against lasers
```

### The Brutal Reality: Armor vs Weapons

**10cm Adamantane Nose Armor (best material, 56 tons):**

| Weapon | Raw Damage | After Armor | Reduction | Verdict |
|--------|------------|-------------|-----------|---------|
| Coilgun Salvo (2.59 GJ) | 2590 MJ | 2440 MJ | **6%** | Still kills frigate |
| Spinal Salvo (29.2 GJ) | 29200 MJ | 27500 MJ | **6%** | Vaporizes frigate |
| Laser (50 MJ) | 50 MJ | 34 MJ | **32%** | Meaningful reduction |
| Torpedo (540 MJ) | 540 MJ | 509 MJ | **6%** | Still hurts |

**What armor thickness would actually help vs kinetics?**

| Adamantane Thickness | vs Coilgun (2.59 GJ) | Mass (nose) | Practical? |
|---------------------|----------------------|-------------|------------|
| 10 cm | 2440 MJ (94%) | 56 tons | Current |
| 50 cm | 1940 MJ (75%) | 280 tons | Heavy |
| 116 cm (1 half-value) | 1295 MJ (50%) | 650 tons | Absurd |
| 232 cm (2 half-values) | 648 MJ (25%) | 1300 tons | Ship IS armor |

**Bottom line:**
- **Lasers**: Armor works, 10cm Adamantane = 32% reduction
- **Kinetics**: Armor is decoration, punches through regardless
- **Survival strategy**: Don't get hit. Evasion > armor.

---

## Laser Weapons

### Point Defense Lasers (Anti-Missile)

| Weapon | Mass (t) | Power (MJ) | Range (km) | Cooldown | Efficiency | Pivot |
|--------|----------|------------|------------|----------|------------|-------|
| **PD Laser Turret** | 20 | 50 | 250 | 5s | 25% | 180° |
| **PD Arc Laser Turret** | 20 | 50 | 300 | 4s | 35% | 180° |
| **PD Phaser Turret** | 20 | 50 | 350 | 3s | 45% | 180° |

**Point Defense Notes:**
- `defenseMode: true` - Can automatically engage missiles/torpedoes
- `attackMode: false` - Cannot target ships (PD only)
- Light speed means no lead calculation needed
- Damage = shotPower * efficiency

### Attack Laser Batteries (Turrets)

| Weapon | Mass (t) | Power (MJ) | Range (km) | Cooldown | Mirror (cm) | Notes |
|--------|----------|------------|------------|----------|-------------|-------|
| **60 cm IR Laser** | 150 | 100 | 600 | 30s | 60 | Small turret |
| **120 cm IR Laser** | 200 | 150 | 700 | 30s | 120 | Medium turret |
| **360 cm IR Laser** | 400 | 250 | 850 | 30s | 360 | Large turret |

### Laser Cannons (Spinal/Fixed)

| Weapon | Mass (t) | Power (MJ) | Range (km) | Cooldown | Mirror (cm) | Pivot |
|--------|----------|------------|------------|----------|-------------|-------|
| **240 cm IR Cannon** | 300 | 200 | 800 | 30s | 240 | 45° |
| **480 cm IR Cannon** | 550 | 300 | 900 | 30s | 480 | 45° |
| **960 cm IR Cannon** | 900 | 400 | 1000 | 30s | 960 | 45° |

### Laser Physics Notes

- **Wavelength**: Shorter = tighter beam (IR 810nm, UV 270nm, X-ray even shorter)
- **Mirror radius**: Larger = longer effective range (diffraction-limited)
- **Beam quality**: 1.0 = perfect, higher = more divergence
- **Jitter**: Pointing accuracy in radians (1e-7 = excellent)
- **Efficiency**: 25-45% - rest becomes waste heat on firing ship
- **Damage type**: X-Ray (countered by X-Ray resistant armor)

### Laser Damage Formula

```
Effective damage = shotPower_MJ * efficiency
Heat generated = shotPower_MJ * (1 - efficiency)
```

For a 60 cm IR Laser Battery:
- Shot power: 100 MJ
- Efficiency: 25%
- Damage delivered: 25 MJ to target
- Heat generated: 75 MJ on own ship

---

## Missiles/Torpedoes

### Attack Missiles

| Missile | Accel (g) | Delta-V (km/s) | Damage (MJ) | Warhead | Mass (kg) | Magazine |
|---------|-----------|----------------|-------------|---------|-----------|----------|
| **Krait** | 4.5 | 3.33 | 240 | Explosive | 1600 | 12 |
| **Anaconda** | 14.9 | 3.76 | 360 | Explosive | 1600 | 16 |
| **Cobra** | 14.9 | 3.17 | 540 | Explosive | 1600 | 16 |
| **Copperhead** | 18.3 | 3.68 | 720 | Explosive | 1600 | 16 |
| **Harlequin** | 14.9 | 4.48 | Penetrator | Penetrator | 1600 | 16 |
| **Viper** | 18.3 | 5.64 | Frag | Fragmentation | 1600 | 16 |

### Missile Components

| Component | Mass Range |
|-----------|------------|
| Fuel | 225-1200 kg |
| Systems | 150-400 kg |
| Warhead | 25-300 kg |
| Total Ammo | 400-1600 kg |

### Missile Performance

- **Targeting Range**: 500-800 km
- **Turn Rate**: 25-40 deg/s
- **Maneuver Angle**: 35-60 degrees
- **Salvo Size**: 6-8 missiles
- **Cooldown**: 3-6 seconds

### Torpedo Mass Budget

For 2x Torpedo Launchers with 16 torpedoes each (Cobra class):
- 32 torpedoes × 1600 kg = **51.2 tons of torpedoes**
- Each launch reduces ship mass by 1.6 tons

## Ship Hulls

### Hull Classes and Structural Integrity

| Hull | Mass (t) | Length (m) | Crew | SI | Est. HP (GJ) | Hardpoints |
|------|----------|-----------|------|-----|--------------|------------|
| **Gunship** | 178 | 50 | 3 | 4 | 0.5 | 1 nose |
| **Escort** | 350 | 50 | 4 | 7 | 0.9 | 2 hull |
| **Corvette** | 400 | 65 | 8 | 8 | 1.0 | 1 nose, 1 hull |
| **Frigate** | 600 | 100 | 20 | 12 | 1.5 | 1 nose, 2 hull |
| **Monitor** | 800 | - | 35 | 16 | 2.0 | - |
| **Destroyer** | 825 | - | 40 | 18 | 2.3 | - |
| **Cruiser** | 1000 | - | 60 | 20 | 2.5 | - |
| **Battlecruiser** | 1200 | - | 70 | 24 | 3.0 | - |
| **Battleship** | 1600 | - | - | 40 | 5.0 | - |
| **Dreadnought** | 2400 | - | - | 48 | 6.0 | - |

### Structural Integrity to Hit Points

**Conversion: ~125 MJ = 1 Structural Integrity**

```python
def calculate_hull_hp(structural_integrity: int) -> float:
    """Convert SI to GJ of damage capacity"""
    MJ_PER_SI = 125  # Estimated from game balance
    return structural_integrity * MJ_PER_SI / 1000  # Return GJ

# Examples:
# Frigate (SI=12): 12 * 125 = 1500 MJ = 1.5 GJ
# Cruiser (SI=20): 20 * 125 = 2500 MJ = 2.5 GJ
```

**What this means in combat:**

| Weapon | Damage per Salvo | Kills Frigate (1.5 GJ)? |
|--------|------------------|------------------------|
| Light Coilgun (5×198 MJ) | 990 MJ | 66% hull damage |
| Coilgun Battery (5×518 MJ) | 2.59 GJ | **Overkill (173%)** |
| Spinal Coilgun (5×5.83 GJ) | 29.2 GJ | **Vaporized (1947%)** |
| Cobra Torpedo | 540 MJ | 36% hull damage |

**Critical insight: A single well-aimed coilgun salvo can kill a frigate.**

### Ship Module Slots (Standard)
- 1x Drive
- 1x Power Plant
- 1x Radiator
- 2-5x Utility
- 3x Armor (Nose, Lateral, Tail)
- 1x Propellant Tank

---

## Recommended Values for AI Commanders

Based on the Terra Invicta data, here are **scaled values for a realistic near-future combat ship**:

### Combat Ship Specification

| System | Value | Justification |
|--------|-------|---------------|
| **Hull Class** | Frigate | 100m length, balanced |
| **Length** | 100 m | Standard frigate |
| **Width** | 20 m | Cylindrical profile |
| **Drive** | Helion Torus x4 | 2.2 MN thrust, 690 km/s EV |
| **Reactor** | Z-Pinch Fusion II | 610 GW output, 95% efficiency |
| **Max Accel** | 4g (39.2 m/s²) | Matches spec requirement |
| **Delta-V** | 300-400 km/s | Realistic for fusion torch |
| **Crew** | 20-30 | Combat operations |

### Complete Mass Breakdown

| Component | Mass (tons) | Notes |
|-----------|-------------|-------|
| **Hull Structure** | 150 | Base frigate frame |
| **Drive (Helion Torus x4)** | 40 | 4x 10t thrusters |
| **Reactor (Z-Pinch II)** | 50 | Scaled for ship size |
| **Radiator (Lithium Spray)** | 10 | 130 MW dissipation |
| **Heatsink (Heavy Lithium)** | 256 | 1050 GJ buffer |
| **Battery (Superconducting)** | 20 | 160 GJ capacity |
| **Weapons** | | |
| - Coilgun Battery Mk3 ×2 | 160 | 3600 rounds total |
| - Spinal Coilgun Mk3 | 200 | 450 rounds |
| - PD Arc Laser ×2 | 40 | Anti-missile |
| - Torpedo Launcher ×2 | 50 | 32 torpedoes |
| **Ammunition** | | |
| - Coilgun ammo | 72 | 3600 × 20kg |
| - Spinal ammo | 45 | 450 × 100kg |
| - Torpedoes | 51 | 32 × 1600kg |
| **Armor** | | |
| - Nose (10pt Adamantane) | 56 | Heavy front protection |
| - Lateral (3pt Composite) | 36 | Side coverage |
| - Tail (2pt Titanium) | 30 | Drive protection |
| **Propellant** | ~400 | For 350 km/s delta-v |
| **Crew/Life Support** | 30 | 30 crew |
| **Misc Systems** | 50 | Sensors, comms, etc. |
| | | |
| **DRY MASS** | ~850 | Without propellant |
| **WET MASS** | ~1250 | Combat ready |

### Weapon Loadout

| Weapon | Quantity | Mass (t) | Notes |
|--------|----------|----------|-------|
| Coilgun Battery Mk3 | 2 | 160 | Main anti-ship (3600 rounds) |
| Spinal Coilgun Mk3 | 1 | 200 | Heavy strike (450 rounds) |
| PD Arc Laser Turret | 2 | 40 | Anti-missile (4s cooldown) |
| Cobra Launcher | 2 | ~50 | 32 torpedoes total |
| **Total Weapons Mass** | - | ~450 | Plus ammo/torpedoes |

### Torpedo Specification (Based on Cobra/Copperhead)

| Parameter | Value |
|-----------|-------|
| Mass | 1600 kg |
| Acceleration | 15-18g |
| Delta-V | 10-15 km/s (improved over TI) |
| Warhead | 500-750 MJ |
| Guidance | Inertial + Terminal IR |
| Magazine | 8 per launcher |

---

---

## Batteries (Energy Storage)

### Human Battery Technologies

| Battery | Capacity (GJ) | Mass (t) | Specific (GJ/t) | Recharge (GJ/s) | Notes |
|---------|---------------|----------|-----------------|-----------------|-------|
| Lithium-Ion | 12 | 11 | 1.09 | 0.005 | Baseline |
| Graphene | 48 | 22 | 2.18 | 0.025 | 2x density |
| Quantum | 80 | 26 | 3.08 | 0.05 | Good efficiency |
| **Superconducting Coil** | 160 | 20 | **8.0** | 0.075 | Best human tech! |

The **Superconducting Coil Battery** is exceptional - 8 GJ/ton is 4x better than Quantum!

### Battery Sizing

For our 600t Frigate with weapons:
- Spinal coilgun: 5.83 GJ per shot × 5 salvos = ~30 GJ needed
- Turret bursts: 0.5 GJ per salvo × 20 salvos = ~10 GJ
- Systems/reserves: ~20 GJ
- **Total: ~60 GJ combat capacity**

**Recommended: 1× Superconducting Coil Battery (160 GJ, 20 tons)**
- Powers 27 spinal shots before depletion
- Recharges at 75 MW (from reactor)
- Full recharge in ~35 minutes

### Battery Code Model

```python
@dataclass
class Battery:
    capacity_gj: float = 160  # Superconducting Coil
    current_charge_gj: float = 160
    recharge_rate_gj_s: float = 0.075  # 75 MW
    mass_tons: float = 20

    def draw_power(self, amount_gj: float) -> bool:
        """Draw power for weapons/systems. Returns False if insufficient."""
        if self.current_charge_gj >= amount_gj:
            self.current_charge_gj -= amount_gj
            return True
        return False

    def recharge(self, dt: float, reactor_available_gw: float):
        """Recharge from reactor power"""
        recharge = min(self.recharge_rate_gj_s * dt, reactor_available_gw * dt)
        self.current_charge_gj = min(self.capacity_gj, self.current_charge_gj + recharge)
```

---

## Thermal Management Systems (Human Tech, Non-Exotic)

### Radiators

#### Fin/Panel Radiators (Fixed, Vulnerable)

| Radiator | Specific Power (kW/kg) | Op. Temp (K) | Emissivity | Vulnerability | Materials |
|----------|------------------------|--------------|------------|---------------|-----------|
| **Aluminum Fin** | 2.5 | 800 | 0.85 | 30 (high) | Metals |
| **Titanium Array** | 5.5 | 1200 | 0.83 | 20 (med) | Noble metals |
| **Molybdenum Pipe** | 4.5 | 1650 | 0.80 | 10 (low) | Metals/Noble |
| **Nanotube Filament** | 6.5 | 1300 | 0.90 | 5 (v.low) | Volatiles |

#### Droplet Radiators (Retractable, Combat-Safe)

| Radiator | Specific Power (kW/kg) | Op. Temp (K) | Emissivity | Vulnerability | Materials |
|----------|------------------------|--------------|------------|---------------|-----------|
| **Tin Droplet** | 8.0 | 1030 | 0.96 | 1 | Metals |
| **Gallium Mist** | 10.0 | 1200 | 0.96 | 1 | Metals/Noble |
| **Lithium Spray** | 13.0 | 1500 | 0.96 | 1 | Metals/Noble |
| **Dusty Plasma** | 18.0 | 2000 | 0.96 | 2 | Metals/Noble |

**Recommended for Combat Ships**:
- **Lithium Spray** (best non-exotic): 13 kW/kg, vulnerability 1
- **Dusty Plasma** (top tier): 18 kW/kg, vulnerability 2

#### Radiator Sizing Example

For a ship with 100 MW continuous waste heat using Lithium Spray (13 kW/kg):
```
Radiator mass = 100,000 kW / 13 kW/kg = 7,700 kg (7.7 tons)
```

### Heat Sinks (Non-Exotic)

| Heat Sink | Capacity (GJ) | Mass (tons) | Specific (GJ/ton) | Materials |
|-----------|---------------|-------------|-------------------|-----------|
| **Water** | 100 | 250 | 0.4 | Water |
| **Potassium** | 110 | 205 | 0.54 | Metals |
| **Sodium** | 370 | 230 | 1.6 | Metals |
| **Lithium** | 525 | 128 | 4.1 | Metals |
| **Molten Salt** | 900 | 485 | 1.85 | Volatiles/Metals |
| **Heavy Water** | 200 | 500 | 0.4 | Water |
| **Heavy Sodium** | 740 | 460 | 1.6 | Metals |
| **Heavy Lithium** | 1050 | 256 | 4.1 | Metals |
| **Heavy Molten Salt** | 1800 | 970 | 1.85 | Volatiles/Metals |

**Best Non-Exotic Options**:
- **Lithium Heat Sink**: 525 GJ at 128 tons (4.1 GJ/ton) - most efficient
- **Heavy Lithium**: 1050 GJ at 256 tons - for larger ships
- **Heavy Molten Salt**: 1800 GJ at 970 tons - highest capacity

---

## Heat/Efficiency Notes

From the drive data:
- **Fusion drive efficiency**: 80-97%
- **Waste heat** = Power * (1 - efficiency)
- Reactor waste heat only generated when reactor is running at that power level

### Reactor Power States

| State | Reactor Output | Waste Heat | Notes |
|-------|----------------|------------|-------|
| Idle | 5-10% | Minimal | Life support, sensors only |
| Cruise | 20-40% | Low | Normal operations |
| Combat Maneuver | 60-80% | Moderate | Weapons + partial thrust |
| Max Burn | 100% | Maximum | Full thrust, all systems |

### Heat Budget Example (600t Frigate)

**Thermal Systems:**
- Lithium Spray Radiator: ~10 tons = 130 MW dissipation
- Heavy Lithium Heat Sink: 256 tons = 1050 GJ buffer

**Heat Sources by State:**

| State | Reactor (GW) | Waste Heat (MW) | Can Dissipate? |
|-------|--------------|-----------------|----------------|
| Idle (10%) | 0.06 | 4.5 | Yes (130 MW capacity) |
| Cruise (30%) | 0.18 | 13.5 | Yes |
| Combat (70%) | 0.42 | 31.5 | Yes |
| Max Burn (100%) | 0.61 | 45.8 | Yes (within margin) |
| **Combat + Weapons** | 0.61 + bursts | 45.8 + bursts | Use heatsink |

**Combat Alpha Strike Heat Budget:**
- 1050 GJ heatsink capacity
- Each coilgun shot: ~36 MJ heat
- Spinal shot: ~3.6 GJ heat
- At 45.8 MW waste (max burn) + weapon heat, heatsink provides ~20 minutes of sustained combat before thermal limits

### Thermal Combat Considerations

1. **Radiators are targets** - Extend only when safe, retract for close combat
2. **Droplet radiators** - Can operate during combat (vulnerability 1-2)
3. **Heat sink buffer** - Absorb weapon heat bursts, dissipate during pauses
4. **Thermal signature** - Hot ships are easier to detect and track

---

## Combat Physics Notes

### Relative Velocity for Kinetic Damage

**Critical mechanic**: Kinetic damage is based on *relative* velocity, not muzzle velocity!

```
Impact velocity = |V_projectile - V_target|
Kinetic Energy = 0.5 * mass * (impact_velocity)^2
```

| Engagement Type | Projectile | Target | Impact Vel | Damage Multiplier |
|-----------------|------------|--------|------------|-------------------|
| Head-on | +7.2 km/s | -4 km/s (closing) | 11.2 km/s | 2.4x |
| Broadside | +7.2 km/s | 0 (perpendicular) | 7.2 km/s | 1.0x |
| Pursuit | +7.2 km/s | +4 km/s (fleeing) | 3.2 km/s | 0.2x |

**Tactical implication**: A head-on joust massively increases damage both ways!

### Point Defense vs Projectile Types

PD effectiveness depends on projectile type:

**Solid Slugs (Coilguns/Railguns):**
Must be **vaporized** - energy required = mass × heat_of_vaporization (~10 MJ/kg)

| Projectile | Mass | Energy to Destroy | PD Shots (17.5 MJ each) |
|------------|------|-------------------|-------------------------|
| Light coilgun slug | 10 kg | 100 MJ | 6 shots |
| Coilgun battery slug | 20 kg | 200 MJ | 12 shots |
| Heavy coilgun slug | 40 kg | 400 MJ | 23 shots |
| Spinal coilgun slug | 100 kg | 1 GJ | 58 shots |

**Torpedoes/Missiles (Electronics + Explosives):**
Much easier to neutralize! PD only needs to:
1. **Fry electronics** → guidance failure → inert ballistic slug
2. **Cook the warhead** → premature detonation at safe distance

| Kill Method | Energy Required | PD Shots | Result |
|-------------|-----------------|----------|--------|
| Electronics kill | ~50-100 MJ | 3-6 | Guidance dead, now inert slug |
| Warhead cook-off | ~200-500 MJ | 12-30 | Premature detonation |

An inert torpedo is just a 1600kg slug moving at ~3-4 km/s - easily evaded with a small burn since it can no longer track.

**Tactical implications:**
- PD lasers ARE effective against torpedoes (disable, don't vaporize)
- Heavy/spinal slugs are nearly immune to PD (solid metal, no electronics)
- Disabled torpedoes become predictable ballistic threats → evade with minor delta-v
- Spinal weapons are devastating precisely because they're unstoppable solid slugs

---

## Utility Modules

### Electronic Warfare

| Module | Mass (t) | Power (MW) | Effect | Notes |
|--------|----------|------------|--------|-------|
| **ECM 1** | 10 | 2 | -20% enemy hit chance | Basic jamming |
| **ECM 2** | 10 | 3 | -40% enemy hit chance | Improved |
| **ECM 3** | 10 | 4 | -60% enemy hit chance | Advanced |
| **Targeting Computer 1** | 10 | 1 | +10% own hit chance | Basic tracking |
| **Targeting Computer 2** | 10 | 1 | +30% own hit chance | Improved |
| **Targeting Computer 3** | 10 | 1 | +50% own hit chance | Advanced |

### ECM vs Targeting Computer

These systems are **multiplicative** on hit chance:

```python
def calculate_hit_chance(base_chance: float,
                         attacker_tc: float,
                         defender_ecm: float) -> float:
    """
    Calculate modified hit chance.

    base_chance: Base accuracy (0.0-1.0)
    attacker_tc: Targeting computer bonus (0.1-0.5)
    defender_ecm: ECM penalty (0.2-0.6)
    """
    modified = base_chance * (1 + attacker_tc) * (1 - defender_ecm)
    return max(0.05, min(0.95, modified))  # 5% floor, 95% ceiling

# Example: 60% base, TC3 (+50%), ECM3 (-60%)
# hit = 0.60 * 1.5 * 0.4 = 0.36 (36%)
# ECM is powerful!

# Example: 60% base, TC3 (+50%), no ECM
# hit = 0.60 * 1.5 * 1.0 = 0.90 (90%)
```

**Tactical implications:**
- ECM is more impactful than Targeting Computers
- Both ships should always mount TC3 if possible
- ECM3 can save your ship against accurate weapons
- At close range (high base accuracy), ECM matters more

---

## Combat Doctrine

### The Alpha Strike Reality

**Core truth: The first solid kinetic hit wins the battle.**

| Weapon System | Salvo Damage | vs Frigate (1.5 GJ) | vs Cruiser (2.5 GJ) |
|---------------|--------------|---------------------|---------------------|
| Light Coilgun ×2 (10 slugs) | 1.98 GJ | **Kill (132%)** | 79% damage |
| Coilgun Battery ×2 (10 slugs) | 5.18 GJ | **Overkill (345%)** | **Kill (207%)** |
| Spinal Coilgun (5 slugs) | 29.2 GJ | **Vaporized** | **Vaporized** |
| Cobra Torpedoes ×8 | 4.32 GJ | **Kill (288%)** | **Kill (173%)** |

### Why This Matters for AI Commanders

1. **No attrition warfare**: This isn't WW2 naval combat with gradual damage
2. **Initiative is everything**: The ship that hits first likely wins
3. **Armor is marginal**: 6% reduction doesn't change the outcome
4. **Evasion > protection**: Not being hit matters more than tanking hits

### Approach Geometry

The approach phase determines combat outcome:

```
APPROACH VECTORS (2D simplified)

HEAD-ON JOUST (Maximum violence)
    Ship A ───────> <─────── Ship B
    - Relative velocity: ADDITIVE (2.4x damage)
    - Both ships vulnerable
    - First accurate hit wins
    - High-risk, high-reward

OBLIQUE APPROACH (Asymmetric)
    Ship A ───────>
                    ╲
                     ╲ Ship B
    - Reduced closing rate
    - One ship can maintain better firing angle
    - Allows for geometry advantage

PURSUIT (Low energy transfer)
    Ship A ───────> Ship B ───────>
    - Relative velocity: SUBTRACTIVE (0.2x damage)
    - Pursuer has firing solution, defender must maneuver
    - Slower damage accumulation
    - Favors the pursued (can focus on escape)
```

### Optimal Combat Doctrine

**For Attacker (must reach station):**
1. Open with spinal weapon at maximum range
2. Use ECM to survive initial exchange
3. If first strike fails, assess damage before committing
4. Consider torpedo salvo to force defensive maneuvering
5. Accept that breakthrough may require sacrifice

**For Defender (must stop attacker):**
1. Position for oblique intercept (maximize own accuracy)
2. Reserve spinal weapon for confirmed solution
3. Use torpedoes early to force evasive burns (costs attacker delta-v)
4. If damaged, consider ramming geometry (closing velocity = mutual kill)

### Combat Timeline

```
RANGE       ACTION                              TIME TO CONTACT (4g approach)
1000 km     Detection confirmed                 ~4 minutes
500 km      Spinal weapons in range            ~2 minutes
200 km      Turret engagement begins           ~50 seconds
100 km      Torpedoes launched                 ~35 seconds
50 km       Point defense active               ~25 seconds
10 km       "Knife fight" range                ~10 seconds
0 km        Either destroyed or breakthrough
```

### Survival Probability

Given the alpha-strike reality, survival depends on:

| Factor | Impact | Notes |
|--------|--------|-------|
| **First accurate hit** | Decisive | Winner likely determined |
| **ECM 3 vs no ECM** | -60% enemy accuracy | Huge survivability boost |
| **TC 3 vs no TC** | +50% own accuracy | First hit more likely |
| **Evasive maneuvers** | Variable | Burns delta-v, complicates targeting |
| **Armor** | 6% vs kinetics | Marginal, won't change outcome |
| **Point defense** | Stops torpedoes | Doesn't stop slugs |

### Ship Design Implications

**Optimized for Alpha Strike:**
- Mount largest spinal weapon available
- TC3 + ECM3 mandatory
- Minimal armor (weight better spent on weapons)
- High thrust for maneuvering (geometry advantage)
- Superconducting battery for reliable weapon power

**Glass Cannon is the Meta:**
The math says: focus on hitting first, not on surviving hits.
