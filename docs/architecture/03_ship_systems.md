# Ship Systems Architecture

## Ship Overview

Both ships are identical at start, consisting of:
- 3 Hull Modules (Forward, Central, Aft)
- Power systems (Reactor, Battery)
- Thermal systems (Heatsink, Radiators)
- Propulsion (Main drive, RCS)
- Weapons (Turrets, Spinal, PD Lasers, Torpedoes)
- Sensors and Command

## Hull Module System

```
     FRONT                    REAR
       |                        |
       v                        v
+-------------+  +-------------+  +-------------+
|   FORWARD   |--+   CENTRAL   +--|    AFT      |
|   MODULE    |  |   MODULE    |  |   MODULE    |
+-------------+  +-------------+  +-------------+
| HP: 100     |  | HP: 150     |  | HP: 100     |
| Armor:Front |  | Armor:Side  |  | Armor:Rear  |
| - Bridge    |  | - Reactor   |  | - Drive     |
| - Sensors   |  | - Battery   |  | - Fuel      |
| - PDL #1    |  | - Turret #1 |  | - Torpedo   |
|             |  | - Turret #2 |  | - PDL #2    |
|             |  | - Heatsink  |  | - Radiators |
+-------------+  +-------------+  +-------------+
```

### Module Data Structure

```python
@dataclass
class Module:
    name: str
    max_hp: int
    current_hp: int
    armor_facing: str  # 'front', 'side', 'rear'
    hardpoints: list[Hardpoint]

    @property
    def is_destroyed(self) -> bool:
        return self.current_hp <= 0
```

### Armor System

| Facing | Rating | Equivalent | Notes |
|--------|--------|------------|-------|
| Front | 100 | 500mm steel | Primary threat axis |
| Side | 40 | 150mm steel | Secondary protection |
| Rear | 20 | 75mm steel | Drive vulnerability |

```python
def calculate_damage_penetration(incoming_damage, armor_rating):
    if incoming_damage > armor_rating:
        return incoming_damage - armor_rating
    else:
        # Armor absorbs, but degrades slightly
        armor_degradation = incoming_damage * 0.05
        return 0, armor_degradation
```

## Power Systems

### Reactor

| Parameter | Value |
|-----------|-------|
| Output | 500 MW thermal |
| Electrical efficiency | 40% |
| Available power | 200 MW electrical |
| Base heat output | 300 MW (constant) |
| Emergency overdrive | 250 MW (150% heat) |

```python
@dataclass
class Reactor:
    max_output: float = 200e6  # 200 MW
    current_output: float = 200e6
    heat_rate: float = 300e6  # 300 MW heat when running
    is_overdrive: bool = False

    def get_power(self) -> float:
        if self.is_overdrive:
            return self.max_output * 1.25
        return min(self.current_output, self.max_output)

    def get_heat(self) -> float:
        if self.is_overdrive:
            return self.heat_rate * 1.5
        return self.heat_rate
```

### Battery

| Parameter | Value |
|-----------|-------|
| Capacity | 500 GJ |
| Recharge rate | 100 MW (from reactor) |
| Discharge rate | 10 GW peak |
| Efficiency | 95% |

```python
@dataclass
class Battery:
    max_capacity: float = 500e9  # 500 GJ
    current_charge: float = 500e9
    max_recharge_rate: float = 100e6  # 100 MW
    max_discharge_rate: float = 10e9  # 10 GW

    def draw_power(self, amount: float, dt: float) -> float:
        """Returns actual power available"""
        max_draw = min(amount, self.max_discharge_rate)
        energy_needed = max_draw * dt

        if energy_needed <= self.current_charge:
            self.current_charge -= energy_needed
            return max_draw
        else:
            available = self.current_charge / dt
            self.current_charge = 0
            return available

    def recharge(self, power: float, dt: float):
        energy = power * dt * 0.95  # Efficiency loss
        self.current_charge = min(self.max_capacity, self.current_charge + energy)
```

### Power Distribution

```python
def update_power(ship, dt):
    # Priority: Life support > Sensors > Weapons > Drive

    reactor_power = ship.reactor.get_power()

    # Life support (always on)
    power_used = ship.life_support.draw

    # Sensors
    if reactor_power - power_used >= ship.sensors.draw:
        power_used += ship.sensors.draw

    # If drive is running, it needs power
    if ship.drive.throttle > 0:
        drive_need = ship.drive.power_draw * ship.drive.throttle
        if reactor_power - power_used >= drive_need:
            power_used += drive_need
        else:
            # Draw from battery
            battery_draw = ship.battery.draw_power(drive_need, dt)
            power_used += battery_draw

    # Weapons draw from battery during combat
    # (Handled separately per-shot)

    # Recharge battery with excess
    excess = reactor_power - power_used
    if excess > 0:
        ship.battery.recharge(excess, dt)
```

## Thermal Systems

### Heat Model

```python
@dataclass
class ThermalState:
    current_heat: float = 0  # Joules stored
    max_heat: float = 100e9  # 100 GJ = destruction

    @property
    def heat_percentage(self) -> float:
        return (self.current_heat / self.max_heat) * 100
```

### Heat Sources

| Source | Heat Rate | Condition |
|--------|-----------|-----------|
| Reactor (base) | 300 MW | Always running |
| Drive (waste) | 25 MW | At max throttle |
| Turret shot | 36 MJ | Per shot |
| Spinal shot | 3.6 GJ | Per shot |
| PD laser | 750 kW | While firing |
| Incoming laser | Variable | When hit |

### Heat Sinks

| Parameter | Value |
|-----------|-------|
| Capacity | 100 GJ |
| Absorption rate | 500 MW |
| Regeneration | Requires radiators |

```python
@dataclass
class Heatsink:
    max_capacity: float = 100e9
    current_fill: float = 0
    absorption_rate: float = 500e6

    def absorb_heat(self, heat: float, dt: float) -> float:
        """Returns heat NOT absorbed"""
        max_absorb = min(self.absorption_rate * dt, self.max_capacity - self.current_fill)
        absorbed = min(heat, max_absorb)
        self.current_fill += absorbed
        return heat - absorbed
```

### Radiators

| Parameter | Extended | Retracted |
|-----------|----------|-----------|
| Dissipation | 10 MW | 1 MW |
| Vulnerability | High | Low |
| Heat limit when extended | N/A | T^4 radiation |

```python
@dataclass
class Radiators:
    extended: bool = True
    max_dissipation_extended: float = 10e6  # 10 MW
    max_dissipation_retracted: float = 1e6  # 1 MW

    def dissipate(self, thermal_state: ThermalState, dt: float):
        rate = self.max_dissipation_extended if self.extended else self.max_dissipation_retracted
        heat_removed = rate * dt
        thermal_state.current_heat = max(0, thermal_state.current_heat - heat_removed)
```

### Heat Effects

| Heat % | Effect |
|--------|--------|
| 0-50% | Normal operations |
| 50-75% | -10% weapon accuracy, crew stress |
| 75-90% | -25% accuracy, random minor failures |
| 90-99% | -50% accuracy, systems shutting down |
| 100% | Ship destroyed (thermal runaway) |

## Sensors

### Detection Ranges

| Target Type | Passive Detection | Active Detection |
|-------------|-------------------|------------------|
| Ship (drive on) | 50,000 km | 100,000 km |
| Ship (drive off) | 5,000 km | 50,000 km |
| Torpedo | 1,000 km | 10,000 km |
| Projectile | 100 km | 1,000 km |

### Tracking Accuracy

```python
def tracking_accuracy(distance, target_size, sensor_quality):
    base_error = distance / 1000  # 1m error per km
    size_factor = 100 / target_size  # Larger = easier
    quality_factor = sensor_quality

    return base_error * size_factor / quality_factor
```

## Complete Ship Class

```python
@dataclass
class Ship:
    name: str
    faction: str  # 'attacker' or 'defender'

    # Physics
    physics: PhysicsState
    mass: float
    dry_mass: float

    # Modules
    modules: dict[str, Module]

    # Power
    reactor: Reactor
    battery: Battery

    # Thermal
    thermal: ThermalState
    heatsink: Heatsink
    radiators: Radiators

    # Propulsion
    drive: MainDrive
    rcs: RCSSystem
    propellant: float

    # Weapons
    turrets: list[CoilgunTurret]
    spinal: SpinalCoilgun
    pd_lasers: list[PointDefenseLaser]
    torpedo_launchers: list[TorpedoLauncher]

    # Sensors
    sensors: SensorSuite

    # Ammunition
    turret_ammo: int
    spinal_ammo: int
    torpedoes: int
```
