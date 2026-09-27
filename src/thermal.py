"""
Thermal management system for the AI Commanders space battle simulator.

This module implements heat sinks, radiator arrays, and thermal management
for space combat simulations. Droplet radiators are vulnerable to damage
and can be retracted during combat at the cost of reduced heat dissipation.

Based on Terra Invicta thermal mechanics with liquid droplet radiator physics.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

class RadiatorState(Enum):
    """Operational state of a droplet radiator."""
    RETRACTED = "retracted"
    EXTENDED = "extended"
    DAMAGED = "damaged"
    DESTROYED = "destroyed"


class RadiatorPosition(Enum):
    """
    Physical position of radiator on ship hull.

    Radiators are located at the TAIL section of the ship, before the drive
    and next to the reactor for efficient heat transfer. They are tucked
    against the hull on all four sides around the tail/engine section.

    When extended, radiators angle 45° backwards toward the engine, reducing
    frontal cross-section and exposure to forward-facing weapons.
    """
    TAIL_PORT = "tail_port"
    TAIL_STARBOARD = "tail_starboard"
    TAIL_DORSAL = "tail_dorsal"
    TAIL_VENTRAL = "tail_ventral"


# Radiator extension angle - angled backwards toward engine
RADIATOR_EXTENSION_ANGLE_DEG = 45.0


# =============================================================================
# HEAT LADDER
# =============================================================================
#
# Heat is stored as a fraction of the heat-sink rating and is NEVER discarded:
# past 100% the lithium runs above its design temperature and the stored heat
# keeps climbing. What happens on the way up (fractions of sink capacity):
#
#   >= HEAT_GUN_HOLD_FRACTION        guns hold fire (weapon_status 'heat_limited');
#                                    point defense keeps firing
#   >= HEAT_AUTO_EXTEND_FRACTION     radiators are forced out (Terra Invicta's
#                                    RetractRadiatorsCommand rule) and cannot be
#                                    retracted until heat falls back below it
#   >= REACTOR_MELTDOWN_HEAT_FRACTION reactor containment fails: the ship dies by
#                                    immediate reactor detonation, cause
#                                    'reactor_overheat'
#
# The drive is never throttled automatically - evasion always stays available.
HEAT_GUN_HOLD_FRACTION = 0.90
HEAT_AUTO_EXTEND_FRACTION = 1.00
REACTOR_MELTDOWN_HEAT_FRACTION = 1.50

# heat_warning events fire once per upward crossing of each level, and a level
# re-arms only after heat falls HEAT_WARNING_REARM_FRACTION below it (no spam
# while heat hovers on a threshold).
HEAT_WARNING_LEVELS = (0.90, 1.00, 1.25)
HEAT_WARNING_REARM_FRACTION = 0.05

# set_radiators(mode="auto"): the rule layer extends the radiators when heat
# reaches the ship's auto_extend_pct (default below) and no torpedo is inbound,
# retracts them while a torpedo is inbound (unless the ladder forces them out)
# and once heat has fallen AUTO_RADIATOR_HYSTERESIS_PCT below the extend line.
AUTO_RADIATOR_DEFAULT_PCT = 65.0
AUTO_RADIATOR_HYSTERESIS_PCT = 25.0
# "Under fire" for the auto mode: a torpedo targeting the ship, closing, with
# an ETA inside this window (the same horizon the heuristic captain dodges at).
AUTO_RADIATOR_TORPEDO_ETA_S = 75.0

# Smoothing for the reported heat rates (exponential moving averages, seconds).
# Shots arrive as impulses (a spinal round dumps 1.8 GJ in one tick), so the
# raw per-tick rate is spiky; these give a readable "net rate" and a load
# estimate that includes gunfire.
HEAT_RATE_SMOOTHING_S = 10.0
HEAT_LOAD_SMOOTHING_S = 30.0

# Drive waste heat at FULL throttle as a multiple of the ship's nominal
# (undamaged, extended) radiator rejection. Terra Invicta sizes a ship's
# radiators to reject the power plant's full-burn waste heat; we keep that
# RULE at game scale (TI-literal is ~306 GW for the ICF VII + Protium torch,
# which a 525 GJ sink holds for 1.7 s). With the 60 t LithiumSpray refit the
# drive therefore makes 780 MW at full burn, i.e. DRIVE_HEAT_FRACTION =
# 780 MW / 306.4 GW ~ 0.25% of the TI-literal figure. Linear in throttle.
DRIVE_HEAT_TO_RADIATOR_RATIO = 1.0


def coerce_radiator_position(position) -> RadiatorPosition:
    """
    Normalise a radiator position into a thermal.RadiatorPosition member.

    The combat module declares its own RadiatorPosition enum with bare
    PORT/STARBOARD/DORSAL/VENTRAL members, and passes those values into the
    ThermalSystemProtocol methods below. Because ``ThermalSystemProtocol`` is a
    ``runtime_checkable`` Protocol it only checks method *names*, so the enum
    mismatch is invisible to isinstance() and every ``dict.get(position)``
    lookup silently returned None - radiators could never be reported extended
    and could never be damaged. Accepting either spelling (plus plain strings)
    fixes the lookup without requiring both modules to change at once.

    Args:
        position: A thermal.RadiatorPosition, a compatible enum from another
            module, or a string such as "PORT" / "tail_port".

    Returns:
        The matching thermal.RadiatorPosition member.

    Raises:
        KeyError: If the position cannot be mapped. Failing loudly is
            deliberate - the previous silent-None behaviour hid a real bug.
    """
    if isinstance(position, RadiatorPosition):
        return position

    # Enums from other modules: prefer the member name, fall back to the value.
    raw = getattr(position, "name", None) or getattr(position, "value", None) or position
    if not isinstance(raw, str):
        raise KeyError(f"Unknown radiator position: {position!r}")

    key = raw.strip().upper().replace("-", "_").replace(" ", "_")
    if not key.startswith("TAIL_"):
        key = f"TAIL_{key}"

    try:
        return RadiatorPosition[key]
    except KeyError:
        raise KeyError(f"Unknown radiator position: {position!r}") from None


@dataclass
class DropletRadiator:
    """
    A single droplet radiator panel that dissipates heat into space.

    Droplet radiators spray hot liquid metal droplets into space, which
    radiate heat before being recaptured. They are efficient but vulnerable
    to damage when extended.

    Attributes:
        position: Physical location on the ship hull.
        max_dissipation_kw: Maximum heat dissipation rate in kilowatts.
        mass_tons: Radiator mass in metric tons.
        state: Current operational state.
        health_percent: Remaining structural integrity (0-100).
    """
    position: RadiatorPosition
    max_dissipation_kw: float
    mass_tons: float
    state: RadiatorState = RadiatorState.RETRACTED
    health_percent: float = 100.0

    @property
    def current_dissipation_kw(self) -> float:
        """
        Calculate current heat dissipation rate based on state and health.

        Returns:
            Current dissipation rate in kilowatts.
            - 0 if retracted or destroyed
            - max_dissipation_kw * health_factor if extended or damaged
        """
        if self.state == RadiatorState.RETRACTED:
            return 0.0
        if self.state == RadiatorState.DESTROYED:
            return 0.0

        # Damaged radiators operate at reduced efficiency
        health_factor = self.health_percent / 100.0
        return self.max_dissipation_kw * health_factor

    def extend(self) -> bool:
        """
        Extend the radiator for maximum heat dissipation.

        Returns:
            True if successfully extended, False if destroyed.
        """
        if self.state == RadiatorState.DESTROYED:
            return False

        if self.health_percent < 100.0:
            self.state = RadiatorState.DAMAGED
        else:
            self.state = RadiatorState.EXTENDED
        return True

    def retract(self) -> bool:
        """
        Retract the radiator for protection during combat.

        Returns:
            True if successfully retracted, False if destroyed.
        """
        if self.state == RadiatorState.DESTROYED:
            return False

        self.state = RadiatorState.RETRACTED
        return True

    def damage(self, energy_gj: float) -> float:
        """
        Apply damage to the radiator from weapon impact.

        Radiators are fragile systems. Each GJ of damage reduces health
        by approximately 20%, making them vulnerable to even glancing hits.

        Args:
            energy_gj: Damage energy in gigajoules.

        Returns:
            Health remaining after damage (0-100).
        """
        # Radiators are fragile - roughly 5 GJ to destroy
        damage_percent = energy_gj * 20.0
        self.health_percent = max(0.0, self.health_percent - damage_percent)

        if self.health_percent <= 0:
            self.state = RadiatorState.DESTROYED
            self.health_percent = 0.0
        elif self.health_percent < 100.0 and self.state != RadiatorState.RETRACTED:
            self.state = RadiatorState.DAMAGED

        return self.health_percent


@dataclass
class RadiatorArray:
    """
    Complete radiator array with panels on all four sides of the ship.

    Ships mount radiator panels on port, starboard, dorsal, and ventral
    surfaces. During combat, radiators can be retracted for protection
    but this limits heat dissipation capacity.

    Attributes:
        radiators: Dictionary mapping positions to DropletRadiator instances.
    """
    radiators: dict[RadiatorPosition, DropletRadiator] = field(default_factory=dict)

    @classmethod
    def from_ship_data(cls, thermal_data: dict) -> RadiatorArray:
        """
        Create a RadiatorArray from ship thermal configuration.

        Args:
            thermal_data: Ship's thermal section from fleet data.

        Returns:
            Configured RadiatorArray with four radiator panels.
        """
        radiator_data = thermal_data.get("radiator", {})
        mass_tons = radiator_data.get("mass_tons", 60.0)
        dissipation_kw_per_kg = radiator_data.get("dissipation_kw_per_kg", 13.0)

        # Total dissipation capacity: mass * dissipation rate
        # Divide equally among 4 radiators
        total_dissipation_kw = mass_tons * 1000 * dissipation_kw_per_kg
        per_radiator_dissipation = total_dissipation_kw / 4.0
        per_radiator_mass = mass_tons / 4.0

        radiators = {}
        for position in RadiatorPosition:
            radiators[position] = DropletRadiator(
                position=position,
                max_dissipation_kw=per_radiator_dissipation,
                mass_tons=per_radiator_mass,
                state=RadiatorState.RETRACTED,
                health_percent=100.0,
            )

        return cls(radiators=radiators)

    @property
    def total_dissipation_kw(self) -> float:
        """
        Calculate total heat dissipation across all radiators.

        Returns:
            Combined dissipation rate in kilowatts.
        """
        return sum(r.current_dissipation_kw for r in self.radiators.values())

    def extend_all(self) -> int:
        """
        Extend all non-destroyed radiators.

        Returns:
            Number of radiators successfully extended.
        """
        count = 0
        for radiator in self.radiators.values():
            if radiator.extend():
                count += 1
        return count

    def retract_all(self) -> int:
        """
        Retract all non-destroyed radiators.

        Returns:
            Number of radiators successfully retracted.
        """
        count = 0
        for radiator in self.radiators.values():
            if radiator.retract():
                count += 1
        return count

    @property
    def nominal_dissipation_kw(self) -> float:
        """Rejection of the whole array undamaged and extended (the design rating)."""
        return sum(r.max_dissipation_kw for r in self.radiators.values())

    @property
    def extended_dissipation_kw(self) -> float:
        """What the array would reject if extended now (damage included)."""
        return sum(
            0.0 if r.state == RadiatorState.DESTROYED
            else r.max_dissipation_kw * r.health_percent / 100.0
            for r in self.radiators.values()
        )

    @property
    def any_extended(self) -> bool:
        """True if any surviving panel is out (EXTENDED, or DAMAGED while out)."""
        return any(
            r.state in (RadiatorState.EXTENDED, RadiatorState.DAMAGED)
            for r in self.radiators.values()
        )

    @property
    def any_retracted(self) -> bool:
        """True if any surviving panel is still tucked in."""
        return any(r.state == RadiatorState.RETRACTED for r in self.radiators.values())

    @property
    def health_percent(self) -> float:
        """Surviving rejection as a percentage of the design rating."""
        nominal = self.nominal_dissipation_kw
        if nominal <= 0:
            return 0.0
        return 100.0 * self.extended_dissipation_kw / nominal

    # Radiator HIT chances live in one place: combat.radiator_hit_chance, used
    # by combat.RadiatorHitResolver for both projectile and torpedo impacts.
    # The array's own probability model (get_hit_probability) disagreed with it
    # and was only ever exercised by this module's __main__, so it was removed.


@dataclass
class HeatSink:
    """
    Thermal mass for absorbing waste heat before radiating to space.

    Heat sinks use materials like lithium to store large amounts of
    thermal energy. This allows ships to temporarily exceed their
    radiator capacity during intense combat operations.

    The rating is a design point, not a wall: heat past capacity_gj is still
    stored (the sink runs hot) and drives the heat ladder - see
    HEAT_GUN_HOLD_FRACTION / HEAT_AUTO_EXTEND_FRACTION /
    REACTOR_MELTDOWN_HEAT_FRACTION. Nothing is ever silently discarded.

    Attributes:
        capacity_gj: Design heat capacity in gigajoules (100% heat).
        current_heat_gj: Current stored heat in gigajoules (may exceed capacity).
    """
    capacity_gj: float
    current_heat_gj: float = 0.0

    @classmethod
    def from_ship_data(cls, thermal_data: dict) -> HeatSink:
        """
        Create a HeatSink from ship thermal configuration.

        Args:
            thermal_data: Ship's thermal section from fleet data.

        Returns:
            Configured HeatSink instance.
        """
        heatsink_data = thermal_data.get("heatsink", {})
        capacity = heatsink_data.get("capacity_gj", 320.0)

        return cls(capacity_gj=capacity, current_heat_gj=0.0)

    @property
    def heat_fraction(self) -> float:
        """Stored heat as a fraction of the rating (may exceed 1.0)."""
        if self.capacity_gj <= 0:
            return 1.0
        return self.current_heat_gj / self.capacity_gj

    @property
    def heat_percent(self) -> float:
        """
        Get current heat level as percentage of capacity.

        Returns:
            Heat percentage; exceeds 100.0 when the sink runs past its rating.
        """
        return self.heat_fraction * 100.0

    @property
    def available_capacity_gj(self) -> float:
        """
        Get remaining heat absorption capacity below the 100% rating.

        Returns:
            Available capacity in gigajoules (0 once the sink is at or past 100%).
        """
        return max(0.0, self.capacity_gj - self.current_heat_gj)

    def absorb(self, heat_gj: float) -> bool:
        """
        Store heat in the sink. Never clips: heat past the rating is kept.

        Args:
            heat_gj: Amount of heat to absorb in gigajoules (<= 0 is ignored).

        Returns:
            True while the sink is still at or below its rating afterwards.
        """
        if heat_gj > 0:
            self.current_heat_gj += heat_gj
        return self.current_heat_gj <= self.capacity_gj


@dataclass
class HeatSource:
    """
    A system that generates waste heat during operation.

    Attributes:
        name: Identifier for the heat source.
        heat_generation_kw: Heat output rate in kilowatts when active.
        active: Whether the source is currently generating heat.
    """
    name: str
    heat_generation_kw: float
    active: bool = False


# Standard heat generation rates (kW)
#
# GAME-BALANCE VALUES, NOT PHYSICAL ONES. For reference, the physical numbers
# from data/fleet_ships.json are roughly:
#   reactor : 306,430 GW at 0.999 efficiency  -> ~306 GW of waste heat
#   drive   : 58.56 MN at ve = 10,256 km/s    -> ~300 TW jet power
# A heat sink would saturate in about a second at the physical reactor figure,
# so the drive's heat is instead tied to ONE rule borrowed from Terra Invicta:
# radiators are sized to reject full-burn waste heat. Drive heat at full
# throttle therefore equals the ship's nominal radiator rejection
# (DRIVE_HEAT_TO_RADIATOR_RATIO), scaled linearly with applied throttle - see
# ThermalSystem.set_drive_throttle. Weapon heat is per shot, charged by the
# power system (power.WeaponCapacitor: shot energy x (1 - efficiency)).
HEAT_GENERATION_RATES = {
    "reactor_idle": 1000.0,          # Systems plant housekeeping load (1 MW)
}


@dataclass
class ThermalSystem:
    """
    Complete thermal management system for a spacecraft.

    Manages heat generation from ship systems, heat absorption into
    sinks, and heat dissipation through radiators. Critical for
    managing ship operations during extended combat.

    Attributes:
        heatsink: The ship's heat sink for thermal storage.
        radiators: The ship's radiator array for heat dissipation.
        heat_sources: Continuous heat sources (reactor housekeeping, drive).
        drive_heat_full_kw: Drive waste heat at full throttle (kW).
        radiator_mode: "manual" (captain extends/retracts) or "auto" (the rule
            layer manages them, see AUTO_RADIATOR_DEFAULT_PCT).
        auto_extend_pct: Heat % at which "auto" mode extends the radiators.
    """
    heatsink: HeatSink
    radiators: RadiatorArray
    heat_sources: List[HeatSource] = field(default_factory=list)
    drive_heat_full_kw: float = 0.0
    radiator_mode: str = "manual"
    auto_extend_pct: float = AUTO_RADIATOR_DEFAULT_PCT

    # Telemetry, refreshed by update(). heat_rate_gw is the smoothed net change
    # of STORED heat (signed); rad_out_gw is what the radiators actually
    # rejected on the last update; load_gw estimates total generation
    # (continuous sources now + smoothed per-shot impulses).
    heat_rate_gw: float = 0.0
    rad_out_gw: float = 0.0
    continuous_gw: float = 0.0
    impulse_gw: float = 0.0
    peak_heat_fraction: float = 0.0
    _impulse_pending_gj: float = field(default=0.0, repr=False)
    _heat_at_last_update_gj: Optional[float] = field(default=None, repr=False)
    _warned_levels: set = field(default_factory=set, repr=False)

    # Warning thresholds (percent) for the is_overheating / is_critical flags
    OVERHEAT_THRESHOLD: float = 80.0
    CRITICAL_THRESHOLD: float = 95.0

    @classmethod
    def from_ship_data(cls, ship_data: dict) -> ThermalSystem:
        """
        Create a ThermalSystem from ship configuration data.

        Initializes heat sink and radiators from fleet data, the reactor's
        housekeeping load, and the drive (full-throttle heat = the radiator
        array's design rejection, see DRIVE_HEAT_TO_RADIATOR_RATIO).

        Args:
            ship_data: Complete ship configuration dictionary.

        Returns:
            Configured ThermalSystem instance.
        """
        thermal_data = ship_data.get("thermal", {})

        heatsink = HeatSink.from_ship_data(thermal_data)
        radiators = RadiatorArray.from_ship_data(thermal_data)
        drive_heat_full_kw = radiators.nominal_dissipation_kw * DRIVE_HEAT_TO_RADIATOR_RATIO

        heat_sources = [
            # Reactor housekeeping (always running)
            HeatSource(
                name="reactor",
                heat_generation_kw=HEAT_GENERATION_RATES["reactor_idle"],
                active=True,
            ),
            # Drive: scaled by applied throttle every tick (set_drive_throttle)
            HeatSource(name="engines", heat_generation_kw=0.0, active=False),
        ]

        return cls(
            heatsink=heatsink,
            radiators=radiators,
            heat_sources=heat_sources,
            drive_heat_full_kw=drive_heat_full_kw,
        )

    @property
    def heat_fraction(self) -> float:
        """Stored heat as a fraction of sink capacity (may exceed 1.0)."""
        return self.heatsink.heat_fraction

    @property
    def heat_percent(self) -> float:
        """
        Get current heat level as percentage of capacity.

        Returns:
            Heat percentage (exceeds 100.0 when the sink runs past its rating).
        """
        return self.heatsink.heat_percent

    @property
    def radiators_extended(self) -> bool:
        """True if any surviving radiator panel is out."""
        return self.radiators.any_extended

    @property
    def is_overheating(self) -> bool:
        """
        Check if ship is in overheating warning state.

        Returns:
            True if heat exceeds 80% of capacity.
        """
        return self.heat_percent >= self.OVERHEAT_THRESHOLD

    @property
    def is_critical(self) -> bool:
        """
        Check if ship is in critical thermal state.

        Returns:
            True if heat exceeds 95% of capacity.
        """
        return self.heat_percent >= self.CRITICAL_THRESHOLD

    @property
    def guns_heat_limited(self) -> bool:
        """Guns hold fire at or above HEAT_GUN_HOLD_FRACTION."""
        return self.heat_fraction >= HEAT_GUN_HOLD_FRACTION

    @property
    def radiators_forced(self) -> bool:
        """At or above HEAT_AUTO_EXTEND_FRACTION the radiators must be out."""
        return self.heat_fraction >= HEAT_AUTO_EXTEND_FRACTION

    @property
    def meltdown(self) -> bool:
        """Reactor containment has failed (REACTOR_MELTDOWN_HEAT_FRACTION)."""
        return self.heat_fraction >= REACTOR_MELTDOWN_HEAT_FRACTION

    def add_heat(self, source_name: str, heat_gj: float) -> bool:
        """
        Add a lump of heat (a shot, a PD burst, a launch) to the sink.

        Args:
            source_name: Name of the heat source (for logging).
            heat_gj: Amount of heat to add in gigajoules.

        Returns:
            True while the sink is still at or below its rating afterwards.
            The heat is stored either way.
        """
        if heat_gj > 0:
            self._impulse_pending_gj += heat_gj
        return self.heatsink.absorb(heat_gj)

    def set_drive_throttle(self, throttle: float) -> None:
        """
        Scale the drive's waste heat to the throttle actually applied.

        Waste heat is proportional to jet power, i.e. to thrust: full throttle
        makes drive_heat_full_kw, a 5% station-keeping burn 5% of it.
        """
        throttle = max(0.0, min(1.0, throttle))
        for source in self.heat_sources:
            if source.name == "engines":
                source.heat_generation_kw = self.drive_heat_full_kw * throttle
                source.active = throttle > 0.0
                return

    def set_source_active(self, source_name: str, active: bool) -> bool:
        """
        Activate or deactivate a heat source.

        Args:
            source_name: Name of the heat source.
            active: Whether to activate or deactivate.

        Returns:
            True if source was found and updated.
        """
        for source in self.heat_sources:
            if source.name == source_name:
                source.active = active
                return True
        return False

    def get_total_heat_generation_kw(self) -> float:
        """
        Calculate total heat generation from all active sources.

        Returns:
            Total heat generation in kilowatts.
        """
        return sum(
            source.heat_generation_kw
            for source in self.heat_sources
            if source.active
        )

    def update(self, dt_seconds: float) -> dict:
        """
        Update thermal system for a time step.

        Generates heat from active sources, then dissipates through
        radiators. Returns status information about the update.

        Net-flux accounting: continuous heat produced this step and heat
        already stored (including shot heat added since the last update) are
        rejected by the radiators up to their throughput; the rest stays in
        the sink - past 100% if need be. Nothing is clipped or discarded.

        Args:
            dt_seconds: Time step in seconds.

        Returns:
            Dictionary with update statistics:
            - heat_generated_gj: Continuous heat produced this step
            - heat_dissipated_gj: Heat radiated this step
            - net_heat_gj: Change in stored heat during this update
            - heat_percent / heat_fraction: Current heat level
            - rad_out_gw: Radiator rejection this step (GW)
            - heat_rate_gw: Smoothed net change of stored heat (GW, signed)
            - is_overheating / is_critical: 80% / 95% flags
        """
        dt = max(0.0, dt_seconds)
        before_gj = self.heatsink.current_heat_gj
        if self._heat_at_last_update_gj is None:
            # First update: shots already absorbed count from the start state.
            self._heat_at_last_update_gj = before_gj - self._impulse_pending_gj

        heat_gen_gj = (self.get_total_heat_generation_kw() / 1_000_000.0) * dt
        dissipation_capacity_gj = (
            self.radiators.total_dissipation_kw / 1_000_000.0
        ) * dt
        heat_dissipated = min(dissipation_capacity_gj, before_gj + heat_gen_gj)
        self.heatsink.current_heat_gj = max(
            0.0, before_gj + heat_gen_gj - heat_dissipated
        )
        after_gj = self.heatsink.current_heat_gj

        if dt > 0.0:
            self.rad_out_gw = heat_dissipated / dt
            self.continuous_gw = heat_gen_gj / dt
            # Smoothed rates. The stored-heat change includes shot impulses
            # added since the previous update, so the net rate is complete.
            actual_rate = (after_gj - self._heat_at_last_update_gj) / dt
            a_rate = 1.0 - math.exp(-dt / HEAT_RATE_SMOOTHING_S)
            a_load = 1.0 - math.exp(-dt / HEAT_LOAD_SMOOTHING_S)
            self.heat_rate_gw += a_rate * (actual_rate - self.heat_rate_gw)
            self.impulse_gw += a_load * (
                self._impulse_pending_gj / dt - self.impulse_gw
            )
        self._impulse_pending_gj = 0.0
        self._heat_at_last_update_gj = after_gj
        self.peak_heat_fraction = max(self.peak_heat_fraction, self.heat_fraction)

        return {
            "heat_generated_gj": heat_gen_gj,
            "heat_dissipated_gj": heat_dissipated,
            # Change in *stored* heat during this update.
            "net_heat_gj": after_gj - before_gj,
            "heat_percent": self.heat_percent,
            "heat_fraction": self.heat_fraction,
            "rad_out_gw": self.rad_out_gw,
            "heat_rate_gw": self.heat_rate_gw,
            "is_overheating": self.is_overheating,
            "is_critical": self.is_critical,
        }

    def crossed_warning_levels(self) -> List[float]:
        """
        Heat-warning levels crossed UPWARD since the last call (each fires once).

        A level re-arms after heat drops HEAT_WARNING_REARM_FRACTION below it.
        """
        frac = self.heat_fraction
        crossed = []
        for level in HEAT_WARNING_LEVELS:
            if level in self._warned_levels:
                if frac < level - HEAT_WARNING_REARM_FRACTION:
                    self._warned_levels.discard(level)
            elif frac >= level:
                self._warned_levels.add(level)
                crossed.append(level)
        return crossed

    @property
    def load_gw(self) -> float:
        """Estimated total heat generation: continuous now + smoothed shot heat."""
        return self.continuous_gw + self.impulse_gw

    def outlook(self, load_gw: Optional[float] = None) -> dict:
        """
        Commander-facing heat outlook at a (default: current) heat load.

        Returns heat level, the smoothed net rate, radiator health and the
        time to 100% (radiators forced out) and to meltdown, both with the
        radiators retracted and extended. Times are seconds, None = never
        (the ship cools or holds at that load).
        """
        load = self.load_gw if load_gw is None else load_gw
        heat_gj = self.heatsink.current_heat_gj
        cap = self.heatsink.capacity_gj
        ext_gw = self.radiators.extended_dissipation_kw / 1_000_000.0

        def _time_to(level_frac: float, net_gw: float) -> Optional[float]:
            gap = level_frac * cap - heat_gj
            if gap <= 0.0:
                return 0.0
            if net_gw <= 1e-9:
                return None
            return gap / net_gw

        panels = {
            pos.value: (0.0 if r.state == RadiatorState.DESTROYED else r.health_percent)
            for pos, r in self.radiators.radiators.items()
        }
        return {
            "heat_percent": self.heat_percent,
            "heat_gj": heat_gj,
            "capacity_gj": cap,
            "net_rate_mw": self.heat_rate_gw * 1000.0,
            "load_mw": load * 1000.0,
            "radiators_extended": self.radiators_extended,
            "radiator_mode": self.radiator_mode,
            "auto_extend_pct": self.auto_extend_pct,
            "radiator_rating_mw": self.radiators.nominal_dissipation_kw / 1000.0,
            "radiator_available_mw": ext_gw * 1000.0,
            "radiator_health_pct": self.radiators.health_percent,
            "radiator_panels": panels,
            "drive_heat_full_mw": self.drive_heat_full_kw / 1000.0,
            "guns_heat_limited": self.guns_heat_limited,
            "time_to_100_retracted_s": _time_to(HEAT_AUTO_EXTEND_FRACTION, load),
            "time_to_meltdown_retracted_s": _time_to(REACTOR_MELTDOWN_HEAT_FRACTION, load),
            "time_to_100_extended_s": _time_to(HEAT_AUTO_EXTEND_FRACTION, load - ext_gw),
            "time_to_meltdown_extended_s": _time_to(REACTOR_MELTDOWN_HEAT_FRACTION, load - ext_gw),
        }

    def get_status(self) -> dict:
        """
        Get comprehensive thermal system status.

        Returns:
            Dictionary with full system status.
        """
        radiator_status = {}
        for pos, radiator in self.radiators.radiators.items():
            radiator_status[pos.value] = {
                "state": radiator.state.value,
                "health_percent": radiator.health_percent,
                "dissipation_kw": radiator.current_dissipation_kw,
            }

        source_status = {}
        for source in self.heat_sources:
            source_status[source.name] = {
                "heat_generation_kw": source.heat_generation_kw,
                "active": source.active,
            }

        return {
            "heatsink": {
                "current_heat_gj": self.heatsink.current_heat_gj,
                "capacity_gj": self.heatsink.capacity_gj,
                "heat_percent": self.heatsink.heat_percent,
            },
            "radiators": {
                "total_dissipation_kw": self.radiators.total_dissipation_kw,
                "mode": self.radiator_mode,
                "auto_extend_pct": self.auto_extend_pct,
                "panels": radiator_status,
            },
            "heat_sources": source_status,
            "is_overheating": self.is_overheating,
            "is_critical": self.is_critical,
        }

    def is_radiator_extended(self, position: RadiatorPosition) -> bool:
        """
        Check if a radiator at the given position is extended.

        Args:
            position: The radiator position to check.

        Returns:
            True if the radiator is extended (or damaged but still active),
            False if retracted or destroyed.

        Raises:
            KeyError: If the position is not a recognised radiator position.
        """
        radiator = self.radiators.radiators.get(coerce_radiator_position(position))
        if radiator is None:
            return False

        return radiator.state in (RadiatorState.EXTENDED, RadiatorState.DAMAGED)

    def get_radiator_dissipation(self, position: RadiatorPosition) -> float:
        """
        Get the current dissipation capacity of a radiator in kW.

        Args:
            position: The radiator position to query.

        Returns:
            Current dissipation capacity in kW (0.0 if destroyed or not found).

        Raises:
            KeyError: If the position is not a recognised radiator position.
        """
        radiator = self.radiators.radiators.get(coerce_radiator_position(position))
        if radiator is None:
            return 0.0

        return radiator.current_dissipation_kw

    def damage_radiator(self, position: RadiatorPosition, energy_gj: float) -> float:
        """
        Apply damage to a radiator at the given position.

        Args:
            position: The radiator position to damage.
            energy_gj: Damage energy in gigajoules.

        Returns:
            Remaining health of the radiator (0-100), or -1 if position invalid.

        Raises:
            KeyError: If the position is not a recognised radiator position.
        """
        radiator = self.radiators.radiators.get(coerce_radiator_position(position))
        if radiator is None:
            return -1.0

        return radiator.damage(energy_gj)

    def apply_radiator_damage(
        self,
        position: RadiatorPosition,
        damage_gj: float
    ) -> tuple[float, bool, float]:
        """
        Apply damage to a radiator and return the results.

        This method implements the ThermalSystemProtocol interface expected
        by the combat system for radiator damage resolution.

        Args:
            position: The radiator position to damage.
            damage_gj: Amount of damage in gigajoules.

        Returns:
            Tuple of (damage_taken_gj, destroyed, dissipation_lost_kw).

        Raises:
            KeyError: If the position is not a recognised radiator position.
        """
        radiator = self.radiators.radiators.get(coerce_radiator_position(position))
        if radiator is None:
            return (0.0, False, 0.0)

        # Get current dissipation before damage
        dissipation_before = radiator.current_dissipation_kw

        # Check if already destroyed
        if radiator.state == RadiatorState.DESTROYED:
            return (0.0, False, 0.0)

        # Apply damage
        prev_health = radiator.health_percent
        radiator.damage(damage_gj)

        # Calculate results
        damage_taken = min(damage_gj, prev_health / 20.0)  # ~20% per GJ
        destroyed = radiator.state == RadiatorState.DESTROYED
        dissipation_after = radiator.current_dissipation_kw
        dissipation_lost = dissipation_before - dissipation_after

        return (damage_taken, destroyed, dissipation_lost)


def load_thermal_system(
    fleet_data_path: str | Path,
    ship_type: str
) -> ThermalSystem:
    """
    Load a thermal system for a specific ship type from fleet data.

    Args:
        fleet_data_path: Path to the fleet_ships.json file.
        ship_type: Type of ship (e.g., 'destroyer', 'cruiser').

    Returns:
        Configured ThermalSystem for the ship.

    Raises:
        KeyError: If ship type not found in fleet data.
        FileNotFoundError: If fleet data file not found.
    """
    with open(fleet_data_path, "r") as f:
        fleet_data = json.load(f)

    ships = fleet_data.get("ships", {})
    if ship_type not in ships:
        raise KeyError(f"Ship type '{ship_type}' not found in fleet data")

    return ThermalSystem.from_ship_data(ships[ship_type])


if __name__ == "__main__":
    # Example usage and basic validation
    import sys

    data_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"

    if not data_path.exists():
        print(f"Fleet data not found at {data_path}")
        sys.exit(1)

    print("AI Commanders Thermal Management System")
    print("=" * 50)

    thermal = load_thermal_system(data_path, "destroyer")
    print("\nDestroyer Thermal System Status:")
    print(f"  Heat Sink Capacity: {thermal.heatsink.capacity_gj} GJ")
    print(f"  Radiator rating: {thermal.radiators.nominal_dissipation_kw / 1000:.0f} MW")
    print(f"  Drive heat at full burn: {thermal.drive_heat_full_kw / 1000:.0f} MW")

    print("\nFull burn, radiators retracted, 60 s steps:")
    thermal.set_drive_throttle(1.0)
    for t in range(0, 600, 60):
        for _ in range(60):
            thermal.update(1.0)
        print(f"  t={t + 60:4d}s heat {thermal.heat_percent:6.1f}%")
    print(f"  Outlook: {thermal.outlook()}")
