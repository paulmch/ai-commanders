"""
Ship geometry system for the AI Commanders space battle simulator.

This module implements cylinder-based ship geometry for:
- Hit location calculation based on impact angles
- Precise hit point determination on ship surface
- Weapon placement and firing arc geometry
- Cross-section calculation for targeting
- Module hit determination based on penetration paths

Ships are modeled as cylinders with tapered nose cones and engine sections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

from .physics import Vector3D
from .combat import HitLocation
from .modules import Module, ModuleLayout


# =============================================================================
# CONSTANTS
# =============================================================================

# Warship proportions (cylinder model)
RADIUS_TO_LENGTH_RATIO = 1 / 8  # Radius is 1/8 of length
NOSE_CONE_LENGTH_FRACTION = 0.15  # 15% of length is nose cone
ENGINE_SECTION_LENGTH_FRACTION = 0.15  # 15% of length is engine section

# Hit location angle thresholds (degrees from forward axis)
NOSE_HIT_ANGLE_THRESHOLD = 30.0  # |angle| < 30 deg from forward = NOSE
TAIL_HIT_ANGLE_THRESHOLD = 150.0  # |angle| > 150 deg from forward = TAIL

# Firing arc half-angles (degrees)
SPINAL_FIRING_ARC_HALF_ANGLE = 2.5  # 5 deg cone total
TURRET_FIRING_ARC_HALF_ANGLE = 90.0  # 180 deg hemisphere
PD_FIRING_ARC_HALF_ANGLE = 180.0  # 360 deg full sphere


# =============================================================================
# ENUMS
# =============================================================================

class WeaponMountType(Enum):
    """Types of weapon mounting positions on a ship."""
    NOSE = "nose"  # Spinal weapons at nose tip
    HULL = "hull"  # Turrets distributed along lateral surface
    ENGINE = "engine"  # Rear-mounted weapons near tail


class WeaponType(Enum):
    """Types of weapons with different firing characteristics."""
    SPINAL = "spinal"  # Forward-facing spinal mount
    TURRET = "turret"  # Rotatable turret mount
    POINT_DEFENSE = "point_defense"  # Full-sphere coverage PD


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class HitPoint:
    """
    Precise hit point information on ship surface.

    Attributes:
        location: General hit location (NOSE, LATERAL, TAIL)
        axial_position_m: Distance from nose along ship axis (0 = nose tip)
        radial_angle_deg: Position on circumference (0-360, 0 = dorsal, 90 = starboard)
        surface_normal: Unit vector pointing outward from surface at hit point
    """
    location: HitLocation
    axial_position_m: float
    radial_angle_deg: float
    surface_normal: Vector3D

    def __str__(self) -> str:
        return (
            f"HitPoint({self.location.value}, "
            f"axial={self.axial_position_m:.1f}m, "
            f"radial={self.radial_angle_deg:.0f}deg)"
        )


@dataclass
class FiringArc:
    """
    Firing arc geometry for a weapon.

    Attributes:
        weapon_position: Position of weapon in ship-local coordinates
        forward_direction: Primary firing direction (unit vector)
        half_angle_deg: Half-angle of firing cone in degrees
        weapon_type: Type of weapon (spinal, turret, point_defense)
        can_fire_full_sphere: True if weapon has 360-degree coverage
    """
    weapon_position: Vector3D
    forward_direction: Vector3D
    half_angle_deg: float
    weapon_type: WeaponType
    can_fire_full_sphere: bool = False

    def can_engage_target(
        self,
        target_direction: Vector3D
    ) -> bool:
        """
        Check if a target direction is within the firing arc.

        Args:
            target_direction: Direction to target (unit vector)

        Returns:
            True if target is within firing arc
        """
        if self.can_fire_full_sphere:
            return True

        # Calculate angle between forward direction and target
        angle_rad = self.forward_direction.angle_to(target_direction)
        angle_deg = math.degrees(angle_rad)

        return angle_deg <= self.half_angle_deg

    def coverage_solid_angle_sr(self) -> float:
        """
        Calculate solid angle coverage in steradians.

        Returns:
            Solid angle coverage (max 4*pi for full sphere)
        """
        if self.can_fire_full_sphere:
            return 4 * math.pi

        # Solid angle of a cone: 2*pi*(1 - cos(half_angle))
        half_angle_rad = math.radians(self.half_angle_deg)
        return 2 * math.pi * (1 - math.cos(half_angle_rad))


@dataclass
class ShipGeometry:
    """
    Cylinder-based geometry model for spacecraft.

    Models a ship as a cylinder with tapered nose cone and engine section.
    Used for hit calculations, weapon placement, and targeting.

    Attributes:
        length_m: Total ship length in meters
        radius_m: Hull radius in meters (estimated as length/8)
        nose_cone_length_m: Length of tapered nose section
        engine_section_length_m: Length of engine section at tail
        ship_type: Type identifier for this ship class
    """
    length_m: float
    radius_m: float
    nose_cone_length_m: float
    engine_section_length_m: float
    ship_type: str = "unknown"

    def __post_init__(self) -> None:
        """Validate geometry parameters."""
        if self.length_m <= 0:
            raise ValueError("Ship length must be positive")
        if self.radius_m <= 0:
            raise ValueError("Ship radius must be positive")

    @classmethod
    def from_ship_data(cls, ship_data: dict) -> ShipGeometry:
        """
        Create geometry from ship data dictionary.

        Args:
            ship_data: Ship data from fleet_ships.json

        Returns:
            Configured ShipGeometry instance
        """
        hull = ship_data.get("hull", {})
        ship_type = hull.get("type", "unknown")
        length_m = hull.get("length_m", 100.0)

        # Calculate derived dimensions
        radius_m = length_m * RADIUS_TO_LENGTH_RATIO
        nose_cone_length_m = length_m * NOSE_CONE_LENGTH_FRACTION
        engine_section_length_m = length_m * ENGINE_SECTION_LENGTH_FRACTION

        return cls(
            length_m=length_m,
            radius_m=radius_m,
            nose_cone_length_m=nose_cone_length_m,
            engine_section_length_m=engine_section_length_m,
            ship_type=ship_type
        )

    @property
    def main_cylinder_length_m(self) -> float:
        """Length of the main cylindrical section."""
        return self.length_m - self.nose_cone_length_m - self.engine_section_length_m

    @property
    def nose_cone_start_m(self) -> float:
        """Axial position where nose cone starts (from nose tip)."""
        return 0.0

    @property
    def main_cylinder_start_m(self) -> float:
        """Axial position where main cylinder starts."""
        return self.nose_cone_length_m

    @property
    def engine_section_start_m(self) -> float:
        """Axial position where engine section starts."""
        return self.length_m - self.engine_section_length_m

    # -------------------------------------------------------------------------
    # Hit Location Calculation
    # -------------------------------------------------------------------------

    def calculate_hit_location(
        self,
        impact_vector: Vector3D,
        ship_orientation: Vector3D
    ) -> HitLocation:
        """
        Determine hit location based on impact angle.

        Args:
            impact_vector: Direction FROM which the projectile comes
                (i.e., projectile velocity direction, NOT reversed)
            ship_orientation: Ship's forward direction (unit vector)

        Returns:
            HitLocation enum (NOSE, LATERAL, or TAIL)
        """
        # Normalize vectors
        impact_dir = impact_vector.normalized()
        ship_fwd = ship_orientation.normalized()

        # Calculate angle between impact direction and ship forward
        # If projectile comes from ahead (opposite to forward), angle ~ 180 deg
        # If projectile comes from behind (same as forward), angle ~ 0 deg
        angle_rad = impact_dir.angle_to(ship_fwd)
        angle_deg = math.degrees(angle_rad)

        # Impact from ahead means projectile direction is opposite to ship forward
        # So angle ~ 180 deg means nose hit
        # Angle ~ 0 deg means tail hit (from behind)

        if angle_deg > (180.0 - NOSE_HIT_ANGLE_THRESHOLD):
            # Impact from ahead (projectile direction opposite to forward)
            return HitLocation.NOSE
        elif angle_deg < (180.0 - TAIL_HIT_ANGLE_THRESHOLD):
            # Impact from behind (projectile direction similar to forward)
            return HitLocation.TAIL
        else:
            # Impact from side
            return HitLocation.LATERAL

    # -------------------------------------------------------------------------
    # Precise Hit Point Calculation
    # -------------------------------------------------------------------------

    def calculate_hit_point(
        self,
        impact_vector: Vector3D,
        ship_pos: Vector3D,
        ship_orientation: Vector3D,
        ship_up: Optional[Vector3D] = None
    ) -> HitPoint:
        """
        Calculate precise hit point on ship surface.

        Args:
            impact_vector: Direction FROM which the projectile comes
            ship_pos: Ship position in world coordinates
            ship_orientation: Ship's forward direction (unit vector)
            ship_up: Ship's up direction (defaults to +Z)

        Returns:
            HitPoint with location, axial position, radial angle, and surface normal
        """
        # Default up vector
        if ship_up is None:
            ship_up = Vector3D.unit_z()

        # Normalize input vectors
        impact_dir = impact_vector.normalized()
        ship_fwd = ship_orientation.normalized()
        ship_up = ship_up.normalized()

        # Calculate ship's right vector
        ship_right = ship_fwd.cross(ship_up).normalized()
        # Re-orthogonalize up vector
        ship_up = ship_right.cross(ship_fwd).normalized()

        # Determine general hit location
        location = self.calculate_hit_location(impact_dir, ship_fwd)

        # Calculate axial position based on hit location
        if location == HitLocation.NOSE:
            # Hit on nose cone
            # Approximate axial position based on impact angle within cone
            angle_from_forward = math.degrees(impact_dir.angle_to(-ship_fwd))
            # Map angle to position within nose cone (steeper angle = further back)
            axial_fraction = min(1.0, angle_from_forward / NOSE_HIT_ANGLE_THRESHOLD)
            axial_position_m = axial_fraction * self.nose_cone_length_m

        elif location == HitLocation.TAIL:
            # Hit on engine section
            angle_from_backward = math.degrees(impact_dir.angle_to(ship_fwd))
            axial_fraction = min(1.0, angle_from_backward / (180.0 - TAIL_HIT_ANGLE_THRESHOLD))
            axial_position_m = (
                self.engine_section_start_m +
                (1.0 - axial_fraction) * self.engine_section_length_m
            )

        else:
            # Hit on lateral surface (main cylinder)
            # Use the axial component of impact direction to estimate position
            # Project impact onto ship's longitudinal axis
            axial_component = impact_dir.dot(ship_fwd)
            # Map from [-1, 1] to cylinder section
            normalized_axial = (axial_component + 1.0) / 2.0  # [0, 1]
            axial_position_m = (
                self.main_cylinder_start_m +
                normalized_axial * self.main_cylinder_length_m
            )

        # Calculate radial angle (where on circumference)
        # Project impact direction onto ship's cross-section plane
        lateral_component = impact_dir - ship_fwd * impact_dir.dot(ship_fwd)

        if lateral_component.magnitude > 1e-6:
            lateral_dir = lateral_component.normalized()
            # Calculate angle from dorsal (up) direction
            cos_angle = lateral_dir.dot(-ship_up)  # Negative because impact comes FROM this direction
            sin_angle = lateral_dir.dot(-ship_right)
            radial_angle_rad = math.atan2(sin_angle, cos_angle)
            radial_angle_deg = math.degrees(radial_angle_rad)
            if radial_angle_deg < 0:
                radial_angle_deg += 360.0
        else:
            # Impact is purely axial
            radial_angle_deg = 0.0

        # Calculate surface normal at hit point
        surface_normal = self._calculate_surface_normal(
            axial_position_m, radial_angle_deg, ship_fwd, ship_up, ship_right
        )

        return HitPoint(
            location=location,
            axial_position_m=axial_position_m,
            radial_angle_deg=radial_angle_deg,
            surface_normal=surface_normal
        )

    def _calculate_surface_normal(
        self,
        axial_position_m: float,
        radial_angle_deg: float,
        ship_fwd: Vector3D,
        ship_up: Vector3D,
        ship_right: Vector3D
    ) -> Vector3D:
        """
        Calculate outward surface normal at a given point.

        For nose cone: normal points outward and slightly forward
        For main cylinder: normal points radially outward
        For engine section: normal points outward and slightly backward
        """
        radial_rad = math.radians(radial_angle_deg)

        # Radial direction in cross-section
        radial_dir = (
            ship_up * math.cos(radial_rad) +
            ship_right * math.sin(radial_rad)
        )

        if axial_position_m < self.nose_cone_length_m:
            # Nose cone - normal tilts forward
            cone_angle = math.atan2(self.radius_m, self.nose_cone_length_m)
            normal = (
                radial_dir * math.cos(cone_angle) -
                ship_fwd * math.sin(cone_angle)
            )
        elif axial_position_m > self.engine_section_start_m:
            # Engine section - normal tilts backward
            cone_angle = math.atan2(self.radius_m, self.engine_section_length_m)
            normal = (
                radial_dir * math.cos(cone_angle) +
                ship_fwd * math.sin(cone_angle)
            )
        else:
            # Main cylinder - purely radial
            normal = radial_dir

        return normal.normalized()

    # -------------------------------------------------------------------------
    # Weapon Placement
    # -------------------------------------------------------------------------

    def get_weapon_position(
        self,
        weapon_mount: str,
        mount_index: int = 0
    ) -> Vector3D:
        """
        Get weapon position in ship-local coordinates.

        Args:
            weapon_mount: Mount type ('nose', 'hull', 'engine', or 'nose_only')
            mount_index: Index for multiple mounts of same type (0 = first)

        Returns:
            Position vector in ship-local coordinates
            (origin at ship center, +X = forward)
        """
        mount_type = weapon_mount.lower().replace("_only", "")

        if mount_type == "nose":
            # Spinal weapons at nose tip, offset slightly for multiple mounts
            forward_pos = self.length_m / 2 - 2.0  # Just behind nose tip
            lateral_offset = mount_index * 1.0  # Slight offset for multiple spinals
            return Vector3D(forward_pos, lateral_offset, 0.0)

        elif mount_type == "hull":
            # Hull turrets distributed along lateral surface
            # Alternate between port and starboard
            num_positions = 6  # Divide hull into 6 positions
            position_fraction = (mount_index % num_positions) / num_positions

            # Axial position along main cylinder
            axial_pos = (
                self.main_cylinder_start_m +
                position_fraction * self.main_cylinder_length_m
            )
            # Convert to ship-local (origin at center)
            axial_local = axial_pos - self.length_m / 2

            # Alternate sides
            if mount_index % 2 == 0:
                lateral = self.radius_m * 0.9  # Slightly inside hull
            else:
                lateral = -self.radius_m * 0.9

            return Vector3D(axial_local, lateral, 0.0)

        elif mount_type == "engine":
            # Engine mounts near tail
            aft_pos = -self.length_m / 2 + self.engine_section_length_m / 2
            lateral_offset = mount_index * 2.0 - 1.0  # Center around axis
            return Vector3D(aft_pos, lateral_offset, 0.0)

        else:
            # Default: any mount goes to hull
            return self.get_weapon_position("hull", mount_index)

    # -------------------------------------------------------------------------
    # Firing Arc Geometry
    # -------------------------------------------------------------------------

    def get_weapon_firing_arc(
        self,
        weapon_position: Vector3D,
        weapon_type: str
    ) -> FiringArc:
        """
        Get firing arc geometry for a weapon at a given position.

        Args:
            weapon_position: Position in ship-local coordinates
            weapon_type: Type string ('spinal', 'turret', 'point_defense', etc.)

        Returns:
            FiringArc with coverage geometry
        """
        wtype = weapon_type.lower()

        # Determine weapon category
        if "spinal" in wtype or "coiler" in wtype and "heavy" not in wtype:
            # Spinal weapons have narrow forward arc
            return FiringArc(
                weapon_position=weapon_position,
                forward_direction=Vector3D.unit_x(),
                half_angle_deg=SPINAL_FIRING_ARC_HALF_ANGLE,
                weapon_type=WeaponType.SPINAL,
                can_fire_full_sphere=False
            )

        elif "pd" in wtype or "point_defense" in wtype or "laser" in wtype:
            # Point defense has full sphere coverage
            return FiringArc(
                weapon_position=weapon_position,
                forward_direction=Vector3D.unit_x(),
                half_angle_deg=PD_FIRING_ARC_HALF_ANGLE,
                weapon_type=WeaponType.POINT_DEFENSE,
                can_fire_full_sphere=True
            )

        else:
            # Default turret - hemisphere facing outward
            # Determine outward direction based on position
            if abs(weapon_position.y) > abs(weapon_position.z):
                # Side-mounted
                if weapon_position.y > 0:
                    outward = Vector3D(0, 1, 0)  # Starboard
                else:
                    outward = Vector3D(0, -1, 0)  # Port
            else:
                # Top/bottom mounted
                if weapon_position.z > 0:
                    outward = Vector3D(0, 0, 1)  # Dorsal
                else:
                    outward = Vector3D(0, 0, -1)  # Ventral

            return FiringArc(
                weapon_position=weapon_position,
                forward_direction=outward,
                half_angle_deg=TURRET_FIRING_ARC_HALF_ANGLE,
                weapon_type=WeaponType.TURRET,
                can_fire_full_sphere=False
            )

    # -------------------------------------------------------------------------
    # Cross-Section Calculation
    # -------------------------------------------------------------------------

    def get_cross_section_area(
        self,
        viewing_angle: Vector3D
    ) -> float:
        """
        Calculate effective cross-section area from a viewing angle.

        This affects hit probability - smaller cross-section = harder to hit.

        Args:
            viewing_angle: Direction FROM which the ship is being viewed
                (unit vector, in ship-local or world coordinates)

        Returns:
            Effective cross-section area in square meters
        """
        view_dir = viewing_angle.normalized()

        # Calculate angle from forward axis
        # View from ahead means view_dir points opposite to forward (+X)
        forward = Vector3D.unit_x()
        angle_rad = view_dir.angle_to(forward)
        angle_deg = math.degrees(angle_rad)

        # Cross-section components
        # Nose-on (angle ~ 180): circular cross-section (pi * r^2)
        # Broadside (angle ~ 90): rectangular cross-section (length * 2*r)

        # Use cosine interpolation for smooth transition
        # angle = 180: cos_factor = -1 (nose-on)
        # angle = 90: cos_factor = 0 (broadside)
        # angle = 0: cos_factor = 1 (tail-on)
        cos_angle = math.cos(angle_rad)

        # Frontal area (circular)
        frontal_area = math.pi * self.radius_m ** 2

        # Side area (rectangular projection)
        side_area = self.length_m * 2 * self.radius_m

        # Blend based on viewing angle
        # abs(cos) = 1: pure frontal/rear view
        # abs(cos) = 0: pure side view
        frontal_factor = abs(cos_angle)
        side_factor = math.sqrt(1 - cos_angle ** 2)  # sin(angle)

        # Effective cross-section
        effective_area = (
            frontal_factor * frontal_area +
            side_factor * side_area
        )

        return effective_area

    def get_frontal_cross_section(self) -> float:
        """
        Get minimum cross-section (nose-on view).

        Returns:
            Frontal area in square meters
        """
        return math.pi * self.radius_m ** 2

    def get_broadside_cross_section(self) -> float:
        """
        Get maximum cross-section (broadside view).

        Returns:
            Side area in square meters
        """
        return self.length_m * 2 * self.radius_m

    @property
    def nose_cross_section_m2(self) -> float:
        """Cross-section when viewing from nose (frontal area)."""
        return self.get_frontal_cross_section()

    @property
    def tail_cross_section_m2(self) -> float:
        """Cross-section when viewing from tail (slightly larger due to engines)."""
        # Tail is slightly larger due to engine bells
        return self.get_frontal_cross_section() * 1.2

    @property
    def lateral_cross_section_m2(self) -> float:
        """Cross-section when viewing from side (broadside area)."""
        return self.get_broadside_cross_section()

    # -------------------------------------------------------------------------
    # Module Hit Determination
    # -------------------------------------------------------------------------

    def get_module_at_hit_point(
        self,
        hit_point: HitPoint,
        module_layout: ModuleLayout
    ) -> Optional[Module]:
        """
        Determine which module would be hit based on penetration path.

        Args:
            hit_point: Precise hit location on ship surface
            module_layout: Ship's internal module layout

        Returns:
            Module that would be hit first, or None if no module in path
        """
        if not module_layout.layers:
            return None

        # Calculate which layer the hit is in based on axial position
        layer_depth = self.length_m / module_layout.total_layers
        layer_index = int(hit_point.axial_position_m / layer_depth)
        layer_index = max(0, min(layer_index, module_layout.total_layers - 1))

        # Get modules at this layer
        modules = module_layout.get_modules_at_layer(layer_index)
        if not modules:
            return None

        # Find module closest to hit point's lateral position
        # Convert radial angle to lateral offset
        radial_rad = math.radians(hit_point.radial_angle_deg)
        # 0 deg = dorsal (z+), 90 deg = starboard (y+)
        # Approximate lateral offset from radial position
        lateral_offset = self.radius_m * math.sin(radial_rad)

        # Find closest module
        closest_module: Optional[Module] = None
        closest_distance = float('inf')

        for module in modules:
            distance = abs(module.position.lateral_offset - lateral_offset)
            # Also consider module size (larger modules more likely to be hit)
            effective_distance = distance - math.sqrt(module.size_m2) / 2
            if effective_distance < closest_distance:
                closest_distance = effective_distance
                closest_module = module

        return closest_module

    def get_modules_in_penetration_path(
        self,
        hit_point: HitPoint,
        module_layout: ModuleLayout,
        penetration_depth_layers: int = 3
    ) -> list[Module]:
        """
        Get all modules along a penetration path from a hit point.

        Args:
            hit_point: Entry point on ship surface
            module_layout: Ship's internal module layout
            penetration_depth_layers: How many layers the projectile can penetrate

        Returns:
            List of modules in order of penetration (first hit to last)
        """
        if not module_layout.layers:
            return []

        modules_hit: list[Module] = []
        layer_depth = self.length_m / module_layout.total_layers

        # Determine starting layer and direction based on hit location
        if hit_point.location == HitLocation.NOSE:
            start_layer = 0
            direction = 1  # Penetrate toward tail
        elif hit_point.location == HitLocation.TAIL:
            start_layer = module_layout.total_layers - 1
            direction = -1  # Penetrate toward nose
        else:
            # Lateral hit - start at calculated layer, penetrate inward
            start_layer = int(hit_point.axial_position_m / layer_depth)
            start_layer = max(0, min(start_layer, module_layout.total_layers - 1))
            direction = 0  # Stay at same layer (lateral penetration)

        # Convert radial angle to lateral offset for lateral hits
        radial_rad = math.radians(hit_point.radial_angle_deg)
        lateral_offset = self.radius_m * math.sin(radial_rad)

        # Traverse layers
        for i in range(penetration_depth_layers):
            if direction == 0:
                # Lateral penetration - check centerline modules
                layer_idx = start_layer
            else:
                layer_idx = start_layer + i * direction

            if layer_idx < 0 or layer_idx >= module_layout.total_layers:
                break

            modules = module_layout.get_modules_at_layer(layer_idx)

            # Find modules in penetration path
            for module in modules:
                if direction == 0:
                    # Lateral hit - check if module is between surface and centerline
                    if abs(module.position.lateral_offset) <= abs(lateral_offset):
                        modules_hit.append(module)
                else:
                    # Axial penetration - check centerline modules
                    if module.position.distance_from_center() < self.radius_m * 0.5:
                        modules_hit.append(module)

        return modules_hit

    # -------------------------------------------------------------------------
    # Surface Area Calculations
    # -------------------------------------------------------------------------

    def get_total_surface_area(self) -> float:
        """
        Calculate total ship surface area.

        Returns:
            Surface area in square meters
        """
        # Nose cone (cone surface area)
        slant_height_nose = math.sqrt(
            self.nose_cone_length_m ** 2 + self.radius_m ** 2
        )
        nose_area = math.pi * self.radius_m * slant_height_nose

        # Main cylinder (lateral surface)
        cylinder_area = 2 * math.pi * self.radius_m * self.main_cylinder_length_m

        # Engine section (cone surface area)
        slant_height_engine = math.sqrt(
            self.engine_section_length_m ** 2 + self.radius_m ** 2
        )
        engine_area = math.pi * self.radius_m * slant_height_engine

        return nose_area + cylinder_area + engine_area

    def get_section_areas(self) -> dict[str, float]:
        """
        Get surface area breakdown by section.

        Returns:
            Dict with nose, lateral, and tail areas in m^2
        """
        slant_nose = math.sqrt(
            self.nose_cone_length_m ** 2 + self.radius_m ** 2
        )
        slant_engine = math.sqrt(
            self.engine_section_length_m ** 2 + self.radius_m ** 2
        )

        return {
            "nose": math.pi * self.radius_m * slant_nose,
            "lateral": 2 * math.pi * self.radius_m * self.main_cylinder_length_m,
            "tail": math.pi * self.radius_m * slant_engine
        }

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __str__(self) -> str:
        return (
            f"ShipGeometry({self.ship_type}: "
            f"L={self.length_m}m, R={self.radius_m:.1f}m)"
        )

    def __repr__(self) -> str:
        return (
            f"ShipGeometry(length_m={self.length_m}, radius_m={self.radius_m}, "
            f"nose_cone_length_m={self.nose_cone_length_m}, "
            f"engine_section_length_m={self.engine_section_length_m}, "
            f"ship_type='{self.ship_type}')"
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_geometry_from_fleet_data(
    ship_type: str,
    fleet_data: dict
) -> ShipGeometry:
    """
    Create geometry for a ship type from fleet data.

    Args:
        ship_type: Ship type identifier (e.g., 'destroyer')
        fleet_data: Loaded fleet_ships.json data

    Returns:
        Configured ShipGeometry

    Raises:
        KeyError: If ship_type not found in fleet data
    """
    ships = fleet_data.get("ships", {})
    if ship_type not in ships:
        raise KeyError(f"Ship type '{ship_type}' not found in fleet data")

    return ShipGeometry.from_ship_data(ships[ship_type])


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_hit_probability_modifier(
    geometry: ShipGeometry,
    viewing_angle: Vector3D
) -> float:
    """
    Calculate hit probability modifier based on target cross-section.

    Normalized such that broadside = 1.0, nose-on < 1.0.

    Args:
        geometry: Target ship geometry
        viewing_angle: Direction to target from attacker

    Returns:
        Hit probability modifier (0.0 to 1.0+)
    """
    current_area = geometry.get_cross_section_area(viewing_angle)
    broadside_area = geometry.get_broadside_cross_section()

    if broadside_area <= 0:
        return 1.0

    return current_area / broadside_area


# =============================================================================
# EXAMPLE USAGE / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    import json
    from pathlib import Path

    print("=" * 70)
    print("AI COMMANDERS GEOMETRY MODULE - SELF TEST")
    print("=" * 70)

    # Load fleet data
    data_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"

    if data_path.exists():
        with open(data_path) as f:
            fleet_data = json.load(f)

        print("\n--- Ship Geometries ---")
        for ship_type in ["corvette", "destroyer", "cruiser", "battleship", "dreadnought"]:
            geom = create_geometry_from_fleet_data(ship_type, fleet_data)
            print(f"\n{ship_type.upper()}:")
            print(f"  Length: {geom.length_m} m")
            print(f"  Radius: {geom.radius_m:.2f} m")
            print(f"  Nose cone: {geom.nose_cone_length_m:.1f} m")
            print(f"  Engine section: {geom.engine_section_length_m:.1f} m")
            print(f"  Main cylinder: {geom.main_cylinder_length_m:.1f} m")
            print(f"  Frontal cross-section: {geom.get_frontal_cross_section():.1f} m^2")
            print(f"  Broadside cross-section: {geom.get_broadside_cross_section():.1f} m^2")
            print(f"  Total surface area: {geom.get_total_surface_area():.1f} m^2")

        # Test hit location calculation
        print("\n--- Hit Location Tests ---")
        destroyer = create_geometry_from_fleet_data("destroyer", fleet_data)
        ship_fwd = Vector3D.unit_x()

        test_cases = [
            (Vector3D(-1, 0, 0), "from ahead"),
            (Vector3D(1, 0, 0), "from behind"),
            (Vector3D(0, 1, 0), "from starboard"),
            (Vector3D(0, -1, 0), "from port"),
            (Vector3D(-0.9, 0.4, 0), "from ahead-starboard"),
        ]

        for impact, desc in test_cases:
            location = destroyer.calculate_hit_location(impact, ship_fwd)
            print(f"  Impact {desc}: {location.value}")

        # Test hit point calculation
        print("\n--- Hit Point Tests ---")
        for impact, desc in test_cases[:3]:
            hit = destroyer.calculate_hit_point(impact, Vector3D.zero(), ship_fwd)
            print(f"  {desc}: {hit}")

        # Test weapon placement
        print("\n--- Weapon Placement ---")
        nose_pos = destroyer.get_weapon_position("nose", 0)
        hull_pos = destroyer.get_weapon_position("hull", 0)
        engine_pos = destroyer.get_weapon_position("engine", 0)
        print(f"  Nose mount 0: {nose_pos}")
        print(f"  Hull mount 0: {hull_pos}")
        print(f"  Engine mount 0: {engine_pos}")

        # Test firing arcs
        print("\n--- Firing Arcs ---")
        spinal_arc = destroyer.get_weapon_firing_arc(nose_pos, "spinal_coiler")
        turret_arc = destroyer.get_weapon_firing_arc(hull_pos, "coilgun")
        pd_arc = destroyer.get_weapon_firing_arc(hull_pos, "pd_laser")
        print(f"  Spinal arc: {spinal_arc.half_angle_deg * 2} deg cone")
        print(f"  Turret arc: {turret_arc.half_angle_deg * 2} deg hemisphere")
        print(f"  PD arc: full sphere = {pd_arc.can_fire_full_sphere}")

        # Test cross-section variation
        print("\n--- Cross-Section by Viewing Angle ---")
        angles = [
            (Vector3D(-1, 0, 0), "nose-on"),
            (Vector3D(0, 1, 0), "broadside"),
            (Vector3D(1, 0, 0), "tail-on"),
            (Vector3D(-0.707, 0.707, 0), "45 deg"),
        ]
        for view, desc in angles:
            area = destroyer.get_cross_section_area(view)
            modifier = calculate_hit_probability_modifier(destroyer, view)
            print(f"  {desc}: {area:.1f} m^2 (hit mod: {modifier:.2f})")

    else:
        print(f"Fleet data not found at {data_path}")

        # Create a test geometry manually
        print("\nCreating test geometry...")
        test_geom = ShipGeometry(
            length_m=100.0,
            radius_m=12.5,
            nose_cone_length_m=15.0,
            engine_section_length_m=15.0,
            ship_type="test"
        )
        print(f"Test geometry: {test_geom}")
        print(f"Frontal area: {test_geom.get_frontal_cross_section():.1f} m^2")
        print(f"Broadside area: {test_geom.get_broadside_cross_section():.1f} m^2")

    print("\n" + "=" * 70)
    print("Geometry tests completed!")
    print("=" * 70)
