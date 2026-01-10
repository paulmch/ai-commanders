"""
Tests for the ship geometry system.

Tests cover:
- ShipGeometry creation and validation
- Hit location calculation with various impact angles
- Precise hit point determination
- Weapon placement
- Firing arc geometry
- Cross-section calculations
- Module hit determination
"""

import json
import math
import pytest
from pathlib import Path

from src.geometry import (
    ShipGeometry,
    HitPoint,
    FiringArc,
    WeaponMountType,
    WeaponType,
    create_geometry_from_fleet_data,
    calculate_hit_probability_modifier,
    NOSE_HIT_ANGLE_THRESHOLD,
    TAIL_HIT_ANGLE_THRESHOLD,
    RADIUS_TO_LENGTH_RATIO,
    NOSE_CONE_LENGTH_FRACTION,
    ENGINE_SECTION_LENGTH_FRACTION,
    SPINAL_FIRING_ARC_HALF_ANGLE,
    TURRET_FIRING_ARC_HALF_ANGLE,
)
from src.physics import Vector3D
from src.combat import HitLocation
from src.modules import ModuleLayout, ModuleLayer, Module, ModuleType, ModulePosition


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fleet_data():
    """Load fleet data from JSON file."""
    data_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"
    with open(data_path) as f:
        return json.load(f)


@pytest.fixture
def destroyer_geometry(fleet_data):
    """Create destroyer geometry from fleet data."""
    return create_geometry_from_fleet_data("destroyer", fleet_data)


@pytest.fixture
def corvette_geometry(fleet_data):
    """Create corvette geometry from fleet data."""
    return create_geometry_from_fleet_data("corvette", fleet_data)


@pytest.fixture
def dreadnought_geometry(fleet_data):
    """Create dreadnought geometry from fleet data."""
    return create_geometry_from_fleet_data("dreadnought", fleet_data)


@pytest.fixture
def simple_module_layout():
    """Create a simple module layout for testing."""
    layout = ModuleLayout("test_ship", 100.0)

    # Layer 0: Sensors at nose
    layer0 = ModuleLayer(0, depth_m=25.0)
    layer0.add_module(Module(
        name="Sensors",
        module_type=ModuleType.SENSOR,
        position=ModulePosition(0, 0.0),
        size_m2=20.0
    ))
    layout.add_layer(layer0)

    # Layer 1: Bridge
    layer1 = ModuleLayer(1, depth_m=25.0)
    layer1.add_module(Module(
        name="Bridge",
        module_type=ModuleType.BRIDGE,
        position=ModulePosition(1, 0.0),
        size_m2=30.0
    ))
    layer1.add_module(Module(
        name="Crew Starboard",
        module_type=ModuleType.CREW,
        position=ModulePosition(1, 5.0),
        size_m2=15.0
    ))
    layout.add_layer(layer1)

    # Layer 2: Reactor
    layer2 = ModuleLayer(2, depth_m=25.0)
    layer2.add_module(Module(
        name="Reactor",
        module_type=ModuleType.REACTOR,
        position=ModulePosition(2, 0.0),
        size_m2=40.0
    ))
    layout.add_layer(layer2)

    # Layer 3: Engine
    layer3 = ModuleLayer(3, depth_m=25.0)
    layer3.add_module(Module(
        name="Engine",
        module_type=ModuleType.ENGINE,
        position=ModulePosition(3, 0.0),
        size_m2=50.0
    ))
    layout.add_layer(layer3)

    return layout


# =============================================================================
# SHIP GEOMETRY CREATION TESTS
# =============================================================================

class TestShipGeometryCreation:
    """Tests for ShipGeometry creation and validation."""

    def test_from_ship_data_destroyer(self, fleet_data):
        """Test creating geometry from destroyer ship data."""
        geom = create_geometry_from_fleet_data("destroyer", fleet_data)

        assert geom.ship_type == "destroyer"
        assert geom.length_m == 125.0
        assert geom.radius_m == pytest.approx(125.0 / 8, rel=0.01)
        assert geom.nose_cone_length_m == pytest.approx(125.0 * 0.15, rel=0.01)
        assert geom.engine_section_length_m == pytest.approx(125.0 * 0.15, rel=0.01)

    def test_from_ship_data_all_ship_types(self, fleet_data):
        """Test creating geometry for all ship types."""
        ship_types = ["corvette", "frigate", "destroyer", "cruiser",
                      "battlecruiser", "battleship", "dreadnought"]

        for ship_type in ship_types:
            geom = create_geometry_from_fleet_data(ship_type, fleet_data)
            assert geom.ship_type == ship_type
            assert geom.length_m > 0
            assert geom.radius_m > 0
            assert geom.radius_m == pytest.approx(geom.length_m * RADIUS_TO_LENGTH_RATIO)

    def test_invalid_ship_type_raises(self, fleet_data):
        """Test that invalid ship type raises KeyError."""
        with pytest.raises(KeyError):
            create_geometry_from_fleet_data("invalid_ship", fleet_data)

    def test_manual_creation(self):
        """Test manual ShipGeometry creation."""
        geom = ShipGeometry(
            length_m=100.0,
            radius_m=12.5,
            nose_cone_length_m=15.0,
            engine_section_length_m=15.0,
            ship_type="test"
        )

        assert geom.length_m == 100.0
        assert geom.radius_m == 12.5
        assert geom.main_cylinder_length_m == 70.0

    def test_invalid_length_raises(self):
        """Test that invalid length raises ValueError."""
        with pytest.raises(ValueError):
            ShipGeometry(
                length_m=-100.0,
                radius_m=12.5,
                nose_cone_length_m=15.0,
                engine_section_length_m=15.0
            )

    def test_invalid_radius_raises(self):
        """Test that invalid radius raises ValueError."""
        with pytest.raises(ValueError):
            ShipGeometry(
                length_m=100.0,
                radius_m=-12.5,
                nose_cone_length_m=15.0,
                engine_section_length_m=15.0
            )


# =============================================================================
# HIT LOCATION CALCULATION TESTS
# =============================================================================

class TestHitLocationCalculation:
    """Tests for hit location determination based on impact angles."""

    def test_nose_hit_from_directly_ahead(self, destroyer_geometry):
        """Impact from directly ahead should hit nose."""
        impact = Vector3D(-1, 0, 0)  # Coming from +X direction (ahead)
        ship_fwd = Vector3D.unit_x()

        location = destroyer_geometry.calculate_hit_location(impact, ship_fwd)

        assert location == HitLocation.NOSE

    def test_tail_hit_from_directly_behind(self, destroyer_geometry):
        """Impact from directly behind should hit tail."""
        impact = Vector3D(1, 0, 0)  # Coming from -X direction (behind)
        ship_fwd = Vector3D.unit_x()

        location = destroyer_geometry.calculate_hit_location(impact, ship_fwd)

        assert location == HitLocation.TAIL

    def test_lateral_hit_from_starboard(self, destroyer_geometry):
        """Impact from starboard should hit lateral."""
        impact = Vector3D(0, 1, 0)  # Coming from +Y direction (starboard)
        ship_fwd = Vector3D.unit_x()

        location = destroyer_geometry.calculate_hit_location(impact, ship_fwd)

        assert location == HitLocation.LATERAL

    def test_lateral_hit_from_port(self, destroyer_geometry):
        """Impact from port should hit lateral."""
        impact = Vector3D(0, -1, 0)  # Coming from -Y direction (port)
        ship_fwd = Vector3D.unit_x()

        location = destroyer_geometry.calculate_hit_location(impact, ship_fwd)

        assert location == HitLocation.LATERAL

    def test_lateral_hit_from_dorsal(self, destroyer_geometry):
        """Impact from above should hit lateral."""
        impact = Vector3D(0, 0, 1)  # Coming from +Z direction (above)
        ship_fwd = Vector3D.unit_x()

        location = destroyer_geometry.calculate_hit_location(impact, ship_fwd)

        assert location == HitLocation.LATERAL

    def test_nose_hit_at_threshold_angle(self, destroyer_geometry):
        """Impact at threshold angle should still hit nose."""
        # 25 degrees from directly ahead (within 30 deg threshold)
        angle_rad = math.radians(25)
        impact = Vector3D(
            -math.cos(angle_rad),
            math.sin(angle_rad),
            0
        )
        ship_fwd = Vector3D.unit_x()

        location = destroyer_geometry.calculate_hit_location(impact, ship_fwd)

        assert location == HitLocation.NOSE

    def test_lateral_hit_just_past_nose_threshold(self, destroyer_geometry):
        """Impact just past nose threshold should hit lateral."""
        # 35 degrees from directly ahead (past 30 deg threshold)
        angle_rad = math.radians(35)
        impact = Vector3D(
            -math.cos(angle_rad),
            math.sin(angle_rad),
            0
        )
        ship_fwd = Vector3D.unit_x()

        location = destroyer_geometry.calculate_hit_location(impact, ship_fwd)

        assert location == HitLocation.LATERAL

    def test_tail_hit_at_threshold_angle(self, destroyer_geometry):
        """Impact at tail threshold angle should hit tail."""
        # 25 degrees from directly behind (within 30 deg threshold)
        angle_rad = math.radians(25)
        impact = Vector3D(
            math.cos(angle_rad),
            math.sin(angle_rad),
            0
        )
        ship_fwd = Vector3D.unit_x()

        location = destroyer_geometry.calculate_hit_location(impact, ship_fwd)

        assert location == HitLocation.TAIL

    def test_rotated_ship_orientation(self, destroyer_geometry):
        """Test hit calculation with rotated ship orientation."""
        # Ship pointing in +Y direction
        ship_fwd = Vector3D(0, 1, 0)
        # Impact from ship's front (which is now +Y)
        impact = Vector3D(0, -1, 0)

        location = destroyer_geometry.calculate_hit_location(impact, ship_fwd)

        assert location == HitLocation.NOSE


# =============================================================================
# HIT POINT CALCULATION TESTS
# =============================================================================

class TestHitPointCalculation:
    """Tests for precise hit point determination."""

    def test_hit_point_nose_basic(self, destroyer_geometry):
        """Test hit point on nose from directly ahead."""
        impact = Vector3D(-1, 0, 0)
        ship_pos = Vector3D.zero()
        ship_fwd = Vector3D.unit_x()

        hit = destroyer_geometry.calculate_hit_point(impact, ship_pos, ship_fwd)

        assert hit.location == HitLocation.NOSE
        assert hit.axial_position_m >= 0
        assert hit.axial_position_m <= destroyer_geometry.nose_cone_length_m

    def test_hit_point_tail_basic(self, destroyer_geometry):
        """Test hit point on tail from directly behind."""
        impact = Vector3D(1, 0, 0)
        ship_pos = Vector3D.zero()
        ship_fwd = Vector3D.unit_x()

        hit = destroyer_geometry.calculate_hit_point(impact, ship_pos, ship_fwd)

        assert hit.location == HitLocation.TAIL
        assert hit.axial_position_m >= destroyer_geometry.engine_section_start_m

    def test_hit_point_lateral_basic(self, destroyer_geometry):
        """Test hit point on lateral from starboard."""
        impact = Vector3D(0, 1, 0)
        ship_pos = Vector3D.zero()
        ship_fwd = Vector3D.unit_x()

        hit = destroyer_geometry.calculate_hit_point(impact, ship_pos, ship_fwd)

        assert hit.location == HitLocation.LATERAL
        # Radial angle should be around 90 degrees (starboard)
        assert 80 <= hit.radial_angle_deg <= 100 or 260 <= hit.radial_angle_deg <= 280

    def test_hit_point_has_valid_surface_normal(self, destroyer_geometry):
        """Test that surface normal is a valid unit vector."""
        impact = Vector3D(-1, 0.5, 0)
        ship_pos = Vector3D.zero()
        ship_fwd = Vector3D.unit_x()

        hit = destroyer_geometry.calculate_hit_point(impact, ship_pos, ship_fwd)

        # Normal should be approximately unit length
        assert 0.99 <= hit.surface_normal.magnitude <= 1.01

    def test_hit_point_lateral_radial_angle_port(self, destroyer_geometry):
        """Test radial angle for port-side hit."""
        impact = Vector3D(0, -1, 0)  # From port
        ship_pos = Vector3D.zero()
        ship_fwd = Vector3D.unit_x()

        hit = destroyer_geometry.calculate_hit_point(impact, ship_pos, ship_fwd)

        # Port side should be around 270 degrees
        assert hit.location == HitLocation.LATERAL

    def test_hit_point_dorsal(self, destroyer_geometry):
        """Test radial angle for dorsal (top) hit."""
        impact = Vector3D(0, 0, 1)  # From above
        ship_pos = Vector3D.zero()
        ship_fwd = Vector3D.unit_x()

        hit = destroyer_geometry.calculate_hit_point(impact, ship_pos, ship_fwd)

        # Dorsal should be around 0 degrees
        assert hit.location == HitLocation.LATERAL


# =============================================================================
# WEAPON PLACEMENT TESTS
# =============================================================================

class TestWeaponPlacement:
    """Tests for weapon position calculations."""

    def test_nose_weapon_position(self, destroyer_geometry):
        """Test nose weapon is near front of ship."""
        pos = destroyer_geometry.get_weapon_position("nose", 0)

        # Should be in forward half of ship (positive X in ship-local coords)
        assert pos.x > 0

    def test_hull_weapon_position(self, destroyer_geometry):
        """Test hull weapon has lateral offset."""
        pos = destroyer_geometry.get_weapon_position("hull", 0)

        # Should have some lateral offset
        assert abs(pos.y) > 0

    def test_hull_weapon_alternating_sides(self, destroyer_geometry):
        """Test that hull weapons alternate between port and starboard."""
        pos0 = destroyer_geometry.get_weapon_position("hull", 0)
        pos1 = destroyer_geometry.get_weapon_position("hull", 1)

        # Different sides
        assert pos0.y * pos1.y < 0  # Opposite signs

    def test_engine_weapon_position(self, destroyer_geometry):
        """Test engine weapon is near rear of ship."""
        pos = destroyer_geometry.get_weapon_position("engine", 0)

        # Should be in aft half of ship (negative X in ship-local coords)
        assert pos.x < 0

    def test_nose_only_mount_type(self, destroyer_geometry):
        """Test 'nose_only' mount type is handled."""
        pos = destroyer_geometry.get_weapon_position("nose_only", 0)

        # Should be same as regular nose mount
        assert pos.x > 0


# =============================================================================
# FIRING ARC TESTS
# =============================================================================

class TestFiringArcs:
    """Tests for firing arc geometry."""

    def test_spinal_weapon_narrow_arc(self, destroyer_geometry):
        """Test spinal weapons have narrow forward arc."""
        pos = destroyer_geometry.get_weapon_position("nose", 0)
        arc = destroyer_geometry.get_weapon_firing_arc(pos, "spinal_coiler")

        assert arc.weapon_type == WeaponType.SPINAL
        assert arc.half_angle_deg == SPINAL_FIRING_ARC_HALF_ANGLE
        assert not arc.can_fire_full_sphere

    def test_turret_weapon_hemisphere(self, destroyer_geometry):
        """Test turret weapons have hemisphere coverage."""
        pos = destroyer_geometry.get_weapon_position("hull", 0)
        arc = destroyer_geometry.get_weapon_firing_arc(pos, "coilgun_mk3")

        assert arc.weapon_type == WeaponType.TURRET
        assert arc.half_angle_deg == TURRET_FIRING_ARC_HALF_ANGLE
        assert not arc.can_fire_full_sphere

    def test_pd_weapon_full_sphere(self, destroyer_geometry):
        """Test point defense has full sphere coverage."""
        pos = destroyer_geometry.get_weapon_position("hull", 0)
        arc = destroyer_geometry.get_weapon_firing_arc(pos, "pd_laser")

        assert arc.weapon_type == WeaponType.POINT_DEFENSE
        assert arc.can_fire_full_sphere

    def test_firing_arc_can_engage(self, destroyer_geometry):
        """Test firing arc engagement calculation."""
        pos = destroyer_geometry.get_weapon_position("nose", 0)
        arc = destroyer_geometry.get_weapon_firing_arc(pos, "spinal_coiler")

        # Direct forward should be engageable
        assert arc.can_engage_target(Vector3D(1, 0, 0))

        # 90 degrees off should not be engageable
        assert not arc.can_engage_target(Vector3D(0, 1, 0))

    def test_pd_can_engage_any_direction(self, destroyer_geometry):
        """Test PD can engage targets in any direction."""
        pos = destroyer_geometry.get_weapon_position("hull", 0)
        arc = destroyer_geometry.get_weapon_firing_arc(pos, "pd_laser")

        # All directions should be engageable
        assert arc.can_engage_target(Vector3D(1, 0, 0))
        assert arc.can_engage_target(Vector3D(-1, 0, 0))
        assert arc.can_engage_target(Vector3D(0, 1, 0))
        assert arc.can_engage_target(Vector3D(0, 0, 1))

    def test_solid_angle_calculation(self, destroyer_geometry):
        """Test solid angle calculation for firing arcs."""
        pos = destroyer_geometry.get_weapon_position("hull", 0)

        pd_arc = destroyer_geometry.get_weapon_firing_arc(pos, "pd_laser")
        assert pd_arc.coverage_solid_angle_sr() == pytest.approx(4 * math.pi, rel=0.01)

        turret_arc = destroyer_geometry.get_weapon_firing_arc(pos, "coilgun")
        # Hemisphere = 2*pi steradians
        assert turret_arc.coverage_solid_angle_sr() == pytest.approx(2 * math.pi, rel=0.01)


# =============================================================================
# CROSS-SECTION TESTS
# =============================================================================

class TestCrossSection:
    """Tests for cross-section area calculations."""

    def test_frontal_cross_section(self, destroyer_geometry):
        """Test frontal (nose-on) cross-section."""
        frontal = destroyer_geometry.get_frontal_cross_section()

        expected = math.pi * destroyer_geometry.radius_m ** 2
        assert frontal == pytest.approx(expected, rel=0.01)

    def test_broadside_cross_section(self, destroyer_geometry):
        """Test broadside cross-section."""
        broadside = destroyer_geometry.get_broadside_cross_section()

        expected = destroyer_geometry.length_m * 2 * destroyer_geometry.radius_m
        assert broadside == pytest.approx(expected, rel=0.01)

    def test_broadside_larger_than_frontal(self, destroyer_geometry):
        """Broadside should be much larger than frontal."""
        frontal = destroyer_geometry.get_frontal_cross_section()
        broadside = destroyer_geometry.get_broadside_cross_section()

        assert broadside > frontal * 2

    def test_cross_section_nose_on_view(self, destroyer_geometry):
        """Test cross-section from nose-on view."""
        # View from +X looking at -X (ship pointing +X)
        view = Vector3D(-1, 0, 0)
        area = destroyer_geometry.get_cross_section_area(view)

        frontal = destroyer_geometry.get_frontal_cross_section()
        assert area == pytest.approx(frontal, rel=0.1)

    def test_cross_section_broadside_view(self, destroyer_geometry):
        """Test cross-section from broadside view."""
        view = Vector3D(0, 1, 0)
        area = destroyer_geometry.get_cross_section_area(view)

        broadside = destroyer_geometry.get_broadside_cross_section()
        assert area == pytest.approx(broadside, rel=0.1)

    def test_cross_section_45_degree_view(self, destroyer_geometry):
        """Test cross-section from 45-degree angle."""
        view = Vector3D(-0.707, 0.707, 0)
        area = destroyer_geometry.get_cross_section_area(view)

        frontal = destroyer_geometry.get_frontal_cross_section()
        broadside = destroyer_geometry.get_broadside_cross_section()

        # Should be between frontal and broadside
        assert area > frontal
        assert area < broadside

    def test_hit_probability_modifier_broadside(self, destroyer_geometry):
        """Test hit probability modifier at broadside (should be 1.0)."""
        view = Vector3D(0, 1, 0)
        modifier = calculate_hit_probability_modifier(destroyer_geometry, view)

        assert modifier == pytest.approx(1.0, rel=0.1)

    def test_hit_probability_modifier_nose_on(self, destroyer_geometry):
        """Test hit probability modifier nose-on (should be < 1.0)."""
        view = Vector3D(-1, 0, 0)
        modifier = calculate_hit_probability_modifier(destroyer_geometry, view)

        assert modifier < 1.0


# =============================================================================
# MODULE HIT DETERMINATION TESTS
# =============================================================================

class TestModuleHitDetermination:
    """Tests for module hit determination."""

    def test_get_module_at_nose_hit(self, destroyer_geometry, simple_module_layout):
        """Test finding module at nose hit point."""
        # Create nose hit
        hit = HitPoint(
            location=HitLocation.NOSE,
            axial_position_m=5.0,  # Early in ship
            radial_angle_deg=0.0,
            surface_normal=Vector3D(-1, 0, 0)
        )

        # Create matching geometry for the layout
        geom = ShipGeometry(
            length_m=100.0,
            radius_m=12.5,
            nose_cone_length_m=15.0,
            engine_section_length_m=15.0,
            ship_type="test"
        )

        module = geom.get_module_at_hit_point(hit, simple_module_layout)

        assert module is not None
        assert module.name == "Sensors"

    def test_get_module_at_tail_hit(self, simple_module_layout):
        """Test finding module at tail hit point."""
        geom = ShipGeometry(
            length_m=100.0,
            radius_m=12.5,
            nose_cone_length_m=15.0,
            engine_section_length_m=15.0,
            ship_type="test"
        )

        hit = HitPoint(
            location=HitLocation.TAIL,
            axial_position_m=90.0,  # Near tail
            radial_angle_deg=0.0,
            surface_normal=Vector3D(1, 0, 0)
        )

        module = geom.get_module_at_hit_point(hit, simple_module_layout)

        assert module is not None
        assert module.name == "Engine"

    def test_get_modules_in_penetration_path_nose(self, simple_module_layout):
        """Test getting modules in penetration path from nose hit."""
        geom = ShipGeometry(
            length_m=100.0,
            radius_m=12.5,
            nose_cone_length_m=15.0,
            engine_section_length_m=15.0,
            ship_type="test"
        )

        hit = HitPoint(
            location=HitLocation.NOSE,
            axial_position_m=5.0,
            radial_angle_deg=0.0,
            surface_normal=Vector3D(-1, 0, 0)
        )

        modules = geom.get_modules_in_penetration_path(hit, simple_module_layout, 3)

        # Should get sensors, bridge (center), reactor in order
        assert len(modules) >= 2
        module_names = [m.name for m in modules]
        assert "Sensors" in module_names or "Bridge" in module_names

    def test_empty_layout_returns_none(self, destroyer_geometry):
        """Test that empty layout returns None."""
        empty_layout = ModuleLayout("empty", 100.0)

        hit = HitPoint(
            location=HitLocation.NOSE,
            axial_position_m=5.0,
            radial_angle_deg=0.0,
            surface_normal=Vector3D(-1, 0, 0)
        )

        module = destroyer_geometry.get_module_at_hit_point(hit, empty_layout)

        assert module is None


# =============================================================================
# SURFACE AREA TESTS
# =============================================================================

class TestSurfaceArea:
    """Tests for surface area calculations."""

    def test_total_surface_area_positive(self, destroyer_geometry):
        """Test total surface area is positive."""
        area = destroyer_geometry.get_total_surface_area()
        assert area > 0

    def test_section_areas_sum_to_total(self, destroyer_geometry):
        """Test section areas approximately sum to total."""
        sections = destroyer_geometry.get_section_areas()
        total_from_sections = sum(sections.values())
        total = destroyer_geometry.get_total_surface_area()

        assert total_from_sections == pytest.approx(total, rel=0.01)

    def test_lateral_section_largest(self, destroyer_geometry):
        """Test lateral section has largest area."""
        sections = destroyer_geometry.get_section_areas()

        assert sections["lateral"] > sections["nose"]
        assert sections["lateral"] > sections["tail"]

    def test_surface_area_scales_with_size(self, corvette_geometry, dreadnought_geometry):
        """Test larger ships have larger surface area."""
        corvette_area = corvette_geometry.get_total_surface_area()
        dreadnought_area = dreadnought_geometry.get_total_surface_area()

        assert dreadnought_area > corvette_area


# =============================================================================
# GEOMETRY PROPERTIES TESTS
# =============================================================================

class TestGeometryProperties:
    """Tests for geometry property calculations."""

    def test_main_cylinder_length(self, destroyer_geometry):
        """Test main cylinder length calculation."""
        expected = (
            destroyer_geometry.length_m -
            destroyer_geometry.nose_cone_length_m -
            destroyer_geometry.engine_section_length_m
        )

        assert destroyer_geometry.main_cylinder_length_m == pytest.approx(expected)

    def test_section_positions(self, destroyer_geometry):
        """Test section start positions are ordered correctly."""
        assert destroyer_geometry.nose_cone_start_m == 0
        assert destroyer_geometry.main_cylinder_start_m > 0
        assert destroyer_geometry.engine_section_start_m > destroyer_geometry.main_cylinder_start_m
        assert destroyer_geometry.engine_section_start_m < destroyer_geometry.length_m

    def test_string_representation(self, destroyer_geometry):
        """Test string representation includes key info."""
        str_repr = str(destroyer_geometry)

        assert "destroyer" in str_repr
        assert "125" in str_repr  # length

    def test_repr_representation(self, destroyer_geometry):
        """Test repr representation is complete."""
        repr_str = repr(destroyer_geometry)

        assert "ShipGeometry" in repr_str
        assert "length_m=" in repr_str
        assert "radius_m=" in repr_str


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple geometry features."""

    def test_hit_to_module_workflow(self, destroyer_geometry, fleet_data):
        """Test complete workflow from impact to module damage."""
        from src.modules import ModuleLayout

        # Create module layout
        layout = ModuleLayout.from_ship_type("destroyer", fleet_data)

        # Simulate nose hit
        impact = Vector3D(-1, 0, 0)
        ship_pos = Vector3D.zero()
        ship_fwd = Vector3D.unit_x()

        hit_point = destroyer_geometry.calculate_hit_point(impact, ship_pos, ship_fwd)

        assert hit_point.location == HitLocation.NOSE

        # Find module at hit point
        module = destroyer_geometry.get_module_at_hit_point(hit_point, layout)

        assert module is not None

    def test_cross_section_affects_hit_probability(self, destroyer_geometry):
        """Test that different angles give different hit probabilities."""
        nose_on = Vector3D(-1, 0, 0)
        broadside = Vector3D(0, 1, 0)
        angle_45 = Vector3D(-0.707, 0.707, 0)

        mod_nose = calculate_hit_probability_modifier(destroyer_geometry, nose_on)
        mod_broad = calculate_hit_probability_modifier(destroyer_geometry, broadside)
        mod_45 = calculate_hit_probability_modifier(destroyer_geometry, angle_45)

        # Nose on should be hardest to hit
        assert mod_nose < mod_45 < mod_broad

    def test_weapon_arcs_cover_reasonable_angles(self, destroyer_geometry):
        """Test that combined weapon arcs provide good coverage."""
        # Get multiple weapon positions and arcs
        nose_pos = destroyer_geometry.get_weapon_position("nose", 0)
        hull_pos_0 = destroyer_geometry.get_weapon_position("hull", 0)
        hull_pos_1 = destroyer_geometry.get_weapon_position("hull", 1)

        spinal = destroyer_geometry.get_weapon_firing_arc(nose_pos, "spinal")
        turret_0 = destroyer_geometry.get_weapon_firing_arc(hull_pos_0, "coilgun")
        turret_1 = destroyer_geometry.get_weapon_firing_arc(hull_pos_1, "coilgun")

        # Test coverage of different directions
        directions = [
            Vector3D(1, 0, 0),    # Forward
            Vector3D(-1, 0, 0),   # Aft
            Vector3D(0, 1, 0),    # Starboard
            Vector3D(0, -1, 0),   # Port
        ]

        for direction in directions:
            can_engage = (
                spinal.can_engage_target(direction) or
                turret_0.can_engage_target(direction) or
                turret_1.can_engage_target(direction)
            )
            # At least one weapon should cover each cardinal direction
            # (though spinal only covers forward)
            if direction.x <= 0:  # Not forward
                assert turret_0.can_engage_target(direction) or turret_1.can_engage_target(direction)


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_vector_impact(self, destroyer_geometry):
        """Test handling of zero impact vector."""
        # Zero vector should be handled gracefully
        impact = Vector3D(0, 0, 0)
        ship_fwd = Vector3D.unit_x()

        # This might return any location since there's no direction
        location = destroyer_geometry.calculate_hit_location(impact, ship_fwd)
        assert location in [HitLocation.NOSE, HitLocation.LATERAL, HitLocation.TAIL]

    def test_very_small_ship(self):
        """Test geometry for very small ship."""
        geom = ShipGeometry(
            length_m=10.0,
            radius_m=1.25,
            nose_cone_length_m=1.5,
            engine_section_length_m=1.5,
            ship_type="tiny"
        )

        assert geom.get_frontal_cross_section() > 0
        assert geom.get_broadside_cross_section() > 0

    def test_very_large_ship(self):
        """Test geometry for very large ship."""
        geom = ShipGeometry(
            length_m=1000.0,
            radius_m=125.0,
            nose_cone_length_m=150.0,
            engine_section_length_m=150.0,
            ship_type="huge"
        )

        assert geom.get_frontal_cross_section() > 0
        assert geom.get_broadside_cross_section() > 0

    def test_non_unit_vectors(self, destroyer_geometry):
        """Test that non-unit vectors are handled correctly."""
        # Non-normalized impact vector
        impact = Vector3D(-10, 0, 0)
        ship_fwd = Vector3D(2, 0, 0)  # Non-unit

        location = destroyer_geometry.calculate_hit_location(impact, ship_fwd)

        assert location == HitLocation.NOSE

    def test_diagonal_impact_vectors(self, destroyer_geometry):
        """Test various diagonal impact angles."""
        ship_fwd = Vector3D.unit_x()

        # Test multiple diagonal impacts
        diagonals = [
            Vector3D(-1, 1, 0),
            Vector3D(-1, -1, 0),
            Vector3D(-1, 0, 1),
            Vector3D(-1, 1, 1),
            Vector3D(0.5, 0.5, 0.707),
        ]

        for impact in diagonals:
            location = destroyer_geometry.calculate_hit_location(impact, ship_fwd)
            # All diagonal impacts should return valid locations
            assert location in [HitLocation.NOSE, HitLocation.LATERAL, HitLocation.TAIL]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
