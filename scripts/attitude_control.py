#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Attitude Control Calculator for AI Commanders

Calculates rotational dynamics for space combat ships:
- Thrust vectoring from main engines (combat rotation while accelerating)
- RCS for rotation when engines are off
"""

import json
import math
from pathlib import Path

# Main drive specs
MAIN_THRUST_MN = 58.56
MAIN_THRUST_N = MAIN_THRUST_MN * 1e6

# Thrust vectoring parameters
COMBAT_DEFLECTION_DEG = 1.0  # Conservative deflection for magnetic nozzle
MAX_DEFLECTION_DEG = 3.0     # Maximum safe deflection

# RCS system (VectorThrusters module treated as RCS)
RCS_TORQUE_NM = 1_500_000  # From TI VectorThrusters module
RCS_MASS_T = 20


def calculate_moment_of_inertia(mass_kg, length_m):
    """
    Calculate moment of inertia for ship rotation.
    Models ship as elongated cylinder.
    """
    # Pitch/Yaw - rotating perpendicular to length
    I_pitch_yaw = (1/12) * mass_kg * (length_m ** 2)

    # Roll - rotating about the long axis (width ≈ length/4)
    radius = length_m / 8  # radius = width/2 = (length/4)/2
    I_roll = (1/2) * mass_kg * (radius ** 2)

    return {
        'pitch_yaw_kg_m2': I_pitch_yaw,
        'roll_kg_m2': I_roll
    }


def calculate_thrust_vectoring(mass_kg, length_m, deflection_deg):
    """
    Calculate rotation from main engine thrust vectoring.

    Assumes:
    - Engines at tail of ship
    - CoM at ~55% from nose (aft-heavy due to reactor)
    - Lever arm = distance from CoM to engine mount
    """
    # Lever arm: CoM at 55% from nose, engines at 100%
    # So lever arm = 45% of length
    lever_arm = length_m * 0.45

    # Lateral force from deflection
    deflection_rad = math.radians(deflection_deg)
    lateral_force = MAIN_THRUST_N * math.sin(deflection_rad)

    # Torque
    torque = lateral_force * lever_arm

    # Moment of inertia
    moi = calculate_moment_of_inertia(mass_kg, length_m)
    I = moi['pitch_yaw_kg_m2']

    # Angular acceleration
    alpha_rad = torque / I
    alpha_deg = math.degrees(alpha_rad)

    return {
        'lever_arm_m': lever_arm,
        'lateral_force_mn': lateral_force / 1e6,
        'torque_mn_m': torque / 1e6,
        'angular_accel_rad_s2': alpha_rad,
        'angular_accel_deg_s2': alpha_deg,
        'moi_kg_m2': I
    }


def calculate_rcs_rotation(mass_kg, length_m):
    """
    Calculate rotation from RCS (VectorThrusters module).
    Works when main engines are off.
    """
    moi = calculate_moment_of_inertia(mass_kg, length_m)
    I = moi['pitch_yaw_kg_m2']

    alpha_rad = RCS_TORQUE_NM / I
    alpha_deg = math.degrees(alpha_rad)

    return {
        'torque_nm': RCS_TORQUE_NM,
        'angular_accel_rad_s2': alpha_rad,
        'angular_accel_deg_s2': alpha_deg
    }


def time_to_rotate(alpha_rad, angle_deg):
    """
    Time to rotate a given angle (accelerate to midpoint, decelerate to stop).
    t = 2 × √(θ / α)
    """
    angle_rad = math.radians(angle_deg)
    return 2 * math.sqrt(angle_rad / alpha_rad)


def max_angular_velocity(alpha_rad, angle_deg):
    """Maximum angular velocity at midpoint of rotation."""
    t = time_to_rotate(alpha_rad, angle_deg)
    return alpha_rad * (t / 2)


def calculate_attitude_control(ship_name, wet_mass_t, length_m):
    """Calculate complete attitude control data for a ship."""
    mass_kg = wet_mass_t * 1000

    # Thrust vectoring at combat deflection (1°)
    tv_combat = calculate_thrust_vectoring(mass_kg, length_m, COMBAT_DEFLECTION_DEG)

    # Thrust vectoring at max deflection (3°)
    tv_max = calculate_thrust_vectoring(mass_kg, length_m, MAX_DEFLECTION_DEG)

    # RCS (engines off)
    rcs = calculate_rcs_rotation(mass_kg, length_m)

    # Rotation times - combat thrust vectoring
    tv_time_45 = time_to_rotate(tv_combat['angular_accel_rad_s2'], 45)
    tv_time_90 = time_to_rotate(tv_combat['angular_accel_rad_s2'], 90)
    tv_time_180 = time_to_rotate(tv_combat['angular_accel_rad_s2'], 180)

    # Rotation times - RCS only
    rcs_time_45 = time_to_rotate(rcs['angular_accel_rad_s2'], 45)
    rcs_time_90 = time_to_rotate(rcs['angular_accel_rad_s2'], 90)
    rcs_time_180 = time_to_rotate(rcs['angular_accel_rad_s2'], 180)

    # Max angular velocities
    tv_max_omega = max_angular_velocity(tv_combat['angular_accel_rad_s2'], 90)
    rcs_max_omega = max_angular_velocity(rcs['angular_accel_rad_s2'], 90)

    # Delta-v costs for maneuvers (thrust vectoring only - RCS uses separate propellant)
    # Mass flow rate: ṁ = F / v_e
    exhaust_vel_ms = 10_256_000  # m/s
    mass_flow_rate = MAIN_THRUST_N / exhaust_vel_ms  # kg/s

    prop_90_kg = mass_flow_rate * tv_time_90
    prop_180_kg = mass_flow_rate * tv_time_180

    # Δv = v_e × ln(m / (m - m_prop))
    dv_90_kps = (exhaust_vel_ms / 1000) * math.log(mass_kg / (mass_kg - prop_90_kg))
    dv_180_kps = (exhaust_vel_ms / 1000) * math.log(mass_kg / (mass_kg - prop_180_kg))

    return {
        'moment_of_inertia': {
            'pitch_yaw_kg_m2': round(tv_combat['moi_kg_m2']),
            'notes': 'Elongated cylinder model, wet mass'
        },
        'thrust_vectoring': {
            'description': 'Main engine nozzle deflection (requires engines firing)',
            'combat_deflection_deg': COMBAT_DEFLECTION_DEG,
            'max_deflection_deg': MAX_DEFLECTION_DEG,
            'lever_arm_m': round(tv_combat['lever_arm_m'], 1),
            'lateral_force_mn': round(tv_combat['lateral_force_mn'], 3),
            'torque_mn_m': round(tv_combat['torque_mn_m'], 2),
            'angular_accel_deg_s2': round(tv_combat['angular_accel_deg_s2'], 3),
            'time_to_rotate_45_deg_s': round(tv_time_45, 1),
            'time_to_rotate_90_deg_s': round(tv_time_90, 1),
            'time_to_rotate_180_deg_s': round(tv_time_180, 1),
            'max_angular_velocity_deg_s': round(math.degrees(tv_max_omega), 2),
            'forward_thrust_efficiency': round(math.cos(math.radians(COMBAT_DEFLECTION_DEG)), 5),
            'delta_v_cost': {
                'rotate_90_deg_kps': round(dv_90_kps, 3),
                'rotate_180_deg_kps': round(dv_180_kps, 3),
                'propellant_90_deg_kg': round(prop_90_kg, 1),
                'propellant_180_deg_kg': round(prop_180_kg, 1)
            }
        },
        'rcs': {
            'description': 'Reaction Control System (works with engines off)',
            'module_name': 'VectorThrusters (as RCS)',
            'mass_tons': RCS_MASS_T,
            'torque_nm': RCS_TORQUE_NM,
            'angular_accel_deg_s2': round(rcs['angular_accel_deg_s2'], 4),
            'time_to_rotate_45_deg_s': round(rcs_time_45, 1),
            'time_to_rotate_90_deg_s': round(rcs_time_90, 1),
            'time_to_rotate_180_deg_s': round(rcs_time_180, 1),
            'max_angular_velocity_deg_s': round(math.degrees(rcs_max_omega), 2)
        }
    }


def main():
    # Load existing fleet data
    fleet_file = Path('/home/plmch/ai-commanders/data/fleet_ships.json')
    with open(fleet_file) as f:
        fleet = json.load(f)

    print("=" * 70)
    print("ATTITUDE CONTROL - THRUST VECTORING VS RCS")
    print("Main engines: 58.56 MN | Combat deflection: 1°")
    print("=" * 70)

    # Calculate for each ship
    for ship_name, ship_data in fleet['ships'].items():
        wet_mass = ship_data['mass_breakdown']['total_wet']
        length = ship_data['hull']['length_m']

        attitude = calculate_attitude_control(ship_name, wet_mass, length)
        ship_data['attitude_control'] = attitude

        tv = attitude['thrust_vectoring']
        rcs = attitude['rcs']

        print(f"\n{ship_name.upper()} ({wet_mass:.0f}t, {length}m)")
        print(f"  Moment of Inertia: {attitude['moment_of_inertia']['pitch_yaw_kg_m2']:,.0f} kg·m²")
        print(f"  Thrust Vectoring (1° deflection, engines ON):")
        print(f"    Lateral force: {tv['lateral_force_mn']:.3f} MN")
        print(f"    Torque: {tv['torque_mn_m']:.2f} MN·m")
        print(f"    Angular accel: {tv['angular_accel_deg_s2']:.3f} deg/s²")
        print(f"    90° turn: {tv['time_to_rotate_90_deg_s']:.1f}s")
        print(f"  RCS (engines OFF):")
        print(f"    Angular accel: {rcs['angular_accel_deg_s2']:.4f} deg/s²")
        print(f"    90° turn: {rcs['time_to_rotate_90_deg_s']:.1f}s")

    # Update metadata
    fleet['attitude_control_systems'] = {
        'thrust_vectoring': {
            'description': 'Main engine nozzle deflection for high-authority rotation',
            'main_thrust_mn': MAIN_THRUST_MN,
            'combat_deflection_deg': COMBAT_DEFLECTION_DEG,
            'max_deflection_deg': MAX_DEFLECTION_DEG,
            'requirement': 'Main engines must be firing'
        },
        'rcs': {
            'description': 'Reaction Control System for maneuvering without main thrust',
            'module_name': 'VectorThrusters',
            'mass_tons': RCS_MASS_T,
            'torque_nm': RCS_TORQUE_NM
        }
    }

    # Remove old key if exists
    if 'attitude_control_modules' in fleet:
        del fleet['attitude_control_modules']

    # Save
    with open(fleet_file, 'w') as f:
        json.dump(fleet, f, indent=2)

    print(f"\n{'=' * 70}")
    print("SUMMARY: 90° ROTATION TIMES")
    print(f"{'=' * 70}")
    print(f"{'Ship':<15} {'Mass (t)':<10} {'TV 1° (s)':<12} {'RCS (s)':<12} {'Ratio':<8}")
    print("-" * 60)

    for ship_name, ship_data in fleet['ships'].items():
        att = ship_data['attitude_control']
        tv_time = att['thrust_vectoring']['time_to_rotate_90_deg_s']
        rcs_time = att['rcs']['time_to_rotate_90_deg_s']
        ratio = rcs_time / tv_time
        print(f"{ship_name:<15} {ship_data['mass_breakdown']['total_wet']:<10.0f} "
              f"{tv_time:<12.1f} {rcs_time:<12.1f} {ratio:<8.1f}x")

    print(f"\nUpdated {fleet_file}")


if __name__ == "__main__":
    main()
