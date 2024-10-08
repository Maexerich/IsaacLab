### GOAL ###
# Open usd-stage which has robot configured.
# Approach analytical solution using this robot, not the URDF, which might contain errors.

import argparse
import sys
import time
from typing import Union
import torch
import numpy as np

from omni.isaac.lab.app import AppLauncher

### ARGPARSE ###
# add argparse arguments
parser = argparse.ArgumentParser(description="Second urdf implementation script.")
# Default to headless mode
if False: # In headless mode: Errors are thrown, then in the analytical approach, mismatch between devices 'cuda' and 'cpu' error
    sys.argv.append("--headless")
    print(
        "\n" * 5, "#" * 65, f"\n ------------------ Running in headless mode ------------------\n", "#" * 65, "\n" * 5
    )
    time.sleep(1)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

### Launch APP ###
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

### IMPORTS ###
# Isaac Sim Imports
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.articulations import ArticulationView
# Isaac Lab Imports
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, ArticulationData, RigidObjectCfg, RigidObject
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.actuators import ImplicitActuatorCfg, ActuatorBaseCfg
from omni.isaac.lab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
# Custom Imports
from utils.control_methods import SimpleAngAccelProfile, LinearVelocity, TorqueProfile, Controller_floatingBase, SoftAngAccelProfile
from utils.data_recorder import DataRecorder, DataRecorder_V2
from utils.simulation_functions import PhysicsSceneModifier, record, instantiate_Articulation, open_stage, get_tail_orientation, apply_forces

def simulate_generic_setup(prim_path: str, sim: sim_utils.SimulationContext, total_time: float, step_size: float,
                           tail_joint_profile: Union[SimpleAngAccelProfile, TorqueProfile, SoftAngAccelProfile],
                           track_joint_profile: Union[LinearVelocity, None],
                           data_recorder: DataRecorder_V2,
                           analytical_function = None):
    """
    This function handles simulation according to passed arguments.
    
    The following steps are taken in this order:
    - instantiate Articulation
    - reset simulation
    - initialize ArticulationView, ArticulationData
    - instantiate appropriate controller
    - run simulation loop:
        - update control input
        - write data to sim
        - physics-step
        - callback-function is called (if provided)
        - record data"""    
    articulation = instantiate_Articulation(prim_path=prim_path)

    sim.reset()

    articulation_view = ArticulationView(prim_paths_expr=prim_path)
    articulation_view.initialize()
    artdata: ArticulationData = articulation.data

    current_time = 0.0

    if track_joint_profile is not None:
        controller = Controller_floatingBase(articulation_view, articulation, artdata, tail_joint_profile, track_joint_profile)
    else:
        controller = None 
        controller = Controller_floatingBase(articulation_view, articulation, artdata, tail_joint_profile, track_joint_profile)
    
    while current_time < total_time:
        controller.update_control_input(current_time=current_time)

        articulation.write_data_to_sim() 

        sim.step()
        current_time += step_size

        articulation.update(dt=step_size)

        if analytical_function is not None: # Analytical approach: find tail position and apply F_D
            analytical_function(current_time, articulation, articulation_view, artdata, data_recorder)

        record(data_recorder=data_recorder, time_seconds=current_time, articulation=articulation, articulation_view=articulation_view, artdata=artdata)

def apply_analytical_drag_force(time_seconds: float, articulation: Articulation, articulation_view: ArticulationView, artdata: ArticulationData, data_recorder: DataRecorder_V2):
    # Function that returns the tail's instantaneous: orientation, rotation axis and rotation magnitude
    tail_motion = get_tail_orientation(time_seconds, articulation, articulation_view, artdata, data_recorder)
    # Function that applies the drag force to the tail
    Wind_vector = torch.tensor([[30.0], [0.0], [0.0]], device='cuda:0').reshape(3,) # m/s
    apply_forces(Wind_vector, time_seconds, articulation, articulation_view, artdata, tail_motion, data_recorder, apply=False)

import os
# Main
def main():
    main_recording()
    ### Setup ###
    stage = open_stage(stage_path = "source/temp/stage_FloatingBase_and_FixedBase_v2.usd")
    # stage_utils.print_stage_prim_paths() # Good for debugging

    # Physics Scene Modifier
    ff_handler = PhysicsSceneModifier()

    # Parameters
    total_time = 0.5 # seconds
    step_size = 1.0 / 580.0 # seconds
    sim_cfg = sim_utils.SimulationCfg(physics_prim_path="/physicsScene", 
                                      device='cpu', ### IF disabled, ForceFields will not have any effect!
                                      dt=step_size,
                                      )
    sim = SimulationContext(sim_cfg)

    ### Joint Profiles ###
    # Tail Joint
    tail_joint_profile = SimpleAngAccelProfile(sim_dt=step_size,
                                                a=100,
                                                t0=0,
                                                t0_t1=0.2,
                                                t1_t2=0.8)
    
    tail_joint_profile_soft = SoftAngAccelProfile(sim_dt=step_size,
                                                  a=100,
                                                  k=3.5,
                                                  t0=0,
                                                  t0_t1=0.2,
                                                  t1_t2=0.8,
                                                  reach_setpoint_gain=0.4)
    # Track Joint (Prismatic)
    track_joint_profile = LinearVelocity(sim_dt=step_size,
                                         control_mode='const',
                                         const_vel=-30.0)
    
    ### Analytical Approach (e.g. FixedBase) ###
    if False:
        # Analytical approach will throw 'device errors' if cpu is enforced in the sim_cfg!
        ff_handler.disable_drag_force_field() # Disables force fields for analytical approach
        sim.set_camera_view(eye=(-2, -1.5, 2.3), target=(1.5, 1.5, 1.5))
        prim_path = "/World/Robot_fixedBase"

        data_recorder = DataRecorder_V2()
        simulate_generic_setup(prim_path, sim, total_time, step_size, tail_joint_profile, None, data_recorder, apply_analytical_drag_force)
        data_recorder.save("source/results/2024_10_07_An_disabled.csv")
    
    ### Force Field Approach (e.g. FloatingBase) ###
    if False:
        # sim.set_camera_view(eye=(-100, -0.1, 2.3), target=(-95, 1.5, 1.5))
        sim.set_camera_view(eye=(-30, -1, 1.5), target=(-12, 3, 2))
        ff_handler.disable_drag_force_field()
        assert sim_cfg.device == 'cpu', "ForceFields will not have any effect if device is not 'cpu'."

        prim_path = "/World/Robot_floatingBase"

        data_recorder = DataRecorder_V2()
        simulate_generic_setup(prim_path, sim, total_time, step_size, tail_joint_profile, track_joint_profile, data_recorder, None)
        # data_recorder.save("source/results/2024_10_06_FF_corner_enabled.csv")
        data_recorder.save("source/results/2024_10_07_FF_disabled.csv")
    
    if False:
        to_simulate_dict = {
                            # "source/results/2024_10_06_FF_windsweep00": 0,
                            # "source/results/2024_10_06_FF_windsweep02": 2,
                            "source/results/2024_10_06_FF_windsweep05": 5,
                            # "source/results/2024_10_06_FF_windsweep10": 10,
                            # "source/results/2024_10_06_FF_windsweep15": 15,
                            # "source/results/2024_10_06_FF_windsweep30": 30,
                            }
        
        def run(path: str, wind_speed: float):
            track_joint_profile = LinearVelocity(sim_dt=step_size, control_mode='const', const_vel=-wind_speed)
            sim.set_camera_view(eye=(-30, -1, 1.5), target=(-12, 3, 2))
            assert sim_cfg.device == 'cpu', "ForceFields will not have any effect if device is not 'cpu'."
            prim_path = "/World/Robot_floatingBase"
            data_recorder = DataRecorder_V2()
            simulate_generic_setup(prim_path, sim, total_time, step_size, tail_joint_profile, track_joint_profile, data_recorder, None)
            data_recorder.save(path)
        
        # Use either of these for-loops and uncomment the other! One is responsible for simulating with and the other without the Drag Force Field
        for key, wind_speed in to_simulate_dict.items():
            # First with drag force field enabled
            path = key + "_enabled.csv"
            run(path, wind_speed)
        # for key, wind_speed in to_simulate_dict.items():
        #     # Then with drag force field disabled
        #     ff_handler.disable_drag_force_field()
        #     path = key + "_disabled.csv"
        #     run(path, wind_speed)

def main_recording():
    ### Setup ###
    stage = open_stage(stage_path = "source/temp/stage_FloatingBase_and_FixedBase_v2.usd")
    # stage_utils.print_stage_prim_paths() # Good for debugging

    # Physics Scene Modifier
    ff_handler = PhysicsSceneModifier()

    # Parameters
    total_time = 10.0 # seconds
    step_size = 1.0 / 60.0 # seconds
    sim_cfg = sim_utils.SimulationCfg(physics_prim_path="/physicsScene", 
                                    #   device='cpu', ### IF disabled, ForceFields will not have any effect!
                                      dt=step_size,
                                      )
    sim = SimulationContext(sim_cfg)

    ### Joint Profiles ###
    # Tail Joint
    tail_joint_profile = SimpleAngAccelProfile(sim_dt=step_size,
                                                a=100,
                                                t0=0,
                                                t0_t1=0.2,
                                                t1_t2=0.8)
    start_time = 1.0
    tail_joint_profile_soft = SoftAngAccelProfile(sim_dt=step_size,
                                                  a=100,
                                                  k=3.5,
                                                  t0=start_time,
                                                  t0_t1=0.2,
                                                  t1_t2=0.4,
                                                  reach_setpoint_gain=0.4)
    # Track Joint (Prismatic)
    track_joint_profile = LinearVelocity(sim_dt=step_size,
                                         control_mode='const',
                                         const_vel=-15.0)

    ff_handler.disable_drag_force_field() # Recording does not matter if force fields are on or off
    sim.set_camera_view(eye=(-45, -1, 1.5), target=(-12, 5, 2))
    prim_path = "/World/Robot_floatingBase"

    articulation = instantiate_Articulation(prim_path=prim_path)

    sim.reset()

    articulation_view = ArticulationView(prim_paths_expr=prim_path)
    articulation_view.initialize()
    artdata: ArticulationData = articulation.data

    current_time = 0.0

    while current_time < total_time:

        # Only now start moving floating base
        tailjoint_input = tail_joint_profile_soft.get_control_setpoint(current_time)
        if tailjoint_input is None:
            # Switch to zero effort control
            articulation_view.switch_control_mode(mode='effort')
            articulation.write_joint_damping_to_sim(torch.tensor([[0.0, 0.0]]))
            articulation.set_joint_effort_target(torch.tensor([[0.0, 0.0]]))
        else:
            # Velocity control
            if current_time > start_time:
                trackjoint_input = track_joint_profile.get_control_setpoint(current_time)
            else:
                trackjoint_input = 0.0
            
            articulation_view.switch_control_mode(mode='velocity')
            articulation.set_joint_velocity_target(torch.tensor([[trackjoint_input, tailjoint_input]]))
        
        articulation.write_data_to_sim()
        sim.step()
        current_time += step_size

    simulation_app.close()
    

if __name__ == "__main__":
    main()

    # Close sim app
    simulation_app.close()