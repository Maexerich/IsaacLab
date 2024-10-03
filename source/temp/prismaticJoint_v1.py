### GOAL ###
# Open usd-stage which has robot configured.
# Approach analytical solution using this robot, not the URDF, which might contain errors.

import argparse
import sys
import time
from typing import Union

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
from utils.control_methods import SimpleAngAccelProfile, LinearVelocity, TorqueProfile, Controller_floatingBase
from utils.data_recorder import DataRecorder, DataRecorder_V2
from utils.simulation_functions import PhysicsSceneModifier, record, instantiate_Articulation, open_stage

def simulate_generic_setup(prim_path: str, sim: sim_utils.SimulationContext, total_time: float, step_size: float,
                           tail_joint_profile: Union[SimpleAngAccelProfile, TorqueProfile],
                           track_joint_profile: Union[LinearVelocity, None],
                           data_recorder: DataRecorder_V2,
                           callback_function = None):
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
        raise NotImplementedError("Controller not implemented for this case.")
    
    while current_time < total_time:
        controller.update_control_input(current_time=current_time)

        articulation.write_data_to_sim() 

        sim.step()
        current_time += step_size

        articulation.update(dt=step_size)

        if callback_function is not None: # Analytical approach: find tail position and apply F_D
            callback_function()

        record(data_recorder=data_recorder, time_seconds=current_time, articulation=articulation, articulation_view=articulation_view, artdata=artdata)


# Main
def main():
    ### Setup ###
    stage = open_stage(stage_path = "source/temp/stage_FloatingBase_and_FixedBase_v1.usd")
    # stage_utils.print_stage_prim_paths() # Good for debugging

    # Physics Scene Modifier
    ff_handler = PhysicsSceneModifier()

    # Parameters
    total_time = 0.5 # seconds
    step_size = 1.0 / 580.0 # seconds
    sim_cfg = sim_utils.SimulationCfg(physics_prim_path="/physicsScene", 
                                    #   device='cpu',
                                      dt=step_size,
                                    #   use_fabric=False,
                                      )
    sim = SimulationContext(sim_cfg)

    ### Joint Profiles ###
    # Tail Joint
    tail_joint_profile = SimpleAngAccelProfile(sim_dt=step_size,
                                                a=200,
                                                t0=0,
                                                t0_t1=0.2,
                                                t1_t2=0.1)
    # Track Joint (Prismatic)
    track_joint_profile = LinearVelocity(sim_dt=step_size,
                                         control_mode='const',
                                         const_vel=-30.0)
    
    ### Analytical Approach (e.g. FixedBase) ###
    if False:
        ff_handler.disable_drag_force_field() # Disables force fields for analytical approach
        sim.set_camera_view(eye=(-1, -0.5, 2.3), target=(1.5, 1.5, 1.5))
        prim_path = "/World/Robot_fixedBase"

        data_recorder = DataRecorder_V2()
        sim.reset()
        simulate_generic_setup(prim_path, sim, total_time, step_size, tail_joint_profile, None, data_recorder)

    
    ### Force Field Approach (e.g. FloatingBase) ###
    # sim.set_camera_view(eye=(-100, -0.1, 2.3), target=(-95, 1.5, 1.5))
    sim.set_camera_view(eye=(-30, -4, 2), target=(-18, 0, 2))

    prim_path = "/World/Robot_floatingBase"

    data_recorder = DataRecorder_V2()
    simulate_generic_setup(prim_path, sim, total_time, step_size, tail_joint_profile, track_joint_profile, data_recorder, None)
    data_recorder.save("source/temp/track_drive_v1.csv")

if __name__ == "__main__":
    main()

    # Close sim app
    simulation_app.close()

    