### GOAL ###
# Import a pre-built usd stage which contains all necessary items for simulation.
# This script should provide a framework for sending control commands as well as modifying
# some parts of the simulation environment (e.g. ForceFields).

import argparse
import sys

import time
import numpy as np
import torch

from omni.isaac.lab.app import AppLauncher

### ARGPARSE ###
# add argparse arguments
parser = argparse.ArgumentParser(description="Second urdf implementation script.")
# Default to headless mode
if False:
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
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

### More imports ###
# Isaac-Sim imports
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.extensions as extensions_utils
from omni.isaac.core.articulations import ArticulationView
# Isaac-Lab imports
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, ArticulationData, RigidObjectCfg, RigidObject
from omni.isaac.lab.actuators import ImplicitActuatorCfg, ActuatorBaseCfg
from omni.isaac.lab.sim import SimulationContext, SimulationCfg
# Custom imports
from utils.data_recorder import DataRecorder
from utils.control_methods import SimpleAngAccelProfile

def open_stage(stage_path: str = "source/temp/scene_creation_using_GUI_v5.usd"):
    stage_utils.open_stage(usd_path=stage_path)
    stage = stage_utils.get_current_stage()
    return stage

def instantiate_Articulation() -> Articulation:
    print(f"[TODO]: This function is hard-coded, be sure to fix!")
    DAMPING = 0.0
    BOX_CFG = ArticulationCfg(
        prim_path="/World/Robot2/Body", # Must point to articulation root
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            # pos=(0, 0, 2.5),
            joint_pos={"TailJoint": -1.5708}, # DO NOT ADD FULL PATH: /World/Robot2/.... ADD ONLY JOINT NAME!
            # joint_pos={"box_to_rod": -0.3},
            # joint_vel={"box_to_rod": 10.0}
        ),
        actuators={
            "TailDrive": ImplicitActuatorCfg(
                joint_names_expr=["TailJoint"], # DO NOT ADD FULL PATH: /World/Robot2/.... ADD ONLY JOINT NAME!
                # friction=0.2,
                friction=0.02,
                damping=0.0,
                # effort_limit=1000.0,
                stiffness=0.0,  # Leave at zero! (velcity and effort control!) # TODO: Why??
                # stiffness=1000.0,
                # velocity_limit=500.0,
            ),
        },
    )
    articulation = Articulation(BOX_CFG)
    return articulation

def load_extension_ForceFields():
    boolean = extensions_utils.enable_extension(extension_name='omni.physx.forcefields')
    if boolean == False:
        raise Exception("Failed to enable ForceFields extension.")
    print(f"ForceFields extension enabled: {boolean}")

class PhysicsSceneModifier:
    """
    Provide a class for reading and modifying values in the physics scene.
    """
    def __init__(self):
        # raise NotImplementedError("This class is not implemented yet.")
        pass

    def attributes(self):
        attributes = prim_utils.get_prim_property(prim_path='/physicsScene', 
                                                  property_name='physxForceField:ForceField1:physxForceFieldLinear:constant')
        print(f"Get value from physicsScene: \n{attributes}")

    def disable_all(self):
        print(f"physicsScene Attributes: \n{prim_utils.get_prim_attribute_names('/physicsScene')}\n\n")
        prim_utils.set_prim_attribute_value(prim_path='/physicsScene', 
                                            attribute_name='physxForceField:ForceField1:physxForceField:enabled',
                                            value=False)
        prim_utils.set_prim_attribute_value(prim_path='/physicsScene',
                                            attribute_name='physxForceField:ForceField2:physxForceField:enabled',
                                            value=False)
        
        print(f"""physxForceField:ForceField1:physxForceField:enabled: 
              \n{prim_utils.get_prim_property(prim_path='/physicsScene', 
                                              property_name='physxForceField:ForceField1:physxForceField:enabled')}""")
        print(f"""physxForceField:ForceField2:physxForceField:enabled: 
              \n{prim_utils.get_prim_property(prim_path='/physicsScene', 
                                              property_name='physxForceField:ForceField2:physxForceField:enabled')}""")
        pass

def record_values(sim: SimulationContext, current_time: float, step_size: float, 
                  articulation: Articulation, artdata: ArticulationData, articulation_view: ArticulationView):
    forces_and_torques = articulation_view.get_measured_joint_forces(indices=[0], joint_indices=[1])[0]
    force_torque_dict = {
        key: forces_and_torques[0][i] for i, key in enumerate(["fx", "fy", "fz", "tx", "ty", "tz"])
    }
    
    values = {
        "friction": artdata.joint_friction[0].item(),
        "damping": artdata.joint_damping[0].item(),
        "pos": artdata.joint_pos[0].item(),
        "vel_setpoint": artdata.joint_vel_target[0].item(),
        "vel": artdata.joint_vel[0].item(),
        # "vel_setpoint": articulation._joint_vel_target_sim[0].item(), # same as artdata.joint_vel_target
        # "vel": articulation_view.get_joint_velocities()[0].item(), # same as artdata.joint_vel (artdata.joint_acc exists)
        "effort_setpoint": artdata.joint_effort_target[0].item(),
        # "effort_setpoint": articulation._joint_effort_target_sim[0].item(), # should be same as artdata.joint_effort_target
        "effort_measured": articulation_view.get_measured_joint_efforts()[0].item(), # looks good
        "effort_applied": artdata.applied_torque[0].item(), # should be same as articularion_view.get_applied_joint_efforts()[0].item()    }
    }
    DATA_RECORDER.record(time_seconds=current_time, values={**values, **force_torque_dict})
    pass

def run_simulation(sim: SimulationContext, total_time: float, step_size: float, articulation: Articulation):
    articulation_view = ArticulationView(prim_paths_expr="/World/Robot2/Body") # TODO: Hard coded
    articulation_view.initialize()
    artdata: ArticulationData = articulation.data

    current_time = 0.0


    ### ANGULAR VELOCITY CONTROL ###
    ang_vel_profile = SimpleAngAccelProfile(sim_dt=step_size,
                                            a=200,
                                            t0=0,
                                            t0_t1=0.4,
                                            t1_t2=0.2,)

    while current_time < total_time:
        # ang_vel = ang_vel_profile.get_ang_vel(count=int(current_time/step_size)) # TODO: Type hinting
        # if ang_vel is not None:
        #     # Ensure damping is set to value =!= 0.0
        #     articulation_view.switch_control_mode(mode="velocity")
        #     articulation.write_joint_damping_to_sim(torch.full_like(articulation.actuators["TailDrive"].damping, 10.0))
        #     # Set target velocity
        #     joint_vel_setpoint = torch.full_like(articulation.actuators["TailDrive"].applied_effort, ang_vel)
        #     articulation.set_joint_velocity_target(joint_vel_setpoint)
        # else:
        #     # Free swinging of tail
        #     # For effort control, stiffness and damping must be 0.0
        #     articulation.set_joint_velocity_target(torch.zeros_like(articulation.actuators["TailDrive"].applied_effort))
        #     articulation_view.switch_control_mode(mode="effort")
        #     articulation.write_joint_damping_to_sim(torch.zeros_like(articulation.actuators["TailDrive"].damping))
        #     # Apply zero effort
        #     articulation.set_joint_effort_target(torch.zeros_like(articulation.actuators["TailDrive"].applied_effort))

        ### TORQUE CONTROL ###
        torque = 10.0
        articulation_view.switch_control_mode(mode="effort")
        articulation.write_joint_damping_to_sim(torch.zeros_like(articulation.actuators["TailDrive"].damping))
        # Apply effort
        articulation.set_joint_effort_target(torch.full_like(articulation.actuators["TailDrive"].applied_effort, torque))
        
        articulation.write_data_to_sim()            


        sim.step()
        current_time += step_size

        articulation.update(dt=step_size) # TODO: Where do I need to call this???

        record_values(sim, current_time, step_size, articulation, artdata, articulation_view)

DATA_RECORDER = DataRecorder(record=True)

def main():
    stage = open_stage()

    # Stage setup:
    # stage_utils.print_stage_prim_paths()

    # Instantiate Articulation
    articulation = instantiate_Articulation()

    load_extension_ForceFields()

    ## Simulation ##
    # Parameters
    total_time = 10.0 # seconds
    step_size = 1.0 / 240.0
    sim_cfg = SimulationCfg(physics_prim_path="/physicsScene",
                            device='cpu',
                            dt=step_size,
                            use_fabric=False)
    sim = SimulationContext(sim_cfg)

    ff_handler = PhysicsSceneModifier()
    ff_handler.disable_all() # Disables force fields, useful when debugging control inputs

    # Start simulation
    sim.reset()
    run_simulation(sim, total_time, step_size, articulation)


if __name__ == "__main__":
    main()
    print("[INFO] Script finished!")

    DATA_RECORDER.save("source/temp/forcefields_sim.csv")
    
    DATA_RECORDER.plot(
        dictionary={
            "Parameters": ["friction", "damping"],
            "Joint Velocity": ["vel_setpoint", "vel"],
            "Joint Position": ["pos"],
            "Joint Torque": ["effort_setpoint", "effort_measured", "effort_applied"],
            "Torques on Body": ["tx", "ty", "tz"],
            "Forces on Body": ["fx", "fy", "fz"],
            # "Tail Orientation [rad]": ["tail_orientation_radians"],
            # "Tail Velocity": ["tail_velocity"],
            # "Substitution Force on Tail": ["F_sub_x", "F_sub_y", "F_sub_z", "F_total"],
            # "Wind constant [m/s]": ["Wind_x", "Wind_y", "Wind_z"], # TODO: Add wind to plot
        },
        save_path="source/temp/forcefields_sim.png"
    )

    simulation_app.close()

