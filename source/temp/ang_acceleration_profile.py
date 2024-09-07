import argparse
import sys

# Data Storage - Class
# import pandas as pd
# import matplotlib.pyplot as plt

import time

from omni.isaac.lab.app import AppLauncher

### ARGPARSE ###
# add argparse arguments
parser = argparse.ArgumentParser(description="Second urdf implementation script.")
# Default to headless mode
if True:
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
import torch
import os

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.articulations import ArticulationView

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, ArticulationData, RigidObjectCfg, RigidObject
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.actuators import ImplicitActuatorCfg, ActuatorBaseCfg
from omni.isaac.lab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
import omni.physx as physx

### Store values ###
from utils.data_recorder import DataRecorder

DATA_RECORDER = DataRecorder(record=True)

### Ang Acceleration Profile ###
from utils.control_methods import SimpleAngAccelProfile

### CFG ###
DAMPING = 10.0
BOX_CFG = ArticulationCfg(
    prim_path="/World/Origin.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/temp/box_w_tail.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            # max_linear_velocity=1000.0,
            # max_angular_velocity=1000.0,
            # max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
            # kinematic_enabled=True
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True, enabled_self_collisions=True, fix_root_link=True
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0.5),
        joint_pos={"box_to_rod": -1.5708},
        # joint_pos={"box_to_rod": -0.3},
        # joint_vel={"box_to_rod": 10.0}
    ),
    actuators={
        "rod_motor": ImplicitActuatorCfg(
            joint_names_expr=["box_to_rod"],
            # friction=0.5,
            friction=0.0,
            damping=DAMPING,
            effort_limit=1000.0,
            stiffness=0.0,  # Leave at zero! (velcity and effort control!)
            velocity_limit=500.0,
        ),
    },
)


### FUNCTIONS ###
def design_scene() -> tuple[dict, list[list[float]]]:
    "Designs the scene."

    # Ground-plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # Lights
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Multiple origins
    z = 1
    origins = [[0, 0, z], [2, 0, z], [4, 0, z], [0, 2, z], [2, 2, z], [4, 2, z]]
    for i, origin in enumerate(origins):
        prin = prim_utils.create_prim(f"/World/Origin{i+1}", "Xform", translation=origin)

    box = Articulation(cfg=BOX_CFG)

    # return the scene information
    scene_entities = {"box": box}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    "Runs the simulator."
    robot = entities["box"]

    # Initialize ArticulationData using physx.ArticulationView
    articulation_view = ArticulationView(prim_paths_expr="/World/Origin.*/Robot")
    articulation_view.initialize()
    ArtData: ArticulationData = robot.data

    # Sim step
    sim_dt = sim.get_physics_dt()
    count = 0
    reset_count = 250

    # Create ang-vel profile
    ang_vel_profile = SimpleAngAccelProfile(sim_dt=sim_dt, t1_t2=1.0)

    # Loop
    while simulation_app.is_running():
        # Reset
        if count % reset_count == 0:
            # If recording, then only perform one simulation loop
            if count == reset_count and DATA_RECORDER.record_bool == True:
                DATA_RECORDER.save(("source/temp/trial_data.csv"))
                return

            count = 0

            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_state_to_sim(root_state)

            joint_pos_default = robot.data.default_joint_pos.clone()
            joint_pos_default += torch.rand_like(joint_pos_default) * 0.1

            joint_vel_default = robot.data.default_joint_vel.clone()
            joint_vel_default += torch.rand_like(joint_vel_default) * 0.4

            robot.write_joint_state_to_sim(position=joint_pos_default, velocity=joint_vel_default, env_ids=None)

            print(f"-- INITIALIZATION DONE --")

            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")

        # Follow ang-vel profile
        ang_vel = ang_vel_profile.get_ang_vel(count=count)
        # print(f"[C: {count}]: Ang-vel: {ang_vel}")

        if ang_vel is not None:
            ### Following Ang-Vel profile ###
            # For velocity control; stiffness must be 0.0, damping must be non-zero
            # (source: omni.isaac.core Articulations Documentation)
            articulation_view.switch_control_mode(mode="velocity")
            robot.write_joint_damping_to_sim(torch.full_like(robot.actuators["rod_motor"].damping, DAMPING))
            # Set joint velocity setpoint
            joint_vel_setpoint = torch.full_like(robot.actuators["rod_motor"].applied_effort, ang_vel)
            robot.set_joint_velocity_target(joint_vel_setpoint)
        else:
            ### Free swing of the tail ###
            # For effort control; stiffness and damping must be 0.0
            robot.set_joint_velocity_target(torch.zeros_like(robot.actuators["rod_motor"].applied_effort))
            articulation_view.switch_control_mode(mode="effort")
            robot.write_joint_damping_to_sim(torch.zeros_like(robot.actuators["rod_motor"].damping))
            # Set zero effort (should let tail swing freely???)
            robot.set_joint_effort_target(torch.zeros_like(robot.actuators["rod_motor"].applied_effort))

        robot.write_data_to_sim()

        # TODO: I observed, that world poses still does not consider rotation of the tail.
        #      Using GUI; can I see the tail's frame of reference rotating??? --> No, it does not rotate.

        sim.step()
        count += 1

        robot.update(sim_dt)

        if count == 200:
            continue

        ### Store values ###
        import numpy as np

        # Forces and torques acting on each
        metadata = articulation_view._metadata
        # print(f"Available joints with indices: {metadata.joint_indices}") # TODO: Remove debug-print
        # joints (are coordinate systems of the bodies)
        joint_indices = 1 + np.array(
            [
                metadata.joint_indices["base_joint"],
                # metadata.joint_indices["box_to_rod"], # TODO: Generalize for multiple joints
            ]
        )
        joint_indices = joint_indices.tolist()
        joint_names = [metadata.joint_names[i - 1] for i in joint_indices]
        # Access joint forces & torques for 0th environment/entity
        forces_and_torques = articulation_view.get_measured_joint_forces(indices=[0], joint_indices=joint_indices)[0]
        force_torque_dict = {
            key: forces_and_torques[0][i] for i, key in enumerate(["fx", "fy", "fz", "tx", "ty", "tz"])
        }

        data = {
            "friction": ArtData.joint_friction[0].item(),
            "damping": ArtData.joint_damping[0].item(),
            "vel_setpoint": robot._joint_vel_target_sim[0].item(),
            "vel_applied": articulation_view.get_joint_velocities()[0].item(),
            "effort_setpoint": robot._joint_effort_target_sim[0].item(),
            "effort_applied": ArtData.applied_torque[0].item(),
            "effort_measured": articulation_view.get_measured_joint_efforts()[0].item(),
        }

        # Concatenate two dictionaries
        data = {**data, **force_torque_dict}

        DATA_RECORDER.store(count * sim_dt, data)
        # DATA_RECORDER.save("source/temp/trial_data.csv")


### MAIN ###
def main():
    "Main function."
    sim_cfg = sim_utils.SimulationCfg()
    sim_cfg.dt = 1.0 / 120.0
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([-0.5, -6.0, 2.3], [2, 4, 1.5])

    # Design scene
    scene_entities, origins = design_scene()
    scene_origins = torch.tensor(origins, device=sim.device)

    # Play simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()

    # DATA_RECORDER.plot(columns_to_plot=["vel_setpoint", "vel_joint", "effort_joint(articulation_view)", "joint_rolltorquebody", "damping", "friction"])
    DATA_RECORDER.plot(
        dictionary={
            "Parameters": ["friction", "damping"],
            "Joint Velocity": ["vel_setpoint", "vel_applied"],
            "Joint Torque": ["effort_setpoint", "effort_measured"],
            "Torques on Body": ["tx", "ty", "tz"],
            "Forces on Body": ["fx", "fy", "fz"],
        }
    )

    # close sim app
    simulation_app.close()
