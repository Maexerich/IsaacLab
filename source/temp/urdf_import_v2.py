import argparse

from copy import deepcopy

from omni.isaac.lab.app import AppLauncher

### ARGPARSE ###
# add argparse arguments
parser = argparse.ArgumentParser(description="Second urdf implementation script.")
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

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.actuators import ImplicitActuatorCfg, ActuatorBaseCfg

### CONFIGURATION ###
# box_cfg = ArticulationCfg(
#     prim_path="/World/Origin.*/Robot",
#     spawn=sim_utils.UrdfFileCfg(
#         usd_dir="source/temp/box_w_tail.usd",
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             rigid_body_enabled=True,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=100.0,
#             enable_gyroscopic_forces=True
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             articulation_enabled=True,
#             enabled_self_collisions=True,
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0, 0, 0.5), joint_pos={"box_to_rod": -1.5708}
#     ),
#     actuators={
#         "rod_motor": ImplicitActuatorCfg(
#             joint_names_expr=["box_to_rod"],
#             effort_limit=400.0,
#             velocity_limit=100.0,
#             stiffness=0.0,
#             damping=10.0,
#         ),
#     }
# )
# box_cfg = RigidObjectCfg(
#     prim_path="/World/Origin.*/Robot",
#     spawn=sim_utils.UrdfFileCfg(
#         usd_dir="source/temp/box_w_tail.usd",
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             rigid_body_enabled=True,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=100.0,
#             enable_gyroscopic_forces=True
#         ),
#     ),
#     init_state=RigidObjectCfg.InitialStateCfg(
#         pos=(0, 0, 0.5), joint_pos={"box_to_rod": -1.5708}
#     ),
#     )
# )


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
    origins = [[0.0, 0.0, 0.0], [1, 0, 0], [2, 0, 0], 
               [0, 1, 0], [1, 1, 0], [2, 1, 0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i+1}", "Xform", translation=origin)

    print(prim_utils.find_matching_prim_paths("/World/Origin.*"))
    
    # box_cfg.prim_path = "/World/Origin.*/Robot"
    # print(f"[M] box_cfg.prim_path: {box_cfg.prim_path}")
    # box = Articulation(cfg=box_cfg)

    box_cfg = ArticulationCfg(
        prim_path="/World/Origin.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="source/temp/box_w_tail.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                enabled_self_collisions=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.5), joint_pos={"box_to_rod": -1.5708}
        ),
        actuators={
            "rod_motor": ActuatorBaseCfg(
                joint_names_expr=["box_to_rod"],
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=10.0,
                ),
            },
    )

    box = Articulation(cfg=box_cfg)


    ### THIS WORKS ###
    cone_cfg = RigidObjectCfg(
        spawn=sim_utils.ConeCfg(
            radius=0.1, height=0.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    cone_cfg.prim_path = "/World/Origin.*/Robot"
    # box = RigidObject(cfg=cone_cfg)
    #########################



    # print(f"[M] box_cfg.prim_path: {box_cfg.prim_path}")
    # print(f"[M] box_cfg.actuators: {box_cfg.actuators}")
    # box = Articulation(cfg=box_cfg)

    # try:
    #     box = Articulation(cfg=box_cfg)
    # except Exception as e:
    #     print(f"[E] Error creating Articulation: {e}")
    #     raise

    # return the scene information
    scene_entities = {"box": box}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    "Runs the simulator."
    robot = entities["box"]

    # Sim step
    sim_dt = sim.get_physics_dt()
    count = 0

    # Loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            count = 0

            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_state_to_sim(root_state)

            joint_pos = robot.data.default_joint_pos.clone()
            print(f"Joint_pos: shape: {joint_pos.shape}, \n     data: \{joint_pos}")
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(position=joint_pos, 
                                           velocity=torch.zeros_like(joint_pos),
                                           env_ids=None)

            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")
        
        # Apply torque to the 'box_to_rod' joint
        num_entities = robot.num_instances
        torque = torch.full((num_entities, 1), 10.0).to('cuda')  # Torque value in Nm
        joint_index, joint_names = robot.find_joints(name_keys="box_to_rod")
        robot.set_joint_effort_target(target=torque, 
                                      joint_ids=joint_index, 
                                      env_ids=None)
        robot.write_data_to_sim()

        sim.step() 
        count += 1

        robot.update(sim_dt)


### MAIN ###
def main():
    "Main function."
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 0.2])

    # Design scene
    scene_entities, origins = design_scene()
    scene_origins = torch.tensor(origins, device=sim.device)

    # Play simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
