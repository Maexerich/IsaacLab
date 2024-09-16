import argparse
import sys
import time
import numpy as np

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
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

### IMPORTS ###
import torch

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.robots import RobotView

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, ArticulationData, RigidObjectCfg, RigidObject
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.actuators import ImplicitActuatorCfg, ActuatorBaseCfg
from omni.isaac.lab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
import omni.physx as physx

### Data recorder ###
from utils.data_recorder import DataRecorder
DATA_RECORDER = DataRecorder(record=True)

### Ang Acceleration Profile
from utils.control_methods import SimpleAngAccelProfile

### Articulation CFG ###
BOX_CFG = ArticulationCfg(
    prim_path="/World/Origin/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/temp/box_w_tail.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            enable_gyroscopic_forces=True
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True,
            enabled_self_collisions=True,
            fix_root_link=True
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        # joint_pos={"box_to_rod": -1.5708},
        joint_pos={"box_to_rod": 0.0},
        # joint_vel={"box_to_rod": 2.0},
    ),
    actuators={
        "motor": ImplicitActuatorCfg(
            joint_names_expr=["box_to_rod"],
            stiffness=0.0, # Zero for velocity and effort control
            damping=0.0,
            friction=0.2,
            effort_limit=1000.0,
            velocity_limit=500.0,
            ),
    },
)

def design_scene():

    # Ground plane
    ground_cfg  = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # Light
    light_cfg = sim_utils.DomeLightCfg(intensity=3000)
    light_cfg.func("/World/Light", light_cfg)

    ### Box with tail ###
    origin = [0.0, 0.0, 0.0]
    p = prim_utils.create_prim("/World/Origin", "Xform", translation=origin)
    # box_prim = sim_utils.spawn_from_usd(prim_path="/World/Origin/Robot", cfg=BOX_CFG, translation=origin)

    # BOX_CFG.prim_path = "/World/Origin/Robot"
    box = Articulation(cfg=BOX_CFG)

    return box, origin

def get_tail_orientation(robot: Articulation, articulation_view: ArticulationView, artdata: ArticulationData):
    """
    Computes and returns key information regarding the orientation and rotation of the tail (or rod) in the world frame.
    Note that this function is custom for the specific robot model used in this script.

    Args:
    - robot (Articulation): The robot object containing the articulated model.
    - articulation_view (ArticulationView): Provides access to joint states, body indices, and kinematic details.
    - artdata (ArticulationData): Contains body position and velocity data in the world frame.

    Returns:
    - dict: A dictionary containing:
      - "tail_orientation": A vector in world coordinates representing the tail's orientation from the rod joint to the end-effector.
      - "rotation_axis": A normalized vector representing the current axis of rotation.
      - "rotation_magnitude": The magnitude of angular velocity around the axis of rotation.
    """
    ### Joint Configuration
    joint_cfg = {
        "position[radians]": articulation_view.get_joint_positions(),
        "velocity[radians/s]": articulation_view.get_joint_velocities(),
    }

    ### Generalized Tail Velocity in world frame
    joint_name = "box_to_rod"
    tail_joint_index = articulation_view.get_joint_index(joint_name=joint_name)
    # Jacobian: maps generalized coordinates to world-frame velocities
    jacobian = articulation_view.get_jacobians()[0][tail_joint_index][:][:]
    gen_tail_velocity = torch.matmul(jacobian, joint_cfg["velocity[radians/s]"])
    gen_tail_velocity = torch.round(input=gen_tail_velocity, decimals=6) # So that in-plane motion is precisely in-plane

    ### Rotation Axis
    w_vec = gen_tail_velocity[3:]
    axis_of_rotation = w_vec/torch.norm(input=w_vec, p=2)
    rotation_magnitude = torch.norm(input=w_vec, p=2)

    ### Vector representing tail orientation in world frame
    body_pos_w = torch.round(input=artdata.body_pos_w[0], decimals=6) # Environment 0, rounding off to 6 decimal places
    def get_body_position_vector(body_name: str):
        body_index = articulation_view.get_body_index(body_name=body_name)
        return body_pos_w[body_index]
    
    # Position vectors in world coordinates
    rod_joint_pos_vec = get_body_position_vector("rod")
    endeffector_pos_vec = get_body_position_vector("endeffector")
    tail_orientation_in_world_coordinates = endeffector_pos_vec - rod_joint_pos_vec

    return {"tail_orientation": tail_orientation_in_world_coordinates, "rotation_axis": axis_of_rotation, "rotation_magnitude": rotation_magnitude}

def apply_forces(robot: Articulation, articulation_view: ArticulationView, artdata: ArticulationData, tail_motion: dict):
    ### Forces ###
    WIND = torch.tensor([-30.0, 0.0, 0.0]) # m/s
    density_air = 1.293 # kg/m^3
    C_drag = 1.1 # [has no unit]

    # x shall be variable along tail length
    tail_length = 0.7 # TODO: Tail length is hard-coded
    tail_length = torch.norm(input=tail_motion["tail_orientation"], p=2)
    x = torch.linspace(0.0, tail_length, steps=200)

def run_simulator(sim: sim_utils.SimulationContext, box: Articulation, origin: torch.Tensor):
    "Runs the simulation."

    robot = box

    articulation_view = ArticulationView(prim_paths_expr="/World/Origin/Robot")
    articulation_view.initialize()
    artdata: ArticulationData = robot.data

    # Sim step
    sim_dt = sim.get_physics_dt()
    count = 0
    reset_count = 250
    
    ang_vel_profile = SimpleAngAccelProfile(sim_dt=sim_dt, t1_t2=1.0)

    # Loop
    while simulation_app.is_running():
        # Rest
        if count % reset_count == 0:
            count = 0

            print(f"        ----------------- Resetting -----------------")
            root_state = artdata.default_root_state.clone()
            root_state[:, :3] += origin
            robot.write_root_state_to_sim(root_state)

            joint_pos_default = artdata.default_joint_pos.clone()
            
            joint_vel_default = artdata.default_joint_vel.clone()

            robot.write_joint_state_to_sim(position=joint_pos_default, velocity=joint_vel_default)
            robot.reset()
            print(f"        -------------- Initialization done -----------")
        
        
        shape = robot.actuators["motor"].applied_effort

        # Effort on tail
        articulation_view.switch_control_mode(mode='effort')
        robot.write_joint_damping_to_sim(torch.full_like(shape, 0.0))
        effort = 10.0
        robot.set_joint_effort_target(torch.full_like(shape, effort))

    
        robot.write_data_to_sim()

        sim.step()
        count += 1

        robot.update(sim_dt)

        if count % 20 == 0:
            pass

        # Get tail motion
        tail_motion = get_tail_orientation(robot, articulation_view, artdata)
        # Apply wind and drag force
        apply_forces(robot, articulation_view, artdata, tail_motion)



# Main
def main():
    sim_cfg = sim_utils.SimulationCfg()
    sim_cfg.dt = 1.0 / 120.0
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([-0.5, -6.0, 2.3], [2, 4, 1.5])

    # Design the scene
    box, origin = design_scene()
    origin = torch.tensor(origin, device=sim.device)

    # Run the simulator
    sim.reset()
    print(f"[Info]: Setup complete. Starting first simulation...")
    run_simulator(sim, box, origin)

if __name__ == "__main__":
    main()

    # Close sim app
    simulation_app.close()

    