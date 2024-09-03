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
from omni.isaac.core.articulations import ArticulationView

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, ArticulationData, RigidObjectCfg, RigidObject
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.actuators import ImplicitActuatorCfg, ActuatorBaseCfg
from omni.isaac.lab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
import omni.physx as physx

### Ang Acceleration Profile ###
import numpy as np
class SimpleAngAccelProfile:
    def __init__(self, 
                 sim_dt: float, 
                 a: float = 200.0, 
                 t0: float = 0.0,
                 t0_t1: float = 0.4,
                 t1_t2: float = 0.2):
        """Simple angular acceleration profile defined as follows;
        alpha(t) = a*(t-t0), for t0 < t < t1
        alpha(t) = a*(t1-t0), for t1 < t < t2
        alpha(t) = 0, otherwise
        
        All variables t are in seconds.
        
        Args:
        - sim_dt: Simulation time-discretization in seconds.
        - a: Angular acceleration in rad/s^2.
        - t0: Start time for acceleration in seconds.
        - t0_t1: Time duration for acceleration in seconds (mathematically: t1-t0).
        - t1_t2: Time for constant angular velocity in seconds (mathematically: t2-t1).
        
        """
        self.sim_dt = sim_dt
        self.acceleration = a
        self.t0 = t0
        self.t1 = t0 + t0_t1
        self.t2 = self.t1 + t1_t2
    
    def get_ang_vel(self, count: int):
        "Returns angular velocity in rad/s at simulation step count."
        current_time = count * self.sim_dt
        if current_time < self.t0:
            return None
        elif current_time < self.t1:
            return self.acceleration * (current_time - self.t0)
        elif current_time < self.t2:
            return self.acceleration * (self.t1 - self.t0)
        else:
            return None

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
            articulation_enabled=True,
            enabled_self_collisions=True,
            fix_root_link=True
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0.5), 
        joint_pos={"box_to_rod": -1.5708}, 
        # joint_pos={"box_to_rod": -0.3},
        # joint_vel={"box_to_rod": 10.0}
    ),
    actuators={
        # "rod_torque": ActuatorBaseCfg(
        #     joint_names_expr=["box_to_rod"],
        #     friction=1.0,
        #     effort_limit=1000.0,
        #     velocity_limit=100.0,
        #     stiffness=0.0,
        #     damping=10.0,
        #     ),
        "rod_motor" : ImplicitActuatorCfg(
            joint_names_expr=["box_to_rod"],
            friction=0.5,
            damping=DAMPING,
            effort_limit=1000.0,
            stiffness=0.0, # Leave at zero! (velcity and effort control!)
            velocity_limit=500.0
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
    origins = [[0, 0, z], [2, 0, z], [4, 0, z], 
               [0, 2, z], [2, 2, z], [4, 2, z]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i+1}", "Xform", translation=origin)    

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
    # articulation_view.enable_dof_force_sensors = True
    
    ArtData : ArticulationData = robot.data

    # Sim step
    sim_dt = sim.get_physics_dt()
    count = 0
    reset_count = 800

    # Create ang-vel profile
    ang_vel_profile = SimpleAngAccelProfile(sim_dt)

    # Loop
    while simulation_app.is_running():
        # Reset
        if count % reset_count == 0:
            count = 0

            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_state_to_sim(root_state)

            joint_pos_default = robot.data.default_joint_pos.clone()
            joint_pos_default += torch.rand_like(joint_pos_default) * 0.1

            joint_vel_default = robot.data.default_joint_vel.clone()
            # joint_vel += torch.rand_like(joint_vel) * 0.4
            
            print(f"-- INITIALIZATION --")
            print(f"Joint_pos:\n    {joint_pos_default}")
            print(f"Joint_vel:\n    {joint_vel_default}")

            robot.write_joint_state_to_sim(position=joint_pos_default, 
                                           velocity=joint_vel_default,
                                           env_ids=None)

            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")
        

        joints = robot.find_joints(["box_to_rod"], preserve_order=True)
        # print(joints)

        # print(f"Robot actuators: {robot.actuators}")
        # print(f"Joint Names: {robot.joint_names}")

        # # Apply an effort
        # base_effort = 100 * ((500-count)/500)
        # efforts = torch.full(robot.data.joint_pos.shape, base_effort)
        # # efforts = efforts.to('cpu')
        # robot.set_joint_effort_target(efforts)

        # Follow ang-vel profile
        ang_vel = ang_vel_profile.get_ang_vel(count=count)
        # print(f"[C: {count}]: Ang-vel: {ang_vel}")
        
        if ang_vel is not None:
            # For velocity control; stiffness must be 0.0, damping must be non-zero
            # (source: omni.isaac.core Articulations Documentation)
            articulation_view.switch_control_mode(mode='velocity')
            robot.write_joint_damping_to_sim(torch.full_like(robot.actuators['rod_motor'].damping, DAMPING))

            joint_vel_setpoint = torch.full_like(robot.actuators['rod_motor'].applied_effort, ang_vel)
            robot.set_joint_velocity_target(joint_vel_setpoint)
        else:
            # DONE: How can I have tail swing freely?

            # For effort control; stiffness and damping must be 0.0
            articulation_view.switch_control_mode(mode='effort')
            robot.write_joint_damping_to_sim(torch.zeros_like(robot.actuators['rod_motor'].damping))

            # Set zero effort (should let tail swing freely???)
            robot.set_joint_effort_target(torch.zeros_like(robot.actuators['rod_motor'].applied_effort))
            
            # print(f"[EXIT]")
            # return
            
        
        print(f"""[C: {count}]: Ang-vel setpoint: {ang_vel}\n 
              measured_joint_efforts: {articulation_view.get_measured_joint_efforts()[0]} \n
              joint_effort_target: {robot._joint_effort_target_sim[0]} \n
              damping: {ArtData.joint_damping[0]} \n""")
        
        if count == 5 or count == 20:
            applied_torque = ArtData.applied_torque
            body_acc_w = ArtData.body_acc_w
            root_state = ArtData.root_state_w

            print(f"\n[{count}]: Applied Torque: {applied_torque}\n")
            print(f"Body Acc: {body_acc_w}\n")
            print(f"Root State: {root_state}\n")

            if count == 20:
                print(f"Early stop")
                return


        robot.write_data_to_sim()        

        # Get values from sim
        ang_vel_sim = articulation_view.get_angular_velocities() # TODO: Why does this not provide ang-vel of tail frame of reference?
        applied_joint_efforts = articulation_view.get_applied_joint_efforts()
        applied_actions = articulation_view.get_applied_actions()
        joint_velocities = articulation_view.get_joint_velocities() # This seems to work too
        measured_joint_efforts = articulation_view.get_measured_joint_efforts() # This provides a sensible value
        local_poses = articulation_view.get_local_poses()
        velocities = articulation_view.get_velocities()
        world_poses = articulation_view.get_world_poses()
        world_scales = articulation_view.get_world_scales()
        joint_pos = articulation_view.get_joint_positions()
        joint_vel = articulation_view.get_joint_velocities()
        joint_state = articulation_view.get_joints_state()
        # TODO: I observed, that world poses still does not consider rotation of the tail.
        #      Using GUI; can I see the tail's frame of reference rotating??? --> No, it does not rotate.
        
        body_names = articulation_view.body_names
        print(f"Body names: {body_names}")
        # return
        
        
        # print(f"""    Ang-vel-sim: {ang_vel_sim[0]}, Applied joint efforts: {applied_joint_efforts[0]}, 
        #       Applied actions: {applied_actions.joint_positions[0]}, Joint velocities: {joint_velocities[0]}, 
        #       Measured joint efforts: {measured_joint_efforts[0]}, \n
        #       world_poses: \n
        #         {world_poses}
        #       world_scales: \n
        #         {world_scales}\n""")

        # if count % 5 == 0:
        #     print(f"[C: {count}]: Ang-vel: {ang_vel}, Effort: {measured_joint_efforts[0].float()}")
        #     print(f"    Vel-Target: {robot._joint_vel_target_sim}")
        # if count % 50 == 0:
        #     # print(f"Current effort at {base_effort} with count = {count}")
        #     # print(f"Robot joint efforts: {robot._joint_effort_target_sim}")
        #     print(f"Current ang-vel at {ang_vel} with count = {count}")
        #     print(f"Robot_joint_efforts: {measured_joint_efforts}")
            

        sim.step() 
        count += 1

        robot.update(sim_dt)


### MAIN ###
def main():
    "Main function."
    sim_cfg = sim_utils.SimulationCfg()
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
    # close sim app
    simulation_app.close()
