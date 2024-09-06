import argparse
import sys

# Data Storage - Class
import pandas as pd
import matplotlib.pyplot as plt

from omni.isaac.lab.app import AppLauncher

### ARGPARSE ###
# add argparse arguments
parser = argparse.ArgumentParser(description="Second urdf implementation script.")
# Default to headless mode
sys.argv.append("--headless")
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
# class DataStorage:
#     """This class handles the temporary and local storage of values from the simulation.
#     Values are stored as dictionaries at runtime, transformed into a pd.DataFrame at the end
#     of simulation and stored locally as a .csv file.

#     Use;
#     First call DataStorage.store() to store values at a given time-step.
#     Then you can call DataStorage.save() to save the stored values as a .csv file.
#     If you want to plot your results directly, you can skip the save() method and call plot() directly.
#     Keep in mind, calling the plot() method will still save the data as a .csv file locally.
    
#     Attributes:
#     - data: Dictionary of dictionaries to store values. First key is time-step in seconds, 
#             the second key is the name of the value."""
#     def __init__(self, record: bool = True):
#         self.data = {}
#         self.df : pd.DataFrame = None

#         self._record = record

#     @property
#     def get_data(self):
#         if self.df is None:
#             print("Data has not been transformed to a DataFrame yet. \nCall the save method first.")
#         return self.df

#     @property
#     def record_bool(self):
#         return self._record
    
#     def store(self, time_seconds: float, values: dict):
#         "Stores values in the data attribute."
#         self.data[time_seconds] = values
    
#     def save(self, path: str):
#         "Saves the data attribute as a .csv file at the given path."
#         import pandas as pd
#         self.df = pd.DataFrame.from_dict(self.data, orient='index')
#         self.df.to_csv(path, index_label='time_seconds')

#     def plot(self, columns_to_plot: list = None):
#         "Creates a matplotlib pop-up plot of the (relevant) stored values. Columns_to_plot is a list of column names."
        
#         if self.df is None:
#             print(f"Saving df first...")
#             self.save("source/temp/trial_data.csv")
        
#         available_columns = self.df.columns
#         # Filter columns if columns_to_plot is provided
#         if columns_to_plot is None:
#             columns_to_plot = available_columns
#         else:
#             # Ensure only valid columns are being plotted
#             columns_to_plot = [col for col in columns_to_plot if col in available_columns]
        
#         print(f"Columns being plotted: {columns_to_plot}")
        
#         # Create a figure and set of subplots
#         num_columns = len(columns_to_plot)
#         fig, axes = plt.subplots(num_columns, 1, figsize=(10, 4 * num_columns))
        
#         # Ensure axes is always iterable
#         if num_columns == 1:
#             axes = [axes]
        
#         # Plot each column in a separate subplot
#         for i, col in enumerate(columns_to_plot):
#             ax = axes[i]
#             ax.plot(self.df.index, self.df[col])
#             ax.set_title(col)
#             ax.set_xlabel('Index')
#             ax.set_ylabel(col)
#             ax.set_ylim(bottom=0)  # Limit y-axis to positive values only
        
#         # Adjust layout to prevent overlap
#         plt.tight_layout()
#         plt.show()

from utils.data_recorder import DataRecorder

DATA_RECORDER = DataRecorder(record=True)

### Ang Acceleration Profile ###
# import numpy as np
# class SimpleAngAccelProfile:
#     def __init__(self, 
#                  sim_dt: float, 
#                  a: float = 200.0, 
#                  t0: float = 0.0,
#                  t0_t1: float = 0.4,
#                  t1_t2: float = 0.2):
#         """Simple angular acceleration profile defined as follows;
#         alpha(t) = a*(t-t0), for t0 < t < t1
#         alpha(t) = a*(t1-t0), for t1 < t < t2
#         alpha(t) = 0, otherwise
        
#         All variables t are in seconds.
        
#         Args:
#         - sim_dt: Simulation time-discretization in seconds.
#         - a: Angular acceleration in rad/s^2.
#         - t0: Start time for acceleration in seconds.
#         - t0_t1: Time duration for acceleration in seconds (mathematically: t1-t0).
#         - t1_t2: Time for constant angular velocity in seconds (mathematically: t2-t1).
        
#         """
#         self.sim_dt = sim_dt
#         self.acceleration = a
#         self.t0 = t0
#         self.t1 = t0 + t0_t1
#         self.t2 = self.t1 + t1_t2
    
#     def get_ang_vel(self, count: int):
#         "Returns angular velocity in rad/s at simulation step count."
#         current_time = count * self.sim_dt
#         if current_time < self.t0:
#             return None
#         elif current_time < self.t1:
#             return self.acceleration * (current_time - self.t0)
#         elif current_time < self.t2:
#             return self.acceleration * (self.t1 - self.t0)
#         else:
#             return None

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
            # friction=0.5,
            friction=0.0,
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
    # articulation_view.enable_dof_force_sensors = True
    
    ArtData : ArticulationData = robot.data

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

        # Follow ang-vel profile
        ang_vel = ang_vel_profile.get_ang_vel(count=count)
        # print(f"[C: {count}]: Ang-vel: {ang_vel}")
        
        if ang_vel is not None:
            ### Following Ang-Vel profile ###
            # For velocity control; stiffness must be 0.0, damping must be non-zero
            # (source: omni.isaac.core Articulations Documentation)
            articulation_view.switch_control_mode(mode='velocity')
            robot.write_joint_damping_to_sim(torch.full_like(robot.actuators['rod_motor'].damping, DAMPING))
            # Set joint velocity setpoint
            joint_vel_setpoint = torch.full_like(robot.actuators['rod_motor'].applied_effort, ang_vel)
            robot.set_joint_velocity_target(joint_vel_setpoint)
        else:
            ### Free swing of the tail ###
            # For effort control; stiffness and damping must be 0.0
            # robot.set_joint_velocity_target(torch.zeros_like(robot.actuators['rod_motor'].applied_effort))
            articulation_view.switch_control_mode(mode='effort')
            robot.write_joint_damping_to_sim(torch.zeros_like(robot.actuators['rod_motor'].damping))
            # Set zero effort (should let tail swing freely???)
            robot.set_joint_effort_target(torch.zeros_like(robot.actuators['rod_motor'].applied_effort))

            
        
        # print(f"""[C: {count}]: Ang-vel setpoint: {ang_vel}\n 
        #       measured_joint_efforts: {articulation_view.get_measured_joint_efforts()[0]} \n
        #       joint_effort_target: {robot._joint_effort_target_sim[0]} \n
        #       damping: {ArtData.joint_damping[0]} \n""")
        
        # if count == 5 or count == 20:
        #     applied_torque = ArtData.applied_torque
        #     body_acc_w = ArtData.body_acc_w
        #     root_state = ArtData.root_state_w

        #     print(f"\n[{count}]: Applied Torque: {applied_torque}\n")
        #     print(f"Body Acc: {body_acc_w}\n")
        #     print(f"Root State: {root_state}\n")

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
        # print(f"Body names: {body_names}")       

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
        joint_indices = 1 + np.array([
            metadata.joint_indices["base_joint"],
            # metadata.joint_indices["box_to_rod"], # TODO: Generalize for multiple joints
        ])
        joint_indices = joint_indices.tolist()
        joint_names = [metadata.joint_names[i-1] for i in joint_indices]
        # Access joint forces & torques for 0th environment/entity
        forces_and_torques = articulation_view.get_measured_joint_forces(indices=[0], joint_indices=joint_indices)[0]      
        force_torque_dict = {key: forces_and_torques[0][i] for i, key in enumerate(["fx", "fy", "fz", "tx", "ty", "tz"])}
        
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

        DATA_RECORDER.store(count*sim_dt, data)
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
    DATA_RECORDER.plot(dictionary={"Parameters": ["friction", "damping"], 
                                   "Joint Velocity": ["vel_setpoint", "vel_applied"], 
                                   "Joint Torque":["effort_setpoint", "effort_measured"],
                                   "Torques on Body": ["tx", "ty", "tz"],
                                   "Forces on Body": ["fx", "fy", "fz"]})

    # close sim app
    simulation_app.close()
