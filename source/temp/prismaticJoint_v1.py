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
from utils.control_methods import SimpleAngAccelProfile, LinearVelocity, TorqueProfile, Controller_floatingBase
from utils.data_recorder import DataRecorder, DataRecorder_V2
from utils.simulation_functions import PhysicsSceneModifier, record, instantiate_Articulation, open_stage

def simulate_generic_setup(prim_path: str, sim: sim_utils.SimulationContext, total_time: float, step_size: float,
                           tail_joint_profile: Union[SimpleAngAccelProfile, TorqueProfile],
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

def get_tail_orientation(time_seconds: float, articulation: Articulation, articulation_view: ArticulationView, artdata: ArticulationData, data_recorder: DataRecorder_V2):
    """
    Computes and returns key information regarding the orientation and rotation of the tail (or rod) in the world frame.
    Note that this function is custom for the specific robot model used in this script.

    Args:
    - articulation (Articulation): The robot object containing the articulated model.
    - articulation_view (ArticulationView): Provides access to joint states, body indices, and kinematic details.
    - artdata (ArticulationData): Contains body position and velocity data in the world frame.

    Returns:
    - dict: A dictionary containing:
      - "tail_orientation": A vector in world coordinates representing the tail's orientation from the rod joint to the end-effector.
      - "rotation_axis": A normalized vector representing the current axis of rotation.
      - "rotation_magnitude": The magnitude of angular velocity around the axis of rotation.
    """
    assert len(articulation.actuators.keys()) == 1, "Only one actuator is expected for this function, as it does not generalize!"
    ### Joint Configuration
    joint_cfg = {
        "position[radians]": articulation_view.get_joint_positions(),
        "velocity[radians/s]": articulation_view.get_joint_velocities(),
    }

    ### Generalized Tail Velocity in world frame
    joint_name = "TailDrive" # TODO: Rename variable to tail_joint_name
    tail_joint_index = articulation_view.get_joint_index(joint_name=joint_name)
    # Jacobian: maps generalized coordinates to world-frame velocities
    jacobian = articulation_view.get_jacobians()[0][tail_joint_index][:][:]
    gen_tail_velocity = torch.matmul(jacobian, joint_cfg["velocity[radians/s]"])
    gen_tail_velocity = torch.round(input=gen_tail_velocity, decimals=6) # So that in-plane motion is precisely in-plane

    ### Rotation Axis
    w_vec = gen_tail_velocity[3:]
    if torch.norm(input=w_vec, p=2) == 0.0:
        axis_of_rotation = torch.tensor([0.0, 0.0, 0.0], device='cuda:0')
    else:
        axis_of_rotation = w_vec/torch.norm(input=w_vec, p=2)
    rotation_magnitude = torch.norm(input=w_vec, p=2)

    ### Vector representing tail orientation in world frame
    body_pos_w = torch.round(input=artdata.body_pos_w[0], decimals=6) # Environment 0, rounding off to 6 decimal places
    def get_body_position_vector(body_name: str):
        body_index = articulation_view.get_body_index(body_name=body_name)
        return body_pos_w[body_index]
    
    # Position vectors in world coordinates
    rod_joint_pos_vec = get_body_position_vector("Tail")
    endeffector_pos_vec = get_body_position_vector("Endeffector")
    tail_orientation_in_world_coordinates = endeffector_pos_vec - rod_joint_pos_vec
    tail_orientation_in_world_coordinates = torch.reshape(tail_orientation_in_world_coordinates, (3,1))

    assert torch.isnan(tail_orientation_in_world_coordinates).any() == False
    assert torch.isnan(axis_of_rotation).any() == False
    assert torch.isnan(rotation_magnitude).any() == False

    return {"tail_orientation": tail_orientation_in_world_coordinates, "rotation_axis": axis_of_rotation, "rotation_magnitude": rotation_magnitude}

def apply_forces(Wind_vector: torch.Tensor,time_seconds: float, articulation: Articulation, articulation_view: ArticulationView, artdata: ArticulationData, 
                 tail_motion: dict, data_recorder: DataRecorder_V2, apply: bool = True):
    ### Parameters ####
    WIND = Wind_vector # m/s
    density_air = 1.225 # kg/m^3
    C_d = 1.1 # [has no unit]
    DIAMETER = 0.1 # m, TODO: Hard-coded value
    LENGTH = tail_motion["tail_orientation"].norm(p=2) # m,
    DISCRETIZATION = 200 # Number of points to discretize tail length
    vec_omega = (tail_motion["rotation_axis"] * tail_motion["rotation_magnitude"]).reshape(3,)
    DEVICE = vec_omega.device
    vec_joint_endeffector = tail_motion["tail_orientation"].reshape(3,) #.to(DEVICE) # Deactivated here and next line, shouldn't have an affect
    dir_joint_endeffector = (vec_joint_endeffector/torch.norm(input=vec_joint_endeffector, p=2)).reshape(3,) #.to(DEVICE)
    array_of_A = np.zeros((1, DISCRETIZATION))
    array_of_Td = np.zeros((DISCRETIZATION, 3))

    ### Functions ###
    def vec_x_at_s(s: float):
        "Returns vector x(s) evaluated at a position along the tail."
        assert 0.0 <= s <= vec_joint_endeffector.norm(p=2)
        return s * dir_joint_endeffector
    
    def v_wind_perceived_at_s(s: float):
        """Returns the velocity of the wind perceived along the tail at position s.
        (opposes velocity of the tail)"""
        return -torch.cross(vec_omega, vec_x_at_s(s))
    
    def L_projected_at_s(plane_perpendicular_to):
        "Returns projected tail-length onto plane perpendicular to argument 'plane_perpendicular_to'."
        # Ensure plane_perpendicular_to is a unit vector
        assert (torch.norm(input=plane_perpendicular_to, p=2) - 1.0) < 1e-6
        L_projected = vec_joint_endeffector - torch.dot(vec_joint_endeffector, plane_perpendicular_to) * plane_perpendicular_to
        return torch.norm(input=L_projected, p=2)
    
    def F_drag_at_s(s: float):
        "Returns the drag force at position s. Returned is a vector."
        v = WIND + v_wind_perceived_at_s(s)
        if torch.norm(input=v, p=2) != 0.0:
            v_dir = v/torch.norm(input=v, p=2)
        else:
            v_dir = v.clone()
        v_squared = torch.norm(input=v, p=2)**2
        # Surface Area A is projected and taken proportional quantity according to DISCRETIZATION
        A = DIAMETER * L_projected_at_s(plane_perpendicular_to=v_dir) / DISCRETIZATION
        array_of_A[0, int(s/(LENGTH/DISCRETIZATION))-1] = A
        F_drag_at_s = 0.5 * density_air * C_d * A * v_squared * v_dir
        return F_drag_at_s
    
    def T_total():
        "Returns the total torque acting on the tail. Returned is a vector."
        T_total = torch.zeros(3).to('cuda')
        for s in torch.linspace(0.0, vec_joint_endeffector.norm(p=2), steps=DISCRETIZATION):
            assert 0.0 <= s <= vec_joint_endeffector.norm(p=2)
            T_d = torch.cross(vec_x_at_s(s), F_drag_at_s(s)) #.to(DEVICE)
            array_of_Td[int(s/(LENGTH/DISCRETIZATION))-1, :] = T_d.cpu()
            T_total += T_d
        return T_total
    
    def F_substitution():
        "Returns the equivalent force acting through the CoM of the tail. Returned is a vector."
        vec_x_at_Lhalf = vec_x_at_s(vec_joint_endeffector.norm(p=2)/2)
        norm_vec_x = torch.norm(input=vec_x_at_Lhalf, p=2)
        Torque_total = T_total()
        data_recorder.record(time_seconds=time_seconds, values={"Wind torque magnitude": Torque_total.norm(p=2).cpu()})
        return -(torch.cross(vec_x_at_Lhalf, 2*Torque_total)/norm_vec_x**2)
    
    ### Apply forces ###
    F_sub = F_substitution()
    F_sub_unit_vector = F_sub/torch.norm(input=F_sub, p=2)
    F_sub = torch.reshape(F_sub, (1, 1, 3))

    if apply:
        # If F_sub is 0.0 for every entry, skip (otherwise external wrench is disabled [no clue what this means...])
        if torch.norm(input=F_sub, p=2) > 0.0:
            articulation.set_external_force_and_torque(forces=F_sub, torques=torch.zeros_like(F_sub), body_ids=[2], env_ids=[0])
            articulation.write_data_to_sim() # TODO: Hardcoded something in source function
        else:
            print(f"[WARNING: {time_seconds}] F_sub is zero: {F_sub[0,0,:]}")
    else:
        pass
    
    data_recorder.record(time_seconds=time_seconds, values={
        "e_F_x": F_sub_unit_vector[0],
        "e_F_y": F_sub_unit_vector[1],
        "e_F_z": F_sub_unit_vector[2],
        "F_sub_x": F_sub[0,0,0],
        "F_sub_y": F_sub[0,0,1],
        "F_sub_z": F_sub[0,0,2],
        "F_total": F_sub[0,0,:].norm(p=2),
        "F_applied?": apply,
        "A_tilde": array_of_A.sum(),
        "Wind_x": WIND[0],
        "Wind_y": WIND[1],
        "Wind_z": WIND[2],
    })

def apply_analytical_drag_force(time_seconds: float, articulation: Articulation, articulation_view: ArticulationView, artdata: ArticulationData, data_recorder: DataRecorder_V2):
    # Function that returns the tail's instantaneous: orientation, rotation axis and rotation magnitude
    tail_motion = get_tail_orientation(time_seconds, articulation, articulation_view, artdata, data_recorder)
    # Function that applies the drag force to the tail
    Wind_vector = torch.tensor([[30.0], [0.0], [0.0]], device='cuda:0').reshape(3,) # m/s
    apply_forces(Wind_vector, time_seconds, articulation, articulation_view, artdata, tail_motion, data_recorder, apply=True)

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
        sim.set_camera_view(eye=(-2, -1.5, 2.3), target=(1.5, 1.5, 1.5))
        prim_path = "/World/Robot_fixedBase"

        data_recorder = DataRecorder_V2()
        simulate_generic_setup(prim_path, sim, total_time, step_size, tail_joint_profile, None, data_recorder, apply_analytical_drag_force)
        data_recorder.save("source/results/DEBUG_AnalyticalApproach_v1.csv")
    
    ### Force Field Approach (e.g. FloatingBase) ###
    if True:
        # sim.set_camera_view(eye=(-100, -0.1, 2.3), target=(-95, 1.5, 1.5))
        sim.set_camera_view(eye=(-30, -4, 2), target=(-18, 0, 2))

        prim_path = "/World/Robot_floatingBase"

        data_recorder = DataRecorder_V2()
        simulate_generic_setup(prim_path, sim, total_time, step_size, tail_joint_profile, track_joint_profile, data_recorder, None)
        data_recorder.save("source/results/DEBUG_FFDragApproach_v1.csv")

if __name__ == "__main__":
    main()

    # Close sim app
    simulation_app.close()

    