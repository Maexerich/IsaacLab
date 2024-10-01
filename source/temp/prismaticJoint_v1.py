### GOAL ###
# Open usd-stage which has robot configured.
# Approach analytical solution using this robot, not the URDF, which might contain errors.

import argparse
import sys
import time
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
import torch
# Isaac Sim Imports
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.extensions as extensions_utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.robots import RobotView
# Isaac Lab Imports
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, ArticulationData, RigidObjectCfg, RigidObject
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.actuators import ImplicitActuatorCfg, ActuatorBaseCfg
from omni.isaac.lab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
# Custom Imports
from utils.control_methods import SimpleAngAccelProfile
from utils.data_recorder import DataRecorder

### Data recorder ###
DATA_RECORDER = DataRecorder(record=True)

def open_stage(stage_path: str = "source/temp/stage_robot_v4_PrismaticJoint.usd"):
    stage_utils.open_stage(usd_path=stage_path)
    stage = stage_utils.get_current_stage()
    return stage

def instantiate_Articulation() -> Articulation:
    print(f"[TODO]: This function is hard-coded, be sure to fix!")
    DAMPING = 0.0
    BOX_CFG = ArticulationCfg(
        prim_path="/World/Robot", # 'Must' point to articulation root (seemingly can be a prim higher in hierarchy too...)
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-5, 0, 0),
            # TODO: joint_pos doesn't work
            joint_pos={"TailJoint": 0.0}, # DO NOT ADD FULL PATH: /World/Robot2/.... ADD ONLY JOINT NAME!
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
            "TrackDrive": ImplicitActuatorCfg(
                joint_names_expr=["Track_PrismaticDrive"],
                friction=0.02,
                damping=10e4,
                stiffness=0.0, # Zero for velocity control, but high for position control
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

def get_tail_orientation(time_seconds: float, robot: Articulation, articulation_view: ArticulationView, artdata: ArticulationData):
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
    joint_name = "TailJoint" # TODO: Rename variable to tail_joint_name
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

    DATA_RECORDER.record(time_seconds=time_seconds, values={
        "tail_orientation_radians": joint_cfg["position[radians]"],
        # "tail_orientation_degrees": np.rad2deg(joint_cfg["position[radians]"].cpu()),
        "tail_velocity": joint_cfg["velocity[radians/s]"],
        })

    assert torch.isnan(tail_orientation_in_world_coordinates).any() == False
    assert torch.isnan(axis_of_rotation).any() == False
    assert torch.isnan(rotation_magnitude).any() == False

    return {"tail_orientation": tail_orientation_in_world_coordinates, "rotation_axis": axis_of_rotation, "rotation_magnitude": rotation_magnitude}

def apply_forces(time_seconds: float, robot: Articulation, articulation_view: ArticulationView, artdata: ArticulationData, tail_motion: dict, apply: bool = True):
    ### Parameters ####
    WIND = torch.tensor([[0.0], [0.0], [0.0]], device='cuda:0').reshape(3,) # m/s
    density_air = 1.293 # kg/m^3
    C_d = 1.1 # [has no unit]
    DIAMETER = 0.1 # m, TODO
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
        # return (s * dir_joint_endeffector).to(DEVICE)
        return s * dir_joint_endeffector
    
    def v_tail_at_s(s: float):
        "Returns the velocity of the tail at position s."
        # vec_x_at_s_eval = vec_x_at_s(s)
        # if vec_omega.device != vec_x_at_s_eval.device:
        #     vec_x_at_s_eval = vec_x_at_s_eval.to(vec_omega.device)
        # return torch.cross(vec_omega, vec_x_at_s(s)).to(DEVICE)
        return torch.cross(vec_omega, vec_x_at_s(s))
    
    def L_projected_at_s(plane_perpendicular_to):
        "Returns projected tail-length onto plane perpendicular to argument 'plane_perpendicular_to'."
        # Ensure plane_perpendicular_to is a unit vector
        assert (torch.norm(input=plane_perpendicular_to, p=2) - 1.0) < 1e-6
        L_projected = vec_joint_endeffector - torch.dot(vec_joint_endeffector, plane_perpendicular_to) * plane_perpendicular_to
        return torch.norm(input=L_projected, p=2)
    
    def F_drag_at_s(s: float):
        "Returns the drag force at position s. Returned is a vector."
        # v = WIND.to(DEVICE) + v_tail_at_s(s).to(DEVICE)
        v = WIND + v_tail_at_s(s)
        if torch.norm(input=v, p=2) != 0.0:
            v_dir = v/torch.norm(input=v, p=2)
        else:
            v_dir = v.clone()
        v_squared = torch.norm(input=v, p=2)**2
        # Surface Area A is projected and taken proportional quantity according to DISCRETIZATION
        A = DIAMETER * L_projected_at_s(plane_perpendicular_to=v_dir) / DISCRETIZATION
        array_of_A[0, int(s/(LENGTH/DISCRETIZATION))-1] = A
        F_drag_at_s = 0.5 * density_air * C_d * A * v_squared * v_dir
        # raise ValueError("v is simply too high???") # TODO
        return F_drag_at_s
    
    def T_total():
        "Returns the total torque acting on the tail. Returned is a vector."
        T_total = torch.zeros(3).to('cuda')
        for s in torch.linspace(0.0, vec_joint_endeffector.norm(p=2), steps=DISCRETIZATION):
            assert 0.0 <= s <= vec_joint_endeffector.norm(p=2)
            T_d = torch.cross(vec_x_at_s(s), F_drag_at_s(s)) #.to(DEVICE)
            array_of_Td[int(s/(LENGTH/DISCRETIZATION))-1, :] = T_d.cpu()
            # T_total += T_d.to('cuda')
            T_total += T_d
            # T_total += torch.cross(vec_x_at_s(s), F_drag_at_s(s))
        return T_total
    
    def F_substitution():
        "Returns the equivalent force acting through the CoM of the tail. Returned is a vector."
        # vec_x_at_Lhalf = vec_x_at_s(vec_joint_endeffector.norm(p=2)/2).to('cuda')
        vec_x_at_Lhalf = vec_x_at_s(vec_joint_endeffector.norm(p=2)/2)
        norm_vec_x = torch.norm(input=vec_x_at_Lhalf, p=2)
        Torque_total = T_total()
        DATA_RECORDER.record(time_seconds=time_seconds, values={"Analytical total Torque": Torque_total.norm(p=2).cpu()})
        return -(torch.cross(vec_x_at_Lhalf, 2*Torque_total)/norm_vec_x**2)
    
    ### Apply forces ###
    F_sub = F_substitution()
    F_sub_unit_vector = F_sub/torch.norm(input=F_sub, p=2)
    F_sub = torch.reshape(F_sub, (1, 1, 3)) # TODO: This really kills the performance (I had .to('cpu') here before)

    if apply:
        # If F_sub is 0.0 for every entry, skip (otherwise external wrench is disabled [no clue what this means...])
        if torch.norm(input=F_sub, p=2) > 0.0:
            robot.set_external_force_and_torque(forces=F_sub, torques=torch.zeros_like(F_sub), body_ids=[2], env_ids=[0])
            robot.write_data_to_sim() # TODO: Hardcoded something in source function
        else:
            print(f"[WARNING: {time_seconds}] F_sub is zero: {F_sub[0,0,:]}")
        # robot.set_external_force_and_torque(forces=F_sub, torques=torch.zeros_like(F_sub), body_ids=[2], env_ids=[0])
        # robot.write_data_to_sim()
    else:
        # print(f"Force not applied {F_sub[0,0,:]}")
        pass
    
    DATA_RECORDER.record(time_seconds=time_seconds, values={
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

def record_robot_forces(time_seconds: float, robot: Articulation, articulation_view: ArticulationView, artdata: ArticulationData):
    # Access joint forces & torques for 0th environment/entity
    forces_and_torques = articulation_view.get_measured_joint_forces(indices=[0], joint_indices=[1])[0]
    force_torque_dict = {
        key: forces_and_torques[0][i] for i, key in enumerate(["fx", "fy", "fz", "tx", "ty", "tz"])
    }
    
    values = {
        "friction": artdata.joint_friction[0].item(),
        "damping": artdata.joint_damping[0].item(),
        "vel_setpoint": robot._joint_vel_target_sim[0].item(),
        "vel_applied": articulation_view.get_joint_velocities()[0].item(),
        "effort_setpoint": robot._joint_effort_target_sim[0].item(),
        "effort_applied": artdata.applied_torque[0].item(),
        "effort_measured": articulation_view.get_measured_joint_efforts()[0].item(),
    }

    DATA_RECORDER.record(time_seconds=time_seconds, values={**values, **force_torque_dict
    })

from utils.data_recorder import DataRecorder_V2
def record(data_recorder: DataRecorder_V2, time_seconds: float, articulation: Articulation, articulation_view: ArticulationView, artdata: ArticulationData):

    # Record joint parameters
    joint_parameters = {
        "friction": artdata.joint_friction,
        "damping": artdata.joint_damping,
        "stiffness": artdata.joint_stiffness,
        "position_setpoint": articulation._joint_pos_target_sim, # TODO: maybe use different method
        "velocity_setpoint": articulation._joint_vel_target_sim,
        "effort_setpoint": articulation._joint_effort_target_sim,
        "effort_measured": articulation_view.get_measured_joint_efforts(),
        "effort_applied": artdata.applied_torque
    }
    data_recorder.record_actuator_params(time_seconds=time_seconds, multi_dim_values=joint_parameters, actuators={"TailDrive": 1, "TrackDrive": 0})

    # Record relevant body positions
    body_names = articulation.body_names
    values = {
        "position": artdata.body_pos_w,
        "velocity": artdata.body_lin_vel_w,
        "acceleration": artdata.body_lin_acc_w,
        "angular_velocity": artdata.body_ang_vel_w,
        "angular_acceleration": artdata.body_ang_acc_w,
    }
    data_recorder.record_body_params(time_seconds, body_names, values)


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
        # prim_utils.set_prim_attribute_value(prim_path='/physicsScene',
        #                                     attribute_name='physxForceField:ForceField2:physxForceField:enabled',
        #                                     value=False)
        
        print(f"""physxForceField:ForceField1:physxForceField:enabled: 
              \n{prim_utils.get_prim_property(prim_path='/physicsScene', 
                                              property_name='physxForceField:ForceField1:physxForceField:enabled')}""")
        # print(f"""physxForceField:ForceField2:physxForceField:enabled: 
        #       \n{prim_utils.get_prim_property(prim_path='/physicsScene', 
        #                                       property_name='physxForceField:ForceField2:physxForceField:enabled')}""")
        pass

from abc import ABC, abstractmethod
from typing import Union
class BaseController(ABC):
    @abstractmethod
    def get_control_setpoint(self, current_time_seconds: float) -> Union[float, None]:
        """Abstract method to be implemented by all controller profiles."""
        pass


class TorqueProfile(BaseController):
    def __init__(self, sim_dt: float, control_mode: str):
        self.sim_dt = sim_dt
        
        # A mapping from control modes to the corresponding methods
        control_methods = {
            'const': self.get_const_torque,
            'ramp': self.get_torque_ramp,
        }
        
        # Dynamically set get_control_input to the selected method
        method = control_methods.get(control_mode)
        
        # Raise an error if the provided control_mode is not valid
        if method is None:
            raise ValueError(f"Unknown control mode: {control_mode}")
        
        # Assign the method to the class
        self.get_control_setpoint = method

    def get_const_torque(self, current_time_seconds: float) -> float:
        # Example of constant torque implementation
        return 10.0  # Example value, you can change this

    def get_torque_ramp(self, current_time_seconds: float) -> float:
        # Example of ramping torque implementation
        return current_time_seconds * 2.0  # Example ramp behavior, change as needed


class LinearVelocity(BaseController):
    def __init__(self, sim_dt: float, control_mode: str = 'const', const_vel: float = -30.0):
        self.sim_dt = sim_dt

        # Define available control methods
        control_methods = {
            'const': self._get_const_velocity,
        }

        # Attempt to get the control method
        method = control_methods.get(control_mode)

        # Raise error if method is not found
        if method is None:
            raise ValueError(f"Unknown control mode: {control_mode}")

        # If a valid method is found, assign it
        self._method = method

    def get_control_setpoint(self, current_time_seconds: float) -> Union[float, None]:
        return self._method(current_time_seconds)
    
    def _get_const_velocity(self, current_time_seconds: float) -> Union[float, None]:
        return -30.0

 
class LinearForce(BaseController):
    def __init__(self):
        pass


class LinearPosition(BaseController):
    def __init__(self):
        pass


from typing import Union
class Controller():
    def __init__(self, articulation_view: ArticulationView, articulation: Articulation, artdata: ArticulationData,
                 rotational_drive_profile: Union[SimpleAngAccelProfile, TorqueProfile], 
                 linear_drive_profile: Union[LinearVelocity, LinearForce, LinearPosition],):
        self.articulation_view = articulation_view
        self.articulation = articulation
        self.artdata = artdata
        self.rotational_drive_profile = rotational_drive_profile
        self.linear_drive_profile = linear_drive_profile

        self._get_joint_gains() # This initializes the dictionary 'self.gains'
        
        # Initialize control modes
        self.joint_control_mode, self.track_control_mode = self._get_control_modes(rotational_drive_profile, linear_drive_profile)

    def _get_joint_gains(self):
        "This function fetches the joint gains. BE AWARE: Includes Hard-Coded values for joint indices (& joint names) and limited to 1 environment!"
        # TODO: Hard-coded for one environment, that explains the '[0]'
        damping = self.artdata.joint_damping[0]
        stiffness = self.artdata.joint_stiffness[0]
        # friction = self.artdata.joint_friction[0] # Not needed for now
        
        self.gains = {}
        for actuator in self.articulation.actuators.keys():
            if actuator == "TrackDrive":
                index = 0 # TODO: This is hard-coded
            elif actuator == "TailDrive":
                index = 1 # TODO: This is hard-coded
            else:
                raise ValueError("Actuator not found.") # TODO: This is hard-coded
            # index = self.articulation_view._joint_indices[actuator] # TODO: If joint names and drives have same naming scheme this can work
            self.gains[actuator] = {"stiffness": stiffness[index].item(), "damping": damping[index].item()}
    
    def _get_control_modes(self, rotational_drive_profile, linear_drive_profile):
        "Based on provided profiles, the type of control is determined."
        # Tail Drive
        if isinstance(rotational_drive_profile, SimpleAngAccelProfile):
            self.joint_control_mode = "velocity"
        elif isinstance(rotational_drive_profile, TorqueProfile):
            self.joint_control_mode = "effort"
        else:
            raise ValueError("Invalid controller provided. If no control input is desired, return Torque-Profile with value 0.0.")
        
        # Track Drive
        if isinstance(linear_drive_profile, LinearVelocity):
            self.track_control_mode = "velocity"
        elif isinstance(linear_drive_profile, LinearForce):
            self.track_control_mode = "effort"
        elif isinstance(linear_drive_profile, LinearPosition):
            self.track_control_mode = "position"
        else:
            raise ValueError("Invalid controller provided. If no control input is desired, return Linear-Force with value 0.0.")
        
        return self.joint_control_mode, self.track_control_mode
    
    def update_control_input(self, current_time: float):
        """
        This function steps the control input and writes the targets into the articulation-buffer.
        Be sure to call the function 'articulation.write_data_to_sim()' after calling this function!

        If a profile returns 'None' as a control input, the control method is changed to 'effort' with zero input.
        """
        tail_drive_input = self.rotational_drive_profile.get_control_setpoint(current_time_seconds=current_time)
        track_drive_input = self.linear_drive_profile.get_control_setpoint(current_time_seconds=current_time)

        # If either is None, switch to control mode 'effort' with zero input
        if tail_drive_input is None:
            self.joint_control_mode = "effort"
            tail_drive_input = 0.0
        if track_drive_input is None:
            self.track_control_mode = "effort"
            track_drive_input = 0.0
        
        # Use a dictionary to assign control inputs based on control mode
        # Track and Joint have individual dictionary, containing all entries.
        zero_dictionary = {"position": 0.0,
                           "velocity": 0.0,
                           "effort": 0.0,
                           "damping": 0.0,
                           "stiffness": 0.0}
        tail_drive_input_dict = zero_dictionary.copy()
        tail_drive_input_dict[self.joint_control_mode] = tail_drive_input
        
        track_drive_input_dict = zero_dictionary.copy()
        track_drive_input_dict[self.track_control_mode] = track_drive_input

        # Set control modes for tail joint and track drive correctly
        self._set_control_mode()

        # Send control input in articulation
        self._send_control_inputs(tail_drive_input_dict, track_drive_input_dict)
    
    def _set_control_mode(self):
        """
        Ensures respective actuators are set to the correct control mode. 
        In addition, damping and stiffness are set appropriately and written to the simulation.
        
        A multitude of indexing is hard-coded, which is not ideal.
        
        Hard-coded default values for damping are set:
        - TailDrive: 10.0
        - TrackDrive: 10e4
        """
        # TODO: 'Joint_indices' arguments are hard-coded!
        # Tail Joint
        self.articulation_view.switch_control_mode(mode=self.joint_control_mode, joint_indices=[1])
        # Track Drive
        self.articulation_view.switch_control_mode(mode=self.track_control_mode, joint_indices=[0])
        
        ### Adjust gains according to control mode ###
        #   (this has to be done to ensure proper control - view documentation for more information)
        stiffness = self.artdata.joint_stiffness
        damping = self.artdata.joint_damping
        # Position control requires: Stiffness: High   Damping: Low
        if (self.track_control_mode == "position"):
            raise NotImplementedError("Position control is not implemented yet.") # TODO: Not Implemented
        if (self.joint_control_mode == "position"):
            raise NotImplementedError("Position control is not implemented yet.") # TODO: Not Implemented

        # Velocity control requires: Stiffness: 0.0   Damping: non-zero
        if (self.joint_control_mode == "velocity"):
            # TODO: Hard-coded: '[0]' comes from 0th environment, '[1]' comes from index for TailDrive
            stiffness[0][1] = 0.0
            damping[0][1] = max(self.gains["TailDrive"]["damping"], 10.0) # TODO: Hard-coded damping value
        if (self.track_control_mode == "velocity"):
            # TODO: Hard-coded: '[0]' comes from 0th environment, '[0]' comes from index for TrackDrive
            stiffness[0][0] = 0.0
            damping[0][0] = max(self.gains["TrackDrive"]["damping"], 10e4) # TODO: Hard-coded damping value

        # Effort control requires: Stiffness: 0.0   Damping: 0.0
        if (self.joint_control_mode == "effort"):
            stiffness[0][1] = 0.0
            damping[0][1] = 0.0
        if (self.track_control_mode == "effort"):
            stiffness[0][0] = 0.0
            damping[0][0] = 0.0
        
        # Write gains to simulation
        self.articulation.write_joint_stiffness_to_sim(stiffness)
        self.articulation.write_joint_damping_to_sim(damping)

    def _send_control_inputs(self, tail_drive_input_dict: dict, track_drive_input_dict: dict):
        # TODO: This is hard-coded for one environment (first dimension of the tensor is 1, meaning '1 environemnt')
        position = torch.tensor([[track_drive_input_dict["position"], tail_drive_input_dict["position"]]], device='cuda:0')
        velocity = torch.tensor([[track_drive_input_dict["velocity"], tail_drive_input_dict["velocity"]]], device='cuda:0')
        effort = torch.tensor([[track_drive_input_dict["effort"], tail_drive_input_dict["effort"]]], device='cuda:0')
        
        self.articulation.set_joint_position_target(position)
        self.articulation.set_joint_velocity_target(velocity)
        self.articulation.set_joint_effort_target(effort)


def run_simulator(sim: sim_utils.SimulationContext, total_time: float, step_size: float, articulation: Articulation):
    "Runs the simulation."
    articulation_view = ArticulationView(prim_paths_expr="/World/Robot")
    articulation_view.initialize()
    artdata: ArticulationData = articulation.data

    current_time = 0.0
    
    data_recorder = DataRecorder_V2()
    
    ### ANGULAR VELOCITY CONTROL ###
    ang_vel_profile = SimpleAngAccelProfile(sim_dt=step_size,
                                            a=200,
                                            t0=2,
                                            t0_t1=0.4,
                                            t1_t2=0.2,)
    
    ### LINEAR VELOCITY CONTROL ###
    linear_vel_profile = LinearVelocity(sim_dt=step_size,
                                        control_mode='const',
                                        const_vel=-30.0)

    controller = Controller(articulation_view=articulation_view,
                            articulation=articulation,
                            artdata=artdata,
                            rotational_drive_profile=ang_vel_profile,
                            linear_drive_profile=linear_vel_profile)

    while current_time < total_time:
        ### ANGULAR VELOCITY CONTROL ###
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
        # torque = 0.0
        # articulation_view.switch_control_mode(mode="effort")
        # articulation.write_joint_damping_to_sim(torch.zeros_like(articulation.actuators["TailDrive"].damping))
        # # Apply effort
        # articulation.set_joint_effort_target(torch.full_like(articulation.actuators["TailDrive"].applied_effort, torque))
        

        ### Linear control ###
        # articulation_view.switch_control_mode(mode="velocity")
        # cart_velocity = -30 # m/s
        # tail_joint_velocity = 0.0 # rad/s
        # vel_setpoint = torch.tensor([[cart_velocity, tail_joint_velocity]], device='cuda:0')
        # articulation.set_joint_velocity_target(vel_setpoint)

        ### Controller ###
        controller.update_control_input(current_time=current_time)

        articulation.write_data_to_sim() 

        sim.step()
        current_time += step_size

        articulation.update(dt=step_size)

        ### Analytical approach
        # Get tail motion
        # tail_motion = get_tail_orientation(current_time, articulation, articulation_view, artdata)
        # # Apply wind and drag force
        # apply_forces(current_time, articulation, articulation_view, artdata, tail_motion, apply=False)

        # record_robot_forces(current_time, articulation, articulation_view, artdata)
        record(data_recorder, current_time, articulation, articulation_view, artdata)



# Main
def main():
    stage = open_stage()
    stage_utils.print_stage_prim_paths()

    # Instantiate articulation
    articulation = instantiate_Articulation()

    load_extension_ForceFields() # Not needed for analytical approach
    
    ### Simulation ###
    # Parameters
    total_time = 10.0 # seconds
    step_size = 1.0 / 60.0 # seconds
    sim_cfg = sim_utils.SimulationCfg(physics_prim_path="/physicsScene", 
                                    #   device='cpu',
                                      dt=step_size,
                                    #   use_fabric=False,
                                      )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(-100, -0.1, 2.3), target=(-95, 1.5, 1.5))

    ff_handler = PhysicsSceneModifier()
    ff_handler.disable_all() # Disables force fields, useful when debugging control inputs

    # Run the simulator
    sim.reset()
    print(f"[Info]: Setup complete. Starting first simulation...")
    run_simulator(sim, total_time, step_size, articulation)

if __name__ == "__main__":
    main()

    # DATA_RECORDER.save("source/temp/tail_position_v2.csv")
    # DATA_RECORDER.plot(
    #     dictionary={
    #         "Parameters": ["friction", "damping"],
    #         "Joint Velocity": ["vel_setpoint", "vel_applied"],
    #         "Joint Torque": ["effort_setpoint", "effort_measured", "effort_applied"],
    #         "Torques on Body": ["tx", "ty", "tz", "Analytical total Torque"],
    #         "Forces on Body": ["fx", "fy", "fz"],
    #         "Tail Orientation [rad]": ["tail_orientation_radians"],
    #         # "Tail Velocity": ["tail_velocity"],
    #         "Substitution Force on Tail": ["F_sub_x", "F_sub_y", "F_sub_z", "F_total"],
    #         "Wind constant [m/s]": ["Wind_x", "Wind_y", "Wind_z"],
    #         "Check(s)": ["F_applied?", "A_tilde"],
    #     },
    #     save_path="source/temp/tail_position_v2.png"
    # )

    # Close sim app
    simulation_app.close()

    