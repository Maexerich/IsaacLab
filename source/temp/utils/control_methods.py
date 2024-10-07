import numpy as np
import torch
from typing import Union

# Issac Sim imports
from omni.isaac.core.articulations import ArticulationView
# Isaac Lab imports
from omni.isaac.lab.assets import Articulation, ArticulationData


# This file contains different classes which can be used to control the actuators.
# The control-class takes the other classes as arguments.


from abc import ABC, abstractmethod
from typing import Union
class BaseController(ABC):
    @abstractmethod
    def get_control_setpoint(self, current_time_seconds: float) -> Union[float, None]:
        """Abstract method to be implemented by all controller profiles."""
        pass

class SimpleAngAccelProfile:
    def __init__(self, sim_dt: float, a: float = 200.0, t0: float = 0.0, t0_t1: float = 0.4, t1_t2: float = 0.2):
        """Simple angular velocity profile w(t) defined as follows;
        for t0 < t < t1:    w(t) = a*(t-t0)
        for t1 < t < t2:    w(t) = a*(t1-t0)
        otherwise:          w(t) = <return None>

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

    def get_ang_vel(self, current_time: float = None, count: int = None):
        "Returns angular velocity in rad/s at simulation step count. variable 'current_time' has precedence over 'count'."
        assert (count is not None) or (current_time is not None)
        if current_time is not None:
            count = int(current_time / self.sim_dt)
        current_time = count * self.sim_dt
        if current_time < self.t0:
            return None
        elif current_time < self.t1:
            return self.acceleration * (current_time - self.t0)
        elif current_time < self.t2:
            return self.acceleration * (self.t1 - self.t0)
        else:
            return None
    
    def get_control_setpoint(self, current_time_seconds: float = None, count: int = None):
        return self.get_ang_vel(current_time_seconds, count)

class SoftAngAccelProfile(BaseController):
    def __init__(self, sim_dt: float, a: float = 100.0, k: float = 5, t0: float = 0.0, t0_t1: float = 0.2, t1_t2: float = 0.2, reach_setpoint_gain: float = 0.8):
        """
        Soft angular acceleration profile w(t) defined as follows:
        for t0 < t < t1*0.8:    w(t) = a*t
        for t1*0.8 < t < reach_setpoint_gain*t2: w(t) smoothly transitions to a*t1.
        for t > t2:             w(t) = <return None>

        Args:
        - sim_dt: Simulation time-discretization in seconds.
        - a: Angular acceleration in rad/s^2.
        - k: Gain for how fast to converge to the constant angular velocity. Higher values = faster convergence
        - t0: Start time for acceleration in seconds.
        - t0_t1: Time duration for acceleration in seconds (mathematically: t1-t0).
        - t1_t2: Time for constant angular velocity in seconds (mathematically: t2-t1).
        - reach_setpoint_gain: Fractional multiplier for t2, controlling the time by which the profile should approximately reach a*t1.
        """
        self.sim_dt = sim_dt
        self.acceleration = a
        self.k = k
        self.t0 = t0
        self.t1 = t0 + t0_t1
        self.t2 = self.t1 + t1_t2
        self.t1_08 = 0.8 * self.t1  # 0.8 * t1
        self.t2_X = reach_setpoint_gain * self.t2  # t2 multiplied by reach_setpoint_gain
        self._last_value = 0.0

    def get_control_setpoint(self, current_time_seconds: float) -> Union[float, None]:
        # If current time is less than t0, return None
        if current_time_seconds < self.t0:
            return None

        # Linear acceleration phase: t0 < t < 0.8 * t1
        elif current_time_seconds < self.t1_08:
            self._last_value = self.acceleration * current_time_seconds
            return self._last_value

        # Smooth transition phase: 0.8 * t1 < t < reach_setpoint_gain * t2
        # elif current_time_seconds < self.t2_X:
        elif current_time_seconds < self.t2:
            # Fraction to track progress between 0.8 * t1 and t2_X
            fraction = (current_time_seconds - self.t1_08) / (self.t2_X - self.t1_08)
            # Smooth transition using exponential decay
            difference_to_setpoint = self.acceleration * self.t1 - self._last_value
            return self._last_value + difference_to_setpoint * (1 - np.exp(-self.k * fraction))

        # After t2: return None
        else:
            return None

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
    """
    This class implements simple profiles for LINEAR velocity control, meant to be used for the TrackDrive prismatic joint.
    """
    def __init__(self, sim_dt: float, control_mode: str = 'const', const_vel: float = -30.0):
        self.sim_dt = sim_dt
        self.const_vel = const_vel

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
        return self.const_vel

class LinearForce(BaseController):
    def __init__(self):
        raise NotImplementedError("LinearForce is not implemented yet.")

class LinearPosition(BaseController):
    def __init__(self):
        raise NotImplementedError("LinearPosition is not implemented yet.")

class DummyProfile(BaseController):
    def __init__(self):
        pass

    def get_control_setpoint(self, current_time_seconds: float) -> Union[float, None]:
        return None

class Controller_floatingBase():
    """
    This class is used to provide control inputs to the actuators. It will only work with '/World/Robot_floatingBase'!
    How the actuators are controlled is determined by the provided arguments and their class types.

    ATTENTION: This class contains a lot of hard-coded stuff, which will not generalize to;
    - multiple environments
    - differently named actuators (actuator_names == joint_names)
    - or different usd-files

    Main functions:
    - __init__(): Initializes the controller with the provided arguments.
    - update_control_input(current_time: float): Steps the control input and writes the targets into the articulation-buffer.
    """
    def __init__(self, articulation_view: ArticulationView, articulation: Articulation, artdata: ArticulationData,
                 rotational_drive_profile: Union[SimpleAngAccelProfile, TorqueProfile, DummyProfile, SoftAngAccelProfile], 
                 linear_drive_profile: Union[LinearVelocity, LinearForce, LinearPosition, DummyProfile],):
        self.articulation_view = articulation_view
        self.articulation = articulation
        self.artdata = artdata
        self.rotational_drive_profile = rotational_drive_profile if rotational_drive_profile is not None else DummyProfile()
        self.linear_drive_profile = linear_drive_profile if linear_drive_profile is not None else DummyProfile()

        self._get_joint_gains() # This initializes the dictionary 'self.gains'
        
        # Initialize control modes
        self.joint_control_mode, self.track_control_mode = self._get_control_modes()

    def _get_joint_gains(self):
        "This function fetches the joint gains. BE AWARE: Includes Hard-Coded values for joint indices (& joint names) and limited to 1 environment!"
        # TODO: Hard-coded for one environment, that explains the '[0]'
        damping = self.artdata.joint_damping[0]
        stiffness = self.artdata.joint_stiffness[0]
        # friction = self.artdata.joint_friction[0] # Not needed for now
        
        self.gains = {}
        
        for actuator, index in self.articulation_view._dof_indices.items():
            self.gains[actuator] = {"stiffness": stiffness[index].item(), "damping": damping[index].item()}
    
    def _get_control_modes(self):
        "Based on provided profiles, the type of control is determined."
        # Tail Drive
        if isinstance(self.rotational_drive_profile, SimpleAngAccelProfile) or isinstance(self.rotational_drive_profile, SoftAngAccelProfile):
            self.joint_control_mode = "velocity"
        elif isinstance(self.rotational_drive_profile, TorqueProfile):
            self.joint_control_mode = "effort"
        else:
            raise ValueError("Invalid controller provided. If no control input is desired, return Torque-Profile with value 0.0.")
        
        # Track Drive
        if isinstance(self.linear_drive_profile, LinearVelocity):
            self.track_control_mode = "velocity"
        elif isinstance(self.linear_drive_profile, Union[LinearForce, DummyProfile]):
            self.track_control_mode = "effort"
        elif isinstance(self.linear_drive_profile, LinearPosition):
            self.track_control_mode = "position"
        # elif isinstance(self.linear_drive_profile, DummyProfile):
        #     self.track_control_mode = None
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

        # Send control input only to those DoF's that exist
        control_input = {}
        for actuator, _ in self.articulation_view._dof_indices.items():
            if actuator == "TailDrive":
                control_input["TailDrive"] = tail_drive_input_dict
            if actuator == "TrackDrive":
                control_input["TrackDrive"] = track_drive_input_dict
        self._send_control_inputs(control_input)
    
    def _set_control_mode(self):
        """
        Ensures respective actuators are set to the correct control mode. 
        In addition, damping and stiffness are set appropriately and written to the simulation.
        
        A multitude of indexing is hard-coded, which is not ideal.
        
        Hard-coded default values for damping are set:
        - TailDrive: 1.0
        - TrackDrive: 10e4
        """
        # TODO: 'Joint_indices' arguments are hard-coded!
        joint_indices = self.articulation_view._dof_indices
        for actuator, index in self.articulation_view._dof_indices.items():
            if actuator == "TailDrive":
                self.articulation_view.switch_control_mode(mode=self.joint_control_mode, joint_indices=[index])
            elif actuator == "TrackDrive":
                self.articulation_view.switch_control_mode(mode=self.track_control_mode, joint_indices=[index])
        # # Tail Joint
        # self.articulation_view.switch_control_mode(mode=self.joint_control_mode, joint_indices=[1])
        # # Track Drive
        # self.articulation_view.switch_control_mode(mode=self.track_control_mode, joint_indices=[0])
        
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
        if (self.joint_control_mode == "velocity") and ("TailDrive" in joint_indices):
            # TODO: Hard-coded: '[0]' comes from 0th environment, '[1]' comes from index for TailDrive
            index = joint_indices['TailDrive']
            stiffness[0][index] = 0.0
            damping[0][index] = max(self.gains["TailDrive"]["damping"], 1.0) # TODO: Hard-coded damping value
        if (self.track_control_mode == "velocity") and ("TrackDrive" in joint_indices):
            # TODO: Hard-coded: '[0]' comes from 0th environment, '[0]' comes from index for TrackDrive
            index = joint_indices['TrackDrive']
            stiffness[0][index] = 0.0
            damping[0][index] = max(self.gains["TrackDrive"]["damping"], 10e4) # TODO: Hard-coded damping value

        # Effort control requires: Stiffness: 0.0   Damping: 0.0
        if (self.joint_control_mode == "effort") and ("TailDrive" in joint_indices):
            index = joint_indices['TailDrive']
            stiffness[0][index] = 0.0
            damping[0][index] = 0.0
        if (self.track_control_mode == "effort") and ("TrackDrive" in joint_indices):
            index = joint_indices['TrackDrive']
            stiffness[0][index] = 0.0
            damping[0][index] = 0.0
        
        # Write gains to simulation
        self.articulation.write_joint_stiffness_to_sim(stiffness)
        self.articulation.write_joint_damping_to_sim(damping)

    def _send_control_inputs(self, control_input: dict):
        # TODO: This is hard-coded for one environment (first dimension of the tensor is 1, meaning '1 environemnt')
        # position = torch.tensor([[track_drive_input_dict["position"], tail_drive_input_dict["position"]]], device='cuda:0')
        # velocity = torch.tensor([[track_drive_input_dict["velocity"], tail_drive_input_dict["velocity"]]], device='cuda:0')
        # effort = torch.tensor([[track_drive_input_dict["effort"], tail_drive_input_dict["effort"]]], device='cuda:0')
        
        # Create a tensor of zeros of shape (0, len(joint_indices)) [first dimension '0' is for 0th environment]
        zero_tensor = torch.zeros((1, len(control_input.keys())), device='cuda:0')

        position = zero_tensor.clone()
        velocity = zero_tensor.clone()
        effort = zero_tensor.clone()

        for actuator in control_input.keys():
            position[0][self.articulation_view._dof_indices[actuator]] = control_input[actuator]["position"]
            velocity[0][self.articulation_view._dof_indices[actuator]] = control_input[actuator]["velocity"]
            effort[0][self.articulation_view._dof_indices[actuator]] = control_input[actuator]["effort"]

        
        self.articulation.set_joint_position_target(position)
        self.articulation.set_joint_velocity_target(velocity)
        self.articulation.set_joint_effort_target(effort)