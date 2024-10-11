import torch
import numpy as np
# Isaac Sim Imports
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.extensions as extensions_utils
from omni.isaac.core.articulations import ArticulationView
# Isaac Lab Imports
from omni.isaac.lab.assets import Articulation, ArticulationData, ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
# Custom Imports
from utils.data_recorder import DataRecorder_V2


### Setup Functions ###
class PhysicsSceneModifier:
    """
    Provide a class for reading and modifying values in the physics scene.
    Ensure that the physics scene is located at '/physicsScene'. (NOT '/PhysicsScene'!)
    This function expects only one force field to be present in the scene, it must be a 'Drag' force field.

    During initialization, the extension 'ForceFields' is loaded by default.
    The extension does not have to be loaded for the stage to work. However, disabling the extension will
    prevent force fields from being active!

    Main functions:
    - get_params(): Fetches (a selection of) relevant parameters of the drag force field.
    - set_params(params: dict): Sets/Modifies the parameters of the drag force field.
    - disable_drag_force_field(): Disables the drag force field.
    """
    def __init__(self, load_force_fields_extension: bool = True):
        if load_force_fields_extension:
            self._load_extension_ForceFields()
        
        try:
            # Get all attributes from '/physicsScene'
            attributes = prim_utils.get_prim_attribute_names('/physicsScene')
        except RuntimeError:
            raise RuntimeError("Failed to get attribute names from '/physicsScene'. Is it '/physicsScene' or '/PhysicsScene'?")
        
        # Ensure there is only one force field defined in the scene;
        assert 'physxForceField:ForceField1:physxForceField:enabled' in attributes, "Expected 'physxForceField:ForceField1' in physicsScene."
        assert 'physxForceField:ForceField2:physxForceField:enabled' not in attributes, "Expected only one force field in physicsScene, found 'physxForceField:ForceField2'."

        # Is the force field also a 'Drag' force field?
        assert 'physxForceField:ForceField1:physxForceFieldDrag:square' in attributes, "Expected 'physxForceField:ForceField1' to be a 'Drag' force field."

        # --> Force Field Check Passed
        self._attributes = attributes
    
    def _load_extension_ForceFields(self):
        "This function enables the ForceFields extension."
        boolean = extensions_utils.enable_extension(extension_name='omni.physx.forcefields')
        if boolean == False:
            raise Exception("Failed to enable ForceFields extension.")
        
    def get_params(self) -> dict:
        "This function fetches (relevant) parameters of the drag force field."
        # All possible attributes can be found with: prim_utils.get_prim_attribute_names('/physicsScene')

        base_string = 'physxForceField:ForceField1:physxForceFieldDrag:'
        enabled = prim_utils.get_prim_property(prim_path='/physicsScene', property_name=base_string + 'enabled')
        surfaceAreaScaledEnabled = prim_utils.get_prim_property(prim_path='/physicsScene', property_name=base_string + 'surfaceAreaScaledEnabled')
        surfaceSampleDensity = prim_utils.get_prim_property(prim_path='/physicsScene', property_name=base_string + 'surfaceSampleDensity')
        linear = prim_utils.get_prim_property(prim_path='/physicsScene', property_name=base_string + 'linear')
        square = prim_utils.get_prim_property(prim_path='/physicsScene', property_name=base_string + 'square')

        params = {"FF_enabled": enabled,
                  "FF_surfaceAreaScaledEnabled": surfaceAreaScaledEnabled,
                  "FF_surfaceSampleDensity": surfaceSampleDensity,
                  "FF_linear": linear,
                  "FF_square": square}
        return params

    def set_params(self, params: dict):
        for key, value in params.items():
            if key not in self._attributes:
                print(f"Permissible keys are: \n{self._attributes}\n")
                raise ValueError(f"Attribute '{key}' not found in '/physicsScene'.")
            
            # Get current value
            current_value = prim_utils.get_prim_property(prim_path='/physicsScene', property_name=key)
            # Set new value
            prim_utils.set_prim_attribute_value(prim_path='/physicsScene', attribute_name=key, value=value)
            # Get newly set value
            new_value = prim_utils.get_prim_property(prim_path='/physicsScene', property_name=key)
            assert new_value == value, f"Failed to set value for '{key}'. Expected: {value}, got: {new_value}"
    
    def disable_drag_force_field(self):
        "Disables the force field '1' in the physics scene. (calls the internal function 'set_params')"
        settings = {"physxForceField:ForceField1:physxForceField:enabled": False}
        self.set_params(settings)

def open_stage(stage_path: str):
    stage_utils.open_stage(usd_path=stage_path)
    stage = stage_utils.get_current_stage()
    return stage

def instantiate_Articulation(prim_path) -> Articulation:
    """This function instantiates the Articulation object.
    It is a custom function to be used only with a specific usd file that meets expectations.
    This function expects specific prim_paths, as well as specific names for the joints. Refer
    to the code for details.
    
    Args:
    - prim_path (str): The path to the articulation root in the stage. Accepted values are:
            '/World/Robot_floatingBase'
            '/World/Robot_fixedBase'
    
    Returns:
    - articulation (Articulation): The Articulation object.
    """
    actuators={
        "TailDrive": ImplicitActuatorCfg(
            joint_names_expr=["TailDrive"], # DO NOT ADD FULL PATH: /World/Robot2/.... ADD ONLY JOINT NAME!
            friction=0.02,
            damping=10, # A high value (10e4) will make joint follow velocity setpoint more closely
            stiffness=0.0,  # Leave at zero for velocity and effort control!
            effort_limit=30, # Nm
        ),
        "TrackDrive": ImplicitActuatorCfg(
            joint_names_expr=["TrackDrive"],
            friction=0.0,
            damping=10e4,
            stiffness=0.0, # Leave at zero for velocity and effort control!
        )
    }

    if prim_path == "/World/Robot_fixedBase":
        # Remove 'Track Drive' entry from dictionary, as this joint does not exist in the fixed base system
        actuators.pop("TrackDrive")
    else:
        pass

    # Articulation Configuration 
    box_cfg = ArticulationCfg(
        prim_path=prim_path, # 'Must' point to articulation root (seemingly can be a prim higher in hierarchy too...)
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={"TailDrive": 0.0}, # DO NOT ADD FULL PATH: /World/Robot2/.... ADD ONLY JOINT NAME!
            joint_vel={"TailDrive": 10.0},
        ),
        actuators=actuators,
    )
    articulation = Articulation(box_cfg)
    return articulation

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
    rod_CoM_pos_vec = get_body_position_vector("Tail") # Position vector of CoM of Tail, not the joint!
    endeffector_pos_vec = get_body_position_vector("Endeffector")
    vec_CoM_tail_to_endeffector = endeffector_pos_vec - rod_CoM_pos_vec # This vector has correct heading, but only half the length
    # Vector currently points from CoM of the tail to the endeffector, which is only half of the tail. We want TailDrive to Endeffector.
    tail_orientation_in_world_coordinates = vec_CoM_tail_to_endeffector * 2.0
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
    C_d = 1.2 # [has no unit]
    DIAMETER = 0.05 # m, TODO: Hard-coded value
    LENGTH = tail_motion["tail_orientation"].norm(p=2) # m,
    DISCRETIZATION = 200 # Number of points to discretize tail length
    vec_omega = (tail_motion["rotation_axis"] * tail_motion["rotation_magnitude"]).reshape(3,)
    DEVICE = vec_omega.device
    vec_joint_endeffector = tail_motion["tail_orientation"].reshape(3,) # Deactivated here and next line, shouldn't have an affect
    dir_joint_endeffector = (vec_joint_endeffector/torch.norm(input=vec_joint_endeffector, p=2)).reshape(3,)
    array_of_A = np.zeros((1, DISCRETIZATION))
    array_of_Td = np.zeros((DISCRETIZATION, 3))
    array_of_F_d = np.zeros((DISCRETIZATION, 3))

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
    
    def A_delta(plane_perpendicular_to):
        "Returns the d/ds(A) at position s. d/ds(A) = diameter * L_projected_at_s()/LENGTH"
        return DIAMETER * L_projected_at_s(plane_perpendicular_to=plane_perpendicular_to)/LENGTH
    
    def F_drag_at_s(s: float):
        "Returns the drag force at position s. Returned is a vector."
        import torch
        v = WIND + v_wind_perceived_at_s(s)
        if torch.norm(input=v, p=2) != 0.0:
            v_dir = v/torch.norm(input=v, p=2)
        else:
            v_dir = v.clone()
        v_squared = torch.norm(input=v, p=2)**2
        # Surface Area A is projected and taken proportional quantity according to DISCRETIZATION
        # A = DIAMETER * L_projected_at_s(plane_perpendicular_to=v_dir) / DISCRETIZATION
        A_d = A_delta(v_dir)
        array_of_A[0, int(s/(LENGTH/DISCRETIZATION))-1] = A_d
        F_drag_at_s = 0.5 * density_air * C_d * A_d * v_squared * v_dir
        F_drag_at_s_my_formula = 0.5 * density_air * C_d * DIAMETER * (WIND.norm(p=2) + vec_omega.norm(p=2) * s)**2 # This formula is correct!
        array_of_F_d[int(s/(LENGTH/DISCRETIZATION))-1, :] = F_drag_at_s.cpu()
        return F_drag_at_s # This value is checked and is correct (magnitude compared with 'F_drag_at_s_my_formula')
    
    def T_total():
        "Returns the total torque acting on the tail. Returned is a vector."
        T_total = torch.zeros(3).to('cuda')
        vec_length = vec_joint_endeffector.norm(p=2)
        s_values = torch.linspace(0.0, vec_length, steps=DISCRETIZATION)
        step_size = vec_length / (DISCRETIZATION - 1)
        
        for s in s_values:
            assert 0.0 <= s <= vec_length
            T_d = torch.cross(vec_x_at_s(s), F_drag_at_s(s))
            array_of_Td[int(s / step_size), :] = T_d.cpu()
            T_total += T_d * step_size
        
        return T_total
    
    def F_Drag_total():
        "Returns the total drag force acting on the tail. Returned is a vector."
        F_total = torch.zeros(3).to('cuda')
        s_values = torch.linspace(0.0, LENGTH, steps=DISCRETIZATION)
        step_size = LENGTH / (DISCRETIZATION - 1)

        for s in s_values:
            assert 0.0 <= s <= LENGTH
            F_d = F_drag_at_s(s)
            F_total += F_d * step_size
        
        return F_total
    
    def F_Drag_total_if_no_incoming_wind():
        "The analytical perfect solution for the drag force acting on the tail, assuming no incoming wind."
        assert torch.norm(input=WIND, p=2) == 0.0
        return 0.5 * density_air * C_d * DIAMETER *  vec_omega.norm(p=2)**2 * 1/3 * LENGTH**3 # Correct
    
    def F_substitution():
        "Returns the equivalent force acting through the CoM of the tail. Returned is a vector."
        vec_x_at_Lhalf = vec_x_at_s(vec_joint_endeffector.norm(p=2)/2)
        norm_vec_x = torch.norm(input=vec_x_at_Lhalf, p=2)
        Torque_total = T_total()
        data_recorder.record(time_seconds=time_seconds, values={"Wind torque magnitude": Torque_total.norm(p=2).cpu()})
        return -(torch.cross(vec_x_at_Lhalf, Torque_total)/norm_vec_x**2)
    
    def get_unit_norm_vector(vector: torch.Tensor):
        "Returns the unit vector of the input vector."
        if torch.norm(input=vector, p=2) == 0.0:
            return vector
        else:
            return vector/torch.norm(input=vector, p=2)
    
    def calculate_F_sub():
        Torque_total = T_total()
        vec_x_at_Lhalf = vec_x_at_s(vec_joint_endeffector.norm(p=2)/2) # Vector Joint to CoM of tail
        # Direction of perceived wind due to tail's rotation
        # v_wind_at_Lhalf = v_wind_perceived_at_s(LENGTH/2)
        dir_wind_tail = get_unit_norm_vector(v_wind_perceived_at_s(LENGTH/2))
        # Direction of wind due to incoming wind
        dir_wind_incoming = get_unit_norm_vector(WIND)

        # These if-statements cover all possible combinations of having/not having a specific wind component
        if torch.norm(input=dir_wind_tail, p=2) == 0.0 and torch.norm(input=dir_wind_incoming, p=2) == 0.0:
            ### Case 1: Incoming wind is 0.0 AND tail is not rotating, e.g. dir_wind_tail.norm(p=2) == 0.0
            # Here, F_sub must naturally be 0.0, as there is no rotation and no incoming wind
            return torch.zeros((3,1)).to(DEVICE)
        elif torch.norm(input=dir_wind_incoming, p=2) == 0.0:
            ### Case 2: Incoming wind is 0.0, e.g. WIND.norm(p=2) == 0.0
            # Here, F_sub must be in the same direction as dir_wind_tail, e.g. F_sub = beta * dir_wind_tail
            matrix_A = torch.cross(vec_x_at_Lhalf, dir_wind_tail).reshape(3,1).to(DEVICE)
            x = torch.linalg.lstsq(matrix_A, Torque_total.reshape(3,1))
            F_sub = x.solution * dir_wind_tail
        elif torch.norm(input=dir_wind_tail, p=2) == 0.0:
            ### Case 3: No tail rotation (dir_wind_tail.norm(p=2) == 0.0), but incoming wind is present
            # Here, F_sub must be in the same direction as dir_wind_incoming, e.g. F_sub = alpha * dir_wind_incoming
            matrix_A = torch.cross(vec_x_at_Lhalf, dir_wind_incoming).reshape(3,1).to(DEVICE)
            x = torch.linalg.lstsq(matrix_A, Torque_total.reshape(3,1))
            F_sub = x.solution * dir_wind_incoming
        else:
            ### Case 4: Both magnitudes are non-zero
            # Here we solve a linear system of equations where F_sub = alpha * dir_wind_incoming + beta * dir_wind_tail
            matrix_A = torch.stack([torch.cross(vec_x_at_Lhalf, dir_wind_incoming), torch.cross(vec_x_at_Lhalf, dir_wind_tail)], dim=1).to(DEVICE)
            x = torch.linalg.lstsq(matrix_A, Torque_total.reshape(3,1))
            F_sub = x.solution[0] * dir_wind_incoming + x.solution[1] * dir_wind_tail

        return F_sub
    
    ### Apply forces ###
    # F_sub = F_substitution() # Incorrect because of assumption that x(s) is perpendicular to F_sub
    # F_sub = F_Drag_total() # Correct, because it 'matches' (depends on DISCRETIZATION) the analytical solution 'F_Drag_total_if_no_incoming_wind()'
    F_sub = calculate_F_sub()
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
    
    F_subcpu = F_sub.cpu()
    W_cpu = WIND.cpu()
    data_recorder.record(time_seconds=time_seconds, values={
        # "e_F_x": F_sub_unit_vector[0].cpu(),
        # "e_F_y": F_sub_unit_vector[1].cpu(),
        # "e_F_z": F_sub_unit_vector[2].cpu(),
        "F_sub_x": F_subcpu[0,0,0].item(),
        "F_sub_y": F_subcpu[0,0,1].item(),
        "F_sub_z": F_subcpu[0,0,2].item(),
        "F_total": F_subcpu[0,0,:].norm(p=2).item(),
        "F_applied?": apply,
        "A_tilde": array_of_A.sum(),
        "Wind_x": W_cpu[0].item(),
        "Wind_y": W_cpu[1].item(),
        "Wind_z": W_cpu[2].item(),
    })


### Data Recording Functions ###
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
        "effort_applied": artdata.applied_torque,
        "position": articulation_view.get_joint_positions(),
        "velocity": articulation_view.get_joint_velocities(),
    }
    data_recorder.record_actuator_params(time_seconds=time_seconds, multi_dim_values=joint_parameters, actuators=articulation_view._dof_indices)

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

    # Incoming forces and torques
    joint_names = articulation_view._metadata.joint_names
    forces_and_torques = {
        "force": articulation_view.get_measured_joint_forces()[:, 1:, :3],
        "torque": articulation_view.get_measured_joint_forces()[:, 1:, 3:],
    }
    data_recorder.record_body_params(time_seconds, joint_names, forces_and_torques)