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
            joint_names_expr=["TailJoint"], # DO NOT ADD FULL PATH: /World/Robot2/.... ADD ONLY JOINT NAME!
            friction=0.02,
            damping=10e4, # A high value (10e4) will make joint follow velocity setpoint more closely
            stiffness=0.0,  # Leave at zero for velocity and effort control!
        ),
        "TrackDrive": ImplicitActuatorCfg(
            joint_names_expr=["Track_PrismaticDrive"],
            friction=0.02,
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
            joint_pos={"TailJoint": 0.0}, # DO NOT ADD FULL PATH: /World/Robot2/.... ADD ONLY JOINT NAME!
        ),
        actuators=actuators,
    )
    articulation = Articulation(box_cfg)
    return articulation

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

    # Incoming forces and torques
    joint_names = articulation_view._metadata.joint_names
    forces_and_torques = {
        "force": articulation_view.get_measured_joint_forces()[:, 1:, :3],
        "torque": articulation_view.get_measured_joint_forces()[:, 1:, 3:],
    }
    data_recorder.record_body_params(time_seconds, joint_names, forces_and_torques)