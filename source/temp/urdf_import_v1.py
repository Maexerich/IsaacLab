import omni.isaac.kit as omniIsaac
from omni.isaac.utils.usd_utils import load_usd_file, get_prim_path
from omni.isaac.utils.legacy_physics import SimParams
from omni.isaac.utils.math.rigid_body import RigidBodyState

import os

cwd = os.getcwd()

# Initialize Isaac Sim
omniIsaac.init()

# Load your URDF model
urdf_path = "/path/to/your/urdf/model.urdf"
usd_path = load_usd_file(urdf_path)

# Get the root prim of the loaded model
root_prim_path = get_prim_path(usd_path)

# Create a simulation context
sim_params = SimParams()
sim_params.substeps = 10
sim_params.fix_dt = True
sim_params.dt = 0.005
sim_context = omniIsaac.SimulationContext(root_prim_path, sim_params)

# Add a torque sensor to a joint (replace "joint_name" with the actual joint name)
joint_prim_path = "/world/your_model/joint_name"
sim_context.add_torque_sensor(joint_prim_path)

# Add externally applied forces using force-fields (replace "force_field_name" with the desired name)
force_field_prim_path = "/world/force_field_name"
force_field_prim = omni.usd.Stage.Get(force_field_prim_path)
force_field_prim.GetCustomData()["force_field_type"] = "constant"
force_field_prim.GetCustomData()["force_field_magnitude"] = 10.0  # Adjust magnitude as needed
force_field_prim.GetCustomData()["force_field_direction"] = [0.0, 0.0, 1.0]  # Adjust direction as needed

# Step the simulation
while sim_context.is_running():
    # Retrieve torque sensor readings
    torque_reading = sim_context.get_torque_sensor(joint_prim_path)

    # Retrieve object state (if needed)
    object_state = sim_context.get_rigid_body_state(root_prim_path)

    # Update your simulation logic here based on sensor readings and object state

    sim_context.step()