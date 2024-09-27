### GOAL ###
# Load v5 scene which was designed using only the GUI of Isaac Sim
# The scene simulates properly when run using the GUI
# The scene contains;
# - A rod with a box attached to it
# - A cube of 50kg mass
# - Linear Force Field, powerful enough to move cube
# - Drag Force Field, which affects rotation of tail
# CONFIG: ForceFields ignore body of robot. Tail is force controlled to have a
#         angular velocity of 800°/s

import argparse
import sys

import time
import numpy as np
import torch

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

import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.extensions as extensions_utils

# from omni.isaac.core.articulations import Articulation
from omni.isaac.core.articulations import ArticulationView

# Open stage
value = stage_utils.open_stage(usd_path="source/temp/scene_creation_using_GUI_v5.usd")
stage = stage_utils.get_current_stage()

### Load USD file into stage Attempt
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, ArticulationData, RigidObjectCfg, RigidObject
from omni.isaac.lab.actuators import ImplicitActuatorCfg, ActuatorBaseCfg
from omni.isaac.lab.sim import SimulationContext, SimulationCfg

# Print all prims in stage
stage_utils.print_stage_prim_paths()

# Get articulation
# articulation = prim_utils.get_prim_at_path(prim_path="/World/Robot/Body") # returns physics prim
DAMPING = 0.0
BOX_CFG = ArticulationCfg(
    prim_path="/World/Robot2/Body", # Must point to articulation root
    spawn=None,
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0, 0, 2.5),
        joint_pos={"TailJoint": -1.5708}, # DO NOT ADD FULL PATH: /World/Robot2/.... ADD ONLY JOINT NAME!
        # joint_pos={"box_to_rod": -0.3},
        # joint_vel={"box_to_rod": 10.0}
    ),
    actuators={
        "rod_motor": ImplicitActuatorCfg(
            joint_names_expr=["TailJoint"], # DO NOT ADD FULL PATH: /World/Robot2/.... ADD ONLY JOINT NAME!
            friction=0.2,
            # friction=0.0,
            damping=DAMPING,
            effort_limit=1000.0,
            stiffness=0.0,  # Leave at zero! (velcity and effort control!)
            velocity_limit=500.0,
        ),
    },
)
articulation = Articulation(BOX_CFG)

# PhysicsScene
# print(f"physicsScene Attributes: \n{prim_utils.get_prim_attribute_names('/physicsScene')}\n\n")
prim_utils.set_prim_attribute_value(prim_path='/physicsScene', attribute_name='physxForceField:ForceField1:physxForceFieldLinear:constant', value=2)
print(f"Get value from physicsScene: \n{prim_utils.get_prim_property(prim_path='/physicsScene', property_name='physxForceField:ForceField1:physxForceFieldLinear:constant')}")
print(f"Value in physicsScene using 'prim_utils.set_prim_attribute_value(prim_path, attribute_name, value)'")

# Load extension
print(f"\n\nLoading extension ForceField")
value = extensions_utils.enable_extension(extension_name='omni.physx.forcefields')
print(f"Extension loaded: {value}")


### Simulation
# Initialize the Simulation Context
sim_cfg = SimulationCfg(physics_prim_path="/physicsScene",
                        device='cpu', # TODO: Try with cpu
                        dt=1.0/240.0,
                        # Solver type currently TGS, maybe try PGS (more accurate)
                        use_fabric=False,)
simulation_context = SimulationContext(sim_cfg)

# Define the total time and step size
total_time = 30.0  # seconds
step_size = 1.0 / 240.0  # 60Hz simulation frequency
current_time = 0.0

# Initialize view here will give error
simulation_context.reset() # Reset before anything else
# Initialize view here it will work

articulation_view =  ArticulationView(prim_paths_expr="/World/Robot2/Body")
articulation_view.initialize()

print(f"Articulation joint names: {articulation.joint_names}")
print(f"Articulation num bodies: {articulation.num_bodies}")
print(f"Articulation joints: {articulation.find_joints(name_keys='.*')}")
print(f"Articulation actuators: {articulation.actuators}")

# Run the simulation loop
while current_time < total_time:
    # Step the simulation
    simulation_context.step()
    
    # Update the current time
    current_time += step_size

    joint_vel_setpoint = torch.full_like(articulation.actuators["rod_motor"].applied_effort, -0.5)
    articulation.set_joint_position_target(joint_vel_setpoint)
    articulation.write_data_to_sim()

    # Half-way through simulaiton, Linear component of Force Field is adjusted to attract slightly
    if current_time > 10.0 and current_time < 10.0 + step_size + 10e-6:
        print(f"Current constant: {prim_utils.get_prim_property(prim_path='/physicsScene', property_name='physxForceField:ForceField1:physxForceFieldLinear:constant')}")
        new_value = -0.2
        print(f"Setting to {new_value}")
        prim_utils.set_prim_attribute_value(prim_path='/physicsScene', attribute_name='physxForceField:ForceField1:physxForceFieldLinear:constant', value=new_value)
        print(f"Current constant: {prim_utils.get_prim_property(prim_path='/physicsScene', property_name='physxForceField:ForceField1:physxForceFieldLinear:constant')}")

print("Simulation finished.")

simulation_app.close()




