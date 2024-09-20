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

import omni.usd
from pxr import Sdf, Gf, Tf
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.extensions as extensions_utils

# Open stage
value = stage_utils.open_stage(usd_path="source/temp/scene_creation_using_GUI_v3.usd")
stage = stage_utils.get_current_stage()

### Load USD file into stage Attempt
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, ArticulationData, RigidObjectCfg, RigidObject
from omni.isaac.lab.actuators import ImplicitActuatorCfg, ActuatorBaseCfg
from omni.isaac.lab.sim import SimulationContext, SimulationCfg

DAMPING = 0.0
BOX_CFG = ArticulationCfg(
    prim_path="/World/Origin/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/Max/IsaacLab/source/temp/box_w_tail.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            # max_linear_velocity=1000.0,
            # max_angular_velocity=1000.0,
            # max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
            # kinematic_enabled=True
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True, 
            enabled_self_collisions=True, 
            fix_root_link=False # Fix Base Root
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 2.5),
        joint_pos={"box_to_rod": -1.5708},
        # joint_pos={"box_to_rod": -0.3},
        # joint_vel={"box_to_rod": 10.0}
    ),
    actuators={
        "rod_motor": ImplicitActuatorCfg(
            joint_names_expr=["box_to_rod"],
            friction=0.2,
            # friction=0.0,
            damping=DAMPING,
            effort_limit=1000.0,
            stiffness=0.0,  # Leave at zero! (velcity and effort control!)
            velocity_limit=500.0,
        ),
    },
)
# robot = Articulation(BOX_CFG)

cone_cfg = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
cone_cfg.func("/World/Objects/ConeRigid", cone_cfg, translation=(-0.2, 0.0, 5.0), orientation=(0.5, 0.0, 0.5, 0.0))

# Print all prims in stage
stage_utils.print_stage_prim_paths()

# PhysicsScene
print(f"PhysicsScene Attributes: \n{prim_utils.get_prim_attribute_names('/PhysicsScene')}\n\n")
print(f"Get value from PhysicsScene: \n{prim_utils.get_prim_property(prim_path='/PhysicsScene', property_name='physxForceField:ForceField1:physxForceFieldLinear:constant')}")
print(f"Value in PhysicsScene using 'prim_utils.set_prim_attribute_value(prim_path, attribute_name, value)'")

# Load extension
print(f"\n\nLoading extension ForceField")
value = extensions_utils.enable_extension(extension_name='omni.physx.forcefields')
print(f"Extension loaded: {value}")


### Simulation
# Initialize the Simulation Context
sim_cfg = SimulationCfg(physics_prim_path="/PhysicsScene",
                        # device='cuda:0',
                        dt=1.0/240.0,
                        use_fabric=False)
simulation_context = SimulationContext(sim_cfg)

# Define the total time and step size
total_time = 30.0  # seconds
step_size = 1.0 / 240.0  # 60Hz simulation frequency
current_time = 0.0

simulation_context.reset() # Reset before anything else

# Run the simulation loop
while current_time < total_time:
    # Step the simulation
    simulation_context.step()
    
    # Update the current time
    current_time += step_size

    # Half-way through simulaiton, Linear component of Force Field is adjusted to attract slightly
    if current_time > 10.0 and current_time < 10.0 + step_size + 10e-6:
        print(f"Current constant: {prim_utils.get_prim_property(prim_path='/PhysicsScene', property_name='physxForceField:ForceField1:physxForceFieldLinear:constant')}")
        new_value = -0.2
        print(f"Setting to {new_value}")
        prim_utils.set_prim_attribute_value(prim_path='/PhysicsScene', attribute_name='physxForceField:ForceField1:physxForceFieldLinear:constant', value=new_value)
        print(f"Current constant: {prim_utils.get_prim_property(prim_path='/PhysicsScene', property_name='physxForceField:ForceField1:physxForceFieldLinear:constant')}")

print("Simulation finished.")

simulation_app.close()




