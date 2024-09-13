import argparse
import sys

import time

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


### Imports ###
from pxr import Gf, Usd
from pxr import UsdGeom, UsdPhysics
from pxr.ForceFieldSchema import PhysxForceFieldAPI, PhysxForceFieldLinearAPI

from omni.isaac.lab.sim import SimulationContext


### Functions ###
def spawn_cones_at_origins(stage, origins):
    for i, origin in enumerate(origins):
        # Create a cone geometry at each origin
        cone_prim_path = f"/World/Origin{i+1}/Cone"
        cone_prim = UsdGeom.Cone.Define(stage, cone_prim_path)
        
        # Position the cone at the origin
        cone_prim.AddTranslateOp().Set(Gf.Vec3f(*origin))
        
        # Make the cone a rigid body
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(cone_prim.GetPrim())
        
        # Optionally, set mass, inertia, etc.
        mass_api = UsdPhysics.MassAPI.Apply(cone_prim.GetPrim())
        mass_api.GetMassAttr().Set(1.0)  # Set mass of 1.0
        
        # Add collision
        UsdPhysics.CollisionAPI.Apply(cone_prim.GetPrim())
        
    # Save the stage
    stage.GetRootLayer().Save()

def add_wind_force_field(stage, wind_strength=1000.0):
    # Define a transform for the wind force field
    wind_prim_path = "/World/WindForceField"
    wind_xform = UsdGeom.Xform.Define(stage, wind_prim_path)

    # Apply the force field API to the transform
    wind_prim = wind_xform.GetPrim()
    wind_field_api = PhysxForceFieldLinearAPI.Apply(wind_prim, "Wind")
    
    # Configure the wind force field
    wind_field_api.CreateConstantAttr(wind_strength)
    wind_field_api.CreateLinearAttr(30.0)  # This could represent the attenuation or other properties
    wind_field_api.CreateInverseSquareAttr(0.0)  # No inverse square attenuation
    
    # Set the wind direction (e.g., blowing along the x-axis)
    wind_field_api.CreateDirectionAttr(Gf.Vec3f(1.0, 0.0, 0.0))
    
    # Enable the force field
    wind_base_api = PhysxForceFieldAPI(wind_prim, "Wind")
    wind_base_api.CreateEnabledAttr(True)
    wind_base_api.CreatePositionAttr(Gf.Vec3f(0.0, 0.0, 0.0))  # Set the position of the wind source
    wind_base_api.CreateRangeAttr(Gf.Vec2f(-1.0, -1.0))  # Infinite range in this case

def run_simulation():
    # Initialize the simulation context
    sim_context = SimulationContext()

    # Load the stage
    stage = Usd.Stage.Open("source/temp/temp_cones.usda")

    # Run the simulation for a certain number of steps
    for _ in range(500):  # Simulate for 240 frames (~4 seconds at 60 fps)
        sim_context.step(render=True)

    # After simulation, save the final stage
    stage.GetRootLayer().Save()


### Main ###
def main():
    # Initialize simulation

    # Stage
    stage = Usd.Stage.CreateNew("source/temp/temp_cones.usda")

    # Spawn cones
    origins = [[0, 0, 1], [2, 0, 1], [4, 0, 1], [0, 2, 1], [2, 2, 1], [4, 2, 1]]
    spawn_cones_at_origins(stage, origins)

    # Add wind force field
    add_wind_force_field(stage)

    # Save the stage
    stage.GetRootLayer().Save()

    # Run simulation
    run_simulation()
    


if __name__ == "__main__":
    main()

    # close ap
    simulation_app.close()