import argparse
import sys
import time

from copy import deepcopy

from omni.isaac.lab.app import AppLauncher

### ARGPARSE ###
# add argparse arguments
parser = argparse.ArgumentParser(description="Second urdf implementation script.")
if False: # pxr package will only work if headless == False
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


### Test ForceField import ###
import omni
from pxr import ForceFieldSchema
from pxr.ForceFieldSchema import PhysxForceFieldAPI, PhysxForceFieldDragAPI, PhysxForceFieldLinearAPI

print("ForceFieldAPI imported successfully")

ForceFieldSchema.PhysxForceFieldDragAPI.__doc__
