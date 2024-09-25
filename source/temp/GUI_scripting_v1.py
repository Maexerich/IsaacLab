# import omni.usd
# from pxr import Sdf, Gf, Tf
# from pxr import Usd, UsdGeom, UsdPhysics, UsdShade



import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.extensions as extensions_utils
from omni.isaac.core.articulations import Articulation, ArticulationView


# Open Stage
value = stage_utils.open_stage(usd_path="C:/Users/Max/IsaacLab/source/temp/scene_creation_using_GUI_v1.usd")
stage = stage_utils.get_current_stage()


# Load Articulation
usd_path = "C:/Users/Max/IsaacLab/source/temp/box_w_tail.usd"
articulation = stage_utils.add_reference_to_stage()
