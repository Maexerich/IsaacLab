import argparse

from copy import deepcopy

from omni.isaac.lab.app import AppLauncher

### ARGPARSE ###
# add argparse arguments
parser = argparse.ArgumentParser(description="Second urdf implementation script.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


### Launch APP ###
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import os
import random
from pxr import Gf, Sdf, Usd
from pxr import UsdGeom, UsdUtils, UsdPhysics
from pxr import PhysxSchema, PhysicsSchemaTools, ForceFieldSchema
import omni.physx.scripts.physicsUtils as physicsUtils
import omni.kit
import omni.physxdemos as demo


class ExplodeForceFieldDemo(demo.Base):
    title = "ForceFieldExplode"
    category = demo.Categories.DEMOS
    short_description = "Suck objects together before they are blown apart."
    description = (
        "Use two spherical force fields to first attract objects to a central point and then explode them apart."
    )

    def create(self, stage):
        numberOfBoxes = 20
        boxSpacing = 2
        boxPathName = "/box"
        groundPathName = "/ground"
        scenePathName = "/World/scene"
        explodePathName = "/World/explode"
        # Physics scene
        up = Gf.Vec3f(0.0)
        up[1] = 1.0
        gravityDirection = -up
        gravityMagnitude = 1000.0
        forceRange = Gf.Vec2f(-1.0)
        center = Gf.Vec3f(0.0)
        center[1] = 400.0
        scene = UsdPhysics.Scene.Define(stage, scenePathName)
        scene.CreateGravityDirectionAttr(gravityDirection)
        scene.CreateGravityMagnitudeAttr(gravityMagnitude)
        # Plane
        physicsUtils.add_ground_plane(stage, groundPathName, "Y", 750.0, Gf.Vec3f(0.0), Gf.Vec3f(0.5))
        # Create the force field prim
        xformPrim = UsdGeom.Xform.Define(stage, explodePathName)
        xformPrim.AddTranslateOp().Set(Gf.Vec3f(0.0, 300.0, 0.0))
        explodePrim = xformPrim.GetPrim()
        suckPrimApi = ForceFieldSchema.PhysxForceFieldSphericalAPI.Apply(explodePrim, "Suck")
        suckPrimApi.CreateConstantAttr(-1e10)
        suckPrimApi.CreateLinearAttr(0.0)
        suckPrimApi.CreateInverseSquareAttr(0.0)
        suckPrimApi.CreateEnabledAttr(False)
        suckPrimApi.CreatePositionAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        suckPrimApi.CreateRangeAttr(Gf.Vec2f(-1.0, -1.0))
        explodePrimApi = ForceFieldSchema.PhysxForceFieldSphericalAPI.Apply(explodePrim, "Explode")
        explodePrimApi.CreateConstantAttr(4e10)
        explodePrimApi.CreateLinearAttr(0.0)
        explodePrimApi.CreateInverseSquareAttr(0.0)
        explodePrimApi.CreateEnabledAttr(False)
        explodePrimApi.CreatePositionAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        explodePrimApi.CreateRangeAttr(Gf.Vec2f(-1.0, -1.0))
        dragPrimApi = ForceFieldSchema.PhysxForceFieldDragAPI.Apply(explodePrim, "Drag")
        dragPrimApi.CreateMinimumSpeedAttr(10.0)
        dragPrimApi.CreateLinearAttr(1.0e6)
        dragPrimApi.CreateSquareAttr(0.0)
        dragPrimApi.CreateEnabledAttr(False)
        dragPrimApi.CreatePositionAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        dragPrimApi.CreateRangeAttr(Gf.Vec2f(-1.0, -1.0))
        # Add the collection
        collectionAPI = Usd.CollectionAPI.ApplyCollection(explodePrim, ForceFieldSchema.Tokens.forceFieldBodies)
        collectionAPI.CreateIncludesRel().AddTarget(stage.GetDefaultPrim().GetPath())
        # Boxes
        boxSize = Gf.Vec3f(100.0)
        boxPosition = Gf.Vec3f(0.0)
        m = (int)(math.sqrt(numberOfBoxes))
        for i in range(m):
            for j in range(m):
                boxPath = boxPathName + str(i) + str(j)
                boxPosition[0] = (i + 0.5 - (0.5 * m)) * boxSpacing * boxSize[0]
                boxPosition[1] = 0.5 * boxSize[1]
                boxPosition[2] = (j + 0.5 - (0.5 * m)) * boxSpacing * boxSize[2]
                boxPrim = physicsUtils.add_rigid_box(stage, boxPath, position=boxPosition, size=boxSize)
        # Animate the force fields on and off
        global time
        time = 0.0

        def force_fields_step(deltaTime):
            global time
            time = time + deltaTime
            if time > 4.1:
                suckPrimApi.GetEnabledAttr().Set(False)
                explodePrimApi.GetEnabledAttr().Set(False)
                dragPrimApi.GetEnabledAttr().Set(False)
            elif time > 4.0:
                suckPrimApi.GetEnabledAttr().Set(False)
                explodePrimApi.GetEnabledAttr().Set(True)
                dragPrimApi.GetEnabledAttr().Set(False)
            elif time > 2.0:
                suckPrimApi.GetEnabledAttr().Set(True)
                explodePrimApi.GetEnabledAttr().Set(False)
                dragPrimApi.GetEnabledAttr().Set(True)
            else:
                suckPrimApi.GetEnabledAttr().Set(True)
                explodePrimApi.GetEnabledAttr().Set(False)
                dragPrimApi.GetEnabledAttr().Set(False)

        def timeline_event(event):
            # on play press
            if event.type == int(omni.timeline.TimelineEventType.PLAY):
                global time
                time = 0.0
            # on stop press
            if event.type == int(omni.timeline.TimelineEventType.STOP):
                pass

        physxInterface = omni.physx.get_physx_interface()
        self._subscriptionId = physxInterface.subscribe_physics_step_events(force_fields_step)
        timelineInterface = omni.timeline.get_timeline_interface()
        stream = timelineInterface.get_timeline_event_stream()
        self._timeline_subscription = stream.create_subscription_to_pop(timeline_event)
