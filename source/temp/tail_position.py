import argparse
import sys
import time
import numpy as np

from omni.isaac.lab.app import AppLauncher

### ARGPARSE ###
# add argparse arguments
parser = argparse.ArgumentParser(description="Second urdf implementation script.")
# Default to headless mode
if True:
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

### IMPORTS ###
import torch

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.robots import RobotView

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, ArticulationData, RigidObjectCfg, RigidObject
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.actuators import ImplicitActuatorCfg, ActuatorBaseCfg
from omni.isaac.lab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
import omni.physx as physx

### Data recorder ###
from utils.data_recorder import DataRecorder
DATA_RECORDER = DataRecorder(record=True)

### Ang Acceleration Profile
from utils.control_methods import SimpleAngAccelProfile

### Articulation CFG ###
BOX_CFG = ArticulationCfg(
    prim_path="/World/Origin/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/temp/box_w_tail.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            enable_gyroscopic_forces=True
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True,
            enabled_self_collisions=True,
            fix_root_link=True
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        # joint_pos={"box_to_rod": -1.5708},
        joint_pos={"box_to_rod": 0.0},
        # joint_vel={"box_to_rod": 2.0},
    ),
    actuators={
        "motor": ImplicitActuatorCfg(
            joint_names_expr=["box_to_rod"],
            stiffness=0.0, # Zero for velocity and effort control
            damping=0.0,
            friction=0.2,
            effort_limit=1000.0,
            velocity_limit=500.0,
            ),
    },
)

def design_scene():

    # Ground plane
    ground_cfg  = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # Light
    light_cfg = sim_utils.DomeLightCfg(intensity=3000)
    light_cfg.func("/World/Light", light_cfg)

    ### Box with tail ###
    origin = [0.0, 0.0, 0.0]
    p = prim_utils.create_prim("/World/Origin", "Xform", translation=origin)
    # box_prim = sim_utils.spawn_from_usd(prim_path="/World/Origin/Robot", cfg=BOX_CFG, translation=origin)

    # BOX_CFG.prim_path = "/World/Origin/Robot"
    box = Articulation(cfg=BOX_CFG)

    return box, origin

def get_tail_orientation(time_seconds: float, robot: Articulation, articulation_view: ArticulationView, artdata: ArticulationData):
    """
    Computes and returns key information regarding the orientation and rotation of the tail (or rod) in the world frame.
    Note that this function is custom for the specific robot model used in this script.

    Args:
    - robot (Articulation): The robot object containing the articulated model.
    - articulation_view (ArticulationView): Provides access to joint states, body indices, and kinematic details.
    - artdata (ArticulationData): Contains body position and velocity data in the world frame.

    Returns:
    - dict: A dictionary containing:
      - "tail_orientation": A vector in world coordinates representing the tail's orientation from the rod joint to the end-effector.
      - "rotation_axis": A normalized vector representing the current axis of rotation.
      - "rotation_magnitude": The magnitude of angular velocity around the axis of rotation.
    """
    ### Joint Configuration
    joint_cfg = {
        "position[radians]": articulation_view.get_joint_positions(),
        "velocity[radians/s]": articulation_view.get_joint_velocities(),
    }

    ### Generalized Tail Velocity in world frame
    joint_name = "box_to_rod"
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
    rod_joint_pos_vec = get_body_position_vector("rod")
    endeffector_pos_vec = get_body_position_vector("endeffector")
    tail_orientation_in_world_coordinates = endeffector_pos_vec - rod_joint_pos_vec
    tail_orientation_in_world_coordinates = torch.reshape(tail_orientation_in_world_coordinates, (3,1))

    DATA_RECORDER.record(time_seconds=time_seconds, values={
        "tail_orientation_radians": joint_cfg["position[radians]"],
        # "tail_orientation_degrees": np.rad2deg(joint_cfg["position[radians]"].cpu()),
        "tail_velocity": joint_cfg["velocity[radians/s]"],
        })

    assert torch.isnan(tail_orientation_in_world_coordinates).any() == False
    assert torch.isnan(axis_of_rotation).any() == False
    assert torch.isnan(rotation_magnitude).any() == False

    return {"tail_orientation": tail_orientation_in_world_coordinates, "rotation_axis": axis_of_rotation, "rotation_magnitude": rotation_magnitude}

def apply_forces_old(time_seconds: float, robot: Articulation, articulation_view: ArticulationView, artdata: ArticulationData, tail_motion: dict, apply: bool = True):
    """
    This function takes the wind and drag and calculates the discretized combined forces acting along the tail.
    Each of these forces induce a torque. 
    This function calculates a substitution-force, which when applied through the CoM of the tail, results in the 
    same total torque.
    This function does not only apply forces resulting in roll-torque, but in arbitrary directions.

    For reader: I use different words for the two different components contributing to total force:
    - 'drag': This is the drag induced by the tail's motion/rotation through the air, neglecting any incoming wind
    - 'wind': This is used to describe an additional wind blowing.
    The superposition of these two vectors provide the total experienced wind at a given location.
    
    """
    ### Forces ###
    WIND = torch.tensor([[-30.0], [0.0], [0.0]], device='cuda:0') # m/s
    density_air = 1.293 # kg/m^3
    C_d = 1.1 # [has no unit]

    ### x shall be variable along tail length
    tail_length = 0.7 # TODO: Tail length is hard-coded
    tail_length = torch.norm(input=tail_motion["tail_orientation"], p=2)
    x = torch.linspace(0.0, tail_length, steps=200)

    ### Drag: due to the tail's motion, ignoring wind
    w_vec = tail_motion["rotation_axis"] * tail_motion["rotation_magnitude"] # Not per se necessary, could just use rotation_axis
    # e_drag = torch.cross(w_vec, torch.reshape(tail_motion["tail_orientation"], (3,1)))
    e_drag = torch.cross(w_vec, tail_motion["tail_orientation"])
    torch.reshape(tail_motion["tail_orientation"], (1, 3))
    e_drag = e_drag/torch.norm(input=e_drag, p=2) # Unit vector pointing in the direction of drag force

    ### Combine Wind and Drag
    if torch.norm(input=WIND, p=2) == 0:
        e_wind = WIND
    else:
        e_wind = WIND/torch.norm(input=WIND, p=2) # Unit vector pointing in direction of additional wind
    e_F = -1* (e_wind + e_drag)/(torch.norm(input=e_wind + e_drag, p=2)) # Unit vector pointing in direction of combined wind and drag force
    
    ### Surface A perpendicular to perceived wind
    # Project tail length onto plane perpendicular to e_F
    # Intuition: Project L onto e_F (L*e_F). Multiply this scalar by e_F to get the projection vector.
    #            Subtract this projection vector from L (= subtract part of L which is parallel to e_F)
    L_projected = tail_motion["tail_orientation"] - torch.dot(tail_motion["tail_orientation"][:,0], e_F[:,0]) * e_F
    # Surface Area A
    diameter = 0.1 # TODO: Diameter
    A_tilde = torch.norm(input=L_projected, p=2) * diameter # Diameter is hard-coded!!!

    ### Numerical Integration
    # Function to integrate: (||v_wind + e_Drag * w * x||)^2 * x
    integrand = torch.zeros_like(x)
    for i in range(len(x)):
        # Combined wind and drag:
        v = WIND + e_drag * tail_motion["rotation_magnitude"] * x[i]
        integrand[i] = torch.norm(input=v, p=2)**2 * x[i]
    # Numerically integrate
    integrated_force = torch.trapz(integrand, x=x)

    ### Equivalent Force through CoM
    F = (density_air * C_d) / tail_length * A_tilde * e_F * integrated_force
    F = torch.reshape(F, (1, 1, 3))

    ### Apply force
    DATA_RECORDER.record(time_seconds=time_seconds, values={
        "e_F_x": e_F[0,0],
        "e_F_y": e_F[1,0],
        "e_F_z": e_F[2,0],
        "F_wind_x": F[0,0,0],
        "F_wind_y": F[0,0,1],
        "F_wind_z": F[0,0,2],
        "A_tilde": A_tilde,
    })
    if apply:
        robot.set_external_force_and_torque(forces=F, torques=torch.zeros_like(F), body_ids=[2], env_ids=[0])
        robot.write_data_to_sim()
    else:
        print(f"Force not applied {F[0,0,:]}")

def apply_forces(time_seconds: float, robot: Articulation, articulation_view: ArticulationView, artdata: ArticulationData, tail_motion: dict, apply: bool = True):
    ### Parameters ####
    WIND = torch.tensor([[-30.0], [0.0], [0.0]], device='cuda:0').reshape(3,) # m/s
    density_air = 1.293 # kg/m^3
    C_d = 1.1 # [has no unit]
    DIAMETER = 0.1 # m, TODO
    LENGTH = tail_motion["tail_orientation"].norm(p=2) # m,
    DISCRETIZATION = 200 # Number of points to discretize tail length
    vec_joint_endeffector = tail_motion["tail_orientation"].reshape(3,)
    dir_joint_endeffector = (vec_joint_endeffector/torch.norm(input=vec_joint_endeffector, p=2)).reshape(3,)
    vec_omega = (tail_motion["rotation_axis"] * tail_motion["rotation_magnitude"]).reshape(3,)

    ### Functions ###
    def vec_x_at_s(s: float):
        "Returns vector x(s) evaluated at a position along the tail."
        assert 0.0 <= s <= vec_joint_endeffector.norm(p=2)
        return s * dir_joint_endeffector
    
    def v_tail_at_s(s: float):
        "Returns the velocity of the tail at position s."
        return torch.cross(vec_omega, vec_x_at_s(s))
    
    def L_projected_at_s(plane_perpendicular_to):
        "Returns projected tail-length onto plane perpendicular to argument 'plane_perpendicular_to'."
        L_projected = vec_joint_endeffector - torch.dot(vec_joint_endeffector, plane_perpendicular_to) * plane_perpendicular_to
        return torch.norm(input=L_projected, p=2)
    
    def F_drag_at_s(s: float):
        "Returns the drag force at position s. Returned is a vector."
        v = WIND + v_tail_at_s(s)
        v_dir = v/torch.norm(input=v, p=2)
        v_squared = torch.norm(input=v, p=2)**2
        # Surface Area A is projected and taken proportional quantity according to DISCRETIZATION
        A = DIAMETER * L_projected_at_s(plane_perpendicular_to=v_dir) * DISCRETIZATION
        F_drag_at_s = 0.5 * density_air * C_d * A * v_squared * v_dir
        return F_drag_at_s
    
    def T_total():
        "Returns the total torque acting on the tail. Returned is a vector."
        T_total = torch.zeros(3).to('cuda')
        for s in torch.linspace(0.0, vec_joint_endeffector.norm(p=2), steps=DISCRETIZATION):
            assert 0.0 <= s <= vec_joint_endeffector.norm(p=2)
            T_total += torch.cross(vec_x_at_s(s), F_drag_at_s(s))
        return T_total
    
    def F_substitution():
        "Returns the equivalent force acting through the CoM of the tail. Returned is a vector."
        vec_x_at_Lhalf = vec_x_at_s(vec_joint_endeffector.norm(p=2)/2)
        norm_vec_x = torch.norm(input=vec_x_at_Lhalf, p=2)
        return (torch.cross(vec_x_at_Lhalf, T_total())/norm_vec_x**2)
    
    ### Apply forces ###
    F_sub = F_substitution()
    F_sub_unit_vector = F_sub/torch.norm(input=F_sub, p=2)
    F_sub = torch.reshape(F_sub, (1, 1, 3))

    if apply:
        # If F_sub is 0.0 for every entry, skip (otherwise external wrench is disabled [no clue what this means...])
        if torch.norm(input=F_sub, p=2) > 0.0:
            robot.set_external_force_and_torque(forces=F_sub, torques=torch.zeros_like(F_sub), body_ids=[2], env_ids=[0])
            robot.write_data_to_sim()
        else:
            print(f"[WARNING: {time_seconds}] F_sub is zero: {F_sub[0,0,:]}")
        # robot.set_external_force_and_torque(forces=F_sub, torques=torch.zeros_like(F_sub), body_ids=[2], env_ids=[0])
        # robot.write_data_to_sim()
    else:
        print(f"Force not applied {F_sub[0,0,:]}")
    
    DATA_RECORDER.record(time_seconds=time_seconds, values={
        "e_F_x": F_sub_unit_vector[0],
        "e_F_y": F_sub_unit_vector[1],
        "e_F_z": F_sub_unit_vector[2],
        "F_sub_x": F_sub[0,0,0],
        "F_sub_y": F_sub[0,0,1],
        "F_sub_z": F_sub[0,0,2],
        "F_applied?": apply,
        # "A_tilde": A_tilde,
    })

def record_robot_forces(time_seconds: float, robot: Articulation, articulation_view: ArticulationView, artdata: ArticulationData):
    # Access joint forces & torques for 0th environment/entity
    forces_and_torques = articulation_view.get_measured_joint_forces(indices=[0], joint_indices=[1])[0]
    force_torque_dict = {
        key: forces_and_torques[0][i] for i, key in enumerate(["fx", "fy", "fz", "tx", "ty", "tz"])
    }
    
    values = {
        "friction": artdata.joint_friction[0].item(),
        "damping": artdata.joint_damping[0].item(),
        "vel_setpoint": robot._joint_vel_target_sim[0].item(),
        "vel_applied": articulation_view.get_joint_velocities()[0].item(),
        "effort_setpoint": robot._joint_effort_target_sim[0].item(),
        "effort_applied": artdata.applied_torque[0].item(),
        "effort_measured": articulation_view.get_measured_joint_efforts()[0].item(),
    }

    DATA_RECORDER.record(time_seconds=time_seconds, values={**values, **force_torque_dict
    })

def run_simulator(sim: sim_utils.SimulationContext, box: Articulation, origin: torch.Tensor):
    "Runs the simulation."

    robot = box

    articulation_view = ArticulationView(prim_paths_expr="/World/Origin/Robot")
    articulation_view.initialize()
    artdata: ArticulationData = robot.data

    # Sim step
    sim_dt = sim.get_physics_dt()
    count = 0
    reset_count = 250
    
    ang_vel_profile = SimpleAngAccelProfile(sim_dt=sim_dt, t1_t2=1.0)

    # Loop
    while simulation_app.is_running():
        # Rest
        if count % reset_count == 0:
            if count == reset_count and DATA_RECORDER.record_bool == True:
                DATA_RECORDER.save("source/temp/trial_data.csv")
                return
            
            count = 0

            print(f"        ----------------- Resetting -----------------")
            root_state = artdata.default_root_state.clone()
            root_state[:, :3] += origin
            robot.write_root_state_to_sim(root_state)

            joint_pos_default = artdata.default_joint_pos.clone()
            
            joint_vel_default = artdata.default_joint_vel.clone()

            robot.write_joint_state_to_sim(position=joint_pos_default, velocity=joint_vel_default)
            robot.reset()
            print(f"        -------------- Initialization done -----------")
        
        
        shape = robot.actuators["motor"].applied_effort

        # Effort on tail
        articulation_view.switch_control_mode(mode='effort')
        robot.write_joint_damping_to_sim(torch.full_like(shape, 0.0))
        effort = 25.0
        robot.set_joint_effort_target(torch.full_like(shape, effort))

        # ang_vel = ang_vel_profile.get_ang_vel(count=count)
        # if ang_vel is not None:
        #     ### Following Ang-Vel profile ###
        #     # For velocity control; stiffness must be 0.0, damping must be non-zero
        #     # (source: omni.isaac.core Articulations Documentation)
        #     articulation_view.switch_control_mode(mode="velocity")
        #     robot.write_joint_damping_to_sim(torch.full_like(robot.actuators["motor"].damping, 10.0))
        #     # Set joint velocity setpoint
        #     joint_vel_setpoint = torch.full_like(robot.actuators["motor"].applied_effort, ang_vel)
        #     robot.set_joint_velocity_target(joint_vel_setpoint)
        # else:
        #     ### Free swing of the tail ###
        #     # For effort control; stiffness and damping must be 0.0
        #     robot.set_joint_velocity_target(torch.zeros_like(robot.actuators["motor"].applied_effort))
        #     articulation_view.switch_control_mode(mode="effort")
        #     robot.write_joint_damping_to_sim(torch.zeros_like(robot.actuators["motor"].damping))
        #     # Set zero effort (should let tail swing freely???)
        #     robot.set_joint_effort_target(torch.zeros_like(robot.actuators["motor"].applied_effort))

    
        robot.write_data_to_sim()

        sim.step()
        count += 1

        robot.update(sim_dt)

        if count % 20 == 0:
            pass

        # Get tail motion
        tail_motion = get_tail_orientation(sim_dt*count, robot, articulation_view, artdata)
        # Apply wind and drag force
        apply_forces(sim_dt*count, robot, articulation_view, artdata, tail_motion, apply=True)

        record_robot_forces(sim_dt*count, robot, articulation_view, artdata)



# Main
def main():
    sim_cfg = sim_utils.SimulationCfg()
    sim_cfg.dt = 1.0 / 240.0
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([-0.5, -6.0, 2.3], [2, 4, 1.5])

    # Design the scene
    box, origin = design_scene()
    origin = torch.tensor(origin, device=sim.device)

    # Run the simulator
    sim.reset()
    print(f"[Info]: Setup complete. Starting first simulation...")
    run_simulator(sim, box, origin)

if __name__ == "__main__":
    main()

    DATA_RECORDER.plot(
        dictionary={
            "Parameters": ["friction", "damping"],
            "Joint Velocity": ["vel_setpoint", "vel_applied"],
            "Joint Torque": ["effort_setpoint", "effort_measured", "effort_applied"],
            "Torques on Body": ["tx", "ty", "tz"],
            "Forces on Body": ["fx", "fy", "fz"],
            "Tail Orientation [rad]": ["tail_orientation_radians"],
            "Tail Velocity": ["tail_velocity"],
            "Substitution Force on Tail": ["F_sub_x", "F_sub_y", "F_sub_z"],
            "F_sub applied?": ["F_applied?"],
        },
        save_path="source/temp/tail_position.png"
    )

    # Close sim app
    simulation_app.close()

    