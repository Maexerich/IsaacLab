# Import necessary modules
import argparse
import time
from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab.sim import SimulationCfg, SimulationContext
from omni.kit.viewport.utility import capture_viewport_to_image, capture_viewport_to_video

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Save an image of the environment
    capture_viewport_to_image(filename="simulation_image.png", resolution=(1920, 1080))
    print("[INFO]: Image captured.")

    # Save a short video (example: 5 seconds video at 30 FPS)
    capture_viewport_to_video(filename="simulation_video.mp4", duration=5.0, fps=30.0, resolution=(1920, 1080))
    print("[INFO]: Video captured.")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
