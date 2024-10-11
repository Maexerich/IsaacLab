This repository contains the code developed within context of my semester project at ETH.
The **final report** of my project can be found in this repository.

# Purpose
Framework for simulation of tails for ANYmal, which considers the tail's
- inertia and
- aerodynamic drag
Inertia is natively considered by the Isaac Sim/Isaac Lab simulation environment.
This repository focuses on implementation of aerodynamic forces.

# Prerequisites
### Difference Isaac Lab vs. Isaac Sim 
When navigating examples or writing new code, be sure to understand the difference between Isaac Lab and Isaac Sim classes/functions.
I would recommend to stick to Isaac Lab as close as you can, as documentation is generally better. However, some functions will also be needed from Isaac Sim, like the class 'articulation_view'.
A typical workflow includes cloning of the Isaac Lab repository and then creating your functions/source code within this repository. It will become apparent; those functions and classes for which the IDE knows the definition (e.g. right-click 'To Definition' works), are Isaac Lab classes.
### How to run anything using Isaac Lab & Isaac Sim (containerized on student computer, normal default omniverse installation)
#### ETH Shared Computer Room (e.g. containerized Isaac Sim)
To run code, you must use the command line. Only then will Isaac Sim behave.
I find this setup terrible, as I found myself using the debug functionality of the VS Code IDE continuously when working (I worked on my private computer at home). The debug feature was the easiest way to gain fast insight, into what different classes have to offer and trying things.
#### Normal installation on Windows/Linux
A normal installation will provide the omniverse launcher, through which Isaac Sim can be started.
When using this way of interfacing with the simulation, you can use (when doing an Isaac Lab project) the 'Play' and 'Debug' buttons from within VS Code.

#### VS Code Extension: Isaac Sim
I suggest installing the VS Code extension Isaac Sim. It offers some code snippets and class attributes from Isaac Sim classes.
(The 'Play' buttons within the extension will not work for the containerized installation, I could not get them to work for my windows installation either)

### Extensions
To enable extensions, either use [python interface](https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html?highlight=enable_extension#omni.isaac.core.utils.extensions.enable_extension) or you can launch Isaac Sim, go to 'Window' > 'Extensions' and then search for the respective extension. Be sure to click on the extension, to view the documentation and then you will have the option to click on 'Autoload' the extension. I am not sure if this is absolutely necessary, but mentioned here just in case.


# How to simulate
Within this project, two different methods of simulating aerodynamic drag are implemented and tested.
Read the report first, before attempting to understand the code.

All simulations were done with the same python file. It is required to have understanding of the file, because a few toggles need to be set for the desired outcome.
Everything is explained in the 'main()' function in the form of comments.

The python file used for all simulations is **source/simulation_script.py**.
Part of my project are also
- **source/results** Plots and csv data from simulations
- **source/data_visualization.ipynb** Jupyter notebook used to debug, visualize and create final plots. ATTENTION: used a python 3.12 venv, with boilerplate installations (numpy, pandas, matplotlib, ipykernel)


# How to make your own simulation
For anyone wanting to do their own implementations, I suggest taking a look first at the [tutorials in the Isaac Lab documentations](https://isaac-sim.github.io/IsaacLab/source/tutorials/index.html), then potentially stepping through my code slowly. My code relies on multiple classes and functions developed by myself (with some occasional ChatGPT help). If the documentation for a respective function is missing, I'm sorry, but just give it to ChatGPT first and see if it figures it out - from the context however I believe my code should be more or less clear.

The workflow that worked for me in the end;
- Use online YouTube videos and documentation to learn the Isaac Sim UI to create my own simulation setup, including configuring all joints
- Use Isaac Lab to load a saved stage (.usd-files), then instantiate your robot (e.g. create a class instance of 'Articulation' and tell it at which path '/World/Robot' the robot can be found within the stage) and then voila, simulation loop can start.



# Acknowledgements
Aside from my supervisors, Robert Baines, Yutnao Ma and Ardian Yusufi, always providing valuable insights, a big thanks goes to Mayank Mittal who has contributed significantly on the implementation side. Without Mayank navigation of the complex Isaac simulation suite would have been close to impossible.