# Data-driven MPC for Micro-grid operation

This folder contains the code the master thesis Data-driven optimization of Micro-grid operation.

Authors:

- Nicolai Hoel
- Theodor Husefest

Supervisors:

- Sebastien Gros
- Phillip Maree


All data used in the experiments can be found in src/data.   
Figures used in thesis can be found in figs.

## Installation

Install the packages needed with ´pip install -r requirements.txt´.


## Running the code

The code runs by using the main file, ´python3 main.py´

The configuration of the code happens in config.yml.

- Simulation horizon - number of hours to simulate the MPC for
- Prediction horizon - horizon of the MPC, in hours
- N_scenarios - Number of scenarios to include. Only 1, 3, 7 and 9 are valid
- Perfect predictions - Bool. If False, prediction methods are used.
- Plot - Bool. If True, figures are plotted. 
- Foldername - If provided, a folder with the name will be created with all plots. If black, nothing happens
- Year, month, day - Which day to start the simulation
- Actions per hour - Sampling time of the MPC. Should always be 6
- Logpath - Which folder the logs are saved
- files - Where to find different files.
- Battery - Specifies the parameters of the battery. Read thesis for details.
- System - Specifies the parameters of the MPC. Read thesis for details.


