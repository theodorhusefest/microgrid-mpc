# Project Thesis Fall 2020

This folder contains the work done for the project thesis fall 2020.

## Installation

Run ´pip install -r requirements.txt´ in either VM or directly on your computer.

## Run

The code used for the optimization is located in src.  
The config-file is used to tune parameters and metadata, and paths to datafiles.  
When starting the program you will get two options. Both has to be confirmed with **y**, anything else is regarded as no.

1. You will get the option to _log_. This will create a new folder called logs, and store all figures as well as important files (config).
2. You can run only openloop. This will use the optimal SOC as the real SOC. If not running openloop, a simulated SOC will be used as measurement.

At the end of the run, plots wil appear with the optimal controls and SOC.

### config.yml

These are the variables that can be configured in the config-file. Going outside range will not necessarily raise errors, only give wierd results.

| name                  | description                                                                                                                                                                             | standard              | range         |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | ------------- |
| simulation_horizion   | Number of hours to run the simulation.                                                                                                                                                  | 24                    | [1,-)         |
| prediction_horizion   | Number of hours to predict for. This number will be multiplied with actions_per_hour.                                                                                                   | 2                     | [1,-)         |
| actions_per_hour      | Control actions per hour. As we get data every 10 minutes, standard value is 6. (60min/h / 10min) = 6.                                                                                  | 6                     | [1-10]        |
| x_initial             | State of charge at T=0                                                                                                                                                                  | 0.9                   | [x_min-x_max] |
| logpath               | Path to logfolder. Will be created if it does not exist. standard path is ignored by gitignore.                                                                                         | "./logs/"             | --            |
| datafile              | Path to where to find real-data. Should be a CSV-file with the columns PV, P1, P2, PV_pred, PL_pred, Spot_pris and Grid_cap. If no predictions are provided, we set pred = real values. | "./data/load_PV3.csv" |               |
| perfect_predictions   | Will set predictions equal to measurements.                                                                                                                                             | True                  | True, False   |
| plot_predictions      | Will plot predictions vs measurements, and save if logging is flagged.                                                                                                                  | False                 | True, False   |
| simulations.pv_power  | Peak power for simulated PV-cells                                                                                                                                                       | 80                    | [0, -)        |
| simulations.pv_noise  | Turns on noise on the simulated PV-signal                                                                                                                                               | False                 | True, False   |
| simulations.pl_power  | Peak power for load                                                                                                                                                                     | 80                    | [0, -)        |
| simulations.pl_noise  | Turns on noise on simulated PL-signal                                                                                                                                                   | False                 | [0, -)        |
| simulations.grid_buy  | Starting price for buying from grid. Will be multiplied with general cost of using grid                                                                                                 | 1.5                   | [0, -)        |
| simulations.grid_sell | Stating price for selling from grid. Will be divided with general cost of using grid                                                                                                    | 1.5                   | [0, -)        |
| system.C_MAX          | Maximum capacity of battery. [kWh]                                                                                                                                                      | 700                   | [0, -)        |
| system.nb_c           | Battery charge coefficient.                                                                                                                                                             | 0.8                   | [0, 1]        |
| system.nb_d           | Battery discharge coefficient                                                                                                                                                           | 0.8                   | [0, 1]        |
| system.x_min          | Hard constraint on minimum SOC                                                                                                                                                          | 0.3                   | [0, 1]        |
| system.x_max          | Hard constraint on maximum SOC                                                                                                                                                          | 0.9                   | [0, 1]        |
| system.Pb_max         | Hard constraint on Maximum power which can charge/discharge battery                                                                                                                     | 1000                  | [0, -)        |
| system.Pg_max         | Hard constraint on maximum power to be drawn/sold to grid                                                                                                                               | 500                   | [0, -)        |
| system.battery_cost   | Cost of using battery                                                                                                                                                                   | 0.1                   | [0, -)        |
| system.grid_cost      | General cost of using the grid                                                                                                                                                          | 100                   | [0, -)        |
| system.verbose        | Flag for printing output from the NLP-solver                                                                                                                                            | False                 | True, False   |

### main.py

Contains the mpc-loop.

### solver.py

Contains the optimization-problem that will be solved every hour.

### simulations

Collects datafile
