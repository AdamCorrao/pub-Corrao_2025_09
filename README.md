# pub-Corrao_2025_09
Example code and datasets for "A modular framework for collaborative human-AI, multi-modal and multi-beamline synchrotron experiments."
-----
# Computing requirements and pixi environment
- Development was done using a Linux (RHEL 8.10) VM with 8 cores and 32 GB RAM.
    - Runtime for wafer simulation is approx. 1 min.
- Experiment campaigns were run via slurm scripts on a computing cluster.
    - Runtime for a campaign is approx. 2 hours.
    - Gaussian process training is the bottleneck for this implementation.
## Getting started with pixi
- If pixi is not yet installed, follow instructions here: https://pixi.sh/latest/installation/
- Clone this repo
- Navigate to the directory where this repo is cloned
- In a terminal run `pixi install` to create the python environment defined by the pixi.toml file which will build the pixi.lock file
- Set the pixi env as the python interpretter (Instructions for IDE / VS code integration: https://pixi.sh/dev/integration/editor/vscode/)
-----
# Directory structure
Three main directories are Data, Notebooks, and Scripts.
- Data: files used for assembling a phase diagram dataset, ternary phase diagram datasets, and simulated combinatorial libraries.
- Notebooks: example Jupyter (.ipynb) notebooks for assembling a phase diagram dataset and simulating a combinatorial library.
- Scripts: (3) scripts for running different experiment campaigns, and (1) slurm script (.sbatch) for running experiment campaign(s) using HPC resources.
-----
## Data
### DRNets
- Files used to assemble a ternary phase diagram containing element weights, phase weights, and X-ray diffraction patterns.
- 3 chemical systems available: Al-Li-Fe oxide, Bi-Cu-V oxide, and Li-Sr-Al oxide.
- All files originate from https://github.com/gomes-lab/DRNets-Nature-Machine-Intelligence
### phasediagram_datasets
- Phase diagram Xarray datasets (netcdf4 files) for the 3 chemical systems available. Compressed (.zip) files provided in place of netcdf4 (.nc) files when filesize exceeds 100 mb.
### simulatedwafer_datasets
- Simulated Al-Li-Fe oxide combinatorial library (Xarray dataset, netcdf4 file) with element weights, crystallographic phase weights, and X-ray diffaction patterns. Compressed (.zip) files provided in place of netcdf4 (.nc) files when filesize exceeds 100 mb.
### Miscellaneous
- gp_prediction_onalldata.csv: Al-Li-Fe oxide results from predicting class labels with a Gaussian process classifier (GPC) trained on all x,y coordinates and ground truth labels (asserted during simulation). Here, predicted_labels are used for calculating accuracy during experiment campaigns since these represent the best model possible with the current GPC kernel and hyperparameters.
-----
## Notebooks
### PhaseDiagramAssembler
- Assemble a phase diagram dataset (Xarray dataset, netcdf4 file) from DRNets files.
### WaferSimulator_3libraries
- Simulate combinatorial library datasets to be used in experiment campaigns.
-----
## Scripts
### AlLiFe_geoseries.py
- Run an experiment campaign where measurements are sampled from a grid of points with evolving resolution.
### AlLiFe_random.py
- Run an experiment campaign where measurements are randomly sampled from unmeasured points.
### AlLiFe_AIdriven.py
- Run an experiment campaign where measurements are sampled from a cumulative distribution function calculated from the GPC uncertainty for unmeasured points combined with an on/off edge characteristic (i.e., 0 = neighboring points all have the same label, 1 = any neighboring point has a different label).
### Slurm scripts
- Run a slurm script (multiapproach_singlerun.sbatch) that runs a python script (full_campaign_options.py)
- Flags allow the user to select a dataset, total number of measurements, data from run to save, experiment approach, and a campaign name.