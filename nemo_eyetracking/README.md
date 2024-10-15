# nemo_eyetracking

Citation: TBA

Main contributors to code: Alex J. Hoogerbrugge, Christoph Strauch, Gregor Baer

Utrecht University, 2023

For help, please contact a j hoogerbrugge@uu nl [replace spaces with dots]

## Structure
This repository is divided into two parts, since they were developed partially independently:
* The main directory, which contains pre-processing of data and parts of the analyses (CS & AJH)
* The `saleval` directory, which contains saliency map and NSS computations (GB, CS & AJH)

## Instructions
All outcome measures are already available within this repository, although some folders will need to be unzipped first. Should you want to re-run the code, follow the steps below.
* Install `nemo_eyetracking/environment.yml`
* For preprocessing, run `nemo_eyetracking/src/main.py`. Follow the instructions within that file.
* Then, run the run_analysis scripts within `nemo_eyetracking/saleval`.
* For plotting, run `nemo_eyetracking/src/plots.py`.  

## Notes
* Note that the preprocessed data is almost 4GB unzipped. The full repository unzipped is around 7GB.
* The raw data which we receive directly from the museum has some caveats, but can be made available upon request. See contact information at the top.

This repository was developed on our respective operating systems and may therefore not produce exactly the same results for users on a different OS - especially when it comes to stochastic functions such as confidence intervals in seaborn.
Installation of conda environments may require some adjustments if on a different OS.