ARPLS PREPROCESSING
===================
See ../README.md for setup, integration with the Lorentzian fit, and run instructions.
This subfolder contains the ARPLS baseline correction workflow.
Defaults: island collection cI, island excitation eI, ExPol 0, ROI1 and ROI2,
input from ../data/raw/ or CU2O_RAW_DATA_DIR, no separate background subtraction.
Spectral range 565–590 nm, integration window 573.5–581.5 nm.
Relative-efficiency correction: ROI2 x 0.728; ROI1 unchanged.
Run from the repository root: python arpls_preprocessing/main_arpls.py
Outputs go to ../results/yellow/results_arpls_cI_eI_no_background/ by default.
To prepare cG1 data, change COLLECTION = "cG1" in config_arpls.py and rerun.
