# Towards Bio-Inspired Robotic Trajectory Planning via Self-Supervised RNN

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17127997.svg)](https://doi.org/10.5281/zenodo.17127997)

This is the code for kinematics experiment from the paper 
[_Towards Bio-Inspired Robotic Trajectory Planning via Self-Supervised RNN_](https://arxiv.org/abs/2507.02171)
presented at the International Conference on Artificial Neural Networks 2025.

## Structure
The contents of this repository are organized as follows:
```
.
├── kinematics
│   ├── kinematics_data.ipynb                   // Data post-processing
│   ├── kinematics_models.ipynb                 // Construction and supervised training 
│   │                                           // of the forward, inverse, and trajectory model
│   ├── result_processing.ipynb                 // Processing of the TM training data and generated trajectories
│   └── traj_training.py                        // Trajectory model training within the self-supervised scheme
├── LICENSE
└── README.md
```

## Dependencies
The experiments is implemented mainly as a series of Jupyter Notebooks.
The forward, inverse, and supervised trajectory neural models are implemented in Keras 3 with the PyTorch back-end.
Due to the compatibility and performance issues, the trajectory model in the self-supervised scheme 
in `traj_training.py` was implemented in pure PyTorch as well.

All the dependencies can be installed by running:
```bash
pip install -r requirements.txt
```

## Citation
If you find this work helpful in your research, please consider citing:

[_TBA_]
