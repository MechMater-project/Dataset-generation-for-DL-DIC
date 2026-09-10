from scipy.io import loadmat
import numpy as np
import os

Ninit=0;  # first initial image
Nref = 200;  #  last initial image
NB_DEF_TRAIN = 160;  # first image for evaluation
NB_DEF_MAX_PER_IMG = 200;  # last deformed image for evaluation

for i in range(Ninit,Nref):
    for j in range(NB_DEF_TRAIN, NB_DEF_MAX_PER_IMG):

        u_mat = f"U_{i:03d}_def_{j:03d}.mat"
        v_mat = f"V_{i:03d}_def_{j:03d}.mat"

        try:
            U = loadmat(u_mat)["U"]
            V = loadmat(v_mat)["V"]

            np.save(f"U_{i:03d}_def_{j:03d}.npy", U)
            np.save(f"V_{i:03d}_def_{j:03d}.npy", V)

            # Delete the MATLAB files
            os.remove(u_mat)
            os.remove(v_mat)

            print(f"Converted and deleted i={i}, j={j:03d}")

        except FileNotFoundError:
            print(f"Missing files for i={i}, j={j:03d}")
