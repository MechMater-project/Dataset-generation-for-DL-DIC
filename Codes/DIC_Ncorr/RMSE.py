import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

rmse = []
relative_rmse = []
pairs = []
max_amp = []
for i in range(0, 200):          # 1 ... 199
    for j in range(160, 200):    # 160 ... 199

        try:
            # Ncorr result
            U = np.load(f"U_{i:03d}_def_{j:03d}.npy")
            V = np.load(f"V_{i:03d}_def_{j:03d}.npy")

            # Ground truth
            U_gt = np.load(f"../Ref_image_{i:03d}_def_{j:03d}_U.npy")
            V_gt = np.load(f"../Ref_image_{i:03d}_def_{j:03d}_V.npy")

            # RMSE
            amp = np.max(np.sqrt(U_gt**2 + V_gt**2))
            rmseUV = np.sqrt(np.mean((U-U_gt)**2) + np.mean((V-V_gt)**2))
            rrmse = rmseUV / amp
            rmse.append(rmseUV)
            relative_rmse.append(rrmse)
            pairs.append((i, j))
            max_amp.append(amp)

        except FileNotFoundError:
            print(f"Missing pair i={i:03d}, j={j:03d}")

print("\n====================================")
print(f"Mean RMSE  : {np.mean(rmse):.6f}")
print(f"Std  RMSE  : {np.std(rmse):.6f}")
print("Median RMSE :", np.median(rmse))
print("90th percentile  :", np.percentile(rmse,90))
print("95th percentile  :", np.percentile(rmse,95))
print("Max RMSE :", np.max(rmse))

rmse = np.array(rmse)
pairs = np.array(pairs)
relative_rmse = np.array(relative_rmse)
# Sort from best (lowest RMSE) to worst (highest RMSE)
idx = np.argsort(rmse)
best_idx = idx[:50]
worst_idx = idx[-50:][::-1]   # reverse so worst is first

# ---------- Save 50 best ----------
with open("50best.txt", "w") as f:
    f.write("Rank\tRef\tDef\tRMSE\n")
    for rank, k in enumerate(best_idx, 1):
        i, j = pairs[k]
        f.write(f"{rank:2d}\t{i:3d}\t{j:03d}\t{rmse[k]:.6f}\n")

# ---------- Save 50 worst ----------
with open("50worst.txt", "w") as f:
    f.write("Rank\tRef\tDef\tRMSE\n")
    for rank, k in enumerate(worst_idx, 1):
        i, j = pairs[k]
        f.write(f"{rank:2d}\t{i:3d}\t{j:03d}\t{rmse[k]:.6f}\n")

print("Saved 50best.txt and 50worst.txt")

idx = np.argsort(relative_rmse)

best = idx[:50]
worst = idx[-50:][::-1]
with open("50best_relative.txt","w") as f:
    f.write("Rank\tRef\tDef\tRMSE\tAmplitude\tRelativeRMSE(%)\n")
    for rank,k in enumerate(best,1):
        i,j = pairs[k]
        f.write(f"{rank}\t{i}\t{j:03d}\t"
                f"{rmse[k]:.5f}\t"
                f"{max_amp[k]:.5f}\t"
                f"{100*relative_rmse[k]:.2f}\n")
# ---------- Save 50 worst ----------
with open("50worst_relative.txt", "w") as f:
    f.write("Rank\tRef\tDef\tRMSE\tAmplitude\tRelativeRMSE(%)\n")
    for rank, k in enumerate(worst_idx, 1):
        i, j = pairs[k]
        f.write(f"{rank}\t{i}\t{j:03d}\t"
                f"{rmse[k]:.5f}\t"
                f"{max_amp[k]:.5f}\t"
                f"{100*relative_rmse[k]:.2f}\n")
