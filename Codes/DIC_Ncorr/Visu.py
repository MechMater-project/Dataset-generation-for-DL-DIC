import sys
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Read command line arguments
# =====================================================
if len(sys.argv) != 3:
    print("Usage: python3 Visu.py <ref_index> <def_index>")
    sys.exit(1)

i = int(sys.argv[1])
j = int(sys.argv[2])

# =====================================================
# Load displacement fields
# =====================================================
U_true = np.load(f"../Ref_image_{i}_def_{j:03d}_U.npy")
V_true = np.load(f"../Ref_image_{i}_def_{j:03d}_V.npy")

U_ncorr = np.load(f"U_{i}_def_{j:03d}.npy")
V_ncorr = np.load(f"V_{i}_def_{j:03d}.npy")

# =====================================================
# Plot
# =====================================================
fig, ax = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

vmin = -1
vmax = 1

# U
im = ax[0,0].imshow(U_true, cmap="jet", origin="upper",
                    vmin=vmin, vmax=vmax)
ax[0,0].set_title("Ground truth U")

im = ax[0,1].imshow(U_ncorr, cmap="jet", origin="upper",
                    vmin=vmin, vmax=vmax)
ax[0,1].set_title("Ncorr U")

fig.colorbar(im, ax=ax[0,:], location="right", shrink=0.9)

# V
im = ax[1,0].imshow(V_true, cmap="jet", origin="upper",
                    vmin=vmin, vmax=vmax)
ax[1,0].set_title("Ground truth V")

im = ax[1,1].imshow(V_ncorr, cmap="jet", origin="upper",
                    vmin=vmin, vmax=vmax)
ax[1,1].set_title("Ncorr V")

fig.colorbar(im, ax=ax[1,:], location="right", shrink=0.9)

# Save
outfile = f"Visu_{i}_{j:03d}.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
print(f"Saved {outfile}")

plt.show()
