import numpy as np
import cv2
import glob
import os
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------
# Parameters
# ---------------------------------------
cell_size2 = [4,8,16,32]      # candidate control-point spacings

# Tuning rationale (verified empirically at 128x128 across cell sizes
# 4/8/16/32, same approach used for Hermite2.py): the peak of the combined
# displacement magnitude sqrt(U^2+V^2) is NOT bounded by max_amplitude,
# because U and V are independent - it can reach up to ~sqrt(2)*max_amplitude
# when both happen to peak near the same pixel. Empirically the *mean* peak
# across many random deformations is ~1.32x max_amplitude. Setting
# max_amplitude = 0.58 * TARGET_MAX_DISP makes the mean (uncapped) peak land
# at ~78% of TARGET_MAX_DISP, so TARGET_MAX_DISP is rarely if ever touched
# (0/300 trials in testing) - the safety cap below is then a genuine safety
# net rather than a near-constant ceiling. If you change TARGET_MAX_DISP,
# just keep max_amplitude = 0.58*TARGET_MAX_DISP.
TARGET_MAX_DISP = 1.0
max_amplitude = 0.58 * TARGET_MAX_DISP   # per-node uniform range, pixels
MAX_TOTAL_DISP = TARGET_MAX_DISP          # hard safety cap on interpolated
                                           # |disp|; set to None to disable
n_deformations = 200
interpolation = cv2.INTER_CUBIC

# Create output folder
os.makedirs("GeneratedBoukhtache", exist_ok=True)

# ==========================================================
# Edge-deformation fix: periodized padding
# ==========================================================
# Same rationale/approach as Hermite2.py: the control-point grid pins U,V
# to zero on its outermost row/column of nodes. If that boundary sits
# exactly on the real image edges, border pixels are never deformed - a
# systematic bias. Fix: pad the reference image by periodizing it
# (BORDER_WRAP tiles the image content around itself), build the grid and
# warp on this larger canvas, then crop back to the original size. The
# zero-displacement boundary now lands inside the periodized padding,
# strictly outside the real image, so every real pixel - including the
# border - sits in the grid's interior and gets a genuine, non-zero,
# smoothly-varying displacement.
#
# pad must be at least as large as the biggest cell size so that even the
# largest-spacing grid's first interior node line falls outside the real
# image; +1 gives a one-pixel margin.
_cell_sizes = cell_size2 if isinstance(cell_size2, (list, tuple)) else [cell_size2]
pad = max(_cell_sizes) + 1


def build_node_grid(n_pixels, h, offset):
    """
    Build node positions covering [0, n_pixels-1] with base spacing h,
    starting from the given phase offset (in [0, h)) instead of always 0.

    This means the grid is not pinned to the same pixel indices on every
    sample, which removes the fixed-grid-alignment bias. The first and
    last elements may end up shorter than h (irregular boundary elements)
    so that the grid still exactly spans the full extended image.
    """
    nodes = np.arange(offset, n_pixels, h)

    if nodes[0] != 0:
        nodes = np.insert(nodes, 0, 0)

    if nodes[-1] != n_pixels - 1:
        nodes = np.append(nodes, n_pixels - 1)

    return nodes


# ---------------------------------------
# Loop over all reference images
# ---------------------------------------
ref_files = sorted(glob.glob("../Ref_MO_128x128/Ref_image_*.bmp"))

for ref_file in ref_files:

    print(f"Processing {ref_file}")

    ref_img = cv2.imread(ref_file, cv2.IMREAD_GRAYSCALE)
    rows, cols = ref_img.shape

    # Periodize the reference image outward by `pad` pixels on every side.
    ref_img_ext = cv2.copyMakeBorder(
        ref_img, pad, pad, pad, pad, borderType=cv2.BORDER_WRAP
    )
    rows_ext, cols_ext = ref_img_ext.shape

    basename = os.path.splitext(os.path.basename(ref_file))[0]

    # Pixel grid covering the extended canvas (built once per reference
    # image, reused every sample - only the node grid/interpolator change).
    Y_ext, X_ext = np.meshgrid(
        np.arange(rows_ext),
        np.arange(cols_ext),
        indexing='ij'
    )
    pts_ext = np.stack([Y_ext.ravel(), X_ext.ravel()], axis=-1)

    # Real-image pixel grid, used for the final warp.
    Y, X = np.meshgrid(
        np.arange(rows),
        np.arange(cols),
        indexing='ij'
    )

    for k in range(n_deformations):

        # Randomly choose the control-point spacing for this deformation
        cell_size = np.random.choice(cell_size2)

        # --------------------------------------------------
        # Randomized-phase control-point grid on the EXTENDED image
        # --------------------------------------------------
        # The first node of the grid sits at (x0, y0), with x0 and y0 each
        # independently drawn once per deformed image from [0, cell_size).
        # This is the random phase that keeps the grid from always landing
        # on the same pixel indices from one sample to the next.

        x0 = np.random.randint(0, cell_size)
        y0 = np.random.randint(0, cell_size)

        x_nodes = build_node_grid(cols_ext, cell_size, x0)
        y_nodes = build_node_grid(rows_ext, cell_size, y0)

        nx = len(x_nodes)
        ny = len(y_nodes)

        # ---------------------------------------
        # Random displacements at grid nodes
        # ---------------------------------------
        U_nodes = np.random.uniform(
            -max_amplitude,
            max_amplitude,
            size=(ny, nx)
        )

        V_nodes = np.random.uniform(
            -max_amplitude,
            max_amplitude,
            size=(ny, nx)
        )

        # Boundary displacements set to zero - this now sits on the
        # periodized padding's outer edge (rows_ext-1 / cols_ext-1), not on
        # the real image border.
        U_nodes[0, :] = 0
        U_nodes[-1, :] = 0
        U_nodes[:, 0] = 0
        U_nodes[:, -1] = 0

        V_nodes[0, :] = 0
        V_nodes[-1, :] = 0
        V_nodes[:, 0] = 0
        V_nodes[:, -1] = 0

        # ---------------------------------------
        # Bilinear interpolation to all pixels of the EXTENDED canvas
        # ---------------------------------------
        interp_U = RegularGridInterpolator(
            (y_nodes, x_nodes),
            U_nodes,
            method='linear',
            bounds_error=False,
            fill_value=0
        )

        interp_V = RegularGridInterpolator(
            (y_nodes, x_nodes),
            V_nodes,
            method='linear',
            bounds_error=False,
            fill_value=0
        )

        U_ext = interp_U(pts_ext).reshape(rows_ext, cols_ext)
        V_ext = interp_V(pts_ext).reshape(rows_ext, cols_ext)

        # --------------------------------------------------
        # Safety cap on combined displacement magnitude
        # --------------------------------------------------
        # Bilinear interpolation between independently-random node values
        # can overshoot max_amplitude at interior pixels (it's a per-node
        # bound, not a bound on the interpolated field), so this rescales
        # any pixel exceeding MAX_TOTAL_DISP, preserving direction. Applied
        # on the extended field so the crop below and the warp stay
        # consistent with the (possibly capped) saved U/V labels.
        if MAX_TOTAL_DISP is not None:
            magnitude = np.sqrt(U_ext ** 2 + V_ext ** 2)
            scale = np.minimum(1.0, MAX_TOTAL_DISP / (magnitude + 1e-6))
            U_ext *= scale
            V_ext *= scale

        # ---------------------------------------
        # Crop back to the real image footprint
        # ---------------------------------------
        U = U_ext[pad:pad+rows, pad:pad+cols]
        V = V_ext[pad:pad+rows, pad:pad+cols]

        # ---------------------------------------
        # Warp image
        # ---------------------------------------
        # Sample from the EXTENDED (periodized) reference image so that
        # warping displacement near/at the real border pulls in genuine
        # (periodized) texture instead of clamped or reflected pixels.
        # (X, Y) are real-image pixel indices; their position inside
        # ref_img_ext is offset by `pad`.
        X_new = (X + pad - U).astype(np.float32)
        Y_new = (Y + pad - V).astype(np.float32)

        deformed = cv2.remap(
            ref_img_ext,
            X_new,
            Y_new,
            interpolation=interpolation,
            borderMode=cv2.BORDER_REFLECT101
        )

        # ---------------------------------------
        # Save outputs
        # ---------------------------------------
        out_name = f"{basename}_def_{k:03d}"

        cv2.imwrite(
            os.path.join("GeneratedBoukhtache", out_name + ".bmp"),
            deformed
        )

        np.save(
            os.path.join("GeneratedBoukhtache", out_name + "_U.npy"),
            U
        )

        np.save(
            os.path.join("GeneratedBoukhtache", out_name + "_V.npy"),
            V
        )

print("Done.")
