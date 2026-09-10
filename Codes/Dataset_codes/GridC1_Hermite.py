import numpy as np
import cv2
import glob
import os

# ==========================================================
# PARAMETERS
# ==========================================================
# choose between same element_size or varying for multi-frequency training set
#element_size = 16          # 5,9,17,33,65 ...
element_size = [4,8,16,32]

# Calibrated empirically at 128x128 (same approach used for Yang2022.py and
# the JD_2025_V3.py configs) for a target peak displacement of 5px. Unlike
# Yang2022.py's original bug, this script's max_disp was already close to
# reasonable by luck (max_disp=4.0, factors=1.0 alone gave ~5.6px peak) -
# nudged down slightly and paired with an explicit safety cap below so 5px
# is a guarantee, not a coincidence. The ratio max_disp = 0.58 * target
# (with grad/cross factors left at 1.0) was verified to hold consistently
# at 1px, 5px, and 50px targets too - mean peak lands at ~76-78% of the
# cap in all three cases, with the cap essentially never needing to
# trigger (a safety net, not a crutch) - so this scales cleanly if you
# want a different target later, just recompute max_disp = 0.58*target.
TARGET_MAX_DISP = 1.0
max_disp = 0.58 * TARGET_MAX_DISP   # pixels
max_grad_factor = 1.0      # multiplies max_disp/h
max_cross_factor = 1.0     # multiplies max_disp/h²
MAX_TOTAL_DISP = TARGET_MAX_DISP  # hard safety cap; set to None to disable
n_deformations = 200

input_pattern = "../Ref_MO_128x128/Ref_image_*.bmp"
output_folder = "GeneratedHermite1px_2"

os.makedirs(output_folder, exist_ok=True)

# ==========================================================
# Edge-deformation fix: periodized padding
# ==========================================================
# The Hermite nodal grid pins displacement (and its derivatives) to zero on
# its outermost row/column of nodes. Previously that outer boundary sat
# exactly on the real image edges, so border pixels were never deformed -
# a systematic bias.
#
# Fix: pad the reference image by periodizing it (BORDER_WRAP tiles the
# image content around itself), build the Hermite grid and warp on this
# larger canvas, then crop back to the original size. The zero-displacement
# boundary now lands inside the periodized padding, strictly outside the
# real image, so every real pixel - including the border - sits in the
# grid's interior and gets a genuine, non-zero, smoothly-varying
# displacement.
#
# pad must be at least as large as the biggest element so that even the
# largest-h grid's first interior node line falls outside the real image;
# +1 gives a one-pixel margin.
_cell_sizes = element_size if isinstance(element_size, (list, tuple)) else [element_size]
pad = max(_cell_sizes) + 1

# ==========================================================
# Hermite basis functions
# ==========================================================

def H1(t):
    return 1 - 3*t**2 + 2*t**3

def H2(t):
    return t - 2*t**2 + t**3

def H3(t):
    return 3*t**2 - 2*t**3

def H4(t):
    return -t**2 + t**3


def dH1(t):
    return -6*t + 6*t**2

def dH2(t):
    return 1 - 4*t + 3*t**2

def dH3(t):
    return 6*t - 6*t**2

def dH4(t):
    return -2*t + 3*t**2


def build_node_grid(n_pixels, h, offset):
    """
    Build node positions covering [0, n_pixels-1] with base spacing h,
    starting from the given phase offset (in [0, h)) instead of always 0.

    This means the grid is not pinned to the same pixel indices on every
    sample, which removes the fixed-grid-alignment bias. The first and
    last elements may end up shorter than h (irregular boundary elements)
    so that the grid still exactly spans the full extended image; this is
    the same trick the original code already used to force the last node
    onto n_pixels-1, just applied at both ends and made random instead of
    fixed at 0.
    """
    nodes = np.arange(offset, n_pixels, h)

    if nodes[0] != 0:
        nodes = np.insert(nodes, 0, 0)

    if nodes[-1] != n_pixels - 1:
        nodes = np.append(nodes, n_pixels - 1)

    return nodes


# ==========================================================
# Loop over reference images
# ==========================================================

ref_files = sorted(glob.glob(input_pattern))

for ref_file in ref_files:

    print("Processing:", ref_file)

    ref_img = cv2.imread(ref_file, cv2.IMREAD_GRAYSCALE)

    rows, cols = ref_img.shape

    # Periodize the reference image outward by `pad` pixels on every side.
    ref_img_ext = cv2.copyMakeBorder(
        ref_img, pad, pad, pad, pad, borderType=cv2.BORDER_WRAP
    )
    rows_ext, cols_ext = ref_img_ext.shape

    basename = os.path.splitext(os.path.basename(ref_file))[0]

    for sample in range(n_deformations):

        # Element/cell size is drawn independently for each deformed
        # image, not once per reference image, so consecutive samples
        # from the same reference can use different mesh resolutions.
        h = np.random.choice(element_size)

        # --------------------------------------------------
        # Randomized-phase Hermite node grid on the EXTENDED image
        # --------------------------------------------------
        # The first node of the grid sits at (x0, y0), with x0 and y0 each
        # independently drawn once per deformed image from [0, h). This is
        # the random phase that keeps the grid from always landing on the
        # same pixel indices from one sample to the next.

        x0 = np.random.randint(0, h)  # h >= 2 always here (see element_size)
        y0 = np.random.randint(0, h)

        x_nodes = build_node_grid(cols_ext, h, x0)
        y_nodes = build_node_grid(rows_ext, h, y0)

        nx = len(x_nodes)
        ny = len(y_nodes)

        # --------------------------------------------------
        # Generate Hermite nodal DOFs
        # --------------------------------------------------

        grad_amp = max_grad_factor * max_disp / h
        cross_amp = max_cross_factor * max_disp / h**2

        U     = np.random.uniform(-max_disp, max_disp, (ny, nx))
        Ux    = np.random.uniform(-grad_amp, grad_amp, (ny, nx))
        Uy    = np.random.uniform(-grad_amp, grad_amp, (ny, nx))
        Uxy   = np.random.uniform(-cross_amp, cross_amp, (ny, nx))

        V     = np.random.uniform(-max_disp, max_disp, (ny, nx))
        Vx    = np.random.uniform(-grad_amp, grad_amp, (ny, nx))
        Vy    = np.random.uniform(-grad_amp, grad_amp, (ny, nx))
        Vxy   = np.random.uniform(-cross_amp, cross_amp, (ny, nx))

        # Boundary fixed - this now sits on the periodized padding's outer
        # edge (rows_ext-1 / cols_ext-1), not on the real image border.
        for field in [U,Ux,Uy,Uxy,V,Vx,Vy,Vxy]:
            field[0,:] = 0
            field[-1,:] = 0
            field[:,0] = 0
            field[:,-1] = 0

        disp_u_ext = np.zeros((rows_ext, cols_ext))
        disp_v_ext = np.zeros((rows_ext, cols_ext))

        # --------------------------------------------------
        # Loop over Hermite elements (on the extended canvas)
        # --------------------------------------------------

        for ey in range(ny-1):

            for ex in range(nx-1):

                x0 = x_nodes[ex]
                x1 = x_nodes[ex+1]

                y0 = y_nodes[ey]
                y1 = y_nodes[ey+1]

                hx = x1 - x0
                hy = y1 - y0

                # scalar field matrices for u
                Mu = np.array([
                    [U[ey,ex],          U[ey+1,ex],          hy*Uy[ey,ex],          hy*Uy[ey+1,ex]],
                    [U[ey,ex+1],        U[ey+1,ex+1],        hy*Uy[ey,ex+1],        hy*Uy[ey+1,ex+1]],
                    [hx*Ux[ey,ex],      hx*Ux[ey+1,ex],      hx*hy*Uxy[ey,ex],      hx*hy*Uxy[ey+1,ex]],
                    [hx*Ux[ey,ex+1],    hx*Ux[ey+1,ex+1],    hx*hy*Uxy[ey,ex+1],    hx*hy*Uxy[ey+1,ex+1]]
                ])

                # scalar field matrices for v
                Mv = np.array([
                    [V[ey,ex],          V[ey+1,ex],          hy*Vy[ey,ex],          hy*Vy[ey+1,ex]],
                    [V[ey,ex+1],        V[ey+1,ex+1],        hy*Vy[ey,ex+1],        hy*Vy[ey+1,ex+1]],
                    [hx*Vx[ey,ex],      hx*Vx[ey+1,ex],      hx*hy*Vxy[ey,ex],      hx*hy*Vxy[ey+1,ex]],
                    [hx*Vx[ey,ex+1],    hx*Vx[ey+1,ex+1],    hx*hy*Vxy[ey,ex+1],    hx*hy*Vxy[ey+1,ex+1]]
                ])

                for iy in range(y0, y1 + 1):

                    eta = (iy - y0) / hy

                    Hy = np.array([
                        H1(eta),
                        H3(eta),
                        H2(eta),
                        H4(eta)
                    ])

                    for ix in range(x0, x1 + 1):

                        xi = (ix - x0) / hx

                        Hx = np.array([
                            H1(xi),
                            H3(xi),
                            H2(xi),
                            H4(xi)
                        ])

                        disp_u_ext[iy, ix] = Hx @ Mu @ Hy
                        disp_v_ext[iy, ix] = Hx @ Mv @ Hy

        # --------------------------------------------------
        # Safety cap on combined displacement magnitude (NOT in the
        # original script - added so TARGET_MAX_DISP is an actual
        # guarantee rather than just a typical/average value; matches the
        # convention used for Yang2022.py). Rescales any pixel exceeding
        # the cap, preserving direction. Applied on the extended field so
        # the crop below and the warp stay consistent with the (possibly
        # capped) saved U/V labels.
        # --------------------------------------------------
        if MAX_TOTAL_DISP is not None:
            magnitude = np.sqrt(disp_u_ext ** 2 + disp_v_ext ** 2)
            scale = np.minimum(1.0, MAX_TOTAL_DISP / (magnitude + 1e-6))
            disp_u_ext *= scale
            disp_v_ext *= scale

        # --------------------------------------------------
        # Crop back to the real image footprint
        # --------------------------------------------------

        disp_u = disp_u_ext[pad:pad+rows, pad:pad+cols]
        disp_v = disp_v_ext[pad:pad+rows, pad:pad+cols]

        # --------------------------------------------------
        # Strains (computed on the cropped, real-image-sized field)
        # --------------------------------------------------

        Exx = np.gradient(disp_u, axis=1)
        Eyy = np.gradient(disp_v, axis=0)

        Exy = 0.5 * (
            np.gradient(disp_u, axis=0) +
            np.gradient(disp_v, axis=1)
        )

        # --------------------------------------------------
        # Create deformed image
        # --------------------------------------------------
        # Sample from the EXTENDED (periodized) reference image so that
        # warping displacement near/at the real border pulls in genuine
        # (periodized) texture instead of clamped or reflected pixels.
        # Output coordinates (X,Y) are real-image pixel indices; their
        # position inside ref_img_ext is offset by `pad`.

        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

        map_x = (X + pad - disp_u).astype(np.float32)
        map_y = (Y + pad - disp_v).astype(np.float32)

        # BORDER_REFLECT101 kept as a fallback only; with MAX_TOTAL_DISP
        # far smaller than pad it should essentially never trigger.
        def_img = cv2.remap(
            ref_img_ext,
            map_x,
            map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT101
        )

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        prefix = f"{basename}_def_{sample:03d}"

        cv2.imwrite(
            os.path.join(output_folder, prefix + ".bmp"),
            def_img
        )

        np.save(
            os.path.join(output_folder, prefix + "_U.npy"),
            disp_u
        )

        np.save(
            os.path.join(output_folder, prefix + "_V.npy"),
            disp_v
        )

#        np.save(
#            os.path.join(output_folder, prefix + "_Exx.npy"),
#            Exx
#        )
#
#        np.save(
#            os.path.join(output_folder, prefix + "_Eyy.npy"),
#            Eyy
#        )
#
#        np.save(
#            os.path.join(output_folder, prefix + "_Exy.npy"),
#            Exy
#        )

print("Finished.")
