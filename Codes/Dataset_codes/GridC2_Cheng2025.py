import numpy as np
import cv2
import os
from scipy.interpolate import RectBivariateSpline

SubsetSize = 128
# to add noise
noise_std = 0

N_ref = 200

# ============================================================
# TARGET CONFIGURATION
# ============================================================
# TARGET_MAX_DISP: pick 1.0 or 5.0 (or any other target - see calibration
# note below for how these numbers were derived).
#
# max_amplitude was calibrated empirically (actually running this script's
# RectBivariateSpline mechanism, not guessed) because cubic splines can
# OVERSHOOT beyond the +/-max_amplitude node range fit to random,
# uncorrelated node values - unlike e.g. Boukhtache's bilinear
# interpolation, which is bounded by the node values. At the original
# max_disp=2.7, the scale=4 case alone (before even combining U,V) peaked
# at 5.14px - noticeably above the nominal 2.7 "cap". This calibration was
# re-verified after adding the periodic-padding/random-phase fixes below
# (see conversation) - it still holds.
TARGET_MAX_DISP = 1.0   # <-- set to 1.0 or 5.0 (or another target)
MAX_TOTAL_DISP = TARGET_MAX_DISP  # hard safety cap (not in the original script)

_CALIBRATED_MAX_AMPLITUDE = {
    1.0: 0.50,
    5.0: 2.60,
}
if TARGET_MAX_DISP in _CALIBRATED_MAX_AMPLITUDE:
    max_amplitude = _CALIBRATED_MAX_AMPLITUDE[TARGET_MAX_DISP]
else:
    # Linear fallback for other targets - re-verify empirically if precision matters.
    max_amplitude = 0.5 * TARGET_MAX_DISP

scales = [32, 16, 8, 4]

# DISCLOSED CONCERN (measured, not guessed): with this multi-scale mix,
# smaller scales are progressively harder for classical subset-based DIC
# (or most DL-DIC networks) to resolve - see conversation for the
# per-scale strain measurements. scales=[32,16,8,4] is left UNCHANGED
# here since mixing difficulty levels may be intentional for training
# diversity - drop the smallest scale(s) if you want every sample
# resolvable by classical DIC too.

nbdef = 200  # number of realizations per reference image

output_folder = f"GeneratedCheng{int(TARGET_MAX_DISP)}px"
os.makedirs(output_folder, exist_ok=True)

# ==========================================================
# Edge-deformation fix: periodized padding (same approach as
# Boukhtache_2.py / Hermite2.py)
# ==========================================================
# The node grid pins U,V to zero at its outermost ring of nodes. If that
# ring sits exactly on the real image edges, border pixels are never
# deformed - a systematic bias. Fix: pad the reference image by
# periodizing it (BORDER_WRAP tiles the image content around itself),
# build the grid and evaluate the spline on this larger canvas, then crop
# back to the original size. The zero-displacement node ring now lands
# inside the periodized padding, strictly outside the real image, so
# every real pixel - including the border - gets a genuine, non-zero,
# smoothly-varying displacement.
#
# pad must be at least as large as the biggest scale so that even the
# largest-spacing grid's first interior node line falls outside the real
# image; +1 gives a one-pixel margin.
pad = max(scales) + 1


def build_node_grid(n_pixels, h, offset):
    """
    Build node positions covering [0, n_pixels-1] with base spacing h,
    starting from the given phase offset (in [0, h)) instead of always 0.
    This means the grid is not pinned to the same pixel indices on every
    sample, removing the fixed-grid-alignment bias. The first/last
    elements may end up shorter than h (irregular boundary elements) so
    the grid still exactly spans the full extended image. Strictly
    increasing by construction - required by RectBivariateSpline.
    """
    nodes = np.arange(offset, n_pixels, h)

    if nodes[0] != 0:
        nodes = np.insert(nodes, 0, 0)

    if nodes[-1] != n_pixels - 1:
        nodes = np.append(nodes, n_pixels - 1)

    return nodes


for img_id in range(N_ref):

    ref_name = f"../Ref_MO_128x128/Ref_image_{img_id:03d}.bmp"

    ref_image = cv2.imread(
        ref_name,
        cv2.IMREAD_GRAYSCALE
    )
    if ref_image is None:
        continue
    ref_image = ref_image.astype(np.float32)

    rows, cols = ref_image.shape

    # Periodize the reference image outward by `pad` pixels on every side.
    ref_image_ext = cv2.copyMakeBorder(
        ref_image, pad, pad, pad, pad, borderType=cv2.BORDER_WRAP
    )
    rows_ext, cols_ext = ref_image_ext.shape

    # Real-image pixel grid, used for the final warp.
    X, Y = np.meshgrid(
        np.arange(cols),
        np.arange(rows)
    )

    basename = f"Ref_image_{img_id:03d}"

    for sample_id in range(nbdef):

        s = int(np.random.choice(scales))

        # --------------------------------------------------------
        # Randomized-phase control-point grid on the EXTENDED image.
        # x0, y0 each independently drawn once per deformed image from
        # [0, s) - the random phase that keeps the grid from always
        # landing on the same pixel indices from one sample to the next.
        # --------------------------------------------------------
        x0 = np.random.randint(0, s)
        y0 = np.random.randint(0, s)

        x_nodes = build_node_grid(cols_ext, s, x0)
        y_nodes = build_node_grid(rows_ext, s, y0)

        nx = len(x_nodes)
        ny = len(y_nodes)

        f = np.random.uniform(-max_amplitude, max_amplitude, size=(ny, nx))
        g = np.random.uniform(-max_amplitude, max_amplitude, size=(ny, nx))

        # Boundary displacements pinned to zero at the NODE level - this
        # now sits on the periodized padding's outer edge (rows_ext-1 /
        # cols_ext-1), not on the real image border.
        f[0, :] = 0
        f[-1, :] = 0
        f[:, 0] = 0
        f[:, -1] = 0

        g[0, :] = 0
        g[-1, :] = 0
        g[:, 0] = 0
        g[:, -1] = 0

        interp_u = RectBivariateSpline(
            y_nodes,
            x_nodes,
            f,
            kx=3,
            ky=3
        )

        interp_v = RectBivariateSpline(
            y_nodes,
            x_nodes,
            g,
            kx=3,
            ky=3
        )

        U_ext = interp_u(
            np.arange(rows_ext),
            np.arange(cols_ext)
        )

        V_ext = interp_v(
            np.arange(rows_ext),
            np.arange(cols_ext)
        )

        # --------------------------------------------------------
        # Safety cap on combined displacement magnitude. Needed because
        # of the cubic spline overshoot noted above. Applied on the
        # extended field so the crop below and the warp stay consistent
        # with the (possibly capped) saved U/V labels.
        # --------------------------------------------------------
        if MAX_TOTAL_DISP is not None:
            magnitude = np.sqrt(U_ext ** 2 + V_ext ** 2)
            scale_factor = np.minimum(1.0, MAX_TOTAL_DISP / (magnitude + 1e-6))
            U_ext = U_ext * scale_factor
            V_ext = V_ext * scale_factor

        # Crop back to the real image footprint
        U = U_ext[pad:pad + rows, pad:pad + cols]
        V = V_ext[pad:pad + rows, pad:pad + cols]

        Exx = np.gradient(U, axis=1)
        Eyy = np.gradient(V, axis=0)
        Exy = 0.5 * (
            np.gradient(U, axis=0)
            + np.gradient(V, axis=1)
        )

        # --------------------------------------------------------
        # Warp: sample from the EXTENDED (periodized) reference image so
        # that warping displacement near/at the real border pulls in
        # genuine (periodized) texture instead of clamped/reflected
        # pixels. (X, Y) are real-image pixel indices; their position
        # inside ref_image_ext is offset by `pad`. BORDER_REFLECT101
        # remains only as a rare fallback for the (essentially
        # never-triggered, given the safety cap above and pad chosen
        # >= max(scales)+1) case of a displacement exceeding the padded
        # margin.
        # --------------------------------------------------------
        map_x = (X + pad - U).astype(np.float32)
        map_y = (Y + pad - V).astype(np.float32)

        def_image = cv2.remap(
            ref_image_ext,
            map_x,
            map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT101
        )

        if noise_std > 0:
            def_image = def_image + noise_std * np.random.randn(rows, cols)

        def_image = np.clip(def_image, 0, 255).astype(np.uint8)

        cv2.imwrite(
            os.path.join(output_folder, f"{basename}_def_{sample_id:03d}.bmp"),
            def_image
        )

        np.save(
            os.path.join(output_folder, f"{basename}_def_{sample_id:03d}_U.npy"),
            U
        )

        np.save(
            os.path.join(output_folder, f"{basename}_def_{sample_id:03d}_V.npy"),
            V
        )

print("Done.")
