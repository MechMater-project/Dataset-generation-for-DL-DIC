import os
import glob
import cv2
import numpy as np

# ==========================================================
# PARAMETERS
# ==========================================================

reference_folder = "Reference"
output_folder = "GeneratedYang"

n_deformations_per_image = 5
noise_std = 0.0  # set to 4.0 to reproduce the paper noise

os.makedirs(output_folder, exist_ok=True)

# ==========================================================
# LOOP OVER REFERENCE IMAGES
# ==========================================================

ref_files = sorted(glob.glob(os.path.join(reference_folder, "*.bmp")))

for ref_file in ref_files:

    basename = os.path.splitext(os.path.basename(ref_file))[0]

    ref_img = cv2.imread(ref_file, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    rows, cols = ref_img.shape

    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    # normalized coordinates for Gaussian functions
    Xn = X / (cols - 1)
    Yn = Y / (rows - 1)

    for sample in range(n_deformations_per_image):

        # --------------------------------------------------
        # Random affine parameters
        # --------------------------------------------------

        tx = np.random.uniform(-4.0, 4.0)
        ty = np.random.uniform(-4.0, 4.0)

        kx = np.random.uniform(0.96, 1.04)
        ky = np.random.uniform(0.96, 1.04)

        theta = np.random.uniform(-0.01, 0.01)

        gamma_x = np.random.uniform(-0.03, 0.03)
        gamma_y = np.random.uniform(-0.03, 0.03)

        # --------------------------------------------------
        # Gaussian contribution for U equation
        # --------------------------------------------------

        ug = np.zeros_like(Xn)

        Nu = np.random.randint(1, 3)

        for _ in range(Nu):

            A = np.random.uniform(0.003, 0.6)
            A *= np.random.choice([-1.0, 1.0])

            sigma_x = np.random.uniform(0.06, 0.5)
            sigma_y = np.random.uniform(0.06, 0.5)

            x0 = np.random.uniform(0.0, 1.0)
            y0 = np.random.uniform(0.0, 1.0)

            ug += A * (cols-1) * np.exp(
                -((Xn - x0) ** 2) / (2 * sigma_x ** 2)
                -((Yn - y0) ** 2) / (2 * sigma_y ** 2)
            )

        # --------------------------------------------------
        # Gaussian contribution for V equation
        # --------------------------------------------------

        vg = np.zeros_like(Xn)

        Nv = np.random.randint(1, 3)

        for _ in range(Nv):

            A = np.random.uniform(0.003, 0.6)
            A *= np.random.choice([-1.0, 1.0])

            sigma_x = np.random.uniform(0.06, 0.5)
            sigma_y = np.random.uniform(0.06, 0.5)

            x0 = np.random.uniform(0.0, 1.0)
            y0 = np.random.uniform(0.0, 1.0)

            vg += A * (rows-1) * np.exp(
                -((Xn - x0) ** 2) / (2 * sigma_x ** 2)
                -((Yn - y0) ** 2) / (2 * sigma_y ** 2)
            )

        # --------------------------------------------------
        # Affine deformation
        # --------------------------------------------------

        U = (kx - 1.0) * X + gamma_x * Y + ug
        V = gamma_y * X + (ky - 1.0) * Y + vg

        # --------------------------------------------------
        # Rotation + translation
        # --------------------------------------------------

        disp_x = tx + np.cos(theta) * U + np.sin(theta) * V
        disp_y = ty - np.sin(theta) * U + np.cos(theta) * V

        # --------------------------------------------------
        # Small strain tensor
        # --------------------------------------------------

        du_dx = np.gradient(disp_x, axis=1)
        du_dy = np.gradient(disp_x, axis=0)

        dv_dx = np.gradient(disp_y, axis=1)
        dv_dy = np.gradient(disp_y, axis=0)

        exx = du_dx
        eyy = dv_dy
        exy = 0.5 * (du_dy + dv_dx)

        # --------------------------------------------------
        # Generate deformed image
        # --------------------------------------------------

        map_x = (X - disp_x).astype(np.float32)
        map_y = (Y - disp_y).astype(np.float32)

        def_img = cv2.remap(
            ref_img,
            map_x,
            map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT101
        )

        if noise_std > 0:
            def_img += noise_std * np.random.randn(rows, cols)

        def_img = np.clip(def_img, 0, 255).astype(np.uint8)

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
            disp_x
        )

        np.save(
            os.path.join(output_folder, prefix + "_V.npy"),
            disp_y
        )

        np.save(
            os.path.join(output_folder, prefix + "_Exx.npy"),
            exx
        )

        np.save(
            os.path.join(output_folder, prefix + "_Eyy.npy"),
            eyy
        )

        np.save(
            os.path.join(output_folder, prefix + "_Exy.npy"),
            exy
        )

print("Dataset generation finished.")
