import cv2
import numpy as np
import os
import random
import json


# ============================================================
# AVAILABLE MODES
# ============================================================
# Kept at module level (rather than only inside generate_displacement_field)
# so it can be reused elsewhere, e.g. to validate config["fixed_mode"].
DISPLACEMENT_TYPES = [
    # --- rigid / homogeneous ---
    "stretch_x", "stretch_y",
    "shear_x", "shear_y",
    "translation", "rotation",
    "isotropic_dilation",     # uniform (thermal-like) expansion
    "poisson_biaxial",        # elastic biaxial stretch coupled by nu

    # --- localized / heterogeneous ---
    "localized_disp",
    "shear_band",
    "high_freq",
    "radial_disp",

    # --- damage / fracture ---
    "necking",
    "barreling",               # compression counterpart to necking
    "bending",
    "torsion",
    "cavity",
    "inclusion",
    "void_coalescence",        # interacting/merging cavities
    "crack_open",
    "crack_slide",
    "crack_tip_KI",            # real LEFM Mode I asymptotic field
    "crack_tip_KII",           # real LEFM Mode II asymptotic field
    "buckling",
    "contact_indentation",     # Hertzian-like localized contact
    "delamination_blister",    # ring-shaped out-of-plane-driven field
]



# ============================================================
# IMAGE WARPING
# ============================================================
def apply_deformation(image, displacement_field):
    """
    Warps a grayscale image according to a per-pixel displacement field
    using backward mapping (remap). This is what turns the reference
    image into the "deformed" image for a synthetic DIC pair.
    """
    h, w = image.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    map_x = (X - displacement_field[..., 0]).astype(np.float32)
    map_y = (Y - displacement_field[..., 1]).astype(np.float32)

    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101
    )


# ============================================================
# MODE SELECTION
# ============================================================
def select_modes(displacement_types, min_modes, max_modes):
    """
    Randomly pick a subset of displacement modes to combine for a given
    sample. Combining several modes simulates the fact that real
    specimens rarely deform under a single "pure" loading condition.
    """
    n = np.random.randint(min_modes, max_modes + 1)
    return set(random.sample(displacement_types, n))


# ============================================================
# MAIN DISPLACEMENT FIELD
# ============================================================
def generate_displacement_field(shape, config, enabled_modes=None):
    """
    Builds a synthetic displacement field (disp_x, disp_y) over a
    normalized [-1, 1] x [-1, 1] grid mapped onto the image shape.

    All tunable numbers (amplitude ranges, localization widths, material
    parameters, etc.) are read from the single `config` dictionary
    defined at the bottom of this file -- edit them there, in one place,
    rather than hunting through this function.

    Modes are additive: several physical mechanisms can be superposed
    to mimic realistic, non-idealized loading. Because superposing many
    modes can produce unrealistically large cumulative displacements,
    the total field magnitude is optionally capped at the end
    (see config["max_total_disp"]).
    """

    h, w = shape
    X, Y = np.meshgrid(np.linspace(-1, 1, w),
                       np.linspace(-1, 1, h))

    disp_x = np.zeros_like(X)
    disp_y = np.zeros_like(Y)

    max_disp = config["max_disp"]

    # Poisson's ratio is randomized per sample within [0, nu_max].
    # It is used by the biaxial, necking/barreling transverse coupling,
    # and crack-tip modes, so it is sampled once here and reused.
    nu = np.random.uniform(0, config["nu_max"])

    # Plane-stress vs plane-strain is also randomized per sample, so a
    # dataset of fracture cases ends up with a mix of both instead of
    # always using the same assumption. Controlled by
    # config["plane_stress_probability"] (0 = always plane strain,
    # 1 = always plane stress, 0.5 = 50/50 mix). Only affects
    # crack_tip_KI / crack_tip_KII.
    plane_stress = np.random.rand() < config["plane_stress_probability"]

    # ========================================================
    # AVAILABLE MODES
    # ========================================================
    displacement_types = DISPLACEMENT_TYPES

    if enabled_modes is None:
        active_modes = select_modes(
            displacement_types,
            config["min_modes"],
            config["max_modes"]
        )
    else:
        active_modes = set(enabled_modes)

    # ========================================================
    # BASIC / HOMOGENEOUS MODES
    # ========================================================

    if "stretch_x" in active_modes:
        # Independent uniaxial stretch along X (no physical coupling to Y)
        a = np.random.uniform(-1, 1) * max_disp
        disp_x += a * X

    if "stretch_y" in active_modes:
        # Independent uniaxial stretch along Y
        a = np.random.uniform(-1, 1) * max_disp
        disp_y += a * Y

    if "shear_x" in active_modes:
        a = np.random.uniform(-1, 1) * max_disp
        disp_x += a * Y

    if "shear_y" in active_modes:
        a = np.random.uniform(-1, 1) * max_disp
        disp_y += a * X

    if "translation" in active_modes:
        tx, ty = np.random.uniform(-config["translation_max"], config["translation_max"], 2)
        disp_x += tx
        disp_y += ty

    if "rotation" in active_modes:
        theta = np.random.uniform(-config["rotation_max"], config["rotation_max"])
        disp_x += -theta * Y
        disp_y += theta * X

    # ------------------------------------------------------
    # ISOTROPIC DILATION
    # Uniform expansion/contraction, same coefficient in X and Y.
    # Models thermal expansion or purely volumetric strain, which
    # independent stretch_x/stretch_y draws cannot represent exactly.
    # ------------------------------------------------------
    if "isotropic_dilation" in active_modes:
        eps = np.random.uniform(-1, 1) * max_disp
        disp_x += eps * X
        disp_y += eps * Y

    # ------------------------------------------------------
    # POISSON-COUPLED BIAXIAL STRETCH
    # Elastic loading along X produces a lateral contraction along Y
    # governed by Poisson's ratio: eps_y = -nu * eps_x.
    # ------------------------------------------------------
    if "poisson_biaxial" in active_modes:
        eps_x = np.random.uniform(-1, 1) * max_disp
        disp_x += eps_x * X
        disp_y += -nu * eps_x * Y

    # ========================================================
    # LOCALIZED GAUSSIAN FIELD
    # ========================================================
    if "localized_disp" in active_modes:
        c = config["localized_disp"]
        for _ in range(np.random.randint(c["n_min"], c["n_max"] + 1)):
            x0, y0 = np.random.uniform(-c["pos_range"], c["pos_range"], 2)
            sigma = np.random.uniform(c["sigma_min"], c["sigma_max"])
            A = np.random.uniform(0.1, max_disp)

            G = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

            disp_x += A * (X - x0) * G
            disp_y += A * (Y - y0) * G

    # ========================================================
    # SHEAR BAND
    # ========================================================
    if "shear_band" in active_modes:
        c = config["shear_band"]
        angle = np.random.uniform(0, np.pi)
        pos = np.random.uniform(-c["pos_range"], c["pos_range"])
        width = np.random.uniform(c["width_min"], c["width_max"])
        A = np.random.uniform(0.1, max_disp)

        band = np.exp(-((X*np.cos(angle) + Y*np.sin(angle) - pos)**2) /
                      (2 * width**2))

        disp_x += A * band * np.cos(angle)
        disp_y += A * band * np.sin(angle)

    # ========================================================
    # HIGH FREQUENCY (buckling precursor / noise-like texture)
    # ========================================================
    if "high_freq" in active_modes:
        c = config["high_freq"]
        k = np.random.uniform(c["k_min"], c["k_max"])
        phi = np.random.uniform(0, 2*np.pi)
        A = np.random.uniform(0.1, max_disp)

        wave = A * np.sin(k * (X + Y) + phi)
        disp_x += wave
        disp_y += wave

    # ========================================================
    # RADIAL FIELD, GLOBAL (image-centered, R^2 growth)
    # NOTE: deliberately different from "cavity" below.
    # radial_disp grows with R^2 and is always centered on the image,
    # so it behaves like a global lens-like expansion/contraction.
    # "cavity" is a local 1/r point-source field with a random center.
    # Keep both, but don't confuse them.
    # ========================================================
    if "radial_disp" in active_modes:
        A = np.random.uniform(-max_disp, max_disp)
        R2 = X**2 + Y**2

        disp_x += A * R2
        disp_y += A * R2

    # ========================================================
    # NECKING (tension)
    # Localized narrowing band: material contracts laterally (Y) while
    # elongating along the loading axis (X) in the necked region. The
    # transverse contraction is scaled by Poisson's ratio (nu) rather
    # than an arbitrary constant, for physical consistency.
    # ========================================================
    if "necking" in active_modes:
        c = config["necking"]
        sigma = np.random.uniform(c["sigma_min"], c["sigma_max"])
        A = np.random.uniform(0.1, max_disp)

        band = np.exp(-X**2 / (2 * sigma**2))

        disp_x += A * X * band
        disp_y += -nu * A * Y * band

    # ------------------------------------------------------
    # BARRELING (compression)
    # Compression counterpart to necking: the specimen bulges outward
    # near mid-height instead of narrowing. Same mathematical structure
    # as necking but with the localization band and outward/inward
    # signs swapped between axes. Transverse coupling also uses nu.
    # ------------------------------------------------------
    if "barreling" in active_modes:
        c = config["barreling"]
        sigma = np.random.uniform(c["sigma_min"], c["sigma_max"])
        A = np.random.uniform(0.1, max_disp)

        band = np.exp(-Y**2 / (2 * sigma**2))

        disp_x += A * X * band          # outward bulge at mid-height
        disp_y += -nu * A * Y * band    # accompanying compression

    # ========================================================
    # BENDING
    # ========================================================
    if "bending" in active_modes:
        kappa = np.random.uniform(-max_disp, max_disp)

        disp_x += -kappa * X * Y
        disp_y += 0.5 * kappa * X**2

    # ========================================================
    # TORSION
    # ========================================================
    if "torsion" in active_modes:
        alpha = np.random.uniform(-max_disp, max_disp)

        R2 = X**2 + Y**2
        disp_x += -alpha * R2 * Y
        disp_y += alpha * R2 * X

    # ========================================================
    # CAVITY EXPANSION (local 1/r point source, random center)
    # ========================================================
    if "cavity" in active_modes:
        c = config["cavity"]
        x0, y0 = np.random.uniform(-c["pos_range"], c["pos_range"], 2)
        A = np.random.uniform(0.1, max_disp)

        r = np.sqrt((X - x0)**2 + (Y - y0)**2)

        disp_x += A * (X - x0) / (r + 1e-6)
        disp_y += A * (Y - y0) / (r + 1e-6)

    # ========================================================
    # INCLUSION (Gaussian-decay point source)
    # ========================================================
    if "inclusion" in active_modes:
        c = config["inclusion"]
        x0, y0 = np.random.uniform(-c["pos_range"], c["pos_range"], 2)
        sigma = np.random.uniform(c["sigma_min"], c["sigma_max"])
        A = np.random.uniform(0.1, max_disp)

        G = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

        disp_x += A * (X - x0) * G
        disp_y += A * (Y - y0) * G

    # ------------------------------------------------------
    # VOID COALESCENCE
    # Several cavities placed close together so their 1/r fields
    # interact and start to merge, as in ductile damage prior to
    # final fracture (distinct from a single isolated cavity).
    # ------------------------------------------------------
    if "void_coalescence" in active_modes:
        c = config["void_coalescence"]
        n_voids = np.random.randint(c["n_min"], c["n_max"] + 1)
        cx, cy = np.random.uniform(-c["cluster_pos_range"], c["cluster_pos_range"], 2)
        spread = np.random.uniform(c["spread_min"], c["spread_max"])

        for _ in range(n_voids):
            x0 = cx + np.random.uniform(-spread, spread)
            y0 = cy + np.random.uniform(-spread, spread)
            A = np.random.uniform(0.1, max_disp) / n_voids  # keep total energy bounded

            r = np.sqrt((X - x0)**2 + (Y - y0)**2)
            disp_x += A * (X - x0) / (r + 1e-6)
            disp_y += A * (Y - y0) / (r + 1e-6)

    # ========================================================
    # CRACK OPENING / SLIDING (smoothed step, kept for backward
    # compatibility / cheap qualitative discontinuity)
    # For a physically accurate near-tip field, prefer
    # "crack_tip_KI" / "crack_tip_KII" below.
    # ========================================================
    if "crack_open" in active_modes:
        c = config["crack_open"]
        x0 = np.random.uniform(-c["pos_range"], c["pos_range"])
        delta = np.random.uniform(c["delta_min"], c["delta_max"])
        A = np.random.uniform(0.1, max_disp)

        disp_y += A * np.tanh((X - x0) / delta)

    if "crack_slide" in active_modes:
        c = config["crack_slide"]
        y0 = np.random.uniform(-c["pos_range"], c["pos_range"])
        delta = np.random.uniform(c["delta_min"], c["delta_max"])
        A = np.random.uniform(0.1, max_disp)

        disp_x += A * np.tanh((Y - y0) / delta)

    # ------------------------------------------------------
    # CRACK TIP FIELDS (linear elastic fracture mechanics)
    # Williams' asymptotic near-tip displacement series for Mode I
    # (opening) and Mode II (in-plane shear), using polar coordinates
    # (r, theta) centered at a randomly placed crack tip, with the
    # crack line extending along the negative-X direction from the tip.
    #
    # kappa = (3 - nu) / (1 + nu)   for plane stress
    # kappa = 3 - 4*nu              for plane strain
    #
    # K_I / K_II here are *synthetic* amplitude parameters (not
    # physical MPa*sqrt(m) values) scaled by max_disp, chosen so the
    # correct sqrt(r) singular shape and angular dependence are
    # reproduced -- the key qualitative feature a tanh-based
    # approximation ("crack_open"/"crack_slide") misses.
    # This is the only place `plane_stress` has any effect.
    # ------------------------------------------------------
    if "crack_tip_KI" in active_modes or "crack_tip_KII" in active_modes:
        c = config["crack_tip"]
        kappa = (3 - nu) / (1 + nu) if plane_stress else (3 - 4 * nu)

        x0, y0 = np.random.uniform(-c["pos_range"], c["pos_range"], 2)
        Xc = X - x0
        Yc = Y - y0
        r = np.sqrt(Xc**2 + Yc**2) + 1e-6
        theta = np.arctan2(Yc, Xc)

        sqrt_term = np.sqrt(r / (2 * np.pi))

        if "crack_tip_KI" in active_modes:
            K_I = np.random.uniform(0.1, max_disp)
            disp_x += K_I * sqrt_term * np.cos(theta / 2) * \
                (kappa - 1 + 2 * np.sin(theta / 2)**2)
            disp_y += K_I * sqrt_term * np.sin(theta / 2) * \
                (kappa + 1 - 2 * np.cos(theta / 2)**2)

        if "crack_tip_KII" in active_modes:
            K_II = np.random.uniform(0.1, max_disp)
            disp_x += K_II * sqrt_term * np.sin(theta / 2) * \
                (kappa + 1 + 2 * np.cos(theta / 2)**2)
            disp_y += -K_II * sqrt_term * np.cos(theta / 2) * \
                (kappa - 1 - 2 * np.sin(theta / 2)**2)

    # ========================================================
    # BUCKLING
    # Superposes a primary wrinkle wavelength with 1-2 weaker
    # higher-order harmonics, which is more representative of real
    # post-buckling patterns than a single pure sine mode.
    # ========================================================
    if "buckling" in active_modes:
        c = config["buckling"]
        n_harmonics = np.random.randint(c["n_harmonics_min"], c["n_harmonics_max"] + 1)
        for i in range(n_harmonics):
            A = np.random.uniform(0.1, max_disp) / (i + 1)  # weaker higher harmonics
            k = np.random.uniform(c["k_min"], c["k_max"]) * (i + 1)
            phi = np.random.uniform(0, 2 * np.pi)
            disp_y += A * np.sin(np.pi * k * X + phi)

    # ------------------------------------------------------
    # CONTACT / INDENTATION (Hertzian-inspired)
    # Localized contact patch (indenter, punch, roller) pressing into
    # the surface: material is pushed inward directly under the
    # contact zone and flows/piles up laterally to the sides. This is
    # a simplified, smooth approximation of a Hertzian contact
    # solution -- not an exact Boussinesq solution -- chosen to be
    # cheap to evaluate while keeping the right qualitative shape.
    # ------------------------------------------------------
    if "contact_indentation" in active_modes:
        c = config["contact_indentation"]
        x0 = np.random.uniform(-c["pos_range"], c["pos_range"])   # contact center along X
        a = np.random.uniform(c["a_min"], c["a_max"])             # contact patch half-width
        A = np.random.uniform(0.1, max_disp)

        contact_profile = np.exp(-((X - x0)**2) / (2 * a**2))

        disp_y += -A * contact_profile                       # push inward under contact
        disp_x += 0.5 * A * (X - x0) / a * contact_profile    # lateral pile-up/sink-in

    # ------------------------------------------------------
    # DELAMINATION / BLISTER (composites)
    # Out-of-plane blistering shows up in-plane as a ring-shaped
    # displacement pattern: near zero at the blister center, peaking
    # at the delamination front, and decaying beyond it. This is
    # distinct from "inclusion" (peaks at the center) and from
    # "cavity" (monotonic 1/r decay, no ring).
    # ------------------------------------------------------
    if "delamination_blister" in active_modes:
        c = config["delamination_blister"]
        x0, y0 = np.random.uniform(-c["pos_range"], c["pos_range"], 2)
        sigma = np.random.uniform(c["sigma_min"], c["sigma_max"])
        A = np.random.uniform(0.1, max_disp)

        r = np.sqrt((X - x0)**2 + (Y - y0)**2)
        ring = (r / sigma) * np.exp(-r**2 / (2 * sigma**2))

        disp_x += A * (X - x0) / (r + 1e-6) * ring
        disp_y += A * (Y - y0) / (r + 1e-6) * ring

    # ========================================================
    # SAFETY CAP ON TOTAL DISPLACEMENT MAGNITUDE
    # When many modes are superposed, the summed field can reach
    # unrealistically large displacements. If config["max_total_disp"]
    # is set (not None), rescale any pixel whose combined displacement
    # magnitude exceeds it, preserving direction. Note this is a clamp
    # applied AFTER superposition -- it is not a normalization of the
    # whole field, and pixels under the threshold are left untouched.
    # ========================================================
    max_total_disp = config["max_total_disp"]
    if max_total_disp is not None:
        magnitude = np.sqrt(disp_x**2 + disp_y**2)
        scale = np.ones_like(magnitude)
        exceed = magnitude > max_total_disp
        scale[exceed] = max_total_disp / (magnitude[exceed] + 1e-6)
        disp_x *= scale
        disp_y *= scale

    return np.stack([disp_x, disp_y], axis=-1)


# ============================================================
# PROCESS IMAGES
# ============================================================
def process_images(config):
    """
    Applies random combinations of the physical deformation modes to
    every .bmp reference image found in config["input_folder"], saving
    the deformed image plus the ground-truth U/V displacement maps
    (useful as DIC training targets) to config["output_folder"].
    """

    input_folder = config["input_folder"]
    output_folder = config["output_folder"]
    num_deformations = config["num_deformations"]

    # If config["fixed_mode"] is set (a mode name, or a list of mode
    # names), every generated sample uses exactly those mode(s) instead
    # of a random combination -- this bypasses min_modes/max_modes
    # entirely and is what you want for isolating one mode at a time
    # (e.g. to test model performance mode-by-mode). Leave it as `null`
    # in the JSON to keep the normal random min_modes..max_modes mixing.
    fixed_mode = config.get("fixed_mode")
    enabled_modes = None
    if fixed_mode is not None:
        enabled_modes = [fixed_mode] if isinstance(fixed_mode, str) else list(fixed_mode)
        unknown = set(enabled_modes) - set(DISPLACEMENT_TYPES)
        if unknown:
            raise ValueError(
                f"config['fixed_mode'] contains unknown mode(s) {unknown}. "
                f"Valid modes are: {DISPLACEMENT_TYPES}"
            )

    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.bmp')]

    for img_file in image_files:
        image = cv2.imread(os.path.join(input_folder, img_file),
                           cv2.IMREAD_GRAYSCALE)

        base_name = os.path.splitext(img_file)[0]

        for i in range(num_deformations):

            disp = generate_displacement_field(image.shape, config, enabled_modes=enabled_modes)

            deformed = apply_deformation(image, disp)

            cv2.imwrite(
                os.path.join(output_folder, f"{base_name}_def_{i:03d}.bmp"),
                deformed
            )

            np.save(
                os.path.join(output_folder, f"{base_name}_def_{i:03d}_U.npy"),
                disp[..., 0]
            )

            np.save(
                os.path.join(output_folder, f"{base_name}_def_{i:03d}_V.npy"),
                disp[..., 1]
            )


# ============================================================
# CONFIGURATION LOADING
# All tunable parameters live in the external "config.json" file
# (next to this script), not in the code. Edit values there to
# change amplitudes, localization widths, material parameters, I/O
# folders, etc. without touching this file.
# ============================================================
def load_config(path="config.json"):
    with open(path, "r") as f:
        config = json.load(f)

    # JSON has no concept of Python's None -- config.json uses `null`
    # for "no cap", which json.load already converts to None, so no
    # extra handling is needed here. This function exists as a single
    # place to add validation later if useful (e.g. checking required
    # keys are present).
    return config


# ============================================================
# RUN EXAMPLE
# ============================================================
if __name__ == "__main__":
    config = load_config("config.json")
    process_images(config)
