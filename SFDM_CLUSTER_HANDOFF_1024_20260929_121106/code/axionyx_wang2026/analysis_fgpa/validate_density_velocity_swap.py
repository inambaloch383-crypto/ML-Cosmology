import numpy as np
import matplotlib.pyplot as plt

from lya_forward_model import (
    make_lya_flux,
    flux_power_ratio,
)


# ============================================================
# SYNTHETIC VALIDATION SETTINGS
# ============================================================

z = 3.0
Lbox_hMpc = 30.0

Nskewer = 64
Npix = 512

T0 = 1.0e4
gamma = 1.55

# ------------------------------------------------------------
# SOFTWARE-VALIDATION VALUE ONLY.
#
# This is NOT yet an adopted observational mean flux.
# ------------------------------------------------------------
target_mean_flux = 0.70

rng = np.random.default_rng(24680)


# ============================================================
# STORAGE
# ============================================================

Delta_N = np.empty(
    (Nskewer, Npix)
)

Delta_W = np.empty(
    (Nskewer, Npix)
)

v_N = np.empty(
    (Nskewer, Npix)
)

v_W = np.empty(
    (Nskewer, Npix)
)


# ============================================================
# FOURIER MODE ARRAY
#
# rfft contains modes:
#
# 0 ... Npix/2
#
# for even Npix.
# ============================================================

nmode = np.arange(
    Npix // 2 + 1,
    dtype=float
)

# Avoid division by zero in velocity construction.
n_safe = nmode.copy()
n_safe[0] = 1.0


# ============================================================
# BUILD MATCHED-PHASE SYNTHETIC FIELDS
#
# N = N-body-like proxy
# W = wave-like
#
# IMPORTANT:
# These are NOT physical SFDM predictions.
#
# They exist only to validate the ability of the pipeline
# to independently swap density and velocity fields.
# ============================================================

for s in range(Nskewer):

    # --------------------------------------------------------
    # Common random Fourier realization
    # --------------------------------------------------------

    real_part = rng.normal(
        size=nmode.size
    )

    imag_part = rng.normal(
        size=nmode.size
    )

    coeff = (
        real_part
        + 1j * imag_part
    )

    # --------------------------------------------------------
    # k = 0 mode:
    # no density DC perturbation
    # --------------------------------------------------------

    coeff[0] = 0.0 + 0.0j

    # --------------------------------------------------------
    # Nyquist coefficient of an rfft representation must
    # be purely real for a real-valued spatial field.
    # --------------------------------------------------------

    coeff[-1] = (
        coeff[-1].real
        + 0.0j
    )


    # ========================================================
    # DENSITY FIELD
    # ========================================================

    # --------------------------------------------------------
    # Broadband baseline transfer shape
    # --------------------------------------------------------

    Tbase = (
        1.0
        /
        np.sqrt(
            1.0
            + (nmode / 8.0)**2
        )
    )


    # --------------------------------------------------------
    # N-body-like density field
    # --------------------------------------------------------

    coeff_N = (
        coeff
        * Tbase
    )

    delta_N = np.fft.irfft(
        coeff_N,
        n=Npix
    )


    # --------------------------------------------------------
    # Wave-like density field
    #
    # Same phases but additional small-scale suppression.
    #
    # This suppression is synthetic and has no direct
    # physical interpretation.
    # --------------------------------------------------------

    Twave_density = np.exp(
        -(nmode / 75.0)**2
    )

    coeff_W = (
        coeff_N
        * Twave_density
    )

    delta_W = np.fft.irfft(
        coeff_W,
        n=Npix
    )


    # --------------------------------------------------------
    # Standardize density amplitudes
    #
    # Keep all Delta > 0 for FGPA.
    # --------------------------------------------------------

    delta_N /= np.std(
        delta_N
    )

    delta_W /= np.std(
        delta_W
    )

    delta_N *= 0.18
    delta_W *= 0.18

    Delta_N[s] = (
        1.0
        + delta_N
    )

    Delta_W[s] = (
        1.0
        + delta_W
    )


    # ========================================================
    # VELOCITY FIELD
    # ========================================================

    # --------------------------------------------------------
    # Construct a correlated potential-flow-like velocity:
    #
    #     v(k) ~ i delta(k) / k
    #
    # This gives the expected approximate phase offset
    # between density and velocity.
    # --------------------------------------------------------

    vcoeff_N = (
        1j
        * coeff_N
        / n_safe
    )

    # --------------------------------------------------------
    # FORMAL REAL-FIELD CONDITIONS
    #
    # k=0:
    # no bulk velocity.
    #
    # Nyquist:
    # force to zero because multiplication by i/k would
    # otherwise create an imaginary Nyquist coefficient,
    # which is incompatible with a real-valued irfft field.
    # --------------------------------------------------------

    vcoeff_N[0] = (
        0.0 + 0.0j
    )

    vcoeff_N[-1] = (
        0.0 + 0.0j
    )


    # --------------------------------------------------------
    # Real-space N-like velocity
    # --------------------------------------------------------

    vv_N = np.fft.irfft(
        vcoeff_N,
        n=Npix
    )

    vv_N /= np.std(
        vv_N
    )

    # Synthetic RMS velocity
    vv_N *= 35.0


    # --------------------------------------------------------
    # Wave-like velocity modification
    #
    # Same basic phases, but deliberately different
    # small-scale velocity structure.
    #
    # Again: this is ONLY a pipeline validation device.
    # --------------------------------------------------------

    velocity_modifier = (
        1.0
        +
        0.35
        * (
            (nmode / 35.0)**2
            /
            (
                1.0
                + (nmode / 35.0)**2
            )
        )
    )

    vcoeff_W = (
        vcoeff_N
        * velocity_modifier
    )

    # --------------------------------------------------------
    # Preserve formal Fourier conditions explicitly
    # --------------------------------------------------------

    vcoeff_W[0] = (
        0.0 + 0.0j
    )

    vcoeff_W[-1] = (
        0.0 + 0.0j
    )


    # --------------------------------------------------------
    # Real-space W-like velocity
    # --------------------------------------------------------

    vv_W = np.fft.irfft(
        vcoeff_W,
        n=Npix
    )

    # Keep the same total RMS so that this validation
    # primarily changes the scale dependence, not the
    # overall velocity amplitude.
    vv_W *= (
        35.0
        / np.std(vv_W)
    )

    v_N[s] = vv_N
    v_W[s] = vv_W


# ============================================================
# INPUT SAFETY CHECKS
# ============================================================

if np.min(Delta_N) <= 0:
    raise RuntimeError(
        "Synthetic Delta_N became non-positive."
    )

if np.min(Delta_W) <= 0:
    raise RuntimeError(
        "Synthetic Delta_W became non-positive."
    )

if not np.all(
    np.isfinite(Delta_N)
):
    raise RuntimeError(
        "Delta_N contains non-finite values."
    )

if not np.all(
    np.isfinite(Delta_W)
):
    raise RuntimeError(
        "Delta_W contains non-finite values."
    )

if not np.all(
    np.isfinite(v_N)
):
    raise RuntimeError(
        "v_N contains non-finite values."
    )

if not np.all(
    np.isfinite(v_W)
):
    raise RuntimeError(
        "v_W contains non-finite values."
    )


# ============================================================
# FOUR DENSITY-VELOCITY COMBINATIONS
#
# First letter  = density source
# Second letter = velocity source
#
# NN = N density + N velocity
# WW = W density + W velocity
# WN = W density + N velocity
# NW = N density + W velocity
# ============================================================

print()
print("=" * 72)
print("RUNNING FOUR-WAY DENSITY-VELOCITY SWAP")
print("=" * 72)


NN = make_lya_flux(
    Delta_N,
    v_N,
    z=z,
    Lbox_hMpc=Lbox_hMpc,
    target_mean_flux=target_mean_flux,
    T0=T0,
    gamma=gamma,
)


WW = make_lya_flux(
    Delta_W,
    v_W,
    z=z,
    Lbox_hMpc=Lbox_hMpc,
    target_mean_flux=target_mean_flux,
    T0=T0,
    gamma=gamma,
)


WN = make_lya_flux(
    Delta_W,
    v_N,
    z=z,
    Lbox_hMpc=Lbox_hMpc,
    target_mean_flux=target_mean_flux,
    T0=T0,
    gamma=gamma,
)


NW = make_lya_flux(
    Delta_N,
    v_W,
    z=z,
    Lbox_hMpc=Lbox_hMpc,
    target_mean_flux=target_mean_flux,
    T0=T0,
    gamma=gamma,
)


# ============================================================
# POWER RATIOS RELATIVE TO NN
# ============================================================

kv, R_WW = flux_power_ratio(
    WW,
    NN
)

_, R_WN = flux_power_ratio(
    WN,
    NN
)

_, R_NW = flux_power_ratio(
    NW,
    NN
)


# ============================================================
# INPUT FIELD SUMMARY
# ============================================================

print()
print("INPUT FIELD SUMMARY")
print("-" * 72)

print(
    f"Delta_N min/max            = "
    f"{Delta_N.min():.6f} / "
    f"{Delta_N.max():.6f}"
)

print(
    f"Delta_W min/max            = "
    f"{Delta_W.min():.6f} / "
    f"{Delta_W.max():.6f}"
)

print(
    f"v_N mean                   = "
    f"{np.mean(v_N):.6e} km/s"
)

print(
    f"v_W mean                   = "
    f"{np.mean(v_W):.6e} km/s"
)

print(
    f"v_N RMS                    = "
    f"{np.std(v_N):.6f} km/s"
)

print(
    f"v_W RMS                    = "
    f"{np.std(v_W):.6f} km/s"
)


# ============================================================
# MEAN-FLUX CALIBRATION
# ============================================================

print()
print("MEAN-FLUX CALIBRATION")
print("-" * 72)

for name, result in [
    ("NN", NN),
    ("WW", WW),
    ("WN", WN),
    ("NW", NW),
]:

    print(
        f"{name}: "
        f"A={result['A']:.12f}   "
        f"<F>={result['mean_flux']:.12f}"
    )


# ============================================================
# DETERMINISM TEST
#
# Repeat the NN calculation.
#
# The result should be bit-for-bit identical because the
# forward model itself contains no stochastic operations.
# ============================================================

NN_repeat = make_lya_flux(
    Delta_N,
    v_N,
    z=z,
    Lbox_hMpc=Lbox_hMpc,
    target_mean_flux=target_mean_flux,
    T0=T0,
    gamma=gamma,
)


flux_repeat_error = np.max(
    np.abs(
        NN_repeat["F"]
        - NN["F"]
    )
)


power_repeat_error = np.max(
    np.abs(
        NN_repeat["PF_kms"]
        - NN["PF_kms"]
    )
)


print()
print("DETERMINISM / REPEATABILITY")
print("-" * 72)

print(
    f"max |F_repeat-F|          = "
    f"{flux_repeat_error:.3e}"
)

print(
    f"max |P_repeat-P|          = "
    f"{power_repeat_error:.3e}"
)


# ============================================================
# SAVE NUMERICAL RESULTS
# ============================================================

output = np.column_stack([
    kv,

    NN["PF_kms"],
    WW["PF_kms"],
    WN["PF_kms"],
    NW["PF_kms"],

    R_WW,
    R_WN,
    R_NW,
])


np.savetxt(
    "density_velocity_swap_validation.txt",
    output,
    header=(
        "kv_s_per_km "
        "PF_NN "
        "PF_WW "
        "PF_WN "
        "PF_NW "
        "WW_over_NN "
        "WN_over_NN "
        "NW_over_NN"
    )
)


# ============================================================
# FIGURE
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(7.0, 6.5),
    sharex=True,
    gridspec_kw={
        "height_ratios": [
            1.5,
            1.0
        ],
        "hspace": 0.05
    }
)


# ============================================================
# TOP PANEL:
# ABSOLUTE DIMENSIONLESS 1D FLUX POWER
# ============================================================

ax1.loglog(
    NN["kv_s_per_km"],
    NN["DeltaF2"],
    linewidth=1.6,
    label="NN"
)

ax1.loglog(
    WW["kv_s_per_km"],
    WW["DeltaF2"],
    linewidth=1.6,
    label="WW"
)

ax1.loglog(
    WN["kv_s_per_km"],
    WN["DeltaF2"],
    linestyle="--",
    linewidth=1.5,
    label="WN"
)

ax1.loglog(
    NW["kv_s_per_km"],
    NW["DeltaF2"],
    linestyle="--",
    linewidth=1.5,
    label="NW"
)


ax1.set_ylabel(
    r"$\Delta_F^2(k_v)$"
)


ax1.legend(
    frameon=False,
    ncol=2
)


# ============================================================
# BOTTOM PANEL:
# RATIOS RELATIVE TO NN
# ============================================================

ax2.axhline(
    1.0,
    color="black",
    linestyle=":",
    linewidth=1.0
)


ax2.semilogx(
    kv,
    R_WW,
    linewidth=1.6,
    label=r"$WW/NN$"
)


ax2.semilogx(
    kv,
    R_WN,
    linestyle="--",
    linewidth=1.5,
    label=r"$WN/NN$"
)


ax2.semilogx(
    kv,
    R_NW,
    linestyle="-.",
    linewidth=1.5,
    label=r"$NW/NN$"
)


ax2.set_xlabel(
    r"$k_v\,[{\rm s\,km^{-1}}]$"
)


ax2.set_ylabel(
    r"$P_F/P_F^{NN}$"
)


ax2.legend(
    frameon=False,
    ncol=3,
    fontsize=8.5
)


# ------------------------------------------------------------
# Validation visualization range
#
# Do not interpret this as the final Ly-alpha science range.
# The final production range will be established through
# convergence tests using the actual simulations.
# ------------------------------------------------------------

ax2.set_xlim(
    2.0e-3,
    2.5e-1
)


# ============================================================
# PLOT STYLE
# ============================================================

for ax in (
    ax1,
    ax2
):

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3,
    )


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    "density_velocity_swap_validation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


print()
print("=" * 72)
print("SAVED")
print("=" * 72)

print(
    "density_velocity_swap_validation.png"
)

print(
    "density_velocity_swap_validation.txt"
)
