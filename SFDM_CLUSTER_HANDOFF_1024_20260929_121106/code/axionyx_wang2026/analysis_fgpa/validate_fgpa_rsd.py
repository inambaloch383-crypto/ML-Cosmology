import numpy as np
import matplotlib.pyplot as plt

from fgpa_core import (
    velocity_box_length,
    raw_fgpa_tau,
    calibrate_mean_flux,
    transmitted_flux,
    flux_power_1d,
    dimensionless_flux_power,
)

from fgpa_rsd import (
    tau_velocity_thermal,
    integrated_tau,
)


# ============================================================
# TEST PARAMETERS
# ============================================================

z = 3.0

Lbox_hMpc = 30.0

Npix = 1024
Nskewer = 16

T0 = 1.0e4
gamma = 1.55

target_mean_flux = 0.70

rng = np.random.default_rng(
    12345
)


# ============================================================
# VELOCITY GRID
# ============================================================

Vbox = velocity_box_length(
    Lbox_hMpc,
    z
)

du = Vbox / Npix

u = (
    np.arange(Npix)
    + 0.5
) * du


# ============================================================
# SYNTHETIC DENSITY + PECULIAR VELOCITY
# ============================================================

Delta = np.empty(
    (Nskewer, Npix)
)

vpec = np.empty_like(
    Delta
)

for s in range(Nskewer):

    p1 = rng.uniform(
        0.0,
        2.0*np.pi
    )

    p2 = rng.uniform(
        0.0,
        2.0*np.pi
    )

    # Density
    delta = (
        0.25
        * np.sin(
            2.0*np.pi*6*u/Vbox
            + p1
        )
        +
        0.10
        * np.sin(
            2.0*np.pi*20*u/Vbox
            + p2
        )
    )

    Delta[s] = (
        1.0 + delta
    )

    # Synthetic LOS peculiar velocity.
    #
    # Cosine gives the expected phase offset relative
    # to the density sine modes.
    vpec[s] = (
        40.0
        * np.cos(
            2.0*np.pi*6*u/Vbox
            + p1
        )
        +
        15.0
        * np.cos(
            2.0*np.pi*20*u/Vbox
            + p2
        )
    )


# ============================================================
# RAW REAL-SPACE FGPA FIELD
# ============================================================

tau_raw = raw_fgpa_tau(
    Delta,
    gamma=gamma
)


# ============================================================
# CASE 1:
# thermal broadening, no peculiar velocity
# ============================================================

vzero = np.zeros_like(
    vpec
)

tau_real, T, b = tau_velocity_thermal(
    Delta,
    vzero,
    du,
    T0=T0,
    gamma=gamma,
)


# ============================================================
# CASE 2:
# constant integer-pixel velocity translation
#
# A periodic constant translation must preserve the
# entire optical-depth field up to a grid roll.
# ============================================================

shift_pixels = 17

vshift_value = (
    shift_pixels
    * du
)

vshift = np.full_like(
    vpec,
    vshift_value
)

tau_shift, _, _ = tau_velocity_thermal(
    Delta,
    vshift,
    du,
    T0=T0,
    gamma=gamma,
)


# ============================================================
# CASE 3:
# spatially varying peculiar velocity
# ============================================================

tau_rsd, _, _ = tau_velocity_thermal(
    Delta,
    vpec,
    du,
    T0=T0,
    gamma=gamma,
)


# ============================================================
# CONSERVATION TEST
#
# Thermal broadening + redshift-space remapping must
# conserve integrated raw optical-depth weight.
# ============================================================

I_raw = integrated_tau(
    tau_raw,
    du
)

I_real = integrated_tau(
    tau_real,
    du
)

I_rsd = integrated_tau(
    tau_rsd,
    du
)

err_real = np.max(
    np.abs(
        I_real / I_raw - 1.0
    )
)

err_rsd = np.max(
    np.abs(
        I_rsd / I_raw - 1.0
    )
)


# ============================================================
# CONSTANT-TRANSLATION TEST
# ============================================================

expected_shift = np.roll(
    tau_real,
    shift_pixels,
    axis=1
)

translation_error = (
    np.max(
        np.abs(
            tau_shift
            - expected_shift
        )
    )
    /
    np.max(
        np.abs(
            expected_shift
        )
    )
)


# ============================================================
# SAME-MEAN-FLUX CALIBRATION
# ============================================================

A_real = calibrate_mean_flux(
    tau_real,
    target_mean_flux
)

A_rsd = calibrate_mean_flux(
    tau_rsd,
    target_mean_flux
)

F_real = transmitted_flux(
    tau_real,
    A_real
)

F_rsd = transmitted_flux(
    tau_rsd,
    A_rsd
)


# ============================================================
# 1D FLUX POWER
# ============================================================

kv_real, PF_real, _ = flux_power_1d(
    F_real,
    du
)

kv_rsd, PF_rsd, _ = flux_power_1d(
    F_rsd,
    du
)

DF2_real = dimensionless_flux_power(
    kv_real,
    PF_real
)

DF2_rsd = dimensionless_flux_power(
    kv_rsd,
    PF_rsd
)


# ============================================================
# PRINT VALIDATION RESULTS
# ============================================================

print()
print("=" * 70)
print("FGPA PHASE B: RSD + THERMAL VALIDATION")
print("=" * 70)

print(
    f"Nskewer                       = "
    f"{Nskewer}"
)

print(
    f"Npix                          = "
    f"{Npix}"
)

print(
    f"du                            = "
    f"{du:.8f} km/s"
)

print(
    f"mean thermal b                = "
    f"{np.mean(b):.8f} km/s"
)

print(
    f"min thermal b                 = "
    f"{np.min(b):.8f} km/s"
)

print(
    f"max thermal b                 = "
    f"{np.max(b):.8f} km/s"
)

print()

print(
    f"tau conservation, v=0        = "
    f"{err_real:.3e}"
)

print(
    f"tau conservation, RSD        = "
    f"{err_rsd:.3e}"
)

print(
    f"constant-shift relative error = "
    f"{translation_error:.3e}"
)

print()

print(
    f"A no-RSD                      = "
    f"{A_real:.12f}"
)

print(
    f"A with RSD                    = "
    f"{A_rsd:.12f}"
)

print(
    f"<F> no-RSD                    = "
    f"{np.mean(F_real):.12f}"
)

print(
    f"<F> with RSD                  = "
    f"{np.mean(F_rsd):.12f}"
)

print(
    f"target <F>                    = "
    f"{target_mean_flux:.12f}"
)

print("=" * 70)


# ============================================================
# VALIDATION FIGURE
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(7.0, 6.2),
    gridspec_kw={
        "height_ratios": [
            1.0,
            1.0
        ],
        "hspace": 0.28
    }
)


# ------------------------------------------------------------
# Flux along first synthetic skewer
# ------------------------------------------------------------

ax1.plot(
    u,
    F_real[0],
    linewidth=0.5,
    label="thermal only"
)

ax1.plot(
    u,
    F_rsd[0],
    linewidth=0.5,
    label="thermal + peculiar velocity"
)

ax1.set_ylabel(
    r"$F$"
)

ax1.set_xlabel(
    r"$u\,[{\rm km\,s^{-1}}]$"
)

ax1.legend(
    frameon=False
)


# ------------------------------------------------------------
# Flux power
#
# Keep observational velocity-space coordinate here.
# ------------------------------------------------------------

#ax2.loglog(
    #kv_real,
    #DF2_real,
    #linewidth=1.5,
    #label="thermal only"


#ax2.loglog(
    #kv_rsd,
   # DF2_rsd,
    #linewidth=1.5,
   # label="thermal + peculiar velocity"

ax2.set_xlabel(
    r"$k_v\,[{\rm s\,km^{-1}}]$"
)

ax2.set_ylabel(
    r"$\Delta_F^2(k_v)$"
)

ax2.legend(
    frameon=False
)


for ax in (
    ax1,
    ax2
):

    ax.tick_params(
        direction="in",
        top=True,
        right=True
    )


fig.tight_layout()

fig.savefig(
    "fgpa_rsd_thermal_validation.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close(fig)


print()
print(
    "Saved: "
    "fgpa_rsd_thermal_validation.png"
)
