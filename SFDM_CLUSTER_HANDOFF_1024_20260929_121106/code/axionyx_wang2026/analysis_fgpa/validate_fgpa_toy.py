import numpy as np
import matplotlib.pyplot as plt

from fgpa_core import (
    Hubble,
    fgpa_beta,
    temperature_from_density,
    thermal_b_kms,
    raw_fgpa_tau,
    calibrate_mean_flux,
    transmitted_flux,
    velocity_box_length,
    flux_power_1d,
    dimensionless_flux_power,
    kvelocity_to_kh,
)


# ============================================================
# TEST SETUP
# ============================================================

z = 3.0

Lbox_hMpc = 30.0

Npix = 1024
Nskewer = 128

gamma = 1.55
T0 = 1.0e4

# ONLY FOR THIS SOFTWARE TEST.
# This is not yet our adopted observational mean flux.
target_mean_flux = 0.70


# ============================================================
# VELOCITY GRID
# ============================================================

Vmax = velocity_box_length(
    Lbox_hMpc,
    z
)

du = Vmax / Npix

u = (
    np.arange(Npix)
    + 0.5
) * du


# ============================================================
# SYNTHETIC PERIODIC DENSITY FIELDS
#
# Deliberately simple and fully controlled.
# ============================================================

rng = np.random.default_rng(12345)

Delta = np.empty(
    (Nskewer, Npix)
)

for s in range(Nskewer):

    phase1 = rng.uniform(
        0.0,
        2.0*np.pi
    )

    phase2 = rng.uniform(
        0.0,
        2.0*np.pi
    )

    delta = (
        0.25
        * np.sin(
            2.0*np.pi*6*u/Vmax
            + phase1
        )
        +
        0.10
        * np.sin(
            2.0*np.pi*20*u/Vmax
            + phase2
        )
    )

    Delta[s] = 1.0 + delta


# ============================================================
# THERMODYNAMICS
# ============================================================

T = temperature_from_density(
    Delta,
    T0=T0,
    gamma=gamma
)

b = thermal_b_kms(T)


# ============================================================
# SIMPLE REAL-SPACE FGPA
#
# NO peculiar velocities yet.
# NO thermal convolution yet.
#
# This phase validates only:
#
# density -> tau
# mean-flux calibration
# flux
# FFT normalization
# ============================================================

tau_raw = raw_fgpa_tau(
    Delta,
    gamma=gamma
)

A = calibrate_mean_flux(
    tau_raw,
    target_mean_flux
)

F = transmitted_flux(
    tau_raw,
    A
)


# ============================================================
# FLUX POWER
# ============================================================

kv, PF, PF_std = flux_power_1d(
    F,
    du
)

DeltaF2 = dimensionless_flux_power(
    kv,
    PF
)

kh = kvelocity_to_kh(
    kv,
    z
)


# ============================================================
# NUMERICAL CHECKS
# ============================================================

print()
print("=" * 65)
print("FGPA TOY VALIDATION")
print("=" * 65)

print(
    f"z                     = {z:.1f}"
)

print(
    f"H(z)                  = "
    f"{Hubble(z):.6f} km/s/Mpc"
)

print(
    f"L_box                 = "
    f"{Lbox_hMpc:.3f} Mpc/h"
)

print(
    f"velocity box Vmax     = "
    f"{Vmax:.6f} km/s"
)

print(
    f"velocity pixel du      = "
    f"{du:.6f} km/s"
)

print(
    f"gamma                  = "
    f"{gamma:.4f}"
)

print(
    f"FGPA beta              = "
    f"{fgpa_beta(gamma):.6f}"
)

print(
    f"T0                     = "
    f"{T0:.1f} K"
)

print(
    f"mean b                 = "
    f"{np.mean(b):.6f} km/s"
)

print(
    f"calibrated A           = "
    f"{A:.12f}"
)

print(
    f"target mean flux       = "
    f"{target_mean_flux:.12f}"
)

print(
    f"measured mean flux     = "
    f"{np.mean(F):.12f}"
)

print(
    f"|difference|           = "
    f"{abs(np.mean(F)-target_mean_flux):.3e}"
)

print(
    f"fundamental kv         = "
    f"{kv[0]:.8e} s/km"
)

print(
    f"fundamental k          = "
    f"{kh[0]:.8f} h/Mpc"
)

print("=" * 65)


# ============================================================
# FIGURE
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(7.0, 6.0),
    gridspec_kw={
        "height_ratios": [1.0, 1.0],
        "hspace": 0.30
    }
)


# First skewer
ax1.plot(
    u,
    Delta[0],
    label=r"$\Delta_b$"
)

ax1.plot(
    u,
    F[0],
    label=r"$F$"
)

ax1.set_xlabel(
    r"$u\,[{\rm km\,s^{-1}}]$"
)

ax1.set_ylabel(
    "Field value"
)

ax1.legend(
    frameon=False
)


# Flux power
ax2.loglog(
    kh,
    DeltaF2
)

ax2.set_xlabel(
    r"$k\,[h\,{\rm Mpc}^{-1}]$"
)

ax2.set_ylabel(
    r"$\Delta_F^2(k)=k_vP_F(k_v)/\pi$"
)

for ax in (ax1, ax2):
    ax.tick_params(
        direction="in",
        top=True,
        right=True
    )

fig.tight_layout()

fig.savefig(
    "fgpa_toy_validation.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close(fig)

print()
print(
    "Saved: fgpa_toy_validation.png"
)
