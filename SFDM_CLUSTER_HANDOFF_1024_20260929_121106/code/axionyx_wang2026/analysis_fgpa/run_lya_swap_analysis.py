#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from lya_forward_model import (
    make_lya_flux,
    flux_power_ratio,
)


# ============================================================
# INPUT READER
# ============================================================

def load_skewers(filename):
    """
    Required NPZ arrays:

        Delta : rho_b / <rho_b>
        vlos  : physical LOS peculiar velocity [km/s]

    Shape:
        (Nskewer, Npix)

    1D arrays are accepted and promoted to one skewer.
    """

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(filename)

    with np.load(filename) as f:

        if "Delta" not in f:
            raise KeyError(
                f"{filename}: missing array 'Delta'"
            )

        if "vlos" not in f:
            raise KeyError(
                f"{filename}: missing array 'vlos'"
            )

        Delta = np.asarray(
            f["Delta"],
            dtype=float
        )

        vlos = np.asarray(
            f["vlos"],
            dtype=float
        )

    if Delta.ndim == 1:
        Delta = Delta[None, :]

    if vlos.ndim == 1:
        vlos = vlos[None, :]

    if Delta.ndim != 2:
        raise ValueError(
            f"{filename}: Delta must be 1D or 2D."
        )

    if vlos.ndim != 2:
        raise ValueError(
            f"{filename}: vlos must be 1D or 2D."
        )

    if Delta.shape != vlos.shape:
        raise ValueError(
            f"{filename}: Delta and vlos shapes differ: "
            f"{Delta.shape} versus {vlos.shape}"
        )

    if not np.all(np.isfinite(Delta)):
        raise ValueError(
            f"{filename}: Delta contains NaN/Inf."
        )

    if not np.all(np.isfinite(vlos)):
        raise ValueError(
            f"{filename}: vlos contains NaN/Inf."
        )

    if np.any(Delta <= 0):
        raise ValueError(
            f"{filename}: Delta must be strictly positive."
        )

    return Delta, vlos


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Four-way Ly-alpha FGPA density/velocity "
        "swap analysis."
    )
)

parser.add_argument(
    "--wave",
    required=True,
    help=(
        "NPZ containing wave arrays "
        "'Delta' and 'vlos'."
    )
)

parser.add_argument(
    "--proxy",
    required=True,
    help=(
        "NPZ containing proxy arrays "
        "'Delta' and 'vlos'."
    )
)

parser.add_argument(
    "--z",
    required=True,
    type=float,
    help="Snapshot redshift."
)

parser.add_argument(
    "--Lbox",
    default=30.0,
    type=float,
    help="Comoving box length in Mpc/h."
)

parser.add_argument(
    "--mean-flux",
    required=True,
    type=float,
    dest="mean_flux",
    help="Target mean transmitted flux."
)

parser.add_argument(
    "--T0",
    default=1.0e4,
    type=float,
    help="Temperature at mean density [K]."
)

parser.add_argument(
    "--gamma",
    default=1.55,
    type=float,
    help="Temperature-density slope."
)

parser.add_argument(
    "--outdir",
    default="lya_results",
    help="Output directory."
)

args = parser.parse_args()


# ============================================================
# GLOBAL INPUT VALIDATION
# ============================================================

if args.z < 0:
    raise ValueError("z must be non-negative.")

if args.Lbox <= 0:
    raise ValueError("Lbox must be positive.")

if not (0.0 < args.mean_flux < 1.0):
    raise ValueError(
        "mean-flux must lie strictly between 0 and 1."
    )

if args.T0 <= 0:
    raise ValueError("T0 must be positive.")


# ============================================================
# LOAD DATA
# ============================================================

Delta_W, v_W = load_skewers(
    args.wave
)

Delta_N, v_N = load_skewers(
    args.proxy
)


if Delta_W.shape != Delta_N.shape:
    raise ValueError(
        "Wave and proxy skewer arrays must have "
        "identical shapes for the matched comparison.\n"
        f"wave  = {Delta_W.shape}\n"
        f"proxy = {Delta_N.shape}"
    )


Nskewer, Npix = Delta_W.shape


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

outdir = Path(
    args.outdir
)

outdir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# RUN FOUR COMBINATIONS
#
# First letter = density
# Second letter = velocity
# ============================================================

print()
print("=" * 74)
print("LYMAN-ALPHA FOUR-WAY DENSITY / VELOCITY ANALYSIS")
print("=" * 74)

print(f"z                = {args.z}")
print(f"Lbox             = {args.Lbox} Mpc/h")
print(f"Nskewer          = {Nskewer}")
print(f"Npix             = {Npix}")
print(f"T0               = {args.T0} K")
print(f"gamma            = {args.gamma}")
print(f"target <F>       = {args.mean_flux}")
print()


def run_case(name, Delta, velocity):

    print(
        f"Running {name} ..."
    )

    return make_lya_flux(
        Delta,
        velocity,
        z=args.z,
        Lbox_hMpc=args.Lbox,
        target_mean_flux=args.mean_flux,
        T0=args.T0,
        gamma=args.gamma,
    )


NN = run_case(
    "NN",
    Delta_N,
    v_N
)

WW = run_case(
    "WW",
    Delta_W,
    v_W
)

WN = run_case(
    "WN",
    Delta_W,
    v_N
)

NW = run_case(
    "NW",
    Delta_N,
    v_W
)


# ============================================================
# CHECK COMMON k GRID
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
# SAVE POWER TABLE
# ============================================================

tag = f"z{args.z:.3f}".replace(
    ".",
    "p"
)

power_file = (
    outdir
    / f"flux_power_swap_{tag}.txt"
)


table = np.column_stack([
    kv,

    NN["PF_kms"],
    WW["PF_kms"],
    WN["PF_kms"],
    NW["PF_kms"],

    NN["DeltaF2"],
    WW["DeltaF2"],
    WN["DeltaF2"],
    NW["DeltaF2"],

    R_WW,
    R_WN,
    R_NW,
])


np.savetxt(
    power_file,
    table,
    header=(
        "kv_s_per_km "
        "PF_NN "
        "PF_WW "
        "PF_WN "
        "PF_NW "
        "DeltaF2_NN "
        "DeltaF2_WW "
        "DeltaF2_WN "
        "DeltaF2_NW "
        "WW_over_NN "
        "WN_over_NN "
        "NW_over_NN"
    )
)


# ============================================================
# SAVE PER-CASE FLUX ARRAYS
#
# Useful later for diagnostics without recomputing FGPA.
# ============================================================

flux_file = (
    outdir
    / f"flux_fields_{tag}.npz"
)

np.savez_compressed(
    flux_file,

    F_NN=NN["F"],
    F_WW=WW["F"],
    F_WN=WN["F"],
    F_NW=NW["F"],

    tau_NN=NN["tau"],
    tau_WW=WW["tau"],
    tau_WN=WN["tau"],
    tau_NW=NW["tau"],

    u_kms=NN["u_kms"],
    kv_s_per_km=kv,
)


# ============================================================
# SUMMARY INFORMATION
# ============================================================

summary = {
    "z": float(args.z),
    "Lbox_hMpc": float(args.Lbox),

    "Nskewer": int(Nskewer),
    "Npix": int(Npix),

    "T0_K": float(args.T0),
    "gamma": float(args.gamma),

    "target_mean_flux": float(
        args.mean_flux
    ),

    "wave_input": str(
        Path(args.wave).resolve()
    ),

    "proxy_input": str(
        Path(args.proxy).resolve()
    ),

    "velocity_box_kms": float(
        NN["Vbox_kms"]
    ),

    "du_kms": float(
        NN["du_kms"]
    ),

    "cases": {}
}


for name, result in [
    ("NN", NN),
    ("WW", WW),
    ("WN", WN),
    ("NW", NW),
]:

    summary["cases"][name] = {
        "A": float(
            result["A"]
        ),

        "mean_flux": float(
            result["mean_flux"]
        ),

        "mean_temperature_K": float(
            np.mean(
                result["temperature_K"]
            )
        ),

        "mean_b_kms": float(
            np.mean(
                result["b_kms"]
            )
        ),
    }


summary_file = (
    outdir
    / f"summary_{tag}.json"
)

with open(
    summary_file,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FIGURE
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(7.0, 6.4),
    sharex=True,
    gridspec_kw={
        "height_ratios": [
            1.5,
            1.0
        ],
        "hspace": 0.05
    }
)


# ------------------------------------------------------------
# Top panel
# ------------------------------------------------------------

ax1.loglog(
    kv,
    NN["DeltaF2"],
    linewidth=1.7,
    label="NN"
)

ax1.loglog(
    kv,
    WW["DeltaF2"],
    linewidth=1.7,
    label="WW"
)

ax1.loglog(
    kv,
    WN["DeltaF2"],
    linestyle="--",
    linewidth=1.5,
    label="WN"
)

ax1.loglog(
    kv,
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


# ------------------------------------------------------------
# Bottom panel
# ------------------------------------------------------------

ax2.axhline(
    1.0,
    color="black",
    linestyle=":",
    linewidth=1.0
)


ax2.semilogx(
    kv,
    R_WW,
    linewidth=1.7,
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


figure_file = (
    outdir
    / f"flux_power_swap_{tag}.png"
)

fig.savefig(
    figure_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print()
print("=" * 74)
print("MEAN-FLUX / NORMALIZATION SUMMARY")
print("=" * 74)

for name, result in [
    ("NN", NN),
    ("WW", WW),
    ("WN", WN),
    ("NW", NW),
]:

    print(
        f"{name}: "
        f"A={result['A']:.10f}   "
        f"<F>={result['mean_flux']:.12f}   "
        f"<b>={np.mean(result['b_kms']):.4f} km/s"
    )


print()
print("=" * 74)
print("OUTPUT")
print("=" * 74)

print(power_file)
print(flux_file)
print(summary_file)
print(figure_file)
