#!/usr/bin/env python3

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from lya_forward_model import (
    make_lya_flux,
    flux_power_ratio,
)


# ============================================================
# INPUT LOADER
# ============================================================

def load_skewers(filename):

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(filename)

    with np.load(filename) as f:

        if "Delta" not in f:
            raise KeyError(
                f"{filename}: missing 'Delta'"
            )

        if "vlos" not in f:
            raise KeyError(
                f"{filename}: missing 'vlos'"
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

    if Delta.shape != vlos.shape:
        raise ValueError(
            f"{filename}: Delta/vlos shape mismatch."
        )

    if Delta.ndim != 2:
        raise ValueError(
            "Arrays must be (Nskewer,Npix)."
        )

    if np.any(Delta <= 0):
        raise ValueError(
            f"{filename}: Delta contains non-positive values."
        )

    if not np.all(np.isfinite(Delta)):
        raise ValueError(
            f"{filename}: Delta contains NaN/Inf."
        )

    if not np.all(np.isfinite(vlos)):
        raise ValueError(
            f"{filename}: vlos contains NaN/Inf."
        )

    return Delta, vlos


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "IGM-parameter robustness grid for the "
        "four-way Ly-alpha density/velocity analysis."
    )
)

parser.add_argument(
    "--wave",
    required=True
)

parser.add_argument(
    "--proxy",
    required=True
)

parser.add_argument(
    "--z",
    required=True,
    type=float
)

parser.add_argument(
    "--Lbox",
    default=30.0,
    type=float
)

parser.add_argument(
    "--T0",
    nargs="+",
    required=True,
    type=float,
    help="List of T0 values in K."
)

parser.add_argument(
    "--gamma",
    nargs="+",
    required=True,
    type=float
)

parser.add_argument(
    "--mean-flux",
    nargs="+",
    required=True,
    type=float,
    dest="mean_flux"
)

parser.add_argument(
    "--outdir",
    default="igm_robustness"
)

args = parser.parse_args()


# ============================================================
# BASIC VALIDATION
# ============================================================

if args.z < 0:
    raise ValueError("z must be non-negative.")

if args.Lbox <= 0:
    raise ValueError("Lbox must be positive.")

for x in args.T0:
    if x <= 0:
        raise ValueError(
            "All T0 values must be positive."
        )

for x in args.mean_flux:
    if not (0.0 < x < 1.0):
        raise ValueError(
            "All mean-flux values must lie in (0,1)."
        )


# ============================================================
# LOAD MATCHED SKEWERS
# ============================================================

Delta_W, v_W = load_skewers(
    args.wave
)

Delta_N, v_N = load_skewers(
    args.proxy
)

if Delta_W.shape != Delta_N.shape:
    raise ValueError(
        "Wave and proxy arrays must have identical shapes."
    )

Nskewer, Npix = Delta_W.shape


# ============================================================
# PARAMETER GRID
# ============================================================

grid = list(
    itertools.product(
        args.T0,
        args.gamma,
        args.mean_flux
    )
)

print()
print("=" * 78)
print("IGM ROBUSTNESS GRID")
print("=" * 78)

print(
    f"z              = {args.z}"
)

print(
    f"Lbox           = {args.Lbox} Mpc/h"
)

print(
    f"Nskewer        = {Nskewer}"
)

print(
    f"Npix           = {Npix}"
)

print(
    f"N combinations = {len(grid)}"
)

print("=" * 78)


# ============================================================
# OUTPUT
# ============================================================

outdir = Path(
    args.outdir
)

outdir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# STORAGE
# ============================================================

records = []

kv_reference = None

ratio_WW_all = []
ratio_WN_all = []
ratio_NW_all = []


# ============================================================
# RUN GRID
# ============================================================

for i, (
    T0,
    gamma,
    mean_flux
) in enumerate(grid, start=1):

    print()
    print("-" * 78)

    print(
        f"[{i:03d}/{len(grid):03d}] "
        f"T0={T0:.1f} K, "
        f"gamma={gamma:.4f}, "
        f"<F>={mean_flux:.6f}"
    )

    # --------------------------------------------------------
    # NN
    # --------------------------------------------------------

    NN = make_lya_flux(
        Delta_N,
        v_N,
        z=args.z,
        Lbox_hMpc=args.Lbox,
        target_mean_flux=mean_flux,
        T0=T0,
        gamma=gamma,
    )

    # --------------------------------------------------------
    # WW
    # --------------------------------------------------------

    WW = make_lya_flux(
        Delta_W,
        v_W,
        z=args.z,
        Lbox_hMpc=args.Lbox,
        target_mean_flux=mean_flux,
        T0=T0,
        gamma=gamma,
    )

    # --------------------------------------------------------
    # WN
    # --------------------------------------------------------

    WN = make_lya_flux(
        Delta_W,
        v_N,
        z=args.z,
        Lbox_hMpc=args.Lbox,
        target_mean_flux=mean_flux,
        T0=T0,
        gamma=gamma,
    )

    # --------------------------------------------------------
    # NW
    # --------------------------------------------------------

    NW = make_lya_flux(
        Delta_N,
        v_W,
        z=args.z,
        Lbox_hMpc=args.Lbox,
        target_mean_flux=mean_flux,
        T0=T0,
        gamma=gamma,
    )

    # --------------------------------------------------------
    # Ratios
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Common k grid check
    # --------------------------------------------------------

    if kv_reference is None:

        kv_reference = kv.copy()

    else:

        if not np.allclose(
            kv,
            kv_reference,
            rtol=1e-12,
            atol=0.0
        ):
            raise RuntimeError(
                "k_v grid changed across IGM models."
            )

    ratio_WW_all.append(
        R_WW
    )

    ratio_WN_all.append(
        R_WN
    )

    ratio_NW_all.append(
        R_NW
    )

    # --------------------------------------------------------
    # Record metadata
    # --------------------------------------------------------

    record = {
        "index": i - 1,

        "T0_K": float(T0),

        "gamma": float(gamma),

        "target_mean_flux": float(
            mean_flux
        ),

        "A_NN": float(
            NN["A"]
        ),

        "A_WW": float(
            WW["A"]
        ),

        "A_WN": float(
            WN["A"]
        ),

        "A_NW": float(
            NW["A"]
        ),

        "mean_flux_NN": float(
            NN["mean_flux"]
        ),

        "mean_flux_WW": float(
            WW["mean_flux"]
        ),

        "mean_flux_WN": float(
            WN["mean_flux"]
        ),

        "mean_flux_NW": float(
            NW["mean_flux"]
        ),

        "mean_b_NN_kms": float(
            np.mean(
                NN["b_kms"]
            )
        ),

        "mean_b_WW_kms": float(
            np.mean(
                WW["b_kms"]
            )
        ),
    }

    records.append(
        record
    )

    print(
        f"  A_NN={NN['A']:.6f}  "
        f"A_WW={WW['A']:.6f}"
    )

    print(
        f"  <F>_NN={NN['mean_flux']:.12f}  "
        f"<F>_WW={WW['mean_flux']:.12f}"
    )


# ============================================================
# CONVERT STORAGE ARRAYS
# ============================================================

ratio_WW_all = np.asarray(
    ratio_WW_all
)

ratio_WN_all = np.asarray(
    ratio_WN_all
)

ratio_NW_all = np.asarray(
    ratio_NW_all
)


# ============================================================
# SAVE FULL GRID
# ============================================================

tag = (
    f"z{args.z:.3f}"
    .replace(".", "p")
)

npz_file = (
    outdir
    / f"igm_grid_{tag}.npz"
)

np.savez_compressed(
    npz_file,

    kv_s_per_km=kv_reference,

    ratio_WW_over_NN=ratio_WW_all,

    ratio_WN_over_NN=ratio_WN_all,

    ratio_NW_over_NN=ratio_NW_all,

    T0_K=np.asarray([
        r["T0_K"]
        for r in records
    ]),

    gamma=np.asarray([
        r["gamma"]
        for r in records
    ]),

    target_mean_flux=np.asarray([
        r["target_mean_flux"]
        for r in records
    ]),
)


# ============================================================
# SAVE JSON METADATA
# ============================================================

json_file = (
    outdir
    / f"igm_grid_{tag}.json"
)

metadata = {
    "z": float(args.z),

    "Lbox_hMpc": float(
        args.Lbox
    ),

    "Nskewer": int(
        Nskewer
    ),

    "Npix": int(
        Npix
    ),

    "wave_input": str(
        Path(args.wave).resolve()
    ),

    "proxy_input": str(
        Path(args.proxy).resolve()
    ),

    "records": records,
}

with open(
    json_file,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# ============================================================
# SIMPLE ROBUSTNESS SUMMARY
#
# No physical k-range cut is imposed here.
# The final trusted k_v range must come from numerical
# convergence of the actual simulation.
# ============================================================

summary_file = (
    outdir
    / f"igm_grid_summary_{tag}.txt"
)

with open(
    summary_file,
    "w"
) as f:

    f.write(
        "# IGM robustness grid\n"
    )

    f.write(
        "# No physical interpretation of the full k-range "
        "should be made before simulation convergence tests.\n"
    )

    f.write(
        "#\n"
    )

    f.write(
        "# index T0_K gamma mean_flux "
        "A_NN A_WW A_WN A_NW\n"
    )

    for r in records:

        f.write(
            f"{r['index']:4d} "
            f"{r['T0_K']:.8e} "
            f"{r['gamma']:.8f} "
            f"{r['target_mean_flux']:.8f} "
            f"{r['A_NN']:.10e} "
            f"{r['A_WW']:.10e} "
            f"{r['A_WN']:.10e} "
            f"{r['A_NW']:.10e}\n"
        )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 78)
print("GRID COMPLETE")
print("=" * 78)

print(
    npz_file
)

print(
    json_file
)

print(
    summary_file
)
