#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/SITE_CONFIG.env"

if [[ "${PROJECT_ROOT}" == "AUTO" ]]; then
    PROJECT_ROOT="$HERE"
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

MUSIC_DIR="$PROJECT_ROOT/code/MUSIC_wang"
BOXLIB_HOME="$PROJECT_ROOT/code/BoxLib_legacy"
AXIONYX_DIR="$PROJECT_ROOT/code/axionyx_wang2026"
AXIONYX_EXEC_DIR="$AXIONYX_DIR/Exec/MDM_NoHydro"

echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "MUSIC_DIR    = $MUSIC_DIR"
echo "BOXLIB_HOME  = $BOXLIB_HOME"
echo "AXIONYX_DIR  = $AXIONYX_DIR"

for x in g++ gfortran make mpicxx; do
    command -v "$x" >/dev/null 2>&1 || {
        echo "ERROR: $x is not in PATH. Load the cluster compiler/MPI modules first."
        exit 2
    }
done

for d in "$MUSIC_DIR" "$BOXLIB_HOME" "$AXIONYX_EXEC_DIR"; do
    [[ -d "$d" ]] || { echo "ERROR: missing directory $d"; exit 2; }
done

mkdir -p "$PROJECT_ROOT/build_logs"

echo
echo "============================================================"
echo "BUILDING MUSIC"
echo "============================================================"
cd "$MUSIC_DIR"

# The user's modified MUSIC Nyx output plugin needs legacy BoxLib.
# The site's FFTW/GSL modules should already be loaded by the administrator.
make clean >/dev/null 2>&1 || true

set +e
make -j"${BUILD_JOBS_MUSIC}" \
    BOXLIB_HOME="$BOXLIB_HOME" \
    FFTW3=yes SINGLE=no \
    2>&1 | tee "$PROJECT_ROOT/build_logs/music_build.log"
music_status=${PIPESTATUS[0]}
set -e

if [[ $music_status -ne 0 || ! -x "$MUSIC_DIR/MUSIC" ]]; then
    echo
    echo "MUSIC BUILD DID NOT COMPLETE."
    echo "Please inspect build_logs/music_build.log."
    echo "Common site-specific issue: the MUSIC Makefile may contain old local"
    echo "include/library paths. Replace them with the cluster's FFTW/GSL module paths,"
    echo "but do not replace the modified source files."
    exit 3
fi

echo "MUSIC executable: $MUSIC_DIR/MUSIC"

echo
echo "============================================================"
echo "BUILDING AXIONYX (NO_HYDRO + FDM + PARTICLES, MPI+OpenMP)"
echo "============================================================"
cd "$AXIONYX_EXEC_DIR"

make clean >/dev/null 2>&1 || true

set +e
make -j"${BUILD_JOBS_AXIONYX}" \
    COMP=gnu USE_MPI=TRUE USE_OMP=TRUE \
    2>&1 | tee "$PROJECT_ROOT/build_logs/axionyx_build.log"
ax_status=${PIPESTATUS[0]}
set -e

if [[ $ax_status -ne 0 || ! -x "$AXIONYX_EXEC_DIR/Nyx3d.gnu.MPI.OMP.ex" ]]; then
    echo
    echo "AXIONYX BUILD DID NOT COMPLETE."
    echo "Please inspect build_logs/axionyx_build.log."
    echo "The local validated build used GNU, MPI, OpenMP, C++20 and FFTW."
    echo "If the site's AMReX make variables differ, please adapt the build command"
    echo "while preserving the included source tree and GNUmakefile physics flags."
    exit 4
fi

echo "AxioNyx executable: $AXIONYX_EXEC_DIR/Nyx3d.gnu.MPI.OMP.ex"

echo
echo "BUILD COMPLETE."
