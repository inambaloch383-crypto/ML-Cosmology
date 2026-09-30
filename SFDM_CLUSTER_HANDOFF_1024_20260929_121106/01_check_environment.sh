#!/usr/bin/env bash
set -u

echo "============================================================"
echo "SFDM cluster environment report"
echo "Date: $(date -Is 2>/dev/null || date)"
echo "Host: $(hostname)"
echo "============================================================"

echo
echo "== Scheduler =="
command -v sbatch || true
command -v srun || true
command -v scontrol || true

echo
echo "== Compiler / build tools =="
for x in gcc g++ gfortran make mpicc mpicxx mpifort python3 pkg-config; do
    printf "%-12s : " "$x"
    command -v "$x" || true
done

echo
echo "== Versions =="
gcc --version 2>/dev/null | head -1 || true
g++ --version 2>/dev/null | head -1 || true
gfortran --version 2>/dev/null | head -1 || true
mpicxx --version 2>/dev/null | head -3 || true
python3 --version 2>/dev/null || true

echo
echo "== pkg-config libraries =="
if command -v pkg-config >/dev/null 2>&1; then
    for p in fftw3 fftw3f gsl zlib; do
        if pkg-config --exists "$p" 2>/dev/null; then
            echo "$p : FOUND  $(pkg-config --modversion "$p" 2>/dev/null)"
        else
            echo "$p : not visible through pkg-config (may still be provided by modules)"
        fi
    done
fi

echo
echo "== Shared libraries visible =="
if command -v ldconfig >/dev/null 2>&1; then
    ldconfig -p 2>/dev/null | grep -Ei 'fftw3|gsl|libmpi|libgfortran|libz\.so' | head -100 || true
fi

echo
echo "== Python validation =="
python3 - <<'PY' 2>/dev/null || true
import sys
print("Python:", sys.version.replace("\n"," "))
try:
    import numpy
    print("NumPy:", numpy.__version__)
except Exception as e:
    print("NumPy: MISSING", e)
PY

echo
echo "== CPU / memory =="
lscpu 2>/dev/null | grep -E 'Architecture|CPU\(s\)|Thread|Core|Socket|NUMA|Model name' || true
free -h 2>/dev/null || true

echo
echo "== Filesystems =="
df -h . "$HOME" 2>/dev/null || true

echo
echo "== Slurm summary =="
sinfo -o '%P %a %l %D %c %m %G' 2>/dev/null | head -50 || true

echo
echo "== Current module list =="
module list 2>&1 || true

echo
echo "============================================================"
echo "Minimum requested stack:"
echo "GCC/G++/GFortran, MPI, make, FFTW3(+OpenMP), GSL, zlib,"
echo "Python3+NumPy, Slurm."
echo "============================================================"
