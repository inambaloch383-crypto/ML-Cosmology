#!/bin/bash
#SBATCH --job-name=AxioNyx_phase1024
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=axionyx_phase1024_%j.out
#SBATCH --error=axionyx_phase1024_%j.err

# ADMIN: THIS RESOURCE LAYOUT IS ONLY A STARTING TEMPLATE.
# Tune nodes/ranks/threads/memory/partition/account for the local machine.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/SITE_CONFIG.env"
[[ "$PROJECT_ROOT" == "AUTO" ]] && PROJECT_ROOT="$HERE"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

RUN_DIR="$PROJECT_ROOT/run_1024"
NYX="$PROJECT_ROOT/code/axionyx_wang2026/Exec/MDM_NoHydro/Nyx3d.gnu.MPI.OMP.ex"
INPUT="$RUN_DIR/inputs_sfdm1024_PHASECHECK"
LOG="$RUN_DIR/run_PHASE1024.log"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

[[ -x "$NYX" ]] || { echo "Missing executable: $NYX"; exit 2; }
[[ -f "$INPUT" ]] || { echo "Missing input: $INPUT"; exit 2; }
[[ -d "$RUN_DIR/IC_SFDM_WAVE_FINAL1024" ]] || { echo "Missing wave IC"; exit 2; }
[[ -d "$RUN_DIR/IC_BARYON_PARTICLES_FINAL1024" ]] || { echo "Missing baryon IC"; exit 2; }

echo "Nodes=${SLURM_JOB_NUM_NODES:-NA}"
echo "MPI tasks=${SLURM_NTASKS:-NA}"
echo "OMP threads=$OMP_NUM_THREADS"
df -h "$RUN_DIR" || true

cd "$RUN_DIR"

srun "$NYX" "$INPUT" 2>&1 | tee "$LOG"
test "${PIPESTATUS[0]}" -eq 0

echo
echo "AxioNyx summary:"
grep -niE 'ERROR|FATAL|abort|NaN|Using MUSIC GENERATED AXION PHASE|PLOTFILE|Advanced|STEP' \
    "$LOG" | tail -150 || true

echo
echo "Produced plotfiles:"
find "$RUN_DIR" -maxdepth 1 -type d -name 'plt_PHASE1024_*' -print | sort

echo
echo "SUCCESS CONDITION FOR HANDOFF STAGE:"
echo "Preserve plt_PHASE1024_00000 for the user's direct phase-alias diagnostic."
