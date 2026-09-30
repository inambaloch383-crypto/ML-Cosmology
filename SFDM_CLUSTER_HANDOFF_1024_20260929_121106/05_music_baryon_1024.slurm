#!/bin/bash
#SBATCH --job-name=MUSIC_baryon1024
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=music_baryon1024_%j.out
#SBATCH --error=music_baryon1024_%j.err

# ADMIN: add cluster-specific --partition/--account/--qos/--mem directives.
# 1024^3 MUSIC may require a high-memory node.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/SITE_CONFIG.env"
[[ "$PROJECT_ROOT" == "AUTO" ]] && PROJECT_ROOT="$HERE"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

MUSIC="$PROJECT_ROOT/code/MUSIC_wang/MUSIC"
CONF="$PROJECT_ROOT/run_1024/music_baryon_particles_FINAL1024.conf"
LOG="$PROJECT_ROOT/run_1024/music_baryon_FINAL1024.log"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

[[ -x "$MUSIC" ]] || { echo "Missing executable: $MUSIC"; exit 2; }
[[ -f "$CONF" ]] || { echo "Missing config: $CONF"; exit 2; }

echo "Host: $(hostname)"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
free -h || true
df -h "$PROJECT_ROOT" || true

cd "$PROJECT_ROOT/code/MUSIC_wang"

srun --ntasks=1 --cpus-per-task="$OMP_NUM_THREADS" \
    "$MUSIC" "$CONF" 2>&1 | tee "$LOG"

test "${PIPESTATUS[0]}" -eq 0

echo
echo "Baryon IC output:"
du -sh "$PROJECT_ROOT/run_1024/IC_BARYON_PARTICLES_FINAL1024" || true
