#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

command -v sbatch >/dev/null 2>&1 || {
    echo "ERROR: sbatch not found."
    exit 2
}

echo "Submitting wave MUSIC job..."
jid_wave=$(sbatch --parsable 04_music_wave_1024.slurm)
echo "wave job = $jid_wave"

echo "Submitting baryon MUSIC job..."
jid_baryon=$(sbatch --parsable 05_music_baryon_1024.slurm)
echo "baryon job = $jid_baryon"

echo "Submitting AxioNyx phase job after BOTH MUSIC jobs succeed..."
jid_nyx=$(sbatch --parsable \
    --dependency="afterok:${jid_wave}:${jid_baryon}" \
    06_axionyx_phasecheck_1024.slurm)
echo "AxioNyx job = $jid_nyx"

echo
echo "Submitted chain:"
echo "  MUSIC wave   : $jid_wave"
echo "  MUSIC baryon : $jid_baryon"
echo "  AxioNyx      : $jid_nyx"
