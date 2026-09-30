#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/SITE_CONFIG.env"

if [[ "${PROJECT_ROOT}" == "AUTO" ]]; then
    PROJECT_ROOT="$HERE"
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

RUN_DIR="$PROJECT_ROOT/run_1024"
TRANSFER_DIR="$PROJECT_ROOT/data/transfer"
TEMPLATE_DIR="$PROJECT_ROOT/config_templates"

mkdir -p "$RUN_DIR"

for f in \
    "$TRANSFER_DIR/sfdm_m215_z120_WAVE_FINAL.dat" \
    "$TRANSFER_DIR/sfdm_m215_z120_BARYON_PARTICLES_FINAL.dat" \
    "$TEMPLATE_DIR/music_sfdm_wave_FINAL1024.conf.template" \
    "$TEMPLATE_DIR/music_baryon_particles_FINAL1024.conf.template" \
    "$TEMPLATE_DIR/inputs_sfdm1024_PHASECHECK.template"
do
    [[ -e "$f" ]] || { echo "ERROR: missing $f"; exit 2; }
done

python3 - "$PROJECT_ROOT" "$RUN_DIR" "$TRANSFER_DIR" "$TEMPLATE_DIR" <<'PY'
from pathlib import Path
import sys

project_root = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
transfer_dir = Path(sys.argv[3])
template_dir = Path(sys.argv[4])

mapping = {
    "__PROJECT_ROOT__": str(project_root),
    "__RUN_DIR__": str(run_dir),
    "__TRANSFER_DIR__": str(transfer_dir),
}

pairs = [
    ("music_sfdm_wave_FINAL1024.conf.template", "music_sfdm_wave_FINAL1024.conf"),
    ("music_baryon_particles_FINAL1024.conf.template", "music_baryon_particles_FINAL1024.conf"),
    ("inputs_sfdm1024_PHASECHECK.template", "inputs_sfdm1024_PHASECHECK"),
]

for src_name, dst_name in pairs:
    s = (template_dir / src_name).read_text()
    for a,b in mapping.items():
        s = s.replace(a,b)
    (run_dir / dst_name).write_text(s)

probin = template_dir / "probin_sfdm1024"
if probin.exists():
    (run_dir / "probin_sfdm1024").write_bytes(probin.read_bytes())

print("Prepared runtime directory:", run_dir)
PY

echo
echo "Prepared:"
ls -lh "$RUN_DIR"/music_*1024.conf "$RUN_DIR"/inputs_sfdm1024_PHASECHECK

echo
echo "Transfer-table checksums:"
sha256sum "$TRANSFER_DIR"/*.dat

echo
echo "IMPORTANT: the administrator should now inspect the generated configs"
echo "before submitting the jobs."
