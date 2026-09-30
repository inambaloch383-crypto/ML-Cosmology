# SFDM 1024^3 cluster handoff

## Immediate objective

This archive is for a **high-resolution IC and wave-phase validation test**, not
for a full production campaign.

The requested chain is:

1. build the user's exact modified MUSIC and AxioNyx code;
2. generate two 1024^3 MUSIC IC directories:
   - `IC_SFDM_WAVE_FINAL1024`
   - `IC_BARYON_PARTICLES_FINAL1024`
3. run AxioNyx for initialization plus one short step;
4. preserve the step-0 plotfile so the user can measure phase aliasing.

The full production wave/proxy/LCDM simulations should not be launched until
this test is reviewed.

## Important reproducibility point

Please use the source trees included in this archive rather than downloading a
fresh generic MUSIC/AxioNyx copy. The MUSIC tree contains project-specific
changes for signed oscillatory transfer-function interpolation and validated
species-specific velocity routing.

## Software environment

The local validated builds used GNU tooling. The cluster should provide a
compatible environment with:

- GCC/G++ with C++20 support (GCC 11 or newer is a reasonable target)
- GFortran
- GNU Make
- MPI (OpenMPI or MPICH)
- FFTW3 double + single precision; OpenMP FFTW libraries are needed by AxioNyx
  (`fftw3`, `fftw3_omp`, `fftw3f`, `fftw3f_omp` or module equivalents)
- GSL + CBLAS (used by MUSIC)
- zlib
- Python 3 + NumPy for validation scripts
- Slurm (`sbatch`, `srun`) for the supplied job templates

AMReX is carried inside the user's AxioNyx source tree. The MUSIC Nyx output
plugin uses the included legacy BoxLib tree.

## Start here

Run:

```bash
bash 01_check_environment.sh | tee environment_report.txt
```

Then load the site's compiler/MPI/FFTW/GSL modules and run it again. Please keep
`environment_report.txt` for the user.

## Build

Edit `SITE_CONFIG.env` only if needed, then run:

```bash
bash 02_build_codes.sh
```

The build script deliberately does not try to guess module names. It uses the
compiler/library environment already loaded by the administrator.

### MUSIC

The included MUSIC code is version 1.53 with project modifications. Its Nyx
output plugin uses legacy BoxLib and GSL/FFTW.

### AxioNyx

The expected executable is:

`code/axionyx_wang2026/Exec/MDM_NoHydro/Nyx3d.gnu.MPI.OMP.ex`

The `GNUmakefile` in that directory already identifies the no-hydro FDM +
particles build. The build helper requests GNU + MPI + OpenMP.

## Runtime preparation

Run:

```bash
bash 03_prepare_runtime.sh
```

This resolves the portable placeholders in the MUSIC and AxioNyx templates to
the archive's actual absolute path.

## Slurm resources

The supplied Slurm files are **templates**.

Please tune:
- partition/QOS/account,
- memory,
- CPUs,
- MPI ranks per node,
- OpenMP threads,
- wall time,
- scratch filesystem.

### MUSIC resource note

The user's current MUSIC workflow is effectively a one-process IC generator
with threaded FFT work. A 1024^3 realization can require substantial memory
because several full 3-D fields coexist in memory. Please use a high-memory
node or modify the build/run layout only after confirming MUSIC's supported
parallel mode on this source tree.

### AxioNyx resource note

AxioNyx is MPI+OpenMP and should use multiple nodes for 1024^3. The supplied
example starts from an 8-node hybrid layout only as a placeholder; please choose
the actual topology from the cluster hardware and scaling policy.

## Job chain

After editing the three `.slurm` files:

```bash
bash 07_submit_chain.sh
```

The two MUSIC jobs are submitted independently. The AxioNyx phase-check job is
submitted with a dependency on both successful MUSIC jobs.

## Success criteria for this stage

The run should produce:
- successful MUSIC logs for wave and baryon ICs,
- both 1024^3 IC directories,
- a successful AxioNyx initialization,
- `plt_PHASE1024_00000` (and optionally the one-step output),
- no fatal/NaN/abort errors.

Do not interpret the high-resolution phase convergence automatically. The user
will run the dedicated phase diagnostic on the actual 1024^3 output.

## Frozen physics values

See `SCIENCE_PARAMETERS.txt`. In particular:

- `z_ini = 120`
- `L = 30 h^-1 Mpc`
- `m_phi = 3.16227766e-22 eV`
- `nyx.m_tt = 3.16227766`
- `ratio_fdm = 0.842618008951438`
- `force_pnorm = 3.1586221390e-11`
- common seed = `12345`

Please do not replace `nyx.m_tt` with the older development value `3.40`.
