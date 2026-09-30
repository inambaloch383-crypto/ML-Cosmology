import numpy as np


# ============================================================
# PROJECT COSMOLOGY
# ============================================================

H0 = 67.4                         # km/s/Mpc
h  = H0 / 100.0

OMEGA_M = 0.31330923051184745
OMEGA_L = 1.0 - OMEGA_M


def Hubble(z, H0=H0, Omega_m=OMEGA_M, Omega_L=OMEGA_L):
    """
    Flat LambdaCDM H(z), sufficient for z~2--4 FGPA work.
    Units: km/s/Mpc.
    """
    return H0 * np.sqrt(
        Omega_m * (1.0 + z)**3
        + Omega_L
    )


# ============================================================
# FGPA THERMODYNAMICS
# ============================================================

def temperature_from_density(Delta, T0=1.0e4, gamma=1.55):
    """
    T = T0 * Delta^(gamma-1)

    Delta = rho_b / rho_b_bar
    T in K.
    """
    Delta = np.asarray(Delta)

    if np.any(Delta <= 0):
        raise ValueError("Delta must be positive.")

    return T0 * Delta**(gamma - 1.0)


def thermal_b_kms(T):
    """
    Doppler parameter

        b = sqrt(2 k_B T / m_p)

    returned in km/s.
    """

    kB = 1.380649e-23       # J/K
    mp = 1.67262192369e-27  # kg

    T = np.asarray(T)

    b_ms = np.sqrt(
        2.0 * kB * T / mp
    )

    return b_ms / 1000.0


def fgpa_beta(gamma=1.55):
    """
    From photoionization equilibrium:

        n_HI ~ Delta^2 * T^-0.7

    together with

        T ~ Delta^(gamma-1)

    gives

        tau ~ Delta^beta

        beta = 2 - 0.7(gamma-1).

    This is the real-space simplified FGPA exponent.
    """

    return 2.0 - 0.7 * (gamma - 1.0)


def raw_fgpa_tau(Delta, gamma=1.55):
    """
    Dimensionless raw optical-depth proxy.

    Overall normalization is intentionally omitted because
    it will later be calibrated with the mean flux:

        F = exp(-A * tau_raw).
    """

    Delta = np.asarray(Delta)

    if np.any(Delta <= 0):
        raise ValueError("Delta must be positive.")

    beta = fgpa_beta(gamma)

    return Delta**beta


# ============================================================
# MEAN-FLUX CALIBRATION
# ============================================================

def mean_flux_from_A(A, tau):
    return np.mean(
        np.exp(-A * tau)
    )


def calibrate_mean_flux(
    tau,
    target_mean_flux,
    tol=1.0e-12,
    maxiter=200
):
    """
    Solve

        < exp(-A tau) > = target_mean_flux

    using a robust bisection search.

    Returns A.
    """

    tau = np.asarray(tau)

    if np.any(tau < 0):
        raise ValueError("Optical depth cannot be negative.")

    if not (0.0 < target_mean_flux < 1.0):
        raise ValueError(
            "target_mean_flux must lie between 0 and 1."
        )

    lo = 0.0
    hi = 1.0

    # Increase upper bound until mean flux is below target
    while mean_flux_from_A(hi, tau) > target_mean_flux:
        hi *= 2.0

        if hi > 1.0e12:
            raise RuntimeError(
                "Could not bracket mean-flux normalization."
            )

    for _ in range(maxiter):

        mid = 0.5 * (lo + hi)

        mf = mean_flux_from_A(
            mid,
            tau
        )

        if abs(mf - target_mean_flux) < tol:
            return mid

        if mf > target_mean_flux:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


# ============================================================
# FLUX
# ============================================================

def transmitted_flux(tau, A):
    return np.exp(
        -A * np.asarray(tau)
    )


def flux_contrast(F):
    """
    delta_F = F/<F> - 1
    """

    F = np.asarray(F)

    meanF = np.mean(F)

    if meanF <= 0:
        raise ValueError(
            "Mean flux must be positive."
        )

    return F / meanF - 1.0


# ============================================================
# VELOCITY-SPACE GEOMETRY
# ============================================================

def velocity_box_length(
    Lbox_hMpc,
    z,
    h=h
):
    """
    Convert a comoving box length in Mpc/h into the
    LOS velocity extent

        Vmax = H(z)/(1+z) * Lbox_comoving

    Units returned: km/s.
    """

    Lbox_Mpc = Lbox_hMpc / h

    return (
        Hubble(z)
        / (1.0 + z)
        * Lbox_Mpc
    )


def kh_to_kvelocity(k_hMpc, z, h=h):
    """
    k [h/Mpc] -> k_v [s/km]

        k_v = k_h * h * (1+z) / H(z)
    """

    return (
        np.asarray(k_hMpc)
        * h
        * (1.0 + z)
        / Hubble(z)
    )


def kvelocity_to_kh(k_velocity, z, h=h):
    """
    k_v [s/km] -> k [h/Mpc]
    """

    return (
        np.asarray(k_velocity)
        * Hubble(z)
        / (h * (1.0 + z))
    )


# ============================================================
# 1D FLUX POWER
# ============================================================

def flux_power_1d(F, du):
    """
    Compute P_F^1D(k) for one or many skewers.

    F shape can be:

        (Npix,)
    or
        (Nskewer, Npix)

    Velocity spacing du is in km/s.

    Returned:
        k  : s/km
        P  : km/s
        std: standard deviation across skewers

    Continuum convention:

        delta_tilde(k) ~ du * FFT(delta_F)

        P(k) = |delta_tilde|^2 / L

             = (du/N) |FFT|^2
    """

    F = np.asarray(F)

    if F.ndim == 1:
        F = F[None, :]

    if F.ndim != 2:
        raise ValueError(
            "F must have shape (Npix,) or (Nskewer,Npix)"
        )

    Nskewer, Npix = F.shape

    meanF = np.mean(F)

    deltaF = (
        F / meanF
        - 1.0
    )

    fft = np.fft.rfft(
        deltaF,
        axis=1
    )

    power_each = (
        du / Npix
        * np.abs(fft)**2
    )

    k = (
        2.0 * np.pi
        * np.fft.rfftfreq(
            Npix,
            d=du
        )
    )

    P = np.mean(
        power_each,
        axis=0
    )

    std = np.std(
        power_each,
        axis=0,
        ddof=1
    ) if Nskewer > 1 else np.zeros_like(P)

    # Remove k=0
    return (
        k[1:],
        P[1:],
        std[1:]
    )


def dimensionless_flux_power(k, P):
    """
    Common 1D dimensionless convention:

        Delta_F^2 = k P / pi
    """

    return (
        np.asarray(k)
        * np.asarray(P)
        / np.pi
    )
