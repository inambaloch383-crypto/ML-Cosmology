import numpy as np

from fgpa_core import (
    raw_fgpa_tau,
    temperature_from_density,
    thermal_b_kms,
)


def _to_2d(x):
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        return x[None, :]

    if x.ndim != 2:
        raise ValueError(
            "Input must have shape (Npix,) "
            "or (Nskewer,Npix)"
        )

    return x


def tau_velocity_thermal(
    Delta,
    vpec,
    du,
    T0=1.0e4,
    gamma=1.55,
    nsigma=5.0,
):
    """
    Construct a periodic redshift-space optical-depth field.

    Parameters
    ----------
    Delta :
        Baryon density contrast rho_b / <rho_b>.

        Shape:
            (Npix,)
        or
            (Nskewer, Npix)

    vpec :
        LOS physical peculiar velocity in km/s.
        Same shape as Delta.

    du :
        Velocity-grid spacing in km/s.

    T0 :
        Temperature at mean density, K.

    gamma :
        Temperature-density index:

            T = T0 Delta^(gamma-1)

    nsigma :
        Thermal Gaussian cutoff in units of b.

    Method
    ------
    Each real-space cell has raw optical-depth weight

        tau_0 ~ Delta^beta

    and Hubble-coordinate centre

        u_H = (j+1/2) du.

    Peculiar velocity shifts it to

        u_s = u_H + v_parallel.

    Its contribution is then thermally broadened with

        K(du) =
        exp[-(Delta u/b)^2] / (sqrt(pi) b)

    using periodic boundary conditions.

    The discrete thermal kernel is renormalized so that
    the integrated optical-depth weight is conserved.
    """

    Delta = _to_2d(Delta)
    vpec = _to_2d(vpec)

    if Delta.shape != vpec.shape:
        raise ValueError(
            "Delta and vpec must have identical shapes."
        )

    if du <= 0:
        raise ValueError(
            "du must be positive."
        )

    if np.any(Delta <= 0):
        raise ValueError(
            "Delta must be positive."
        )

    Nskewer, Npix = Delta.shape

    Vbox = Npix * du

    # --------------------------------------------------------
    # Raw FGPA optical-depth field
    # --------------------------------------------------------
    tau0 = raw_fgpa_tau(
        Delta,
        gamma=gamma,
    )

    # --------------------------------------------------------
    # Temperature and thermal Doppler parameter
    # --------------------------------------------------------
    T = temperature_from_density(
        Delta,
        T0=T0,
        gamma=gamma,
    )

    b = thermal_b_kms(T)

    tau_s = np.zeros_like(
        Delta,
        dtype=float
    )

    sqrt_pi = np.sqrt(np.pi)

    # --------------------------------------------------------
    # Deposit every real-space cell into velocity space
    # --------------------------------------------------------
    for s in range(Nskewer):

        for j in range(Npix):

            # Hubble-flow velocity coordinate
            uH = (
                j + 0.5
            ) * du

            # Redshift-space centre
            uc = (
                uH
                + vpec[s, j]
            ) % Vbox

            bj = b[s, j]

            # Number of grid cells required for thermal kernel
            radius = max(
                1,
                int(
                    np.ceil(
                        nsigma * bj / du
                    )
                )
            )

            # Nearest velocity-grid centre
            jc = int(
                np.floor(
                    uc / du
                )
            )

            # Normally radius << Npix/2.
            # Keep a safe fallback.
            if 2 * radius + 1 >= Npix:

                idx = np.arange(
                    Npix
                )

            else:

                idx = (
                    np.arange(
                        jc - radius,
                        jc + radius + 1
                    )
                    % Npix
                )

            ui = (
                idx + 0.5
            ) * du

            # Periodic minimum-image separation
            d = (
                (
                    ui
                    - uc
                    + 0.5 * Vbox
                )
                % Vbox
                - 0.5 * Vbox
            )

            kernel = (
                np.exp(
                    -(d / bj)**2
                )
                / (
                    sqrt_pi
                    * bj
                )
            )

            # Renormalize discrete finite kernel:
            #
            # sum K_i du = 1
            norm = (
                np.sum(kernel)
                * du
            )

            if norm <= 0:
                raise RuntimeError(
                    "Invalid thermal kernel normalization."
                )

            kernel /= norm

            # Source element carries
            #
            # tau0_j * du
            #
            # integrated optical-depth weight.
            contribution = (
                tau0[s, j]
                * du
                * kernel
            )

            np.add.at(
                tau_s[s],
                idx,
                contribution
            )

    return tau_s, T, b


def integrated_tau(tau, du):
    """
    Integral of tau over velocity coordinate.
    """

    tau = _to_2d(tau)

    return (
        np.sum(
            tau,
            axis=1
        )
        * du
    )
