import numpy as np

from fgpa_core import (
    velocity_box_length,
    calibrate_mean_flux,
    transmitted_flux,
    flux_power_1d,
    dimensionless_flux_power,
)

from fgpa_rsd import (
    tau_velocity_thermal,
)


def make_lya_flux(
    Delta,
    vlos,
    z,
    Lbox_hMpc,
    target_mean_flux,
    T0=1.0e4,
    gamma=1.55,
):
    """
    Ly-alpha FGPA forward model.

    Parameters
    ----------
    Delta :
        Baryon density field rho_b/<rho_b>.
        Shape = (Nskewer, Npix) or (Npix,).

    vlos :
        Physical LOS peculiar velocity [km/s].
        Same shape as Delta.

    z :
        Snapshot redshift.

    Lbox_hMpc :
        Comoving box length [Mpc/h].

    target_mean_flux :
        Desired <F>.

    T0 :
        Temperature at mean density [K].

    gamma :
        T-density slope:
            T = T0 * Delta^(gamma-1)

    Returns
    -------
    dictionary containing tau, F, A, temperature,
    thermal width, velocity grid, and 1D flux power.
    """

    Delta = np.asarray(
        Delta,
        dtype=float
    )

    vlos = np.asarray(
        vlos,
        dtype=float
    )

    if Delta.ndim == 1:
        Delta = Delta[None, :]

    if vlos.ndim == 1:
        vlos = vlos[None, :]

    if Delta.shape != vlos.shape:
        raise ValueError(
            "Delta and vlos must have identical shapes."
        )

    Nskewer, Npix = Delta.shape

    # --------------------------------------------------------
    # Convert the comoving box into LOS velocity length
    # --------------------------------------------------------
    Vbox = velocity_box_length(
        Lbox_hMpc,
        z
    )

    du = Vbox / Npix

    u = (
        np.arange(Npix)
        + 0.5
    ) * du

    # --------------------------------------------------------
    # Redshift-space + thermally broadened optical depth
    # --------------------------------------------------------
    tau, T, b = tau_velocity_thermal(
        Delta,
        vlos,
        du,
        T0=T0,
        gamma=gamma,
    )

    # --------------------------------------------------------
    # Match mean transmitted flux
    # --------------------------------------------------------
    A = calibrate_mean_flux(
        tau,
        target_mean_flux
    )

    F = transmitted_flux(
        tau,
        A
    )

    # --------------------------------------------------------
    # 1D flux power
    # --------------------------------------------------------
    kv, PF, PF_std = flux_power_1d(
        F,
        du
    )

    DeltaF2 = dimensionless_flux_power(
        kv,
        PF
    )

    return {
        "z": z,
        "Lbox_hMpc": Lbox_hMpc,
        "Vbox_kms": Vbox,
        "du_kms": du,

        "u_kms": u,

        "tau": tau,
        "F": F,

        "A": A,
        "mean_flux": np.mean(F),

        "temperature_K": T,
        "b_kms": b,

        "kv_s_per_km": kv,
        "PF_kms": PF,
        "PF_std_kms": PF_std,
        "DeltaF2": DeltaF2,
    }


def flux_power_ratio(result_num, result_den):
    """
    Ratio of two matched P_F^1D results.

    Requires identical k_v grids.
    """

    k1 = result_num[
        "kv_s_per_km"
    ]

    k2 = result_den[
        "kv_s_per_km"
    ]

    if (
        k1.shape != k2.shape
        or not np.allclose(
            k1,
            k2,
            rtol=1e-12,
            atol=0.0
        )
    ):
        raise ValueError(
            "Flux-power k grids do not match."
        )

    return (
        k1,
        result_num["PF_kms"]
        / result_den["PF_kms"]
    )
