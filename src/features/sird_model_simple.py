# =============================================================================
# Plots a SIRD model according input parameters
# =============================================================================


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def deriv(y, t, N, beta, gamma, omega):
    S, I, R, D = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I - omega * I
    dRdt = gamma * I
    dDdt = omega * I
    return dSdt, dIdt, dRdt, dDdt


def SIRD_model_simple(N, I0, Ro, T_r, omega, T):
    """
    Plots a simple SIRD model without interaction between populations.

    Parameters
    ----------
    N : int
        Initial population.
    I0 : int
        Initial infected.
    Ro : float
        Reproduction number.
    T_r : int
        Recovery time.
    omega : float
        Mortality rate.
    T : int
        Simulation time.

    Returns
    -------
    None.

    """

    # Initial number of infected, recovered and deceased individuals
    I0, R0, D0 = I0, 0, 0
    S0 = N - I0 - R0  # The rest are susceptible
    T_c = T_r / Ro  # Typical time betwen contacts

    # Transition rates
    beta = T_c ** (-1)
    gamma = T_r ** (-1)

    t = np.linspace(0, T, T)  # t in units of time

    # Initial conditions
    y0 = S0, I0, R0, D0
    # Integrate the differential equations over t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, omega))
    S, I, R, D = ret.T

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    plt.plot(t, S, label="Susceptible")
    plt.plot(t, I, label="Infected")
    plt.plot(t, R, label="Recovered")
    plt.plot(t, D, label="Deceased")
    plt.title("SIRD MODEL", fontsize=20)
    ax.set_ylabel("Individuals")
    ax.set_xlabel("Time")
    plt.legend()
    plt.tight_layout()
    
    return S, I, R, D


if __name__ == '__main__':
    # Runs a simulation ant plot the model
    SIRD_model_simple(47000000, 1, 4, 10, 0.02, 360)
