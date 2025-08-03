import numpy as np
from scipy.integrate import odeint
import pandas as pd

def sir_model(y, t, beta, gamma):
    """SIR differential equations."""
    S, I, R = y
    N = S + I + R
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def solve_sir(N, I0, R0, beta, gamma, days):
    """
    Solves the SIR model.
    
    Args:
        N: Total population
        I0: Initial infected
        R0: Initial recovered
        beta: Infection rate
        gamma: Recovery rate
        days: Number of days to simulate
        
    Returns:
        t: Time grid
        S, I, R: Arrays of susceptible, infected, recovered
    """
    S0 = N - I0 - R0
    y0 = [S0, I0, R0]
    t = np.linspace(0, days, days)
    
    solution = odeint(sir_model, y0, t, args=(beta, gamma))
    S, I, R = solution.T
    return t, S, I, R

def make_sir_dataframe(t, S, I, R):
    """Create a pandas DataFrame from SIR solution."""
    return pd.DataFrame({
        'day': t,
        'susceptible': S,
        'infected': I,
        'recovered': R
    })
