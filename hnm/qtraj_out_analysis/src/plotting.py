import numpy as np
from scipy.special import genlaguerre

def coulomb_basis_function(n, l, r, m, alpha):
    """Generate Coulomb basis function for quantum state (n, l)."""
    beta = 2.0 / n
    rho = beta * m * alpha * r
    norm = np.sqrt(beta**3 * np.math.factorial(n - l - 1) / (2 * n * np.math.factorial(n + l)))
    lag = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    return norm * rho**l * np.exp(-0.5 * rho)

def initial_wavefunction(r, n, l, init_type, m, alpha, init_width=None):
    """Set initial wave function based on type."""
    if init_type == 0:  # Coulomb basis
        return coulomb_basis_function(n, l, r, m, alpha)
    elif init_type == 1:  # Gaussian
        if init_width is None:
            raise ValueError("init_width required for Gaussian initial condition")
        r_scaled = r * m * alpha
        return r**(l + 1) * np.exp(-r_scaled**2 / init_width**2)
    else:
        raise ValueError("Unknown initial condition type")
