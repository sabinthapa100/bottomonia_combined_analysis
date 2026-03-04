import numpy as np
from glauber import OpticalGlauber, SystemSpec

class GlauberWrapper:
    """
    Wrapper for OpticalGlauber to provide convenience methods used in the notebooks,
    such as calculating Leff for a list of centrality bins.
    """
    def __init__(self, system, roots, A, sigma_nn_mb=None):
        self.spec = SystemSpec(system, roots, A, sigma_nn_mb=sigma_nn_mb)
        self.gl = OpticalGlauber(self.spec, verbose=True)
        self.system_type = system

    def leff_minbias_pA(self):
        """Calculates min-bias Leff for pA (Proton-Nucleus)."""
        # Always call the pA method from the underlying glauber instance
        # irrespective of system type, to match notebook usage where 
        # pA minbias is used as a reference or baseline.
        return self.gl.leff_minbias_pA()

    def leff_minbias_dA(self):
        """Calculates min-bias Leff for dA."""
        return self.gl.leff_minbias_dA()

    def leff_bins_pA(self, cent_bins, method="optical"):
        """
        Calculates Leff for a list of centrality bins [(min, max), ...].
        Returns a dictionary { "min-max%": Leff_value }.
        """
        results = {}
        for (c_min, c_max) in cent_bins:
            c0 = c_min / 100.0
            c1 = c_max / 100.0
            label = f"{int(c_min)}-{int(c_max)}%"
            
            if self.system_type == "pA":
                val = self.gl.leff_bin_pA(c0, c1, method=method)
            elif self.system_type == "dA":
                # For dA, use the dA specific method
                val = self.gl.leff_bin_dA(c0, c1, method=method)
            else:
                # Fallback or error
                val = 0.0
                
            results[label] = val
        return results

    # Forward other attribute accesses to the underlying OpticalGlauber instance
    def __getattr__(self, name):
        return getattr(self.gl, name)
