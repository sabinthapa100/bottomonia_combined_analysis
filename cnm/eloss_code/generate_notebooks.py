
import json
import os

def create_notebook(filename, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filename, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Created {filename}")

# --- Common Imports Cell ---
cell_imports = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import sys\n",
        "import os\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Add module path\n",
        "sys.path.append(os.path.abspath(\"../eloss_code\"))\n",
        "\n",
        "import eloss_cronin_centrality_test as EC\n",
        "import plotting_utils as PU\n",
        "from particle import Particle, PPSpectrumParams\n",
        "\n",
        "print(\"Modules imported.\")"
    ]
}

# --- 05_c (LHC) ---
cells_lhc = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# LHC pPb Energy Loss & Broadening (Test Module)\n",
            "Verifying `eloss_cronin_centrality_test.py` against Golden Standard logic.\n",
            "This notebook uses the reconstructed module to compute $R_{pA}$ for pPb at 5.02 TeV."
        ]
    },
    cell_imports,
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Setup J/psi and System\n",
            "Jpsi = Particle(family=\"charmonia\", state=\"1S\", mass_override_GeV=3.097,\n",
            "                pp_params=PPSpectrumParams(p0=4.5, m=3.0, n=5.6))\n",
            "\n",
            "params = EC.get_default_parameters(\"pPb5\")\n",
            "roots = params[\"roots_GeV\"]\n",
            "glauber = params[\"glauber\"]\n",
            "qp = params[\"qp_base\"]\n",
            "cent_bins = params[\"cent_bins\"]\n",
            "\n",
            "print(f\"System: pPb {roots} GeV\")\n",
            "print(f\"Centralities: {cent_bins}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Compute R_pA vs y (Binned Band)\n",
            "y_edges = np.linspace(-5.0, 5.0, 41)\n",
            "pt_range = (0.0, 30.0)\n",
            "\n",
            "print(\"Computing R_pA vs y (this may take a minute)...\")\n",
            "y_cent, bands_y, labels_y = EC.rpa_band_vs_y(\n",
            "    Jpsi, roots, qp, glauber, cent_bins, \n",
            "    y_edges, pt_range, components=(\"eloss_broad\",),\n",
            "    Ny_bin=16, Npt_bin=24\n",
            ")\n",
            "print(\"Done.\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Plot R_pA vs y\n",
            "fig, ax = PU.plot_RpA_vs_y_band(\n",
            "    y_cent, *bands_y[\"eloss_broad\"], labels_y, \n",
            "    note=r\"pPb $\\sqrt{s_{NN}}=5.02$ TeV\"\n",
            ")\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Compute R_pA vs pT (Binned Band) -- Forward Rapidity\n",
            "pT_edges = np.array([0,1,2,3,4,5,6,7,8,9,10,12,15,20,30], float)\n",
            "y_range_fwd = (2.03, 3.53)\n",
            "\n",
            "print(\"Computing R_pA vs pT (Forward)...\")\n",
            "pT_cent, bands_pt, labels_pt = EC.rpa_band_vs_pT(\n",
            "    Jpsi, roots, qp, glauber, cent_bins, \n",
            "    pT_edges, y_range_fwd, components=(\"eloss_broad\",),\n",
            "    Ny_bin=16, Npt_bin=24\n",
            ")\n",
            "print(\"Done.\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. Plot R_pA vs pT\n",
            "fig, ax = PU.plot_RpA_vs_pT_band(\n",
            "    pT_cent, *bands_pt[\"eloss_broad\"], labels_pt, \n",
            "    note=r\"$2.03 < y < 3.53$\"\n",
            ")\n",
            "plt.show()"
        ]
    }
]

create_notebook("eloss_notebooks/05_c_eloss_cronin_pA_LHC.ipynb", cells_lhc)

# --- 06_c (RHIC) ---
cells_rhic = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# RHIC dAu Energy Loss & Broadening (Test Module)\n",
            "Verifying `eloss_cronin_centrality_test.py` against Golden Standard logic.\n",
            "This notebook uses the reconstructed module to compute $R_{dA}$ for dAu at 200 GeV."
        ]
    },
    cell_imports,
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Setup J/psi and System\n",
            "# RHIC J/psi approx parameters\n",
            "Jpsi = Particle(family=\"charmonia\", state=\"1S\", mass_override_GeV=3.097,\n",
            "                pp_params=PPSpectrumParams(p0=4.0, m=3.0, n=5.6)) # Check p0 for RHIC?\n",
            "\n",
            "params = EC.get_default_parameters(\"dAu200\")\n",
            "roots = params[\"roots_GeV\"]\n",
            "glauber = params[\"glauber\"]\n",
            "qp = params[\"qp_base\"]\n",
            "cent_bins = params[\"cent_bins\"]\n",
            "\n",
            "print(f\"System: dAu {roots} GeV\")\n",
            "print(f\"Centralities: {cent_bins}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Compute R_dA vs y (Binned Band)\n",
            "y_edges = np.linspace(-3.0, 3.0, 31)\n",
            "pt_range = (0.0, 10.0)\n",
            "\n",
            "print(\"Computing R_dA vs y...\")\n",
            "y_cent, bands_y, labels_y = EC.rpa_band_vs_y(\n",
            "    Jpsi, roots, qp, glauber, cent_bins, \n",
            "    y_edges, pt_range, components=(\"eloss_broad\",),\n",
            "    Ny_bin=16, Npt_bin=24\n",
            ")\n",
            "print(\"Done.\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Plot R_dA vs y\n",
            "fig, ax = PU.plot_RpA_vs_y_band(\n",
            "    y_cent, *bands_y[\"eloss_broad\"], labels_y, \n",
            "    note=r\"dAu $\\sqrt{s_{NN}}=200$ GeV\"\n",
            ")\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Compute R_dA vs pT (Binned Band) -- Mid Rapidity\n",
            "pT_edges = np.array([0,1,2,3,4,5,6,7,8,10], float)\n",
            "y_range_mid = (-0.35, 0.35)\n",
            "\n",
            "print(\"Computing R_dA vs pT (Mid)...\")\n",
            "pT_cent, bands_pt, labels_pt = EC.rpa_band_vs_pT(\n",
            "    Jpsi, roots, qp, glauber, cent_bins, \n",
            "    pT_edges, y_range_mid, components=(\"eloss_broad\",),\n",
            "    Ny_bin=16, Npt_bin=24\n",
            ")\n",
            "print(\"Done.\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. Plot R_dA vs pT\n",
            "fig, ax = PU.plot_RpA_vs_pT_band(\n",
            "    pT_cent, *bands_pt[\"eloss_broad\"], labels_pt, \n",
            "    note=r\"$|y| < 0.35$\"\n",
            ")\n",
            "plt.show()"
        ]
    }
]

create_notebook("eloss_notebooks/06_c_eloss_cronin_dA_RHIC.ipynb", cells_rhic)
