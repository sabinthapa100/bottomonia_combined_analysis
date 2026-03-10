# Final "Step-by-Step" CNM Scripts

This directory contains the finalized production scripts for generating $R_{pA}$ plots for RHIC and LHC, incorporating:
1.  **nPDF Effects**: Using EPPS21 nPDFs with centrality dependence.
2.  **Energy Loss & Broadening**: Using the Arleo-Peigné coherent energy loss model.
3.  **Combination**: Combining these effects in quadrature.

These scripts bypass the monolithic `CNMCombine` class to provide a clear, "step-by-step" calculation flow that is easier to inspect and debug.

## Prerequisites

You must run these scripts in your **Conda Research Environment**.
Ensure the following packages are installed:
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `torch` (for accelerated energy loss integration)

## Available Scripts

### 1. RHIC d+Au 200 GeV
**Script:** `step_by_step_rhic.py`

*   **System:** d+Au at $\sqrt{s_{NN}} = 200$ GeV.
*   **Centrality Bins:** 0-20%, 20-40%, 40-60%, 60-100%.
*   **Rapidity Windows:** Backward (-2.2, -1.2), Mid (-0.35, 0.35), Forward (1.2, 2.2).
*   **Calculations:**
    *   $R_{dAu}$ vs Rapidity (y)
    *   $R_{dAu}$ vs Transverse Momentum ($p_T$)
    *   $R_{dAu}$ vs Centrality

**To Run:**
```bash
python final_scripts/step_by_step_rhic.py
```

### 2. LHC p+Pb 5.02 TeV
**Script:** `step_by_step_lhc.py`

*   **System:** p+Pb at $\sqrt{s_{NN}} = 5.02$ TeV.
*   **Centrality Bins:** 0-20%, 20-40%, 40-60%, 60-80%, 80-100%.
*   **Rapidity Windows:** Backward (-4.46, -2.96), Mid (-1.37, 0.43), Forward (2.03, 3.53).
*   **Calculations:**
    *   $R_{pPb}$ vs Rapidity (y)
    *   $R_{pPb}$ vs Transverse Momentum ($p_T$)
    *   $R_{pPb}$ vs Centrality

**To Run:**
```bash
python final_scripts/step_by_step_lhc.py
```

## Output

Results are saved to `outputs/step_by_step_rhic/` and `outputs/step_by_step_lhc/`.
For each kinematic variable (y, pT, centrality), the output includes:
*   **PNG/PDF Plots**: High-quality "Gold Standard" figures showing individudal components (nPDF, Eloss, Broadening) and the Total band.
