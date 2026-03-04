# CNM Notebooks Implementation Plan

## Objective
Create validation and production notebooks for CNM effects (nPDF, Energy Loss, pT Broadening) in `cnm_notebook/`.

## location
`cnm_notebook/` folder.

## Notebooks to Create
1.  **`05_d_cnm_pA_LHC.ipynb`**: pPb 8.16 TeV Analysis
2.  **`06_d_cnm_dA_RHIC.ipynb`**: dAu 200 GeV Analysis

## Components
For each system, calculate and store:
*   `npdf`: nPDF effects (EPPS21)
*   `eloss`: Coherent energy loss
*   `broad`: Cronin pT broadening
*   `eloss_broad`: Combined Energy Loss × Broadening
*   `cnm`: Total CNM = nPDF × (Energy Loss × Broadening)

## Plotting Requirements
**Styles:**
*   `eloss`: Black (light/thin)
*   `broad`: Black (dashed, light)
*   `eloss_broad`: Gray (darker)
*   `npdf`: Magenta
*   `cnm`: Blue (or flexible premium color)

**Plots:**
1.  **RpA vs y** (pT-integrated):
    *   Bands for all components.
    *   Centrality bins + Min Bias.
2.  **RpA vs pT** (in 3 rapidity windows):
    *   Bands for all components.
    *   Centrality bins + Min Bias.
3.  **RpA vs Centrality** (in 3 rapidity windows):
    *   Min Bias: Dashed horizontal line (or band).
    *   Centrality Dependent: Solid points/lines.

## Implementation Steps
1.  **Use `cnm_combine_fast`**: Leverage the optimized vectorized module for fast computation.
2.  **Centrality Handling**: ensure `cent_bins` match standard definitions (0-20, 20-40, etc.).
3.  **Rapidity Windows**:
    *   LHC: `[-4.46, -2.96]`, `[-1.37, 0.43]`, `[2.03, 3.53]` (CMS-like) or user specified.
    *   RHIC: `[-2.2, -1.2]`, `[-0.35, 0.35]`, `[1.2, 2.2]` (PHENIX-like).
4.  **Validation**: Verify against "Golden Standard" logic (via `cnm_combine` consistency).

## Execution
Generate notebooks using `generate_cnm_notebooks.py` script to ensure consistency and avoid syntax errors.
