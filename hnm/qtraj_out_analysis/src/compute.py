import logging
import os
import numpy as np
from typing import Optional, List

from qtraj_analysis.io import read_whitespace_table, parse_records
from qtraj_analysis.matching import build_observables
from qtraj_analysis.glauber import GlauberInterpolator, load_glauber_from_input_base
from qtraj_analysis.binning import compute_raa_vs_b
from qtraj_analysis.feeddown import (
    build_feeddown_matrix,
    solve_primordial_sigmas,
    compute_raa_with_feeddown_vs_b,
)

def run_analysis(
    datafile: str,
    energy_label: str,
    data_dir_label: str,
    input_base: str,
    bmb: Optional[float],
    npart_vals: Optional[np.ndarray],
    sigmas_exp: np.ndarray,
    c0: float,
    outdir: str,
    logger: logging.Logger,
    glauber_system: Optional[str] = None,
):
    """
    End-to-end analysis pipeline:
      1. Read + parse datafile
      2. Build 6-state observables (S/P matching)
      3. Load Glauber tables
      4. Compute RAA vs b (6-state primordial)
      5. Apply feed-down to get RAA vs Npart (inc. 9-state)
      6. Save CSVs and Plots
    """
    os.makedirs(outdir, exist_ok=True)

    # 1) Read + parse
    logger.info("Reading datafile: %s", datafile)
    try:
        table = read_whitespace_table(datafile, logger)
        records = parse_records(table, logger)
    except Exception as e:
        logger.error("Failed to read/parse datafile: %s", e)
        raise

    # 2) Build observables
    try:
        obs = build_observables(records, logger)
    except ValueError as e:
        logger.error("Failed to build observables: %s", e)
        raise

    # 2b) Determine bVals used in simulation (unique, excluding bMB)
    b_all = np.unique(np.round(np.array([o.b for o in obs], dtype=np.float64), 6))
    if bmb is not None:
        # float tolerance check
        bvals = b_all[np.abs(b_all - bmb) > 1e-4]
    else:
        bvals = b_all

    logger.info("Unique b-values found (post bMB removal): %d", len(bvals))
    logger.debug("b-values: %s", bvals)

    # 3) Load Glauber tables
    # Assumes standard structure: input_base/glauber-data/bvscData.tsv
    bvsc_path = os.path.join(input_base, "glauber-data", "bvscData.tsv")
    nbin_path = os.path.join(input_base, "glauber-data", "nbinvsbData.tsv")
    
    # Check files exist
    if not os.path.exists(bvsc_path):
        raise FileNotFoundError(f"Glauber file missing: {bvsc_path}")
    if not os.path.exists(nbin_path):
        raise FileNotFoundError(f"Glauber file missing: {nbin_path}")

    glauber_model, glauber_spec = load_glauber_from_input_base(
        input_base,
        logger,
        observed_bvals=bvals,
        npart_vals=npart_vals,
        system_key=glauber_system,
    )
    glauber = GlauberInterpolator(glauber_model)
    if glauber_spec is not None:
        logger.info(
            "Using canonical Glauber mapping for %s %s from %s",
            glauber_spec.system,
            glauber_spec.energy_label,
            glauber_spec.source_notebook,
        )

    # 4) Compute RAA vs b (6-state)
    raa_vs_b = compute_raa_vs_b(obs, logger=logger, bmb=bmb)

    # 5) Feed-down setup
    feeddown = build_feeddown_matrix()
    
    # Solve primordial sigmas from experimental inclusive
    try:
        sigmas_prim = solve_primordial_sigmas(feeddown, sigmas_exp)
    except np.linalg.LinAlgError as e:
        logger.error("Failed to invert feeddown matrix: %s", e)
        raise

    # 6) Compute RAA with feeddown vs b -> Npart
    try:
        npart, raa9, b_used = compute_raa_with_feeddown_vs_b(
            raa_vs_b=raa_vs_b,
            glauber=glauber,
            feeddown=feeddown,
            sigmas_primordial=sigmas_prim,
            logger=logger,
        )
    except Exception as e:
        logger.error("Error computing RAA with feeddown: %s", e)
        raise

    # Sort by Npart increasing (Mathematica usually sorts this way for plotting)
    order = np.argsort(npart)
    npart_s = npart[order]
    raa9_s = raa9[order]

    # 7) Save numeric outputs (CSV)
    # RAA vs Npart
    csv_path = os.path.join(outdir, f"raavsnpart_{data_dir_label}_{energy_label.replace(' ', '_')}.csv")
    header = "Npart,RAA_1S,RAA_2S,RAA_1P0,RAA_1P1,RAA_1P2,RAA_3S,RAA_2P0,RAA_2P1,RAA_2P2"
    
    # raa9_s is shape (nbins, 9)
    # stack npart_s as first column
    arr = np.column_stack([npart_s, raa9_s])
    np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")
    logger.info("Saved CSV: %s", csv_path)

    # 8) Plot if the legacy plotting helper is available.
    fig_path = os.path.join(outdir, f"raavsnpart_{data_dir_label}_{energy_label.replace(' ', '_')}.pdf")
    try:
        from qtraj_analysis.plotting import plot_raa_vs_npart  # type: ignore
    except (ImportError, AttributeError):
        plot_raa_vs_npart = None

    if plot_raa_vs_npart is None:
        logger.warning(
            "Legacy plot helper qtraj_analysis.plotting.plot_raa_vs_npart is unavailable; "
            "skipping raw compute.py PDF output."
        )
    else:
        plot_raa_vs_npart(
            npart=npart_s,
            raa9=raa9_s,
            outpath=fig_path,
            title=f"{energy_label} {data_dir_label}\nInclusive $R_{{AA}}$ vs $N_{{part}}$",
            logger=logger,
        )
    
    # 9) Save raw RAA vs B (6-state) for verification
    csv_b_path = os.path.join(outdir, f"raavsb_{data_dir_label}_{energy_label.replace(' ', '_')}.csv")
    header_b = "b,RAA_1S,RAA_2S,RAA_1P,RAA_3S,RAA_2P,RAA_1D,SEM_1S,SEM_2S,SEM_1P,SEM_3S,SEM_2P,SEM_1D"
    
    # raa_vs_b.bvals shape (nbins,)
    # raa_vs_b.raa6_mean shape (nbins, 6)
    # raa_vs_b.raa6_sem shape (nbins, 6)
    arr_b = np.column_stack([
        raa_vs_b.bvals, 
        raa_vs_b.raa6_mean, 
        raa_vs_b.raa6_sem
    ])
    np.savetxt(csv_b_path, arr_b, delimiter=",", header=header_b, comments="")
    logger.info("Saved CSV: %s", csv_b_path)
