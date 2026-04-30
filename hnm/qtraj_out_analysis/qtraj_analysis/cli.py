import argparse
import logging
import sys
import numpy as np

from qtraj_analysis.compute import run_analysis
from qtraj_analysis.glauber import list_canonical_glauber_systems

def setup_logger(verbosity: int) -> logging.Logger:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger("qtraj_analysis")

def parse_float_list(s: str) -> np.ndarray:
    try:
        return np.array([float(x) for x in s.split(",")], dtype=np.float64)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float list: {s}")

def main():
    parser = argparse.ArgumentParser(
        description="QTraj Data Analysis Pipeline"
    )
    
    parser.add_argument(
        "--datafile",
        required=True,
        help="Path to qtraj output file (.gz or plain)"
    )
    
    parser.add_argument(
        "--input-base",
        required=True,
        help="Base directory containing result folder structure (must look for glauber-data/ inside)"
    )
    
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for CSVs and plots"
    )

    parser.add_argument(
        "--energy-label",
        default="RHIC 200 GeV",
        help="Label for plots (e.g. 'RHIC 200 GeV')"
    )
    
    parser.add_argument(
        "--data-dir-label",
        default="run1",
        help="Short label for file naming (e.g. 'run_kappa5')"
    )

    parser.add_argument(
        "--bmb",
        type=float,
        default=None,
        help="Impact parameter for min-bias to exclude (optional)"
    )
    
    parser.add_argument(
        "--npart-vals",
        type=parse_float_list,
        help="Optional comma-separated list of Npart values corresponding to the discrete b-values in the datafile"
    )

    parser.add_argument(
        "--glauber-system",
        choices=list_canonical_glauber_systems(),
        help="Use the canonical thesis Glauber mapping for one of the published HNM systems"
    )
    
    parser.add_argument(
        "--sigmas-exp",
        type=parse_float_list,
        required=True,
        help="Comma-separated list of 9 inclusive pp cross-sections (1S, 2S, 1P(chi_b0,1,2), 3S, 2P(chi_b0,1,2))"
    )
    
    parser.add_argument(
        "--c0",
        type=float,
        default=0.25,
        help="Centrality weight parameter c0 (default 0.25)"
    )

    parser.add_argument(
        "-v", "--verbosity",
        type=int,
        default=1,
        help="Verbosity level: 0=WARNING, 1=INFO, 2=DEBUG"
    )
    
    args = parser.parse_args()
    logger = setup_logger(args.verbosity)
    
    logger.info("Starting QTraj Analysis")
    logger.info("Datafile: %s", args.datafile)
    
    if len(args.sigmas_exp) != 9:
        logger.error("Error: --sigmas-exp must have exactly 9 entries. Got %d.", len(args.sigmas_exp))
        sys.exit(1)

    try:
        run_analysis(
            datafile=args.datafile,
            energy_label=args.energy_label,
            data_dir_label=args.data_dir_label,
            input_base=args.input_base,
            bmb=args.bmb,
            npart_vals=args.npart_vals,
            sigmas_exp=args.sigmas_exp,
            c0=args.c0,
            outdir=args.outdir,
            logger=logger,
            glauber_system=args.glauber_system,
        )
    except Exception as e:
        logger.error("Analysis failed: %s", e)
        # Verify if verbose for stacktrace
        if args.verbosity >= 2:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    logger.info("Analysis completed successfully.")

if __name__ == "__main__":
    main()
