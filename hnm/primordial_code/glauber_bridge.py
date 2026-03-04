#!/usr/bin/env python3
"""Generate primordial-compatible Glauber TSV maps from eloss optical Glauber."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import argparse
import numpy as np

from eloss_code.glauber import OpticalGlauber, SystemSpec


@dataclass(frozen=True)
class GlauberBridgeConfig:
    roots_gev: float = 8160.0
    target_a: int = 208
    system: str = "pA"
    bmax_fm: float = 20.0
    nb: int = 401
    include_npart: bool = True
    verbose: bool = False


def _write_two_col(path: Path, x: np.ndarray, y: np.ndarray, header: str) -> None:
    arr = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    np.savetxt(path, arr, fmt="%.10e", header=header)


def generate_primordial_glauber_maps(
    output_dir: Path | str,
    *,
    cfg: GlauberBridgeConfig = GlauberBridgeConfig(),
) -> Dict[str, Path]:
    """
    Write primordial map files:
      - bvscData.tsv      : b[fm], cdf in [0,1]
      - nbinvsbData.tsv   : b[fm], <Ncoll>(b) conditional on inelastic events
      - npartvsbData.tsv  : optional pA approximation 1 + <Ncoll>(b)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    spec = SystemSpec(system=cfg.system, roots_GeV=cfg.roots_gev, A=cfg.target_a)
    gl = OpticalGlauber(spec, verbose=cfg.verbose, bmax_fm=cfg.bmax_fm, nb=cfg.nb)

    if cfg.system != "pA":
        raise ValueError(f"This bridge currently writes primordial pA maps only, got: {cfg.system}")

    b = gl.b_grid
    c = gl.cdf_pA

    # <Ncoll>(b) conditional on at least one inelastic interaction.
    lam = gl.sigma_nn_fm2 * np.maximum(gl.TpA_b, 0.0)
    pinel = 1.0 - np.exp(-lam)
    ncoll_cond = np.divide(
        lam,
        pinel,
        out=np.zeros_like(lam, dtype=float),
        where=pinel > 0.0,
    )

    out = {
        "bvsc": output_path / "bvscData.tsv",
        "nbin": output_path / "nbinvsbData.tsv",
    }
    _write_two_col(out["bvsc"], b, c, "b_fm c_fraction")
    _write_two_col(out["nbin"], b, ncoll_cond, "b_fm ncoll_conditional_optical")

    if cfg.include_npart:
        # pA approximation used to avoid primordial fallback defaults.
        npart_cond = 1.0 + ncoll_cond
        out["npart"] = output_path / "npartvsbData.tsv"
        _write_two_col(out["npart"], b, npart_cond, "b_fm npart_conditional_approx_1_plus_ncoll")

    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="Output directory for primordial-style glauber TSV files.")
    p.add_argument("--roots-gev", type=float, default=8160.0)
    p.add_argument("--a", type=int, default=208)
    p.add_argument("--bmax-fm", type=float, default=20.0)
    p.add_argument("--nb", type=int, default=401)
    p.add_argument("--no-npart", action="store_true", help="Do not write npartvsbData.tsv.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = GlauberBridgeConfig(
        roots_gev=args.roots_gev,
        target_a=args.a,
        bmax_fm=args.bmax_fm,
        nb=args.nb,
        include_npart=not args.no_npart,
        verbose=args.verbose,
    )
    out = generate_primordial_glauber_maps(args.out, cfg=cfg)
    for key, path in out.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
