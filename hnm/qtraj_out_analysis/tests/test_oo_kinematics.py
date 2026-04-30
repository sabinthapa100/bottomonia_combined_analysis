"""OO kinematics presets used by run_oo_5360_production."""
from __future__ import annotations

import numpy as np

from qtraj_analysis.kinematics_presets import (
    OO_INTEGRATED_Y_WINDOW,
    OO_PT_MAX_FOR_Y,
    OO_PT_RAPIDITY_WINDOWS,
)


def test_oo_midrapidity_is_pm24():
    mid = next(w for w in OO_PT_RAPIDITY_WINDOWS if w[0] == "midrapidity")
    assert mid[1] == (-2.4, 2.4)
    assert OO_INTEGRATED_Y_WINDOW == (-2.4, 2.4)


def test_oo_pt_max_for_y_is_30():
    assert OO_PT_MAX_FOR_Y == 30.0


def test_oo_pt_edges_covers_0_30():
    from qtraj_analysis.kinematics_presets import OO_PT_EDGES

    assert OO_PT_EDGES[0] == 0.0
    assert OO_PT_EDGES[-1] == 30.0
    assert len(OO_PT_EDGES) == 31


def test_oo_y_edges_cover_cms_acceptance():
    from qtraj_analysis.kinematics_presets import OO_Y_EDGES

    assert np.isclose(OO_Y_EDGES[0], -2.4)
    assert np.isclose(OO_Y_EDGES[-1], 2.4)
