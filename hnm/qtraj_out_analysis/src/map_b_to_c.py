import logging
import numpy as np
from qtraj_analysis.glauber import load_glauber, GlauberInterpolator

# Silence logging
logger = logging.getLogger("dummy")
logger.setLevel(logging.ERROR)

b_vals = np.array([0.0, 2.23108, 4.07936, 5.76909, 7.4707, 8.84676, 10.0347, 11.0956, 12.0634, 12.9592, 13.8166, 15.0803])
# Dummy npart
npart_vals = np.zeros_like(b_vals)

glauber_model = load_glauber(
    "qtraj_outputs/RHIC/AuAu/200GeV/glauber-data/bvscData.tsv",
    "qtraj_outputs/RHIC/AuAu/200GeV/glauber-data/nbinvsbData.tsv",
    b_vals,
    npart_vals,
    logger
)
g = GlauberInterpolator(glauber_model)

c_vals = g.b_to_c(b_vals)
print("b -> Centrality:")
for b, c in zip(b_vals, c_vals):
    print(f"{b:8.4f} -> {c*100:6.2f}%")
