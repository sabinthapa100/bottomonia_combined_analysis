# QTraj-NLO Data Analysis: Complete Technical Documentation

## 1. Data File Formats

### 1.1 Raw qtraj-nlo Output (datafile.gz)

The C++ code outputs paired rows:
```
Row 1 (metadata):  b   ???  ???  ???  pt  ???  y   ???
Row 2 (values):   v1  v2   v3   v4   v5  v6  L  qweight
```

**Example from PbPb κ=3:**
```
0       -0.00106795  1.24364   1.17102  13.9363  0.155985  -2.42081   (meta)
8.3054251443705e-05  8.200950082759101e-05  ...  40  0               (values, L=0)
```

- Meta row: 7 columns (b is column 0, pt column 4, y column 6)
- Values row: 8 columns [v1..v6, L, qweight]

### 1.2 Averaged File (datafile-avg.gz)

After running `processEvents.py`, the format becomes paired rows with errors:
```
Row 1 (metadata):  b   ???  ???  ???  pt  ???  y   ???
Row 2 (values):   mean1  err1  mean2  err2  ...  mean6  err6  ntraj  L
```

**PbPb κ=3 example (14 columns):**
```
# 12 values (6 means + 6 errors) + ntraj + L
0        -0.00106795  1.24364  1.17102  13.9363  0.155985  -2.42081   (meta)
8.3054251443705e-05  8.200950082759101e-05  ...  40  0          (L=0 S-wave)
0.0  0.0  0.0  0.0  2.625925e-12  2.592893e-12  ...  40  1          (L=1 P-wave)
```

### 1.3 Simplified Format (8 columns - OO noReg)

The Oxygen-Oxygen data has simpler 8-column format:
```
# Meta row: b, ..., pt, ..., y, ... (7 columns)
# Values row: v1 v2 v3 v4 v5 v6 L qweight (8 columns)

4.49691  -0.0246835  0.814679  1.25894  1.34493  0.607815  1.25894  (meta)
0.689017  0.297435    0        0.17277  0        0        0    1    (values)
```

### 1.4 Column Mapping Summary

| Column | Index | Description |
|--------|-------|-------------|
| **Meta[0]** | 0 | Impact parameter b (fm) |
| **Meta[4]** | 4 | Transverse momentum pT (GeV) |
| **Meta[6]** | 6 | Rapidity y |
| **Values[0:6]** | 0-5 | Survival probabilities S_1S through S_1D |
| **Values[-1]** | -1 | L (angular momentum: 0=S-wave, 1=P-wave) |
| **Values[-2]** | -2 | ntraj or qweight (number of quantum trajectories) |

---

## 2. Physics Mathematics

### 2.1 State Indexing

**6-State Basis (from qtraj-nlo):**
```
Index:  0      1      2     3      4     5
State:  1S    2S    1P    3S    2P   1D
```

**9-State Basis (after hyperfine splitting):**
```
Index:  0      1     2       3       4       5     6       7       8
State:  1S    2S   1P0     1P1     1P2     3S   2P0     2P1     2P2
        └──────┬──────┘ └────────────┬────────────┘     └───────┬───────┘
              1P (duplicated 3x)          2P (duplicated 3x)
```

### 2.2 Hyperfine Splitting (6 → 9 states)

```python
def split_hyperfine_6_to_9(surv6):
    """
    Mathematica: splitHyperfine[x_] := {x[[1]],x[[2]],x[[3]],x[[3]],x[[3]],x[[4]],x[[5]],x[[5]],x[[5]]}
    """
    return np.array([
        surv6[0],  # 1S
        surv6[1],  # 2S
        surv6[2], surv6[2], surv6[2],  # 1P0, 1P1, 1P2
        surv6[3],  # 3S
        surv6[4], surv6[4], surv6[4],  # 2P0, 2P1, 2P2
    ])
```

### 2.3 Feeddown Matrix

The feeddown matrix describes sequential decays χ_b → Υ + γ:

```python
# decayMatrix[i,j] = branching ratio j → i (j feeds into i)
decayMatrix = np.array([
    [1,    0,      0,      0,     0,     0,      0,      0,     0],   # 1S (no feeddown)
    [0.2645, 1,     0,      0,     0,     0,      0,      0,     0],   # 2S ← 1S
    [0.0194, 0,     1,      0,     0,     0,      0,      0,     0],   # χ_b0(1P) ← 1S
    [0.352,  0,     0,      1,     0,     0,      0,      0,     0],   # χ_b1(1P) ← 1S
    [0.18,   0,     0,      0,     1,     0,      0,      0,     0],   # χ_b2(1P) ← 1S
    [0.0657, 0.106, 0,      0,     0,     1,      0,      0,     0],   # 3S ← 1S,2S
    [0.0038, 0.0138, 0,     0,     0,     0,      1,      0,     0],   # χ_b0(2P) ← 1S,2S
    [0.1153, 0.181, 0,     0.0091, 0,    0,      0,      1,     0],   # χ_b1(2P)
    [0.077,  0.089, 0,     0,     0.0051, 0,     0,      0,     1],   # χ_b2(2P)
])
feeddown = decayMatrix.T  # Transpose for matrix multiplication
```

### 2.4 Inclusive Cross Section Calculation

**From primordial to inclusive:**
```python
# σ_inclusive = F · (σ_direct ⊙ S_primordial)
# where ⊙ is element-wise product

def compute_inclusive(surv6, feeddown, sigmas_direct):
    """
    Compute inclusive cross sections from survival probabilities.
    
    Args:
        surv6: 6-element survival probability vector
        feeddown: 9×9 feeddown matrix
        sigmas_direct: 9-element direct cross section vector (nb)
    
    Returns:
        sigma_inclusive: 9-element inclusive cross section vector
    """
    # Split to 9 states
    surv9 = split_hyperfine_6_to_9(surv6)
    
    # Element-wise product: σ_direct ⊙ S
    yields = sigmas_direct * surv9
    
    # Feeddown: F . yields
    sigma_inclusive = feeddown @ yields
    
    return sigma_inclusive
```

### 2.5 Nuclear Modification Factor (R_AA)

```python
def compute_raa(surv6, feeddown, sigmas_direct, nbin, ncoll=1):
    """
    R_AA = σ_inclusive(AuAu) / σ_inclusive(pp)
         = (F · (σ_direct ⊙ S)) / (F · σ_direct)
    
    For minimum bias, nbin comes from Glauber model.
    """
    # Numerator: F · (σ_direct ⊙ S) · N_bin
    yields = sigmas_direct * split_hyperfine_6_to_9(surv6)
    num = feeddown @ yields * nbin * 0.1  # 0.1 converts fm^2 to nb
    
    # Denominator: F · σ_direct · N_bin  
    denom_vec = feeddown @ sigmas_direct * nbin * 0.1
    
    # R_AA
    raa = num / denom_vec
    
    return raa
```

### 2.6 Direct vs Inclusive States

- **Direct (primordial)**: Υ(1S), Υ(2S), Υ(3S) produced directly in initial state
- **Inclusive**: Direct + feeddown from χ_b states

The R_AA for excited states (2S, 3S) is reduced because:
1. Less feeding from higher states
2. Larger suppression in medium (due to larger binding radius)

---

## 3. Modular Reader Architecture

### 3.1 Design Goals

1. **Handle ALL qtraj output formats** (raw, averaged, with/without errors)
2. **Handle ANY number of impact parameters** (single b, multiple b, min-bias)
3. **Clear error messages** when format is unrecognized
4. **Extensible** for future formats

### 3.2 File Format Detection

```python
def detect_format(n_cols_meta, n_cols_values):
    """
    Detect the data file format based on column counts.
    
    Returns:
        'paired_raw':       8 columns (raw qtraj-nlo)
        'paired_averaged':  14 columns (processed with errors)  
        'paired_simple':    8 columns (simplified OO format)
        'unknown':          otherwise
    """
    if n_cols_meta == 7:
        if n_cols_values == 14:
            return 'paired_averaged'   # PbPb κ=3, κ=4 format
        elif n_cols_values == 8:
            return 'paired_simple'     # OO format
        elif n_cols_values >= 12:
            return 'paired_raw_mathematica'  # Mathematica re/im pairs
    return 'unknown'
```

### 3.3 Core Reading Functions

```python
def read_qtraj_file(filepath, verbose=True):
    """
    Universal reader for any qtraj-nlo output file.
    
    Handles:
    - .gz compressed or plain text
    - Various column formats
    - Auto-detects paired row structure
    
    Returns:
        List of (meta, values) tuples
    """
    import gzip
    import numpy as np
    
    opener = gzip.open if filepath.endswith('.gz') else open
    
    records = []
    with opener(filepath, 'rt') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Check if paired rows (even number)
    if len(lines) % 2 != 0:
        raise ValueError(f"Odd number of lines ({len(lines)}). Expected paired rows.")
    
    for i in range(0, len(lines), 2):
        meta = np.array([float(x) for x in lines[i].split()])
        vals = np.array([float(x) for x in lines[i+1].split()])
        records.append((meta, vals))
    
    if verbose:
        print(f"Read {len(records)} records from {filepath}")
    
    return records
```

### 3.4 State Extraction

```python
def extract_surv6(values_row, format_type):
    """
    Extract 6-state survival vector from values row.
    
    Different formats extract different columns:
    - 'paired_averaged': take means at indices [0,2,4,6,8,10]
    - 'paired_simple': take first 6 values [0:6]
    - 'paired_raw_mathematica': take real parts at indices [0,2,4,6,8,10]
    """
    if format_type == 'paired_averaged':
        # 14 cols: mean,err,mean,err,... => take means at 0,2,4,6,8,10
        indices = [0, 2, 4, 6, 8, 10]
    else:
        # 8 cols or 12+ cols: take first 6
        indices = [0, 1, 2, 3, 4, 5]
    
    return values_row[indices]
```

### 3.5 L Wave Matching

```python
def match_S_P_wave(records):
    """
    Match S-wave (L=0) and P-wave (L=1) records with same metadata.
    
    Constructs 6-state vector:
    S_1S = S.v6[0]
    S_2S = S.v6[1]
    S_1P = P.v6[2]   # Note: from P-wave record!
    S_3S = S.v6[3]
    S_2P = P.v6[4]   # Note: from P-wave record!
    S_1D = S.v6[5]
    """
    from collections import defaultdict
    
    grouped = defaultdict(dict)
    for meta, vals in records:
        b = meta[0]
        pt = meta[4]
        y = meta[6]
        L = int(vals[-1])
        
        key = (round(b,6), round(pt,6), round(y,6))
        grouped[key][L] = vals
    
    results = []
    for key, byL in grouped.items():
        if 0 not in byL or 1 not in byL:
            continue  # Skip if missing S or P wave
        
        s_wave = byL[0]
        p_wave = byL[1]
        
        surv6 = np.array([
            s_wave[0],  # S_1S
            s_wave[1],  # S_2S
            p_wave[2],  # S_1P (from P-wave!)
            s_wave[3],  # S_3S
            p_wave[4],  # S_2P (from P-wave!)
            s_wave[5],  # S_1D
        ])
        
        meta = np.array(key)
        results.append((meta, surv6))
    
    return results
```

---

## 4. Observable Calculations

### 4.1 R_AA vs Impact Parameter

```python
def compute_raa_vs_b(matched_records, feeddown, sigmas_direct, nbin_per_b):
    """
    Compute R_AA for each impact parameter b.
    
    Returns:
        bvals: array of b values
        raa9: array of shape (n_b, 9) for 9 states
    """
    # Group by b
    b_groups = defaultdict(list)
    for meta, surv6 in matched_records:
        b = meta[0]
        b_groups[round(b, 6)].append(surv6)
    
    # Average survival at each b
    bvals = sorted(b_groups.keys())
    raa9 = []
    
    for b in bvals:
        surv6_avg = np.mean(b_groups[b], axis=0)
        raa = compute_raa(surv6_avg, feeddown, sigmas_direct, nbin_per_b[b])
        raa9.append(raa)
    
    return np.array(bvals), np.array(raa9)
```

### 4.2 R_AA vs pT

```python
def compute_raa_vs_pt(matched_records, pt_edges, feeddown, sigmas_direct):
    """
    Compute R_AA in pT bins (integrated over y and b).
    """
    # Filter and bin
    pt_centers = 0.5 * (pt_edges[:-1] + pt_edges[1:])
    raa_bins = []
    
    for i in range(len(pt_edges) - 1):
        p0, p1 = pt_edges[i], pt_edges[i+1]
        
        bin_surv6 = [s for m, s in matched_records if p0 <= m[1] < p1]
        
        if bin_surv6:
            surv6_avg = np.mean(bin_surv6, axis=0)
            # Use nbin=1 for min-bias integration
            raa = compute_raa(surv6_avg, feeddown, sigmas_direct, nbin=1.0)
            raa_bins.append(raa)
        else:
            raa_bins.append(np.full(9, np.nan))
    
    return pt_centers, np.array(raa_bins)
```

---

## 5. Glauber Model Integration

### 5.1 PbPb 5.023 TeV

From `glauber-data/bvscData.tsv` and `nbinvsbData.tsv`:
- b range: 0 - ~20 fm
- N_bin(b) from optical Glauber
- N_part(b) provided as lookup table

### 5.2 OO 5.36 TeV

From `OxygenOxygen5360/glauber-data/`:
- b range: 0 - ~15 fm
- N_bin(b), N_part(b) both available

### 5.3 Interpolation

```python
class GlauberInterpolator:
    def __init__(self, bvsc_df, nbin_df, npart_df=None):
        # Build interpolation functions
        self.b_to_nbin = interp1d(nbin_df.b, nbin_df.Nbin, fill_value='extrapolate')
        self.b_to_npart = interp1d(npart_df.b, npart_df.Npart, fill_value='extrapolate') if npart_df is not None else None
        self.b_to_centrality = interp1d(bvsc_df.b, bvsc_df.c, fill_value='extrapolate')
```

---

## 6. Expected Results

### 6.1 PbPb 5 TeV (κ=3)

From Mathematica `raavsnpart-lhc3d-k3.m`:

| N_part | R_AA(1S) | R_AA(2S) | R_AA(3S) |
|--------|----------|----------|----------|
| 0.97   | 1.000    | 1.000    | 1.000    |
| 3.81   | 0.998    | 0.998    | 0.994    |
| 9.67   | 0.970    | 0.938    | 0.888    |
| 21.3   | 0.943    | 0.724    | 0.566    |
| 41.1   | 0.757    | 0.384    | 0.269    |
| 70.8   | 0.619    | 0.217    | 0.141    |
| 112.4  | 0.565    | 0.149    | 0.082    |
| 168.5  | 0.493    | 0.084    | 0.059    |
| 243.5  | 0.421    | 0.080    | 0.050    |
| 315.9  | 0.425    | 0.099    | 0.039    |
| 375.0  | 0.370    | 0.065    | 0.046    |
| 406.1  | 0.373    | 0.058    | 0.042    |

### 6.2 pT Binning (Mathematica vs Python)

| Bin | Mathematica | Python (current) |
|-----|-------------|------------------|
| 0-5 | 0.4306     | 0.6153          |
| 5-10| 0.4453     | 0.6603          |
| 10-15| 0.4756    | 0.6737          |
| 15-20| 0.4651    | 0.6325          |
| 20-25| 0.4638    | 0.5963          |
| 25-30| 0.4177    | 0.6885          |

**Issue**: Different binning (5 GeV vs 2 GeV). Need to configure binning to match Mathematica.

---

## 7. Known Issues and Fixes

### 7.1 wReg Bug in OO Data

The `processEvents.py` script drops column 6 (primordial d6) when averaging:
```python
meanData = np.mean(dataArray, axis=0).tolist()[:-2]  # Drops last 2: d6, ntraj
```

**Fix**: Re-read raw datafile.gz and manually add primordial contribution:
```python
# L=0: S_1S += mean(d6_prim)
# L=1: S_1P += mean(d6_prim)
```

### 7.2 Import Path Issues

Current script adds wrong path to sys.path:
```python
# Current (broken):
sys.path.insert(0, os.path.join(REPO_ROOT, "hnm", "qtraj-nlo", "qtraj_out_analysis", "src"))

# Should be:
sys.path.insert(0, os.path.join(REPO_ROOT, "hnm", "qtraj-nlo", "qtraj_out_analysis"))
```

Then import as:
```python
from qtraj_analysis.src.io import read_whitespace_table
# OR add 'src' to path and use:
# from qtraj_analysis.io import read_whitespace_table
```

---

## 8. Summary: What Each Module Does

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **io.py** | Read data files | `read_whitespace_table()`, `parse_records()` |
| **matching.py** | Match S/P waves | `build_observables()` |
| **feeddown.py** | Feeddown matrix | `build_feeddown_matrix()`, `apply_feeddown_to_raa6()` |
| **binning.py** | Bin observables | `compute_raa_vs_b()`, `compute_raa_vs_pt()`, `compute_raa_vs_y()` |
| **glauber.py** | Glauber model | `load_glauber()`, `GlauberInterpolator` |
| **schema.py** | Data structures | `Record`, `TrajectoryObs`, `RaaVsBResult` |

---

*Generated: April 2026*
*For verification against Mathematica outputs in arXiv-2305.17841v2*
