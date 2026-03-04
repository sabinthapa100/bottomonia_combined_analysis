# Bottomonia Combined Analysis

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This repository contains the **Bottomonia Combined Analysis** codebase developed by **Sabin Thapa**. It provides a comprehensive framework for studying charmonium production in heavy‑ion collisions, covering both **Cold Nuclear Matter (CNM)** and **Hot Nuclear Matter (HNM)** effects, and delivering combined nuclear modification factor predictions for LHC energies -- focused on Oxygen Oxygen 5.36 TeV LHC.

For the inputs, in inputs --> nPDF, you can download Oxygen (or any other) nPDF set from here, https://research.hip.fi/qcdtheory/nuclear-pdfs/epps21/

### Scientific Scope

1. **NM Effect Calculations**
   - **nPDF modifications** (via `npdf_code` and associated notebooks)
   - **Energy loss** and **transverse momentum broadening** (implemented in `eloss_code`)
   - **Future extension**: nuclear absorption models will be integrated.
2. **HNM (QGP) Effect Calculations**
   - **Dissociation** of charmonium states in the quark‑gluon plasma.
   - **Regeneration** mechanisms based on recombination models.
3. **Combined Nuclear Modification Factor (R<sub>AA</sub>)**
   - Production of a unified R<sub>AA</sub> that incorporates both CNM and HNM contributions for **LHC** (√s = 5.02 TeV) and **RHIC** (√s = 200 GeV) collision systems.
4. **Experimental Data Integration**
   - Curated datasets from **RHIC** (PHENIX, STAR) and **LHC** (ALICE, CMS) are provided in `experimental_data_code`.
   - Utilities to load, interpolate, and compare theory curves with data.
5. **Upcoming Project / Paper**
   - The codebase is structured to support the preparation of a **peer‑reviewed manuscript** detailing the combined CNM+HNM analysis, with ready‑to‑use figures, tables, and systematic uncertainty evaluations.

## Project Structure

```

```

## Installation

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Contact

**Sabin Thapa** – [sabin.thapa@kent.edu](mailto:sthapa3@kent.edu)

---

*Generated on 2026‑01‑21.*
