# OpenMECP

<p align="center">
  <img src="./images/omecp-logo.svg" alt="OpenMECP" style="width: 45%; max-width: 400px;">
</p>

**OpenMECP** is a high-performance Rust implementation of the MECP (Minimum Energy Crossing Point)
optimizer for locating crossing points between two potential energy surfaces (PES) in quantum
chemistry calculations. OpenMECP is developed and maintained by Dr. Le Nhan Pham.

> **Status**: Beta testing phase — OpenMECP is highly robust and ready for production MECP calculations.
> All supported features are actively tested and improved.

## What is an MECP?

A Minimum Energy Crossing Point is the lowest-energy geometry at which two electronic states
become degenerate. MECPs are essential for understanding:

- **Photochemical reactions** — light-driven processes crossing between electronic states
- **Spin-forbidden processes** — reactions changing spin multiplicity
- **Intersystem crossing** — transitions between singlet and triplet manifolds
- **Conical intersections** — degeneracy points governing ultrafast photochemistry
- **Non-adiabatic dynamics** — processes beyond the Born–Oppenheimer approximation

## Key Algorithm

OpenMECP implements the algorithm reported by Harvey et al. in
*Theor. Chem. Acc.* **99**, 95–99 (1998).

The MECP effective gradient combines two orthogonal components:

**f-vector** — drives the energy difference to zero:

$$\mathbf{f} = (E_1 - E_2)\,\hat{\mathbf{x}}$$

**g-vector** — minimizes the average energy perpendicular to the gradient difference:

$$\mathbf{g} = \mathbf{f}_1 - (\hat{\mathbf{x}} \cdot \mathbf{f}_1)\,\hat{\mathbf{x}}$$

where the normalized gradient difference is:

$$\hat{\mathbf{x}} = \frac{\mathbf{f}_1 - \mathbf{f}_2}{|\mathbf{f}_1 - \mathbf{f}_2|}$$

## Feature Highlights

| Category | Features |
|---|---|
| **Optimizers** | BFGS, GDIIS, GEDIIS, Sequential Hybrid, GDIIS_blend |
| **Hessian Updates** | PSB (default), BFGS, Bofill, Powell, adaptive mix |
| **QM Programs** | Gaussian, ORCA, Custom JSON interface |
| **Constraints** | Bond, angle, dihedral (Lagrange multipliers) |
| **Scans** | 1D and 2D PES scans |
| **Path Methods** | LST interpolation, coordinate driving, NEB |
| **Run Modes** | Normal, Read, NoRead, Stable, InterRead |
| **Configuration** | Hierarchical INI config with sensible defaults |

## Quick Navigation

- New to OpenMECP? Start with [Installation](installation.md) then [Quick Start](quick-start.md).
- Ready to run? See the [Input File Format](input-format/README.md) and [Keywords Reference](input-format/keywords.md).
- Want to understand the algorithms? See [MECP Algorithm](algorithms/README.md).
- Looking for the full API? See [API Reference](api-reference.md).

## Citation

If you use OpenMECP in your research, please cite this preprint:

>Pham, Le Nhan. OpenMECP: *A High-Performance Rust Implementation for the Rigorous Location of Minimum Energy Crossing Points in Chemical Dynamics.* 2026, https://doi.org/10.13140/RG.2.2.21309.73443.
