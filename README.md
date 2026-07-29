# SBA-sampler
A Python implementation of an interpretable, local-environment-coverage strategy for selecting compact and chemically diverse training structures for machine-learned interatomic potentials (MLIPs).

The sampler represents each candidate structure as a set of discrete local chemical-environment types and selects structures according to marginal coverage gain rather than configuration count alone. The current defaults are designed for disordered Fe–C systems, but the center elements, cutoff radii, clustering tolerances, and coverage criteria are configurable.

This repository accompanies the manuscript:

Interpretable Local-Environment Coverage Strategy Enables Data-Efficient Neuroevolutionary Potentials for Disordered Fe–C Systems
Overview

The code addresses a common problem in MLIP development: a large candidate pool can contain many geometrically different structures but only a limited number of distinct local chemical environments. Labeling all candidates with density-functional theory (DFT) is therefore expensive and often redundant.

The sampler performs the following tasks:

1. filters structures containing unphysically short bonds;
2. extracts multi-cutoff local fingerprints centered on selected elements;
3. assigns chemically interpretable metadata to each fingerprint;
4. clusters similar raw fingerprints into effective environment types;
5. represents every structure as a set of effective environment types;
6. removes near-duplicate structures using Jaccard similarity;
7. prioritizes rare environment types;
8. performs greedy set-cover selection;
9. removes redundant selected structures without losing target coverage;
10. optionally supplements an existing training set according to its uncovered environment gap; and
11. generates manifests, summaries, checkpoints, caches, and publication-style plots.
The implementation supports two primary modes:

- Initial training-set selection: select a compact subset from one candidate pool.
- Coverage-gap supplementation: compare a candidate pool with an existing probe/training set and select structures that add missing environments.
**#Requirements**
