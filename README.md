# SBA-sampler
A Python implementation of an interpretable, local-environment-coverage strategy for selecting compact and chemically diverse training structures for machine-learned interatomic potentials (MLIPs).

The sampler represents each candidate structure as a set of discrete local chemical-environment types and selects structures according to marginal coverage gain rather than configuration count alone. The current defaults are designed for disordered Fe–C systems, but the center elements, cutoff radii, clustering tolerances, and coverage criteria are configurable.

This repository accompanies the manuscript:

Interpretable Local-Environment Coverage Strategy Enables Data-Efficient Neuroevolutionary Potentials for Disordered Fe–C Systems
