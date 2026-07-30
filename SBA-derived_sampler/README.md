# SBA-Derived_Sampler

# Structures data

The XYZ trajectory is available in the GitHub Release `data-v1.0`.


## Table of Contents

---

## Overview

The code addresses a common problem in MLIP development: a large candidate pool can contain many geometrically different structures but only a limited number of distinct local chemical environments. Labeling all candidates with density-functional theory (DFT) is therefore expensive and often redundant.

The implementation supports two primary modes:

- **Initial training-set selection:** select a compact subset from one candidate pool.
- **Coverage-gap supplementation:** compare a candidate pool with an existing probe/training set and select structures that add missing environments.

---

### Requirements

- Python 3.9+
- NumPy
- SciPy
- ASE
- `spdkit`

## 

--- 

## Workflow

### 1. Physical pre-filter

A candidate is rejected if any interatomic distance satisfies

```text
d_ij < c × (r_i + r_j)
```

where:

- `d_ij` is the minimum-image distance between atoms `i` and `j`;
- `r_i` and `r_j` are ASE covalent radii; and
- `c` is `bond_filter_coeff`, equal to `0.8` by default.

Rejected structures can be exported to a separate XYZ file together with a CSV manifest.

### 2. Multi-cutoff local fingerprint extraction

For each configured cutoff radius and center element, the sampler uses `spdkit` to create a central local environment and extract a raw topology fingerprint.

Default settings:

```python
rcut_list = [2.7, 3.0]
center_elements = ("Fe", "C")
```

Each raw fingerprint is associated with a chemical metadata key containing:

- cutoff radius;
- center element;
- coordination number; and
- sorted neighboring-element composition.

### 3. Raw-fingerprint compression

Raw fingerprints sharing the same chemical metadata key are further grouped into effective environment types.

When geometric clustering is enabled, the feature vector contains:

- sorted center–neighbor bond lengths;
- mean neighbor–neighbor angle; and
- standard deviation of neighbor–neighbor angles.

Bond and angular features are normalized by `bond_tol` and `angle_tol`. The default backend is SciPy Ward hierarchical clustering.

### 4. Frozen effective-type reference frame

The mapping can be stored as:

```text
fp_to_eff_frozen.pkl
```

Reusing this file keeps the effective environment-type space fixed across later runs. **This is important when comparing an initial dataset with later candidate pools: the meaning of an effective type should not change merely because a different subset is being processed.**

If a new raw fingerprint is absent from the frozen mapping, it is assigned a deterministic singleton effective type.

### 5. Structure-level Jaccard pre-deduplication

Every structure is represented by the set of effective environment types it contains.

For two structures with environment sets `A` and `B`, the Jaccard similarity is

```text
J(A, B) = |A ∩ B| / |A ∪ B|
```

Structures above the configured similarity threshold are grouped, and one representative is retained. The code can use different thresholds for rare- and common-environment passes.

**Note:** The essence of **Jaccard deduplication** lies in removing duplicates based on the structure of local environment sets, rather than based on atomic coordinates or global geometry.Each structure is represented by an "effective local environment type set".

structure A = {e1, e2, e3, e4, e5}
structure B = {e1, e2, e3, e4, e6}
structure C = {e7, e8, e9}

Here, e1, e2... are not individual atoms, but rather the effective environmental types obtained after local fingerprint extraction and clustering. Although the atomic coordinates of structures A and B may not be exactly the same, the local chemical environments they contain are highly similar. If both A and B are sent for DFT analysis, the additional information obtained may be very little.

### 6. Two-tier greedy coverage

Environment types are divided into two groups:

- **rare environments:** present in no more than `rare_env_freq_threshold` structures;
- **common environments:** all remaining types.

The default selection has two passes:

1. cover all reachable rare types;
2. cover common types until `common_env_coverage_pct` is reached.

At each step, the lazy greedy set-cover routine selects the structure that covers the largest number of currently uncovered target types.

### 7. Post-selection pruning

After greedy selection, the code removes structures whose covered target types remain represented by other selected structures.

An optional post-cover Jaccard pass further removes a structure only when:

1. it is sufficiently similar to another remaining structure; and
2. all of its target environment types remain covered after removal.

This coverage-safety condition protects uniquely represented rare environments.

### 8. Probe-set gap supplementation

When `probe_file` is provided, the code extracts environments from the existing probe/training structures and computes

```text
coverage gap = candidate-universe types - probe-covered types
```

Only candidate structures containing gap types are considered for Phase-2 supplementation.

The code can then construct a compact joint set from:

- the original probe structures; and
- the selected Phase-2 supplement.

---

## Input Data

The script reads all structures using:

```python
ase.io.read(path, index=":")
```

Any ASE-readable multi-structure format may work, but the present workflow is designed around multi-frame extended XYZ files.

### Candidate pool: `large_file`

`large_file` is the candidate trajectory used to:

- construct the local-environment universe; and
- select representative structures.

Each frame should include:

- element symbols;
- Cartesian coordinates;
- a valid simulation cell;
- periodic-boundary information, when applicable; and
- the elements listed in `center_elements`.

A cutoff is skipped for a frame if:

- the cell is singular; or
- any cell-vector length is smaller than the cutoff radius.

The current script does not provide a command-line parser. Configure a run by editing the `Config(...)` block at the bottom of `SBA-derived_sampler.py`.

### Mode A: select an initial training set

```python
if __name__ == "__main__":
    cfg = Config(
        large_file="data/candidate_pool.xyz",
        probe_file="",
        output_dir="results/initial_selection",

        rcut_list=[2.7, 3.0],
        center_elements=("Fe", "C"),

        use_geom_cluster=True,
        geom_cluster_backend="ward",
        strict_geom_cluster_backend=True,
        bond_tol=0.05,
        angle_tol=5.0,

        bond_filter_enabled=True,
        bond_filter_coeff=0.8,

        jaccard_dedup_enabled=True,
        jaccard_dedup_threshold=0.85,
        jaccard_dedup_threshold_common=0.70,
        jaccard_dedup_size_band_frac=0.25,

        two_tier_cover_enabled=True,
        rare_env_freq_threshold=5,
        common_env_coverage_pct=85.0,

        post_cover_jaccard_dedup_enabled=True,
        post_cover_jaccard_threshold=0.85,
        post_cover_jaccard_size_band_frac=0.25,

        initial_train_enabled=True,
        initial_train_budget=0,

        freeze_fp_to_eff_enabled=True,
        checkpoint_enabled=True,
        resume_from_checkpoint=True,
        publication_plot_enabled=True,
    )
    main(cfg)
```

Run:

```bash
python SBA-derived_sampler.py
```

Main selected structure file:

```text
results/initial_selection/initial_training_selected.xyz 
```

### Existing training/probe set: `probe_file`

`probe_file` is optional.

- Use `probe_file=""` for initial-set selection.
- Provide a path to an existing training/reference trajectory for gap supplementation.

When `preserve_input_xyz_format=True`, selected frames are copied directly from the source text so that comments and extended-XYZ properties are preserved.

### Mode B: supplement an existing training set

```python
if __name__ == "__main__":
    cfg = Config(
        large_file="data/new_candidate_pool.xyz",
        probe_file="data/current_training_set.xyz",
        output_dir="sampler_iter-1",

        rcut_list=[2.7, 3.0],
        center_elements=("Fe", "C"),

        phase2_budget=20000,
        probe_phase2_joint_min_cover_enabled=True,

        # Keep all fingerprint and clustering settings consistent
        # with the run that created the reference type space.
        freeze_fp_to_eff_enabled=True,
    )
    main(cfg)
```

Run:

```bash
python SBA-derived_sampler.py
```

Principal outputs:

```text
results/iteration_01/phase2_supplement.xyz
results/iteration_01/probe_phase2_joint_min_cover_selected.xyz
```



---

**Optional CIF export**

When `export_cluster_env_cifs=True`, representative local clusters are exported with an index containing:

- effective environment type;
- raw fingerprint;
- source structure index;
- center atom;
- center element; and
- cutoff radius.

Use `cluster_env_cif_max_per_cluster` and `cluster_env_cif_max_total` to prevent very large exports.

**Figures and CIF export**

| Parameter                    | Default | Description                                 |
| ---------------------------- | ------- | ------------------------------------------- |
| `publication_plot_enabled`   | `True`  | Generate summary figures                    |
| `plot_universe_summary`      | `True`  | Plot fingerprint compression                |
| `plot_selection_convergence` | `True`  | Plot coverage convergence                   |
| `plot_topology_state_map`    | `True`  | Plot qualitative environment-state maps     |
| `plot_gap_supplement_panels` | `True`  | Plot gap-supplement panels                  |
| `plot_joint_summary`         | `True`  | Plot joint-cover summary                    |
| `export_cluster_env_cifs`    | `False` | Export representative local clusters as CIF |



## Contact

```text
Corresponding author: Liying An
Email: anliying20@mails.ucas.ac.cn
Institution: State Key Laboratory of Coal Conversion, Institute of Coal Chemistry
```
