#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compact topology-driven Fe-C local-environment sampler.

Reduction stack (applied in order)
------------------------------------
Route A  Jaccard pre-deduplication
         Before greedy cover, candidate structures whose effective-environment
         sets have Jaccard similarity >= jaccard_dedup_threshold are grouped and
         only the representative with the largest env-set is forwarded.

Route C  Two-tier greedy cover
         Splits effective environment types into "rare" (freq <= rare_env_freq_threshold)
         and "common" tiers.  Pass 1 covers all rare types (mandatory, 100 %).
         Pass 2 covers common types up to common_env_coverage_pct using the more
         aggressive jaccard_dedup_threshold_common.

Route D  Post-cover coverage-aware Jaccard dedup
         After the greedy cover produces K selected structures, a second Jaccard
         pass runs directly on those K structures.  A structure S is removed only
         when BOTH conditions hold:
           (a) there exists another selected structure T with J(S, T) >= threshold, AND
           (b) every env type in S is also covered by at least one other selected structure.

Root-cause fix: Frozen fp_to_eff mapping
         The Ward geometry clustering in universe.compress() is NOT idempotent:
         fp_to_eff depends on fp_geom averages computed from ALL structures in the
         universe.  When re-running on a K-structure subset, fp_geom values shift,
         Ward boundaries move, formerly distinct eff types merge, and structures that
         uniquely covered them become redundant.

         Fix: after the first full run, save fp_to_eff to freeze_fp_to_eff_file.
         On subsequent runs (including re-runs on selected subsets), this mapping is
         loaded and applied directly — Ward re-clustering is skipped entirely.
         This makes the effective-type space a FIXED REFERENCE FRAME that does not
         change when the structure set changes.

         New Config fields:
           freeze_fp_to_eff_enabled  bool   default True
           freeze_fp_to_eff_file     str    default "fp_to_eff_frozen.pkl"
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator, PercentFormatter
    MATPLOTLIB_OK = True
except Exception as _e:
    MATPLOTLIB_OK = False
    plt = None
    Patch = None
    MaxNLocator = None
    PercentFormatter = None
    warnings.warn(f"[WARN] matplotlib unavailable; plots will be skipped: {_e}")

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x=None, *args, **kwargs):
        return x if x is not None else []

try:
    import scipy
    from scipy.cluster.hierarchy import fclusterdata
    SCIPY_OK = True
    SCIPY_VERSION = getattr(scipy, "__version__", "unknown")
except Exception:
    scipy = None
    fclusterdata = None
    SCIPY_OK = False
    SCIPY_VERSION = "unavailable"

try:
    from ase import Atoms
    from ase.data import covalent_radii
    from ase.io import read, write
    ASE_OK = True
except Exception as _e:
    Atoms = Any  # type: ignore
    covalent_radii = None
    read = None
    write = None
    ASE_OK = False
    ASE_IMPORT_ERROR = _e

_spdkit = None
_dwim = None


def require_ase() -> None:
    if not ASE_OK:
        raise SystemExit(f"[FATAL] ASE is required to run this script: {ASE_IMPORT_ERROR}")


def require_spdkit() -> Tuple[Any, Any]:
    global _spdkit, _dwim
    if _spdkit is None or _dwim is None:
        try:
            import spdkit as _spdkit_mod
            from spdkit import dwim as _dwim_mod
            _spdkit = _spdkit_mod
            _dwim = _dwim_mod
        except Exception as e:
            raise SystemExit(f"[FATAL] spdkit is required to extract local fingerprints: {e}")
    return _spdkit, _dwim


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # Input / output
    large_file: str = "MD-300.xyz"
    probe_file: str = ""
    output_dir: str = "topo_sampler_probe_phase2_out"

    preserve_input_xyz_format: bool = True
    write_annotated_xyz: bool = False
    annotated_xyz_suffix: str = "_annotated"

    # Cache / checkpoint
    force_rebuild_universe: bool = False
    universe_cache_file: str = "universe.pkl"
    checkpoint_enabled: bool = True
    resume_from_checkpoint: bool = True
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 500
    checkpoint_keep_after_success: bool = False

    probe_fps_cache_enabled: bool = True
    probe_fps_cache_file: str = "probe_fps_cache.pkl"
    probe_fps_checkpoint_interval: int = 100

    # cluster_env_cifs export
    export_cluster_env_cifs: bool = False
    cluster_env_cif_dir: str = "cluster_env_cifs"
    cluster_env_cif_max_per_cluster: int = 0
    cluster_env_cif_max_total: int = 0
    cluster_env_cif_vacuum: float = 8.0
    cluster_env_cif_distance_eps: float = 1.0e-6
    cluster_env_cif_clear_existing: bool = True

    # Fingerprint extraction and compression
    rcut_list: List[float] = field(default_factory=lambda: [2.7, 3.0])
    center_elements: Tuple[str, ...] = ("Fe", "C")
    use_geom_cluster: bool = True
    geom_cluster_backend: str = "ward"
    strict_geom_cluster_backend: bool = True
    bond_tol: float = 0.03
    angle_tol: float = 3.0
    geom_min_frac: float = 0.5
    geom_max_samples: int = 5
    topology_similarity_threshold: float = 1.0

    # Physical pre-filter
    bond_filter_enabled: bool = True
    bond_filter_coeff: float = 0.8

    # Export structures rejected by the physical pre-filter
    # When bond_filter_enabled=True, structures rejected by BondFilter are
    # written to output_dir / nonphysical_xyz_file, and their source indices
    # are written to output_dir / nonphysical_manifest_csv.
    write_nonphysical_xyz: bool = True
    nonphysical_xyz_file: str = "nonphysical_bond_filtered.xyz"
    nonphysical_manifest_csv: str = "nonphysical_bond_filtered_manifest.csv"

    # ── Route A: Jaccard pre-deduplication ────────────────────────────────────
    jaccard_dedup_enabled: bool = True
    jaccard_dedup_threshold: float = 0.85
    jaccard_dedup_threshold_common: float = 0.70
    jaccard_dedup_size_band_frac: float = 0.25

    # ── Route C: Two-tier greedy cover ────────────────────────────────────────
    two_tier_cover_enabled: bool = True
    rare_env_freq_threshold: int = 5
    common_env_coverage_pct: float = 85.0

    # ── Route D: Post-cover coverage-aware Jaccard dedup ─────────────────────
    # Runs AFTER the greedy cover on the K selected structures.
    # Removes structure S when:
    #   (a) another selected structure T has J(S,T) >= post_cover_jaccard_threshold, AND
    #   (b) all env types in S are covered by at least one OTHER selected structure.
    #
    # This closes the logical gap where the greedy cover selects two structures
    # that are very similar (e.g. J=0.96) because each uniquely covered one env
    # type originally, but that unique env is also covered by a third structure.
    #
    # post_cover_jaccard_threshold:
    #   Should be >= jaccard_dedup_threshold.  Lower = more aggressive post-pruning.
    #   Default: same as jaccard_dedup_threshold so both passes use the same criterion.
    #
    # The function guarantees: after pruning, for any two remaining structures A, B,
    #   J(A, B) < threshold  OR  at least one of them has a unique env type
    #   (not covered by any other remaining structure).
    post_cover_jaccard_dedup_enabled: bool = True
    post_cover_jaccard_threshold: float = 0.85
    post_cover_jaccard_size_band_frac: float = 0.25

    # ── Frozen fp_to_eff mapping  (ROOT-CAUSE FIX) ────────────────────────────
    # ROOT CAUSE OF RE-RUN REDUNDANCY
    # ─────────────────────────────────────────────────────────────────────────
    # The Ward geometry clustering in universe.compress() is NOT idempotent:
    # the fp_to_eff assignment depends on fp_geom averages computed from ALL
    # structures present in the universe.  When the script is re-run on a
    # subset K of the originally selected structures, fp_geom values shift
    # (fewer samples), Ward cluster boundaries move, and pairs of raw fps that
    # were in DIFFERENT clusters (distinct eff types) can fall into the SAME
    # cluster.  Structures that uniquely covered those formerly-distinct eff
    # types now cover the same eff type and become redundant.
    #
    # The fix: after the first full run, save the fp_to_eff mapping to a file.
    # On subsequent runs (including re-runs on subsets), load this frozen
    # mapping and apply it directly, bypassing Ward re-clustering entirely.
    # This makes the effective-type space a FIXED REFERENCE FRAME that does
    # not change when the structure set changes.
    #
    # freeze_fp_to_eff_enabled:
    #   True  → save mapping after first run; load and reuse on subsequent runs.
    #   False → always re-cluster (original behaviour; NOT recommended if you
    #           ever plan to re-run on a subset).
    #
    # freeze_fp_to_eff_file:
    #   Path to the frozen mapping pickle, relative to output_dir.
    #   The file is created automatically on the first run when it does not
    #   exist, and loaded automatically on subsequent runs when it does exist.
    #   Delete it to force a fresh clustering (e.g. after changing bond_tol /
    #   angle_tol / rcut_list).
    freeze_fp_to_eff_enabled: bool = True
    freeze_fp_to_eff_file: str = "fp_to_eff_frozen.pkl"

    # Single-set initial training selection
    initial_train_enabled: bool = True
    initial_train_budget: int = 0
    initial_train_target_coverage_pct: float = 100.0
    initial_train_output_xyz: str = "initial_training_selected.xyz"
    initial_train_manifest_csv: str = "initial_training_manifest.csv"
    initial_train_report_csv: str = "initial_training_report.csv"
    initial_train_summary_json: str = "initial_training_summary.json"

    # Phase2 gap supplement
    phase2_budget: int = 20000
    exclude_probe_indices_from_phase2_candidates: bool = False
    phase2_output_xyz: str = "phase2_supplement.xyz"
    phase2_manifest_csv: str = "phase2_supplement_manifest.csv"

    # Probe + Phase2 joint minimum cover
    probe_phase2_joint_min_cover_enabled: bool = True
    probe_phase2_joint_output_xyz: str = "probe_phase2_joint_min_cover_selected.xyz"
    probe_phase2_joint_manifest_csv: str = "probe_phase2_joint_min_cover_manifest.csv"
    probe_phase2_joint_report_csv: str = "probe_phase2_joint_min_cover_report.csv"
    probe_phase2_joint_summary_json: str = "probe_phase2_joint_min_cover_summary.json"
    probe_phase2_joint_protect_probe_only_raw: bool = True
    probe_phase2_joint_prefer_phase2_on_tie: bool = True
    probe_phase2_joint_prune_probe_first: bool = True

    # Reports
    write_sampler_report: bool = True
    report_file: str = "report.txt"
    rare_fingerprint_max_structures: int = 1

    # Publication figures
    publication_plot_enabled: bool = True
    publication_plot_dir: str = "publication_figures"

    plot_universe_summary: bool = True
    plot_selection_convergence: bool = True
    plot_topology_state_map: bool = True
    plot_gap_supplement_panels: bool = True
    plot_gap_supplement_max_curve_points: int = 650
    plot_joint_summary: bool = True

    save_individual_panels: bool = True
    individual_panel_dir: str = "individual_panels"
    individual_panel_dpi: int = 300
    individual_panel_format: str = "png"

    plot_font_family: str = "DejaVu Serif"
    plot_title_size: int = 18
    plot_suptitle_size: int = 16
    plot_label_size: int = 15
    plot_tick_size: int = 13
    plot_legend_size: int = 11
    plot_font_weight: str = "bold"
    plot_axis_linewidth: float = 1.6

    plot_hash_dim: int = 128
    plot_topology_max_types: int = 12000
    plot_state_color_before: str = "#008837"
    plot_state_color_new: str = "#D7191C"
    plot_state_color_uncovered: str = "#BDBDBD"
    plot_state_alpha_before: float = 0.50
    plot_state_alpha_new: float = 0.90
    plot_state_alpha_uncovered: float = 0.16
    plot_state_size_before_scale: float = 0.85
    plot_state_size_new_scale: float = 1.45

    verbose: bool = True


# =============================================================================
# Small utilities
# =============================================================================

def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return _jsonable(asdict(obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_jsonable(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _file_fingerprint(path: str) -> Dict[str, Any]:
    p = Path(path)
    try:
        st = p.stat()
        return {"path": str(p.resolve()), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    except FileNotFoundError:
        return {"path": str(p), "missing": True}


def _make_signature(name: str, payload: Dict[str, Any]) -> str:
    blob = json.dumps(
        {"name": name, "payload": _jsonable(payload)},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _atomic_pickle_dump(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{int(time.time() * 1e6)}")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    tmp.replace(path)


def _atomic_pickle_load(path: Path) -> Optional[Any]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        warnings.warn(f"[Checkpoint][WARN] Cannot read {path}; ignored: {e}")
        return None


def _load_checkpoint(path: Path, signature: str) -> Optional[Dict[str, Any]]:
    data = _atomic_pickle_load(path)
    if not isinstance(data, dict):
        return None
    if data.get("signature") != signature:
        print(f"[Checkpoint] Ignore stale checkpoint because signature changed: {path}")
        return None
    return data


def _save_checkpoint(path: Path, signature: str, payload: Dict[str, Any]) -> None:
    _atomic_pickle_dump(
        {"signature": signature, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), **payload},
        path,
    )


def _maybe_remove_checkpoint(path: Optional[Path], keep: bool) -> None:
    if path is not None and not keep:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def _safe_name(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9_.=+:-]+", "_", str(text))
    if len(text) <= max_len:
        return text
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{text[:max_len-11]}_{h}"


def _rcut_from_fp_or_eff(key: str, default: float = 3.0) -> float:
    m = re.search(r"(?:^|:)r(\d+(?:\.\d+)?)", key)
    if m:
        return float(m.group(1))
    m = re.search(r"^r(\d+(?:\.\d+)?)", key)
    return float(m.group(1)) if m else default


def _center_from_eff(key: str) -> str:
    m = re.search(r":(Fe|C|[A-Z][a-z]?):cn", key)
    return m.group(1) if m else "?"


def _universe_signature(cfg: Config) -> str:
    return _make_signature("compact_universe_v1", {
        "large_file": _file_fingerprint(cfg.large_file),
        "rcut_list": cfg.rcut_list,
        "center_elements": cfg.center_elements,
        "use_geom_cluster": cfg.use_geom_cluster,
        "geom_cluster_backend": cfg.geom_cluster_backend,
        "strict_geom_cluster_backend": cfg.strict_geom_cluster_backend,
        "scipy_ok": SCIPY_OK,
        "scipy_version": SCIPY_VERSION,
        "bond_tol": cfg.bond_tol,
        "angle_tol": cfg.angle_tol,
        "geom_min_frac": cfg.geom_min_frac,
        "geom_max_samples": cfg.geom_max_samples,
        "topology_similarity_threshold": cfg.topology_similarity_threshold,
        "bond_filter_enabled": cfg.bond_filter_enabled,
        "bond_filter_coeff": cfg.bond_filter_coeff,
        "export_cluster_env_cifs": cfg.export_cluster_env_cifs,
    })


def _frozen_fp_to_eff_signature(cfg: Config) -> str:
    """
    Signature for the frozen fp_to_eff mapping.

    Covers only the clustering hyperparameters, NOT the input file.  This
    allows the same frozen mapping to be reused across different runs that use
    the same clustering settings (e.g. re-running on a selected subset), while
    still invalidating the cache when the user changes bond_tol, angle_tol,
    rcut_list, or the clustering backend.
    """
    return _make_signature("frozen_fp_to_eff_v1", {
        "rcut_list": cfg.rcut_list,
        "center_elements": cfg.center_elements,
        "use_geom_cluster": cfg.use_geom_cluster,
        "geom_cluster_backend": cfg.geom_cluster_backend,
        "strict_geom_cluster_backend": cfg.strict_geom_cluster_backend,
        "scipy_ok": SCIPY_OK,
        "scipy_version": SCIPY_VERSION,
        "bond_tol": cfg.bond_tol,
        "angle_tol": cfg.angle_tol,
        "geom_min_frac": cfg.geom_min_frac,
        "topology_similarity_threshold": cfg.topology_similarity_threshold,
    })


def save_frozen_fp_to_eff(path: Path, cfg: Config, fp_to_eff: Dict[str, str]) -> None:
    """Persist the fp_to_eff mapping so that re-runs on subsets use the SAME type space."""
    sig = _frozen_fp_to_eff_signature(cfg)
    _atomic_pickle_dump({
        "signature": sig,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_entries": len(fp_to_eff),
        "fp_to_eff": fp_to_eff,
    }, path)
    print(f"[FrozenFpToEff] Saved {len(fp_to_eff):,} entries → {path}")


def load_frozen_fp_to_eff(path: Path, cfg: Config) -> Optional[Dict[str, str]]:
    """
    Load a previously saved fp_to_eff mapping.

    Returns None if the file does not exist or the signature has changed
    (indicating that clustering hyperparameters have been modified and the
    mapping needs to be recomputed).
    """
    data = _atomic_pickle_load(path)
    if not isinstance(data, dict):
        return None
    sig = _frozen_fp_to_eff_signature(cfg)
    if data.get("signature") != sig:
        print(f"[FrozenFpToEff] Signature mismatch — clustering hyperparameters changed. "
              f"Will recompute and overwrite: {path}")
        return None
    mapping = data.get("fp_to_eff")
    if not isinstance(mapping, dict):
        return None
    print(f"[FrozenFpToEff] Loaded {len(mapping):,} entries from {path}")
    return mapping


def _probe_signature(cfg: Config, universe_sig: str) -> str:
    return _make_signature("compact_probe_v1", {
        "probe_file": _file_fingerprint(cfg.probe_file),
        "rcut_list": cfg.rcut_list,
        "center_elements": cfg.center_elements,
        "universe_signature": universe_sig,
        "protect_probe_only_raw": cfg.probe_phase2_joint_protect_probe_only_raw,
    })


def _phase2_signature(cfg: Config, universe_sig: str, probe_sig: str) -> str:
    return _make_signature("compact_phase2_v1", {
        "universe_signature": universe_sig,
        "probe_signature": probe_sig,
        "phase2_budget": cfg.phase2_budget,
        "exclude_probe_indices_from_phase2_candidates": cfg.exclude_probe_indices_from_phase2_candidates,
        "jaccard_dedup_enabled": cfg.jaccard_dedup_enabled,
        "jaccard_dedup_threshold": cfg.jaccard_dedup_threshold,
        "jaccard_dedup_threshold_common": cfg.jaccard_dedup_threshold_common,
        "jaccard_dedup_size_band_frac": cfg.jaccard_dedup_size_band_frac,
        "two_tier_cover_enabled": cfg.two_tier_cover_enabled,
        "rare_env_freq_threshold": cfg.rare_env_freq_threshold,
        "common_env_coverage_pct": cfg.common_env_coverage_pct,
        "post_cover_jaccard_dedup_enabled": cfg.post_cover_jaccard_dedup_enabled,
        "post_cover_jaccard_threshold": cfg.post_cover_jaccard_threshold,
        "post_cover_jaccard_size_band_frac": cfg.post_cover_jaccard_size_band_frac,
    })


def _initial_train_signature(cfg: Config, universe_sig: str) -> str:
    return _make_signature("compact_initial_train_v1", {
        "universe_signature": universe_sig,
        "initial_train_budget": cfg.initial_train_budget,
        "initial_train_target_coverage_pct": cfg.initial_train_target_coverage_pct,
        "initial_train_output_xyz": cfg.initial_train_output_xyz,
        "jaccard_dedup_enabled": cfg.jaccard_dedup_enabled,
        "jaccard_dedup_threshold": cfg.jaccard_dedup_threshold,
        "jaccard_dedup_threshold_common": cfg.jaccard_dedup_threshold_common,
        "jaccard_dedup_size_band_frac": cfg.jaccard_dedup_size_band_frac,
        "two_tier_cover_enabled": cfg.two_tier_cover_enabled,
        "rare_env_freq_threshold": cfg.rare_env_freq_threshold,
        "common_env_coverage_pct": cfg.common_env_coverage_pct,
        "post_cover_jaccard_dedup_enabled": cfg.post_cover_jaccard_dedup_enabled,
        "post_cover_jaccard_threshold": cfg.post_cover_jaccard_threshold,
        "post_cover_jaccard_size_band_frac": cfg.post_cover_jaccard_size_band_frac,
    })


# =============================================================================
# XYZ frame copying helpers
# =============================================================================

def _read_xyz_frame_strings(path: str, wanted: Set[int]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if not wanted:
        return out
    wanted = set(wanted)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        idx = 0
        while True:
            first = f.readline()
            if not first:
                break
            if not first.strip():
                continue
            try:
                n = int(first.strip().split()[0])
            except Exception as e:
                raise RuntimeError(
                    f"Cannot parse XYZ frame {idx} atom count in {path!r}: {first!r}"
                ) from e
            lines = [first]
            comment = f.readline()
            if not comment:
                raise RuntimeError(f"Unexpected EOF after atom count at frame {idx} in {path!r}")
            lines.append(comment)
            for _ in range(n):
                line = f.readline()
                if not line:
                    raise RuntimeError(f"Unexpected EOF inside frame {idx} in {path!r}")
                lines.append(line)
            if idx in wanted:
                out[idx] = "".join(lines)
                if len(out) == len(wanted):
                    break
            idx += 1
    missing = sorted(wanted - set(out))
    if missing:
        raise IndexError(
            f"Missing frame indices in {path}: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    return out


def write_selected_xyz_from_source(
    source_file: str, selected_indices: Sequence[int], output_file: Path
) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frames = _read_xyz_frame_strings(source_file, set(int(i) for i in selected_indices))
    with open(output_file, "w", encoding="utf-8") as out:
        for idx in selected_indices:
            out.write(frames[int(idx)])


def write_mixed_xyz(
    probe_file: str,
    large_file: str,
    selected_items: Sequence[Tuple[str, int]],
    output_file: Path,
    preserve_text: bool = True,
) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    probe_ids = {idx for src, idx in selected_items if src == "probe"}
    large_ids = {idx for src, idx in selected_items if src == "phase2"}
    if preserve_text:
        probe_frames = _read_xyz_frame_strings(probe_file, probe_ids) if probe_ids else {}
        large_frames = _read_xyz_frame_strings(large_file, large_ids) if large_ids else {}
        with open(output_file, "w", encoding="utf-8") as out:
            for src, idx in selected_items:
                out.write(probe_frames[idx] if src == "probe" else large_frames[idx])
        return
    require_ase()
    atoms_out = []
    probe_structs = list(read(probe_file, index=":")) if probe_ids else []
    large_structs = list(read(large_file, index=":")) if large_ids else []
    for src, idx in selected_items:
        atoms_out.append((probe_structs if src == "probe" else large_structs)[idx])
    write(str(output_file), atoms_out)


def write_nonphysical_xyz_after_bond_filter(
    cfg: Config,
    structs_large: Sequence[Atoms],
    physical_idx: Set[int],
    out_dir: Path,
) -> List[int]:
    """
    Write structures rejected by BondFilter to a separate XYZ file.

    The output preserves the original XYZ text frames when
    cfg.preserve_input_xyz_format=True, so comments/properties in the source
    XYZ are kept unchanged.  A companion CSV records the original large_file
    frame indices.
    """
    if not cfg.write_nonphysical_xyz:
        return []

    if not cfg.bond_filter_enabled:
        print("[NonPhysicalXYZ] bond_filter_enabled=False; skip nonphysical XYZ export.")
        return []

    nonphysical_indices = [
        int(i) for i in range(len(structs_large)) if int(i) not in physical_idx
    ]

    xyz_path = out_dir / cfg.nonphysical_xyz_file
    manifest_path = out_dir / cfg.nonphysical_manifest_csv
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if nonphysical_indices:
        if cfg.preserve_input_xyz_format:
            write_selected_xyz_from_source(cfg.large_file, nonphysical_indices, xyz_path)
        else:
            require_ase()
            write(str(xyz_path), [structs_large[i] for i in nonphysical_indices])
    else:
        # Create an empty file so downstream workflows can rely on the path.
        xyz_path.write_text("", encoding="utf-8")

    rows = [
        {
            "source": "large_file",
            "source_idx": int(idx),
            "reason": "rejected_by_BondFilter_min_interatomic_distance",
        }
        for idx in nonphysical_indices
    ]
    write_csv(
        manifest_path,
        rows,
        fieldnames=["source", "source_idx", "reason"],
    )

    print(
        f"[NonPhysicalXYZ] nonphysical structures: {len(nonphysical_indices):,}; "
        f"XYZ → {xyz_path}; manifest → {manifest_path}"
    )
    return nonphysical_indices


# =============================================================================
# Route A: Jaccard pre-deduplication
# =============================================================================

def jaccard_dedup_candidates(
    candidates: Dict[int, Set[str]],
    threshold: float = 0.85,
    size_band_frac: float = 0.25,
    prefer_larger_set: bool = True,
    verbose: bool = True,
    tag: str = "",
) -> Dict[int, Set[str]]:
    """
    Reduce candidate structures by grouping those with near-identical
    effective-environment sets, keeping only one representative per group.
    """
    if not candidates:
        return candidates
    label = f"[JaccardDedup{('/' + tag) if tag else ''}]"
    if threshold >= 1.0:
        seen: Dict[frozenset, int] = {}
        reduced: Dict[int, Set[str]] = {}
        for k, envs in candidates.items():
            key = frozenset(envs)
            if key not in seen:
                seen[key] = k
                reduced[k] = envs
        if verbose:
            n_in, n_out = len(candidates), len(reduced)
            print(f"{label} {n_in:,} → {n_out:,} (exact-dedup, removed {n_in - n_out:,})")
        return reduced

    threshold = float(threshold)
    size_band_frac = float(size_band_frac)
    keys = list(candidates.keys())
    n = len(keys)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    size_to_local: Dict[int, List[int]] = defaultdict(list)
    sizes: List[int] = []
    for local_i, k in enumerate(keys):
        sz = len(candidates[k])
        size_to_local[sz].append(local_i)
        sizes.append(sz)

    sorted_sizes = sorted(size_to_local.keys())
    for s_idx, sz in enumerate(sorted_sizes):
        bucket_a = size_to_local[sz]
        max_sz = int(sz * (1.0 + size_band_frac)) + 1
        partner_buckets: List[List[int]] = [bucket_a]
        for s2_idx in range(s_idx + 1, len(sorted_sizes)):
            if sorted_sizes[s2_idx] > max_sz:
                break
            partner_buckets.append(size_to_local[sorted_sizes[s2_idx]])

        for i in range(len(bucket_a)):
            ia = bucket_a[i]
            ea = candidates[keys[ia]]
            len_ea = len(ea)
            if len_ea == 0:
                continue
            for j in range(i + 1, len(bucket_a)):
                ib = bucket_a[j]
                if _find(ia) == _find(ib):
                    continue
                eb = candidates[keys[ib]]
                inter = len(ea & eb)
                union_sz = len_ea + len(eb) - inter
                if union_sz > 0 and inter / union_sz >= threshold:
                    _union(ia, ib)

        for pb_idx in range(1, len(partner_buckets)):
            bucket_b = partner_buckets[pb_idx]
            for ia in bucket_a:
                if len(candidates[keys[ia]]) == 0:
                    continue
                ea = candidates[keys[ia]]
                len_ea = len(ea)
                for ib in bucket_b:
                    if _find(ia) == _find(ib):
                        continue
                    eb = candidates[keys[ib]]
                    inter = len(ea & eb)
                    union_sz = len_ea + len(eb) - inter
                    if union_sz > 0 and inter / union_sz >= threshold:
                        _union(ia, ib)

    group_best: Dict[int, int] = {}
    for local_i in range(n):
        root = _find(local_i)
        if root not in group_best:
            group_best[root] = local_i
        else:
            prev = group_best[root]
            if prefer_larger_set:
                if sizes[local_i] > sizes[prev]:
                    group_best[root] = local_i
            else:
                if keys[local_i] < keys[prev]:
                    group_best[root] = local_i

    kept_keys = {keys[li] for li in group_best.values()}
    reduced = {k: v for k, v in candidates.items() if k in kept_keys}
    if verbose:
        print(
            f"{label} {n:,} → {len(reduced):,} "
            f"(threshold={threshold:.2f}, removed {n - len(reduced):,}, "
            f"size_band={size_band_frac:.2f})"
        )
    return reduced


# =============================================================================
# Route D: Post-cover coverage-aware Jaccard dedup
# =============================================================================

def post_cover_jaccard_dedup(
    selected_indices: List[int],
    index_to_envs: Dict[int, Set[str]],
    threshold: float = 0.85,
    size_band_frac: float = 0.25,
    verbose: bool = True,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Coverage-aware Jaccard deduplication on the FINAL selected structures.

    Unlike the pre-cover Jaccard dedup (Route A), this operates on the
    already-selected K structures and guarantees that no coverage is lost:

    A structure S is removed when BOTH of the following hold:
      (a) there exists another REMAINING structure T with J(S, T) >= threshold
      (b) every env type in S is covered by at least one other REMAINING structure

    Condition (a) alone (pure Jaccard) could remove a structure that uniquely
    covers some rare env type.  Condition (b) is the coverage safety guard.

    Processing order: structures with the FEWEST unique env types (envs not
    shared with any other selected structure) are considered for removal first,
    ensuring that we preferentially remove the least informative duplicates.

    Parameters
    ----------
    selected_indices : list of structure indices from the greedy cover
    index_to_envs    : mapping struct_idx -> set of effective env strings
                       (should use the SAME env sets as the cover used)
    threshold        : Jaccard similarity threshold (same meaning as Route A)
    size_band_frac   : comparison band (same meaning as Route A)

    Returns
    -------
    pruned_indices : subset of selected_indices after deduplication
    stats          : diagnostic dict
    """
    if not selected_indices or threshold <= 0:
        return list(selected_indices), {"removed": 0, "kept": len(selected_indices)}

    # Build local working state: kept set and per-struct env coverage counts.
    kept: List[int] = list(selected_indices)
    envs_of = {k: set(index_to_envs.get(k, set())) for k in kept}

    # Coverage counts across ALL currently kept structures.
    coverage: Counter = Counter()
    for k in kept:
        coverage.update(envs_of[k])

    def _unique_envs(k: int) -> Set[str]:
        """Env types in structure k not covered by any other kept structure."""
        return {e for e in envs_of[k] if coverage[e] == 1}

    def _jaccard(a: int, b: int) -> float:
        ea, eb = envs_of[a], envs_of[b]
        inter = len(ea & eb)
        union = len(ea) + len(eb) - inter
        return inter / union if union > 0 else 0.0

    n_initial = len(kept)
    n_removed = 0

    # Iterative pruning: continue until no more removals are possible.
    # Sort candidates by ascending number of unique envs (remove least
    # informative first), breaking ties by descending set size (prefer to
    # keep larger env sets).
    changed = True
    while changed:
        changed = False
        # Recompute order each outer iteration.
        order = sorted(
            kept,
            key=lambda k: (len(_unique_envs(k)), -len(envs_of[k])),
        )
        for k in order:
            # Condition (b): all envs must be redundantly covered.
            if any(coverage[e] < 2 for e in envs_of[k]):
                continue  # k has at least one unique env; cannot remove

            # Condition (a): find a similar kept structure.
            sz_k = len(envs_of[k])
            has_similar = False
            for other in kept:
                if other == k:
                    continue
                sz_o = len(envs_of[other])
                # Quick size-band check before computing full Jaccard.
                if sz_o == 0:
                    continue
                ratio = sz_k / sz_o if sz_o > sz_k else sz_o / sz_k
                if ratio < (1.0 - size_band_frac):
                    continue
                if _jaccard(k, other) >= threshold:
                    has_similar = True
                    break

            if not has_similar:
                continue  # no similar partner; keep

            # Both conditions met: remove k.
            for e in envs_of[k]:
                coverage[e] -= 1
            kept.remove(k)
            n_removed += 1
            changed = True
            break  # restart loop with updated kept/coverage

    stats: Dict[str, Any] = {
        "n_before": n_initial,
        "n_after": len(kept),
        "removed": n_removed,
        "threshold": threshold,
        "size_band_frac": size_band_frac,
    }
    if verbose:
        print(
            f"[PostCoverJaccardDedup] {n_initial:,} → {len(kept):,} "
            f"(removed {n_removed:,}, threshold={threshold:.2f})"
        )
    return kept, stats


# =============================================================================
# Route C helpers
# =============================================================================

def _classify_env_types(
    target: Set[str],
    universe: "TopologyUniverse",
    rare_freq_threshold: int,
) -> Tuple[Set[str], Set[str]]:
    rare_target: Set[str] = set()
    common_target: Set[str] = set()
    for e in target:
        freq = len(set(universe.eff_fp_structs.get(e, [])))
        if freq <= rare_freq_threshold:
            rare_target.add(e)
        else:
            common_target.add(e)
    return rare_target, common_target


def _build_tier_candidates(
    candidates: Dict[int, Set[str]],
    tier_target: Set[str],
    exclude_indices: Optional[Set[int]] = None,
) -> Dict[int, Set[str]]:
    excl = exclude_indices or set()
    result: Dict[int, Set[str]] = {}
    for k, envs in candidates.items():
        if k in excl:
            continue
        tier_envs = envs & tier_target
        if tier_envs:
            result[k] = tier_envs
    return result


# =============================================================================
# Fingerprints and universe construction
# =============================================================================

class BondFilter:
    def __init__(self, coeff: float = 0.75):
        self.coeff = float(coeff)

    def is_physical(self, atoms: Atoms) -> bool:
        if len(atoms) <= 1:
            return True
        nums = atoms.get_atomic_numbers()
        dm = atoms.get_all_distances(mic=True)
        for i in range(len(atoms)):
            ri = covalent_radii[int(nums[i])]
            for j in range(i + 1, len(atoms)):
                rmin = (ri + covalent_radii[int(nums[j])]) * self.coeff
                if dm[i, j] < rmin:
                    return False
        return True

    def filter_indices(
        self,
        structs: Sequence[Atoms],
        checkpoint_path: Optional[Path] = None,
        checkpoint_signature: Optional[str] = None,
        resume: bool = True,
        interval: int = 500,
    ) -> Set[int]:
        ok: Set[int] = set()
        next_idx = 0
        interval = max(1, int(interval))
        if checkpoint_path and checkpoint_signature and resume:
            ckpt = _load_checkpoint(checkpoint_path, checkpoint_signature)
            if ckpt is not None:
                ok = set(int(i) for i in ckpt.get("ok", []))
                next_idx = int(ckpt.get("next_idx", 0))
                print(f"[Checkpoint] BondFilter resume: {next_idx:,}/{len(structs):,}")
        for idx in tqdm(range(next_idx, len(structs)), desc="[BondFilter]"):
            if self.is_physical(structs[idx]):
                ok.add(idx)
            if checkpoint_path and checkpoint_signature and ((idx + 1) % interval == 0):
                _save_checkpoint(checkpoint_path, checkpoint_signature,
                                 {"ok": sorted(ok), "next_idx": idx + 1})
        if checkpoint_path and checkpoint_signature:
            _save_checkpoint(checkpoint_path, checkpoint_signature,
                             {"ok": sorted(ok), "next_idx": len(structs)})
        print(f"[BondFilter] physical structures: {len(ok):,}/{len(structs):,}")
        return ok


class FingerprintExtractor:
    def __init__(self, rcut_list: List[float], center_elements: Tuple[str, ...]):
        self.rcut_list = [float(x) for x in rcut_list]
        self.center_elements = set(center_elements)

    @staticmethod
    def _cell_ok(atoms: Atoms, rcut: float) -> bool:
        try:
            cell = atoms.get_cell()
            if abs(np.linalg.det(cell)) < 1e-6:
                return False
            return not np.any(np.linalg.norm(cell, axis=1) < rcut)
        except Exception:
            return False

    @staticmethod
    def _meta_key(m: Any, center_sym: str, rcut: float) -> Optional[str]:
        try:
            all_syms = [
                getattr(a, "symbol", getattr(a, "element", "?"))
                for a in dict(m.atoms()).values()
            ]
            neighbors = list(all_syms)
            for k, s in enumerate(neighbors):
                if s == center_sym:
                    neighbors.pop(k)
                    break
            neighbors.sort()
            return f"r{rcut:.1f}:{center_sym}:cn{len(neighbors)}:{''.join(neighbors)}"
        except Exception:
            return None

    @staticmethod
    def _geom_feature(
        m: Any, center_atom_num: int, fallback_pos: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        try:
            atoms_dict = dict(m.atoms())
            if len(atoms_dict) < 2:
                return None
            center_label = f"O{center_atom_num}"
            center_pos = None
            neigh = []
            for _, a in atoms_dict.items():
                pos = np.asarray(a.position, dtype=np.float64)
                if getattr(a, "label", "") == center_label:
                    center_pos = pos
                else:
                    neigh.append(pos)
            if center_pos is None and fallback_pos is not None:
                all_pos = np.array(
                    [np.asarray(a.position, dtype=np.float64) for a in atoms_dict.values()]
                )
                idx_min = int(np.argmin(np.linalg.norm(all_pos - fallback_pos, axis=1)))
                center_pos = all_pos[idx_min]
                neigh = [all_pos[k] for k in range(len(all_pos)) if k != idx_min]
            if center_pos is None:
                vals = list(atoms_dict.values())
                center_pos = np.asarray(vals[0].position, dtype=np.float64)
                neigh = [np.asarray(a.position, dtype=np.float64) for a in vals[1:]]
            if not neigh:
                return None
            neigh_arr = np.array(neigh, dtype=np.float64)
            vecs = neigh_arr - center_pos
            norms = np.linalg.norm(vecs, axis=1)
            bond_lengths = np.sort(norms)
            dirs = vecs / np.maximum(norms[:, None], 1e-10)
            n = len(dirs)
            if n > 1:
                cos_mat = np.clip(dirs @ dirs.T, -1.0, 1.0)
                ii, jj = np.triu_indices(n, k=1)
                angles = np.degrees(np.arccos(cos_mat[ii, jj]))
                mu_theta = float(angles.mean())
                sig_theta = float(angles.std()) if len(angles) > 1 else 0.0
            else:
                mu_theta, sig_theta = 90.0, 0.0
            return np.concatenate([bond_lengths, [mu_theta, sig_theta]]).astype(np.float32)
        except Exception:
            return None

    def extract(
        self,
        atoms: Atoms,
        struct_idx: Optional[int] = None,
        example_limit_per_fp: Optional[int] = 3,
    ) -> Tuple[Set[str], Dict[str, str], Dict[str, List[Tuple[int, int, str]]], Dict[str, np.ndarray]]:
        spdkit, dwim = require_spdkit()
        fps: Set[str] = set()
        meta_map: Dict[str, str] = {}
        examples: Dict[str, List[Tuple[int, int, str]]] = {}
        geom_cache: Dict[str, np.ndarray] = {}
        try:
            mol = spdkit.from_ase_atoms(atoms)
            mol_cache = dict(mol.atoms())
        except Exception:
            return fps, meta_map, examples, geom_cache
        for rcut in self.rcut_list:
            if not self._cell_ok(atoms, rcut):
                continue
            try:
                env = dwim.ChemicalEnvironment(mol, rcut)
            except Exception:
                continue
            for i in mol.numbers():
                a = mol_cache.get(i)
                if a is None:
                    continue
                sym = getattr(a, "symbol", getattr(a, "element", ""))
                if sym not in self.center_elements:
                    continue
                try:
                    m = env.create_central_molecule(i)
                    m.rebond(ignore_pbc=True)
                    if len(m.numbers()) <= 1:
                        continue
                    raw_fp = f"r{rcut:.1f}:{m.fingerprint()}"
                    fps.add(raw_fp)
                    if raw_fp not in meta_map:
                        mk = self._meta_key(m, sym, rcut)
                        if mk:
                            meta_map[raw_fp] = mk
                    if struct_idx is not None:
                        ex = examples.setdefault(raw_fp, [])
                        if example_limit_per_fp is None or len(ex) < example_limit_per_fp:
                            ex.append((int(struct_idx), int(i), str(sym)))
                    if raw_fp not in geom_cache:
                        try:
                            fallback = np.asarray(a.position, dtype=np.float64)
                        except Exception:
                            fallback = None
                        feat = self._geom_feature(m, int(i), fallback)
                        if feat is not None:
                            geom_cache[raw_fp] = feat
                except Exception:
                    continue
        return fps, meta_map, examples, geom_cache


class TopologyUniverse:
    def __init__(self):
        self.all_fps: Set[str] = set()
        self.struct_fps: Dict[int, Set[str]] = {}
        self.fp_structs: Dict[str, List[int]] = defaultdict(list)
        self.meta_map: Dict[str, str] = {}
        self.fp_examples: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
        self.examples_are_complete: bool = False
        self._geom_samples: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.fp_geom: Dict[str, np.ndarray] = {}
        self.eff_universe: Set[str] = set()
        self.fp_to_eff: Dict[str, str] = {}
        self.eff_struct_fps: Dict[int, Set[str]] = {}
        self.eff_fp_structs: Dict[str, List[int]] = defaultdict(list)
        self.compress_stats: Dict[str, Any] = {}

    def state_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.fp_structs = defaultdict(list, self.fp_structs)
        self.fp_examples = defaultdict(list, self.fp_examples)
        self._geom_samples = defaultdict(list, self._geom_samples)
        self.eff_fp_structs = defaultdict(list, self.eff_fp_structs)

    def build(
        self,
        structs: Sequence[Atoms],
        extractor: FingerprintExtractor,
        physical_idx: Optional[Set[int]] = None,
        geom_max_samples: int = 5,
        collect_all_examples: bool = False,
        checkpoint_path: Optional[Path] = None,
        checkpoint_signature: Optional[str] = None,
        resume: bool = True,
        interval: int = 500,
    ) -> None:
        n_failed = 0
        next_idx = 0
        interval = max(1, int(interval))
        if checkpoint_path and checkpoint_signature and resume:
            ckpt = _load_checkpoint(checkpoint_path, checkpoint_signature)
            if ckpt is not None and isinstance(ckpt.get("universe_state"), dict):
                self.load_state_dict(ckpt["universe_state"])
                next_idx = int(ckpt.get("next_idx", len(self.struct_fps)))
                n_failed = int(ckpt.get("n_failed", 0))
                print(f"[Checkpoint] Universe resume: {next_idx:,}/{len(structs):,}")
        self.examples_are_complete = collect_all_examples
        example_limit = None if collect_all_examples else 3
        print(f"[Universe] Extract fingerprints from {len(structs):,} structures")
        for idx in tqdm(range(next_idx, len(structs)), desc="[Universe.extract]"):
            if physical_idx is not None and idx not in physical_idx:
                self.struct_fps[idx] = set()
            else:
                fps, meta, ex, geom = extractor.extract(structs[idx], idx, example_limit)
                if not fps:
                    n_failed += 1
                self.struct_fps[idx] = set(fps)
                for fp in fps:
                    self.all_fps.add(fp)
                    self.fp_structs[fp].append(idx)
                self.meta_map.update(meta)
                for fp, lst in ex.items():
                    self.fp_examples[fp].extend(lst)
                for fp, feat in geom.items():
                    if len(self._geom_samples[fp]) < geom_max_samples:
                        self._geom_samples[fp].append(feat)
            if checkpoint_path and checkpoint_signature and ((idx + 1) % interval == 0):
                self._finalize_geom_samples()
                _save_checkpoint(checkpoint_path, checkpoint_signature, {
                    "universe_state": self.state_dict(), "next_idx": idx + 1, "n_failed": n_failed,
                })
        self._finalize_geom_samples()
        if checkpoint_path and checkpoint_signature:
            _save_checkpoint(checkpoint_path, checkpoint_signature, {
                "universe_state": self.state_dict(), "next_idx": len(structs), "n_failed": n_failed,
            })
        print(f"[Universe] raw fingerprints: {len(self.all_fps):,}; failed/empty structures: {n_failed:,}")

    def _finalize_geom_samples(self) -> None:
        self.fp_geom.clear()
        for fp, arrs in self._geom_samples.items():
            if arrs:
                try:
                    self.fp_geom[fp] = np.mean(np.stack(arrs), axis=0).astype(np.float32)
                except Exception:
                    pass

    def compress(self, cfg: Config, frozen_fp_to_eff: Optional[Dict[str, str]] = None) -> None:
        """
        Convert raw fingerprints to effective environment types.

        frozen_fp_to_eff
            When provided, skip Ward re-clustering entirely and apply this
            pre-computed mapping directly.  Any raw fingerprint present in the
            current universe but missing from the frozen mapping is treated as a
            new singleton eff type (named ``<meta_key>:frozen_new:<sha1[:8]>``).
            This is the root-cause fix for re-run redundancy: the effective-type
            space is a FIXED REFERENCE FRAME independent of which structures are
            currently in the universe.
        """
        if frozen_fp_to_eff is not None:
            print(
                f"[Universe] Apply frozen fp_to_eff mapping "
                f"({len(frozen_fp_to_eff):,} entries, skipping Ward re-clustering)"
            )
            fp_to_eff: Dict[str, str] = {}
            n_matched = 0
            n_new = 0
            for fp in self.all_fps:
                if fp in frozen_fp_to_eff:
                    fp_to_eff[fp] = frozen_fp_to_eff[fp]
                    n_matched += 1
                else:
                    # Raw fingerprint not seen in the original run; assign a
                    # deterministic singleton eff type based on its content.
                    mk = self.meta_map.get(fp, fp)
                    short_hash = hashlib.sha1(fp.encode("utf-8")).hexdigest()[:8]
                    fp_to_eff[fp] = f"{mk}:frozen_new:{short_hash}"
                    n_new += 1
            self.fp_to_eff = fp_to_eff
            self.eff_universe = set(fp_to_eff.values())
            self.compress_stats = {
                "method": "frozen_fp_to_eff",
                "n_raw": len(self.all_fps),
                "n_eff": len(self.eff_universe),
                "n_matched_from_frozen": n_matched,
                "n_new_singletons": n_new,
                "reduction_pct": (
                    1.0 - len(self.eff_universe) / max(1, len(self.all_fps))
                ) * 100.0,
            }
            self._build_eff_layer()
            print(
                f"[Universe] effective fingerprints: {len(self.eff_universe):,}; "
                f"matched={n_matched:,}, new_singletons={n_new:,}"
            )
            return

        print("[Universe] Compress raw fingerprints into effective environment types")
        if cfg.use_geom_cluster and str(cfg.geom_cluster_backend).lower() in (
            "ward", "scipy_ward", "hierarchical", "hierarchical_ward"
        ):
            if not SCIPY_OK:
                raise RuntimeError("Strict Ward geometry clustering requires scipy.")
        meta_groups: Dict[str, List[str]] = defaultdict(list)
        for fp in self.all_fps:
            meta_groups[self.meta_map.get(fp, fp)].append(fp)
        stats: Dict[str, Any] = {
            "method": "similarity_clustering_no_frequency",
            "n_raw": len(self.all_fps),
            "n_meta_types": len(meta_groups),
            "scipy_ok": SCIPY_OK,
            "scipy_version": SCIPY_VERSION,
            "geom_cluster_backend_requested": cfg.geom_cluster_backend,
            "strict_geom_cluster_backend": cfg.strict_geom_cluster_backend,
            "n_geom_ward_groups": 0,
            "n_geom_ward_fps": 0,
            "n_geom_component_groups": 0,
            "n_geom_component_fps": 0,
            "n_topology_clustered": 0,
            "n_no_geom": 0,
            "n_singletons": 0,
            "geom_fallback_errors": [],
        }
        fp_to_eff = {}
        for mk, fps_list in tqdm(sorted(meta_groups.items()), desc="[Universe.cluster]"):
            fps_list = sorted(fps_list)
            labels = self._cluster_meta_group(
                fps_list, mk, cfg.use_geom_cluster, cfg.bond_tol, cfg.angle_tol,
                cfg.geom_min_frac, cfg.topology_similarity_threshold,
                cfg.geom_cluster_backend, cfg.strict_geom_cluster_backend, stats,
            )
            for fp, lbl in zip(fps_list, labels):
                fp_to_eff[fp] = f"{mk}:sim{int(lbl):05d}"
            if len(fps_list) == 1:
                stats["n_singletons"] += 1
        self.fp_to_eff = fp_to_eff
        self.eff_universe = set(fp_to_eff.values())
        self.compress_stats = {
            **stats,
            "n_eff": len(self.eff_universe),
            "reduction_pct": (1.0 - len(self.eff_universe) / max(1, len(self.all_fps))) * 100.0,
        }
        self._build_eff_layer()
        print(
            f"[Universe] effective fingerprints: {len(self.eff_universe):,}; "
            f"reduction={self.compress_stats['reduction_pct']:.2f}%"
        )

    def _cluster_meta_group(
        self, fps_list, meta_key, use_geom_cluster, bond_tol, angle_tol,
        geom_min_frac, topology_similarity_threshold, geom_cluster_backend,
        strict_geom_cluster_backend, stats,
    ) -> List[int]:
        if len(fps_list) <= 1:
            return [1] * len(fps_list)
        has_geom = [fp for fp in fps_list if fp in self.fp_geom]
        no_geom = [fp for fp in fps_list if fp not in self.fp_geom]
        stats["n_no_geom"] += len(no_geom)
        use_geometry = (
            use_geom_cluster and has_geom
            and (len(has_geom) / len(fps_list) >= geom_min_frac)
        )
        labels_map: Dict[str, int] = {}
        if use_geometry:
            labels_geom = self._cluster_geom_features(
                has_geom, bond_tol, angle_tol,
                geom_cluster_backend, strict_geom_cluster_backend, stats, meta_key,
            )
            for fp, lbl in zip(has_geom, labels_geom):
                labels_map[fp] = int(lbl)
            if no_geom:
                labels_topo = self._cluster_topology_strings(no_geom, topology_similarity_threshold)
                offset = max(labels_geom) if labels_geom else 0
                for fp, lbl in zip(no_geom, labels_topo):
                    labels_map[fp] = int(offset + lbl)
                    stats["n_topology_clustered"] += 1
            return [labels_map[fp] for fp in fps_list]
        stats["n_topology_clustered"] += len(fps_list)
        return self._cluster_topology_strings(fps_list, topology_similarity_threshold)

    def _cluster_geom_features(
        self, fps_list, bond_tol, angle_tol, backend, strict, stats, meta_key,
    ) -> List[int]:
        if len(fps_list) <= 1:
            return [1] * len(fps_list)
        feats = np.stack([self.fp_geom[fp] for fp in fps_list]).astype(np.float64)
        n_dim = feats.shape[1]
        n_bond = max(0, n_dim - 2)
        X = feats.copy()
        X[:, :n_bond] /= max(float(bond_tol), 1e-10)
        X[:, n_bond:] /= max(float(angle_tol), 1e-10)
        threshold = float(math.sqrt(n_dim))
        backend_norm = str(backend or "ward").lower()
        if backend_norm in ("ward", "scipy_ward", "hierarchical", "hierarchical_ward"):
            if not SCIPY_OK or fclusterdata is None:
                raise RuntimeError(f"SciPy Ward requested for {meta_key}, but scipy is unavailable.")
            try:
                labels = fclusterdata(X, t=threshold, criterion="distance",
                                      metric="euclidean", method="ward").tolist()
                stats["n_geom_ward_groups"] += 1
                stats["n_geom_ward_fps"] += len(fps_list)
                return [int(x) for x in labels]
            except Exception as e:
                msg = f"SciPy Ward failed for meta_key={meta_key!r}, n={len(fps_list)}, dim={n_dim}: {e}"
                stats.setdefault("geom_fallback_errors", []).append(msg)
                if strict:
                    raise RuntimeError(msg + "\nStrict Ward mode enabled.") from e
                warnings.warn(msg + " -- fallback to distance components")
                labels = self._cluster_by_distance_components(X, threshold)
                stats["n_geom_component_groups"] += 1
                stats["n_geom_component_fps"] += len(fps_list)
                return labels
        if backend_norm in ("components", "component", "connected_components", "distance_components"):
            if strict:
                raise RuntimeError("Distance-component backend incompatible with strict=True.")
            labels = self._cluster_by_distance_components(X, threshold)
            stats["n_geom_component_groups"] += 1
            stats["n_geom_component_fps"] += len(fps_list)
            return labels
        raise ValueError(f"Unknown geom_cluster_backend={backend!r}")

    @staticmethod
    def _cluster_by_distance_components(X: np.ndarray, threshold: float) -> List[int]:
        n = len(X)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[rb] = ra
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(X[i] - X[j]) <= threshold:
                    union(i, j)
        root_to_label: Dict[int, int] = {}
        labels: List[int] = []
        for i in range(n):
            r = find(i)
            if r not in root_to_label:
                root_to_label[r] = len(root_to_label) + 1
            labels.append(root_to_label[r])
        return labels

    @staticmethod
    def _fp_tokens(fp: str) -> Set[str]:
        return set(re.findall(r"[A-Za-z]+|\d+\.\d+|\d+|[^\sA-Za-z0-9]", fp))

    @classmethod
    def _topology_similarity(cls, fp_a: str, fp_b: str) -> float:
        seq_sim = SequenceMatcher(None, fp_a, fp_b).ratio()
        tok_a, tok_b = cls._fp_tokens(fp_a), cls._fp_tokens(fp_b)
        jac = len(tok_a & tok_b) / max(1, len(tok_a | tok_b)) if (tok_a or tok_b) else 1.0
        return 0.6 * seq_sim + 0.4 * jac

    @classmethod
    def _cluster_topology_strings(cls, fps_list: List[str], similarity_threshold: float) -> List[int]:
        if len(fps_list) <= 1:
            return [1] * len(fps_list)
        n = len(fps_list)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[rb] = ra
        for i in range(n):
            for j in range(i + 1, n):
                if cls._topology_similarity(fps_list[i], fps_list[j]) >= similarity_threshold:
                    union(i, j)
        root_to_label: Dict[int, int] = {}
        labels: List[int] = []
        for i in range(n):
            r = find(i)
            if r not in root_to_label:
                root_to_label[r] = len(root_to_label) + 1
            labels.append(root_to_label[r])
        return labels

    def _build_eff_layer(self) -> None:
        self.eff_struct_fps.clear()
        self.eff_fp_structs.clear()
        self.eff_fp_structs = defaultdict(list)
        for idx, fps in self.struct_fps.items():
            effs = {self.fp_to_eff[fp] for fp in fps if fp in self.fp_to_eff}
            self.eff_struct_fps[idx] = effs
            for eff in effs:
                self.eff_fp_structs[eff].append(idx)


# =============================================================================
# Probe extraction, Phase2, joint cover
# =============================================================================

@dataclass
class ProbeData:
    struct_envs: Dict[int, Set[str]]
    covered_eff: Set[str]
    probe_only_raw: Set[str]
    n_probe_structures: int


def extract_probe_data(
    cfg: Config,
    probe_structs: Sequence[Atoms],
    extractor: FingerprintExtractor,
    universe: TopologyUniverse,
    signature: str,
    out_dir: Path,
) -> ProbeData:
    cache_path = out_dir / cfg.probe_fps_cache_file
    ckpt_path = (
        out_dir / cfg.checkpoint_dir / "probe_extract.pkl"
        if cfg.checkpoint_enabled else None
    )
    if cfg.probe_fps_cache_enabled:
        cached = _atomic_pickle_load(cache_path)
        if isinstance(cached, dict) and cached.get("signature") == signature:
            print(f"[Probe] Load cached probe fingerprints: {cache_path}")
            return ProbeData(
                struct_envs={int(k): set(v) for k, v in cached["struct_envs"].items()},
                covered_eff=set(cached["covered_eff"]),
                probe_only_raw=set(cached.get("probe_only_raw", [])),
                n_probe_structures=int(cached["n_probe_structures"]),
            )
    struct_envs: Dict[int, Set[str]] = {}
    covered_eff: Set[str] = set()
    probe_only_raw: Set[str] = set()
    next_idx = 0
    if ckpt_path and cfg.resume_from_checkpoint:
        ckpt = _load_checkpoint(ckpt_path, signature)
        if ckpt is not None:
            struct_envs = {int(k): set(v) for k, v in ckpt.get("struct_envs", {}).items()}
            covered_eff = set(ckpt.get("covered_eff", []))
            probe_only_raw = set(ckpt.get("probe_only_raw", []))
            next_idx = int(ckpt.get("next_idx", 0))
            print(f"[Checkpoint] Probe resume: {next_idx:,}/{len(probe_structs):,}")
    for idx in tqdm(range(next_idx, len(probe_structs)), desc="[Probe.extract]"):
        raw_fps, _, _, _ = extractor.extract(probe_structs[idx], idx, example_limit_per_fp=0)
        envs: Set[str] = set()
        for fp in raw_fps:
            eff = universe.fp_to_eff.get(fp)
            if eff is not None:
                envs.add(eff); covered_eff.add(eff)
            elif cfg.probe_phase2_joint_protect_probe_only_raw:
                tag = f"probe_only:{fp}"; envs.add(tag); probe_only_raw.add(tag)
        struct_envs[idx] = envs
        if ckpt_path and ((idx + 1) % max(1, cfg.probe_fps_checkpoint_interval) == 0):
            _save_checkpoint(ckpt_path, signature, {
                "struct_envs": {k: sorted(v) for k, v in struct_envs.items()},
                "covered_eff": sorted(covered_eff),
                "probe_only_raw": sorted(probe_only_raw),
                "next_idx": idx + 1,
            })
    data = ProbeData(struct_envs, covered_eff, probe_only_raw, len(probe_structs))
    payload = {
        "signature": signature,
        "struct_envs": {k: sorted(v) for k, v in struct_envs.items()},
        "covered_eff": sorted(covered_eff),
        "probe_only_raw": sorted(probe_only_raw),
        "n_probe_structures": len(probe_structs),
    }
    if cfg.probe_fps_cache_enabled:
        _atomic_pickle_dump(payload, cache_path)
    if ckpt_path:
        _save_checkpoint(ckpt_path, signature, {k: v for k, v in payload.items() if k != "signature"})
    print(f"[Probe] covered effective envs: {len(covered_eff):,}; probe-only raw envs: {len(probe_only_raw):,}")
    return data


def lazy_greedy_cover(
    candidate_envs: Dict[Any, Set[str]],
    target_envs: Set[str],
    budget: int = 0,
    prefer_key=None,
    desc: str = "[GreedyCover]",
    checkpoint_path: Optional[Path] = None,
    checkpoint_signature: Optional[str] = None,
    resume: bool = True,
    interval: int = 100,
    stop_coverage_pct: float = 100.0,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    selected: List[Any] = []
    covered: Set[str] = set()
    manifest: List[Dict[str, Any]] = []
    budget = int(budget or 0)
    interval = max(1, int(interval))
    stop_coverage_pct = max(0.0, min(100.0, float(stop_coverage_pct)))
    if checkpoint_path and checkpoint_signature and resume:
        ckpt = _load_checkpoint(checkpoint_path, checkpoint_signature)
        if ckpt is not None:
            selected = ckpt.get("selected", [])
            selected = [tuple(x) if isinstance(x, list) else x for x in selected]
            manifest = ckpt.get("manifest", [])
            for key in selected:
                covered |= candidate_envs.get(key, set()) & target_envs
            print(f"[Checkpoint] {desc} resume: selected={len(selected):,}, covered={len(covered):,}/{len(target_envs):,}")
    selected_set = set(selected)
    uncovered = set(target_envs) - covered
    def reached_stop():
        return (100.0 * len(covered) / max(1, len(target_envs))) >= stop_coverage_pct
    if not uncovered or reached_stop():
        return selected, manifest
    import heapq
    def tie_rank(key):
        return prefer_key(key) if prefer_key else key
    heap = []
    for key, envs in candidate_envs.items():
        if key in selected_set:
            continue
        gain = len(envs & uncovered)
        if gain > 0:
            heapq.heappush(heap, (-gain, tie_rank(key), key))
    pbar = tqdm(total=len(uncovered), desc=desc)
    while uncovered and heap and (budget <= 0 or len(selected) < budget) and not reached_stop():
        old_neg_gain, _, key = heapq.heappop(heap)
        if key in selected_set:
            continue
        actual_gain_set = candidate_envs.get(key, set()) & uncovered
        actual_gain = len(actual_gain_set)
        if actual_gain <= 0:
            continue
        if actual_gain != -old_neg_gain:
            heapq.heappush(heap, (-actual_gain, tie_rank(key), key)); continue
        selected.append(key); selected_set.add(key)
        uncovered -= actual_gain_set; covered |= actual_gain_set
        rec = {"rank": len(selected), "candidate": repr(key), "new_gain": actual_gain,
               "covered": len(covered), "uncovered": len(uncovered),
               "coverage_pct": 100.0 * len(covered) / max(1, len(target_envs))}
        manifest.append(rec); pbar.update(actual_gain)
        if checkpoint_path and checkpoint_signature and (len(selected) % interval == 0):
            _save_checkpoint(checkpoint_path, checkpoint_signature, {"selected": selected, "manifest": manifest})
    pbar.close()
    if checkpoint_path and checkpoint_signature:
        _save_checkpoint(checkpoint_path, checkpoint_signature, {"selected": selected, "manifest": manifest})
    return selected, manifest


def _two_tier_cover(
    candidates: Dict[int, Set[str]],
    target: Set[str],
    universe: TopologyUniverse,
    cfg: Config,
    desc_prefix: str = "",
    ckpt_rare: Optional[Path] = None,
    ckpt_common: Optional[Path] = None,
    ckpt_sig_rare: Optional[str] = None,
    ckpt_sig_common: Optional[str] = None,
    budget: int = 0,
) -> Tuple[List[int], List[Dict[str, Any]], Dict[str, Any]]:
    rare_target, common_target = _classify_env_types(target, universe, cfg.rare_env_freq_threshold)
    print(f"[TwoTier{desc_prefix}] rare: {len(rare_target):,}, common: {len(common_target):,} "
          f"(freq_threshold={cfg.rare_env_freq_threshold})")

    # Pass 1: rare
    rare_cands = _build_tier_candidates(candidates, rare_target)
    n_rare_before = len(rare_cands)
    if cfg.jaccard_dedup_enabled and rare_cands:
        rare_cands = jaccard_dedup_candidates(
            rare_cands, threshold=cfg.jaccard_dedup_threshold,
            size_band_frac=cfg.jaccard_dedup_size_band_frac,
            prefer_larger_set=True, verbose=cfg.verbose, tag="rare",
        )
    rare_selected: List[int] = []
    rare_manifest: List[Dict[str, Any]] = []
    if rare_target and rare_cands:
        raw, rare_manifest = lazy_greedy_cover(
            rare_cands, rare_target, budget=0, prefer_key=lambda k: int(k),
            desc=f"[Pass1.rare{desc_prefix}]",
            checkpoint_path=ckpt_rare, checkpoint_signature=ckpt_sig_rare,
            resume=cfg.resume_from_checkpoint, interval=cfg.checkpoint_interval,
            stop_coverage_pct=100.0,
        )
        rare_selected = [int(x) for x in raw]
    rare_covered_common: Set[str] = set()
    for idx in rare_selected:
        rare_covered_common |= candidates.get(idx, set()) & common_target

    # Pass 2: common
    already = set(rare_selected)
    remaining_common = common_target - rare_covered_common
    common_cands = _build_tier_candidates(candidates, common_target, exclude_indices=already)
    n_common_before = len(common_cands)
    if cfg.jaccard_dedup_enabled and common_cands:
        common_cands = jaccard_dedup_candidates(
            common_cands, threshold=cfg.jaccard_dedup_threshold_common,
            size_band_frac=cfg.jaccard_dedup_size_band_frac,
            prefer_larger_set=True, verbose=cfg.verbose, tag="common",
        )
    common_selected: List[int] = []
    common_manifest: List[Dict[str, Any]] = []
    if remaining_common and common_cands:
        restricted = {k: v & remaining_common for k, v in common_cands.items() if v & remaining_common}
        if restricted:
            raw2, common_manifest = lazy_greedy_cover(
                restricted, remaining_common,
                budget=int(budget) if budget > 0 else 0,
                prefer_key=lambda k: int(k),
                desc=f"[Pass2.common{desc_prefix}]",
                checkpoint_path=ckpt_common, checkpoint_signature=ckpt_sig_common,
                resume=cfg.resume_from_checkpoint, interval=cfg.checkpoint_interval,
                stop_coverage_pct=cfg.common_env_coverage_pct,
            )
            common_selected = [int(x) for x in raw2]

    selected = rare_selected + common_selected
    for r in rare_manifest: r["pass"] = "rare"
    for r in common_manifest: r["pass"] = "common"
    combined_manifest = rare_manifest + common_manifest

    # ── Fix: recalculate global (cross-pass) rank and total-target coverage ──
    # Per-pass manifests carry per-pass rank (both start at 1) and per-pass
    # coverage_pct (both start at 0 %).  Without this recalculation the
    # convergence plot shows two separate curves because the x-axis resets
    # and the y-axis drops back to 0 % at the pass boundary.
    covered_global_running: Set[str] = set()
    for g_rank, (m_entry, s_idx) in enumerate(zip(combined_manifest, selected), start=1):
        covered_global_running |= candidates.get(s_idx, set()) & target
        m_entry["rank_global"] = g_rank
        m_entry["covered_global"] = len(covered_global_running)
        m_entry["uncovered_global"] = max(0, len(target) - len(covered_global_running))
        m_entry["coverage_pct_global"] = (
            100.0 * len(covered_global_running) / max(1, len(target))
        )

    covered_all: Set[str] = covered_global_running

    tier_stats = {
        "n_rare_target": len(rare_target), "n_common_target": len(common_target),
        "rare_candidates_before_dedup": n_rare_before,
        "common_candidates_before_dedup": n_common_before,
        "rare_selected": len(rare_selected), "common_selected": len(common_selected),
        "total_selected": len(selected),
        "covered_rare": len(covered_all & rare_target),
        "covered_common": len(covered_all & common_target),
        "covered_total": len(covered_all),
        "coverage_rare_pct": 100.0 * len(covered_all & rare_target) / max(1, len(rare_target)),
        "coverage_common_pct": 100.0 * len(covered_all & common_target) / max(1, len(common_target)),
        "coverage_total_pct": 100.0 * len(covered_all) / max(1, len(target)),
        "rare_freq_threshold": cfg.rare_env_freq_threshold,
        "common_env_coverage_pct_target": cfg.common_env_coverage_pct,
    }
    print(
        f"[TwoTier{desc_prefix}] "
        f"rare {len(covered_all & rare_target):,}/{len(rare_target):,} "
        f"({tier_stats['coverage_rare_pct']:.1f}%) | "
        f"common {len(covered_all & common_target):,}/{len(common_target):,} "
        f"({tier_stats['coverage_common_pct']:.1f}%) | "
        f"total selected: {len(selected):,}"
    )
    return selected, combined_manifest, tier_stats


def _apply_post_cover_dedup(
    selected_indices: List[int],
    full_candidates: Dict[int, Set[str]],
    target: Set[str],
    cfg: Config,
    label: str = "",
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Apply Route D (post-cover coverage-aware Jaccard dedup) to selected_indices.

    The env sets passed to post_cover_jaccard_dedup are the FULL effective-env
    sets of each selected structure (not restricted to target), so Jaccard
    similarity is computed on the same basis as Route A.  Coverage safety is
    checked against the target set only.
    """
    if not cfg.post_cover_jaccard_dedup_enabled or not selected_indices:
        return list(selected_indices), {"removed": 0, "kept": len(selected_indices)}

    # Use full env sets (all effective envs of the structure, not just target-restricted)
    # for Jaccard comparison to be consistent with Route A.
    index_to_full_envs = {k: set(full_candidates.get(k, set())) for k in selected_indices}
    # But coverage is checked against target-restricted envs.
    index_to_target_envs = {k: v & target for k, v in index_to_full_envs.items()}

    pruned, stats = post_cover_jaccard_dedup(
        selected_indices,
        index_to_envs=index_to_target_envs,   # coverage safety uses target-restricted sets
        threshold=cfg.post_cover_jaccard_threshold,
        size_band_frac=cfg.post_cover_jaccard_size_band_frac,
        verbose=cfg.verbose,
    )
    return pruned, stats


def run_initial_training_selection(
    cfg: Config,
    universe: TopologyUniverse,
    signature: str,
    out_dir: Path,
) -> Tuple[List[int], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Select an initial training set from the single large_file structure pool.

    Reduction stack applied in order:
      Route A  Jaccard pre-dedup on candidates
      Route C  Two-tier greedy cover (rare mandatory, common partial)
      prune    prune_redundant_items (env-count based)   ← previously missing
      Route D  Post-cover coverage-aware Jaccard dedup   ← NEW
    """
    target = set(universe.eff_universe)
    full_candidates: Dict[int, Set[str]] = {
        int(idx): set(envs) & target
        for idx, envs in universe.eff_struct_fps.items()
        if envs and (set(envs) & target)
    }
    print(f"[InitialTrain] target effective envs: {len(target):,}")
    print(f"[InitialTrain] candidate structures with envs: {len(full_candidates):,}")

    if not target or not full_candidates:
        summary: Dict[str, Any] = {
            "mode": "single_set_initial_training", "enabled": bool(cfg.initial_train_enabled),
            "target_envs": len(target), "candidate_structures_with_envs": len(full_candidates),
            "selected_structures": 0, "covered_envs": 0, "uncovered_envs": len(target),
            "coverage_pct": 0.0 if target else 100.0,
            "budget": int(cfg.initial_train_budget or 0),
            "two_tier_cover_enabled": cfg.two_tier_cover_enabled,
        }
        return [], [], summary

    ckpt_dir = out_dir / cfg.checkpoint_dir

    # ── Routes A + C ─────────────────────────────────────────────────────────
    if cfg.two_tier_cover_enabled:
        selected_indices, manifest, tier_stats = _two_tier_cover(
            full_candidates, target, universe, cfg,
            desc_prefix=".initial_train",
            ckpt_rare=ckpt_dir / "initial_train_cover_rare.pkl" if cfg.checkpoint_enabled else None,
            ckpt_common=ckpt_dir / "initial_train_cover_common.pkl" if cfg.checkpoint_enabled else None,
            ckpt_sig_rare=signature + ":rare",
            ckpt_sig_common=signature + ":common",
            budget=cfg.initial_train_budget,
        )
    else:
        candidates = full_candidates
        n_before = len(candidates)
        if cfg.jaccard_dedup_enabled:
            candidates = jaccard_dedup_candidates(
                candidates, threshold=cfg.jaccard_dedup_threshold,
                size_band_frac=cfg.jaccard_dedup_size_band_frac,
                prefer_larger_set=True, verbose=cfg.verbose,
            )
        ckpt_path = ckpt_dir / "initial_train_cover.pkl" if cfg.checkpoint_enabled else None
        raw, manifest = lazy_greedy_cover(
            candidates, target, budget=cfg.initial_train_budget,
            prefer_key=lambda k: int(k), desc="[InitialTrain.cover]",
            checkpoint_path=ckpt_path, checkpoint_signature=signature,
            resume=cfg.resume_from_checkpoint, interval=cfg.checkpoint_interval,
            stop_coverage_pct=cfg.initial_train_target_coverage_pct,
        )
        selected_indices = [int(x) for x in raw]
        tier_stats = {
            "candidates_before_dedup": n_before,
            "candidates_after_dedup": len(candidates),
        }

    # ── prune_redundant (previously missing in initial_train path) ────────────
    n_before_prune = len(selected_indices)
    item_envs_for_prune = {(int(k),): full_candidates.get(int(k), set()) for k in selected_indices}
    # Use the simpler direct coverage-count pruning on a flat list.
    counts: Counter = Counter()
    for idx in selected_indices:
        counts.update(full_candidates.get(idx, set()) & target)
    pruned_after_greedy: List[int] = list(selected_indices)
    for idx in list(selected_indices):
        envs = full_candidates.get(idx, set()) & target
        if envs and all(counts[e] >= 2 for e in envs):
            pruned_after_greedy.remove(idx)
            for e in envs:
                counts[e] -= 1
    selected_indices = pruned_after_greedy
    n_prune_removed = n_before_prune - len(selected_indices)
    if cfg.verbose and n_prune_removed > 0:
        print(f"[InitialTrain.prune] {n_before_prune:,} → {len(selected_indices):,} "
              f"(removed {n_prune_removed:,} redundant)")

    # ── Route D: post-cover coverage-aware Jaccard dedup ─────────────────────
    selected_indices, post_dedup_stats = _apply_post_cover_dedup(
        selected_indices, full_candidates, target, cfg, label="InitialTrain"
    )

    covered: Set[str] = set()
    for idx in selected_indices:
        covered |= full_candidates.get(idx, set()) & target
    uncovered = target - covered
    summary = {
        "mode": "single_set_initial_training",
        "enabled": bool(cfg.initial_train_enabled),
        "target_envs": len(target),
        "candidate_structures_with_envs": len(full_candidates),
        "selected_structures": len(selected_indices),
        "covered_envs": len(covered),
        "uncovered_envs": len(uncovered),
        "coverage_pct": 100.0 * len(covered) / max(1, len(target)),
        "budget": int(cfg.initial_train_budget or 0),
        "target_coverage_pct": float(cfg.initial_train_target_coverage_pct),
        "two_tier_cover_enabled": cfg.two_tier_cover_enabled,
        "prune_redundant_removed": n_prune_removed,
        "post_cover_jaccard_removed": post_dedup_stats.get("removed", 0),
        "output_xyz": cfg.initial_train_output_xyz,
        "manifest_csv": cfg.initial_train_manifest_csv,
        **tier_stats,
    }
    print(
        f"[InitialTrain] final selected={len(selected_indices):,}; "
        f"covered={len(covered):,}/{len(target):,} ({summary['coverage_pct']:.3f}%)"
    )
    return selected_indices, manifest, summary


def run_phase2_gap_supplement(
    cfg: Config,
    universe: TopologyUniverse,
    probe_data: ProbeData,
    signature: str,
    out_dir: Path,
) -> Tuple[List[int], List[Dict[str, Any]], Set[str]]:
    gap = set(universe.eff_universe) - set(probe_data.covered_eff)
    print(f"[Phase2] gap effective envs to supplement: {len(gap):,}")
    if not gap:
        return [], [], set()

    full_candidates: Dict[int, Set[str]] = {}
    probe_like = set(probe_data.struct_envs.keys()) if cfg.exclude_probe_indices_from_phase2_candidates else set()
    for idx, envs in universe.eff_struct_fps.items():
        if idx in probe_like:
            continue
        g = envs & gap
        if g:
            full_candidates[int(idx)] = set(g)
    print(f"[Phase2] candidate structures with gap envs: {len(full_candidates):,}")

    ckpt_dir = out_dir / cfg.checkpoint_dir

    if cfg.two_tier_cover_enabled:
        selected_indices, manifest, _ = _two_tier_cover(
            full_candidates, gap, universe, cfg, desc_prefix=".phase2",
            ckpt_rare=ckpt_dir / "phase2_gap_rare.pkl" if cfg.checkpoint_enabled else None,
            ckpt_common=ckpt_dir / "phase2_gap_common.pkl" if cfg.checkpoint_enabled else None,
            ckpt_sig_rare=signature + ":rare",
            ckpt_sig_common=signature + ":common",
            budget=cfg.phase2_budget,
        )
    else:
        candidates = full_candidates
        if cfg.jaccard_dedup_enabled:
            candidates = jaccard_dedup_candidates(
                candidates, threshold=cfg.jaccard_dedup_threshold,
                size_band_frac=cfg.jaccard_dedup_size_band_frac,
                prefer_larger_set=True, verbose=cfg.verbose,
            )
        ckpt_path = ckpt_dir / "phase2_gap.pkl" if cfg.checkpoint_enabled else None
        raw, manifest = lazy_greedy_cover(
            candidates, gap, budget=cfg.phase2_budget, prefer_key=lambda k: int(k),
            desc="[Phase2.cover]",
            checkpoint_path=ckpt_path, checkpoint_signature=signature,
            resume=cfg.resume_from_checkpoint, interval=cfg.checkpoint_interval,
        )
        selected_indices = [int(x) for x in raw]

    # Route D: post-cover dedup
    selected_indices, post_stats = _apply_post_cover_dedup(
        selected_indices, full_candidates, gap, cfg, label="Phase2"
    )

    covered_gap: Set[str] = set()
    for idx in selected_indices:
        covered_gap |= full_candidates.get(idx, set())
    print(f"[Phase2] selected={len(selected_indices):,}; covered_gap={len(covered_gap):,}/{len(gap):,}")
    return selected_indices, manifest, covered_gap


def prune_redundant_items(
    selected_items: List[Tuple[str, int]],
    item_envs: Dict[Tuple[str, int], Set[str]],
    target: Set[str],
    prune_probe_first: bool = True,
) -> List[Tuple[str, int]]:
    counts = Counter()
    for item in selected_items:
        counts.update(item_envs.get(item, set()) & target)
    def order_key(item):
        src, idx = item
        if prune_probe_first:
            return (0 if src == "probe" else 1, idx)
        return (0 if src == "phase2" else 1, idx)
    kept = list(selected_items)
    for item in sorted(list(selected_items), key=order_key):
        envs = item_envs.get(item, set()) & target
        if envs and all(counts[e] >= 2 for e in envs):
            kept.remove(item)
            for e in envs:
                counts[e] -= 1
    return kept


def run_probe_phase2_joint_cover(
    cfg: Config,
    probe_data: ProbeData,
    universe: TopologyUniverse,
    phase2_indices: Sequence[int],
    out_dir: Path,
) -> Tuple[List[Tuple[str, int]], Dict[str, Any]]:
    item_envs: Dict[Tuple[str, int], Set[str]] = {}
    for pidx, envs in probe_data.struct_envs.items():
        if envs: item_envs[("probe", int(pidx))] = set(envs)
    for lidx in phase2_indices:
        envs = universe.eff_struct_fps.get(int(lidx), set())
        if envs: item_envs[("phase2", int(lidx))] = set(envs)
    target: Set[str] = set()
    for envs in item_envs.values():
        target |= envs
    print(f"[Joint] candidates={len(item_envs):,}; target envs={len(target):,}")
    def prefer_key(item):
        src, idx = item
        if cfg.probe_phase2_joint_prefer_phase2_on_tie:
            return (0 if src == "phase2" else 1, idx)
        return (0 if src == "probe" else 1, idx)
    selected, manifest = lazy_greedy_cover(item_envs, target, budget=0,
                                           prefer_key=prefer_key, desc="[Joint.cover]")
    selected_items = [(str(src), int(idx)) for src, idx in selected]
    pruned_items = prune_redundant_items(
        selected_items, item_envs, target,
        prune_probe_first=cfg.probe_phase2_joint_prune_probe_first,
    )
    covered = set()
    for item in pruned_items:
        covered |= item_envs.get(item, set()) & target
    n_probe_in = len(probe_data.struct_envs)
    n_phase2_in = len(phase2_indices)
    n_probe_kept = sum(1 for s, _ in pruned_items if s == "probe")
    n_phase2_kept = sum(1 for s, _ in pruned_items if s == "phase2")
    summary = {
        "probe_input": n_probe_in, "phase2_input": n_phase2_in,
        "joint_candidates_with_envs": len(item_envs),
        "target_envs": len(target),
        "selected_before_pruning": len(selected_items),
        "selected_after_pruning": len(pruned_items),
        "covered_after_pruning": len(covered),
        "coverage_pct": 100.0 * len(covered) / max(1, len(target)),
        "kept_probe_structures": n_probe_kept, "kept_phase2_structures": n_phase2_kept,
        "removed_probe_structures": n_probe_in - n_probe_kept,
        "removed_phase2_structures": n_phase2_in - n_phase2_kept,
        "compression_pct_vs_probe_plus_phase2": (
            1.0 - len(pruned_items) / max(1, n_probe_in + n_phase2_in)
        ) * 100.0,
        "manifest": manifest,
    }
    print(f"[Joint] final={len(pruned_items):,}; coverage={summary['coverage_pct']:.3f}%")
    return pruned_items, summary


# =============================================================================
# Reports, CSV, CIF export
# =============================================================================

def write_csv(
    path: Path, rows: Sequence[Dict[str, Any]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fields: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in fields: fields.append(k)
        fieldnames = fields
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def phase2_manifest_rows(selected_indices, manifest, universe) -> List[Dict[str, Any]]:
    rows = []
    for rank, idx in enumerate(selected_indices, start=1):
        m = manifest[rank - 1] if rank - 1 < len(manifest) else {}
        rows.append({
            "rank": rank, "source": "large_file", "source_idx": int(idx),
            "n_effective_envs_in_structure": len(universe.eff_struct_fps.get(int(idx), set())),
            "pass": m.get("pass", ""),
            "new_gain": m.get("new_gain", ""), "covered": m.get("covered", ""),
            "uncovered": m.get("uncovered", ""), "coverage_pct": m.get("coverage_pct", ""),
        })
    return rows


def initial_training_manifest_rows(selected_indices, manifest, universe) -> List[Dict[str, Any]]:
    rows = []
    for rank, idx in enumerate(selected_indices, start=1):
        m = manifest[rank - 1] if rank - 1 < len(manifest) else {}
        rows.append({
            "rank": rank, "source": "large_file", "source_idx": int(idx),
            "n_effective_envs_in_structure": len(universe.eff_struct_fps.get(int(idx), set())),
            "pass": m.get("pass", ""),
            "new_gain": m.get("new_gain", ""), "covered": m.get("covered", ""),
            "uncovered": m.get("uncovered", ""), "coverage_pct": m.get("coverage_pct", ""),
        })
    return rows


def joint_manifest_rows(items, probe_data, universe) -> List[Dict[str, Any]]:
    rows = []
    for rank, (src, idx) in enumerate(items, start=1):
        envs = probe_data.struct_envs.get(idx, set()) if src == "probe" else universe.eff_struct_fps.get(idx, set())
        rows.append({"rank": rank, "source": src, "source_idx": int(idx), "n_envs": len(envs)})
    return rows


def write_report(
    cfg: Config, out_dir: Path, n_large: int, n_probe: int,
    physical_idx: Set[int], universe: TopologyUniverse,
    probe_data: Optional[ProbeData], phase2_indices: Sequence[int],
    covered_gap: Set[str], joint_summary: Optional[Dict[str, Any]],
    initial_train_summary: Optional[Dict[str, Any]] = None,
) -> None:
    if not cfg.write_sampler_report:
        return
    raw_struct_count = Counter({fp: len(set(s)) for fp, s in universe.fp_structs.items()})
    eff_struct_count = Counter({eff: len(set(s)) for eff, s in universe.eff_fp_structs.items()})
    rare_raw = sum(1 for c in raw_struct_count.values() if c <= cfg.rare_fingerprint_max_structures)
    rare_eff = sum(1 for c in eff_struct_count.values() if c <= cfg.rare_fingerprint_max_structures)
    fe_raw = sum(1 for fp in universe.all_fps if ":Fe:cn" in universe.meta_map.get(fp, ""))
    c_raw = sum(1 for fp in universe.all_fps if ":C:cn" in universe.meta_map.get(fp, ""))
    lines = [
        "Compact probe + Phase2 topology sampler report",
        "=" * 60,
        f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"large_file: {cfg.large_file}", f"probe_file: {cfg.probe_file}",
        f"output_dir: {cfg.output_dir}", "",
        "[Input]",
        f"large_structures_total: {n_large:,}",
        f"large_structures_physical_after_bond_filter: {len(physical_idx):,}",
        f"large_structures_nonphysical_after_bond_filter: {max(0, n_large - len(physical_idx)):,}",
        f"write_nonphysical_xyz: {cfg.write_nonphysical_xyz}",
        f"nonphysical_xyz_file: {out_dir / cfg.nonphysical_xyz_file if cfg.write_nonphysical_xyz and cfg.bond_filter_enabled else ''}",
        f"nonphysical_manifest_csv: {out_dir / cfg.nonphysical_manifest_csv if cfg.write_nonphysical_xyz and cfg.bond_filter_enabled else ''}",
        f"probe_structures_total: {n_probe:,}", "",
        "[Universe]",
        f"raw_fingerprints: {len(universe.all_fps):,}",
        f"rare_raw_fingerprints: {rare_raw:,}",
        f"Fe_centered_raw_fingerprints: {fe_raw:,}",
        f"C_centered_raw_fingerprints: {c_raw:,}",
        f"effective_fingerprints: {len(universe.eff_universe):,}",
        f"rare_effective_fingerprints: {rare_eff:,}",
        f"compression_reduction_pct: {universe.compress_stats.get('reduction_pct', 0):.3f}",
        f"geom_cluster_backend: {cfg.geom_cluster_backend}",
        f"scipy_ok: {SCIPY_OK}", f"scipy_version: {SCIPY_VERSION}", "",
        "[Reduction stack]",
        f"Route A  jaccard_dedup_enabled: {cfg.jaccard_dedup_enabled}",
        f"         jaccard_dedup_threshold (rare): {cfg.jaccard_dedup_threshold}",
        f"         jaccard_dedup_threshold_common: {cfg.jaccard_dedup_threshold_common}",
        f"         jaccard_dedup_size_band_frac: {cfg.jaccard_dedup_size_band_frac}",
        f"Route C  two_tier_cover_enabled: {cfg.two_tier_cover_enabled}",
        f"         rare_env_freq_threshold: {cfg.rare_env_freq_threshold}",
        f"         common_env_coverage_pct: {cfg.common_env_coverage_pct}",
        f"Route D  post_cover_jaccard_dedup_enabled: {cfg.post_cover_jaccard_dedup_enabled}",
        f"         post_cover_jaccard_threshold: {cfg.post_cover_jaccard_threshold}",
        f"         post_cover_jaccard_size_band_frac: {cfg.post_cover_jaccard_size_band_frac}",
    ]
    if initial_train_summary is not None:
        its = initial_train_summary
        lines += [
            "", "[Single-set initial training selection]",
            f"initial_train_target_envs: {its.get('target_envs', 0):,}",
            f"initial_train_candidate_structures_with_envs: {its.get('candidate_structures_with_envs', 0):,}",
            f"initial_train_selected_structures: {its.get('selected_structures', 0):,}",
            f"initial_train_prune_redundant_removed: {its.get('prune_redundant_removed', 0):,}",
            f"initial_train_post_cover_jaccard_removed: {its.get('post_cover_jaccard_removed', 0):,}",
            f"initial_train_covered_envs: {its.get('covered_envs', 0):,}",
            f"initial_train_uncovered_envs: {its.get('uncovered_envs', 0):,}",
            f"initial_train_coverage_pct: {its.get('coverage_pct', 0):.6f}",
        ]
        if its.get("two_tier_cover_enabled"):
            lines += [
                f"  n_rare_target: {its.get('n_rare_target', 0):,}",
                f"  n_common_target: {its.get('n_common_target', 0):,}",
                f"  rare_selected: {its.get('rare_selected', 0):,}",
                f"  common_selected: {its.get('common_selected', 0):,}",
                f"  coverage_rare_pct: {its.get('coverage_rare_pct', 0):.3f}",
                f"  coverage_common_pct: {its.get('coverage_common_pct', 0):.3f}",
            ]
    if probe_data is not None:
        lines += [
            "", "[Probe]",
            f"probe_covered_effective_fingerprints: {len(probe_data.covered_eff):,}",
            f"probe_only_raw_fingerprints: {len(probe_data.probe_only_raw):,}",
        ]
    lines += [
        "", "[Phase2 gap supplement]",
        f"phase2_selected_structures: {len(phase2_indices):,}",
        f"phase2_covered_gap_effective_fingerprints: {len(covered_gap):,}",
    ]
    if joint_summary is not None:
        lines += [
            "", "[Probe + Phase2 joint minimum cover]",
            f"joint_target_envs: {joint_summary.get('target_envs', 0):,}",
            f"joint_selected_after_pruning: {joint_summary.get('selected_after_pruning', 0):,}",
            f"joint_coverage_pct: {joint_summary.get('coverage_pct', 0):.6f}",
            f"joint_kept_probe_structures: {joint_summary.get('kept_probe_structures', 0):,}",
            f"joint_kept_phase2_structures: {joint_summary.get('kept_phase2_structures', 0):,}",
            f"joint_compression_pct_vs_probe_plus_phase2: {joint_summary.get('compression_pct_vs_probe_plus_phase2', 0):.3f}",
        ]
    (out_dir / cfg.report_file).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ase_center_index(center_atom_num: int, n_atoms: int) -> Optional[int]:
    if 0 <= center_atom_num < n_atoms: return int(center_atom_num)
    if 1 <= center_atom_num <= n_atoms: return int(center_atom_num - 1)
    return None


def make_local_cluster(atoms, center_idx, rcut, vacuum, eps):
    require_ase()
    if center_idx is None or center_idx < 0 or center_idx >= len(atoms): return None
    vecs = atoms.get_distances(center_idx, list(range(len(atoms))), mic=True, vector=True)
    dists = np.linalg.norm(vecs, axis=1)
    keep = [i for i, d in enumerate(dists) if d <= rcut + eps]
    if not keep: return None
    symbols = [atoms[i].symbol for i in keep]
    rel = np.array([vecs[i] for i in keep], dtype=float)
    span = np.ptp(rel, axis=0) if len(rel) > 1 else np.zeros(3)
    cell_len = float(max(np.max(span) + vacuum, 2 * rcut + vacuum, vacuum))
    return Atoms(symbols=symbols, positions=rel + cell_len / 2.0,
                 cell=[cell_len]*3, pbc=False)


def export_cluster_env_cifs(cfg, structs_large, universe, out_dir):
    if not cfg.export_cluster_env_cifs: return
    require_ase()
    cif_root = out_dir / cfg.cluster_env_cif_dir
    if cfg.cluster_env_cif_clear_existing and cif_root.exists(): shutil.rmtree(cif_root)
    cif_root.mkdir(parents=True, exist_ok=True)
    eff_to_raw: Dict[str, List[str]] = defaultdict(list)
    for raw, eff in universe.fp_to_eff.items(): eff_to_raw[eff].append(raw)
    total = 0; index_rows = []
    max_total = int(cfg.cluster_env_cif_max_total or 0)
    max_per = int(cfg.cluster_env_cif_max_per_cluster or 0)
    print(f"[CIF] Export cluster environment CIFs to {cif_root}")
    for eff in tqdm(sorted(eff_to_raw), desc="[CIF.export]"):
        if max_total and total >= max_total: break
        per_count = 0; eff_dir = cif_root / _safe_name(eff, 80)
        for raw in sorted(eff_to_raw[eff]):
            rcut = _rcut_from_fp_or_eff(raw)
            for struct_idx, center_atom_num, sym in universe.fp_examples.get(raw, []):
                if (max_per and per_count >= max_per) or (max_total and total >= max_total): break
                ase_idx = _ase_center_index(int(center_atom_num), len(structs_large[int(struct_idx)]))
                if ase_idx is None: continue
                cluster = make_local_cluster(structs_large[int(struct_idx)], ase_idx, rcut,
                                             cfg.cluster_env_cif_vacuum, cfg.cluster_env_cif_distance_eps)
                if cluster is None: continue
                eff_dir.mkdir(parents=True, exist_ok=True)
                fname = f"env_{total:07d}_s{int(struct_idx)}_a{ase_idx}_{sym}_r{rcut:.1f}.cif"
                path = eff_dir / fname; write(str(path), cluster)
                index_rows.append({"cif_file": str(path.relative_to(cif_root)),
                                   "effective_fp": eff, "raw_fp": raw,
                                   "source_idx": int(struct_idx),
                                   "center_atom_index": int(ase_idx),
                                   "center_symbol": sym, "rcut": rcut})
                total += 1; per_count += 1
            if max_per and per_count >= max_per: break
    write_csv(cif_root / "cluster_env_cifs_index.csv", index_rows)
    print(f"[CIF] exported CIF files: {total:,}")


# =============================================================================
# Publication figures (abbreviated — same as previous version)
# =============================================================================

def _plot_log(msg, cfg): 
    if getattr(cfg, "verbose", True): print(msg)


def apply_publication_plot_style(cfg):
    if not MATPLOTLIB_OK: return
    try:
        plt.rcParams.update({
            "font.family": getattr(cfg, "plot_font_family", "DejaVu Sans"),
            "font.weight": getattr(cfg, "plot_font_weight", "bold"),
            "axes.titleweight": getattr(cfg, "plot_font_weight", "bold"),
            "axes.labelweight": getattr(cfg, "plot_font_weight", "bold"),
            "axes.titlesize": int(getattr(cfg, "plot_title_size", 18)),
            "axes.labelsize": int(getattr(cfg, "plot_label_size", 15)),
            "xtick.labelsize": int(getattr(cfg, "plot_tick_size", 13)),
            "ytick.labelsize": int(getattr(cfg, "plot_tick_size", 13)),
            "legend.fontsize": int(getattr(cfg, "plot_legend_size", 11)),
            "figure.titlesize": int(getattr(cfg, "plot_suptitle_size", 16)),
            "axes.linewidth": float(getattr(cfg, "plot_axis_linewidth", 1.6)),
            "savefig.dpi": int(getattr(cfg, "individual_panel_dpi", 300)),
            "pdf.fonttype": 42, "ps.fonttype": 42,
        })
    except Exception: pass


def style_axes_publication(ax, cfg):
    fw = getattr(cfg, "plot_font_weight", "bold")
    try:
        ax.title.set_fontsize(int(getattr(cfg, "plot_title_size", 18))); ax.title.set_fontweight(fw)
        ax.xaxis.label.set_fontsize(int(getattr(cfg, "plot_label_size", 15))); ax.xaxis.label.set_fontweight(fw)
        ax.yaxis.label.set_fontsize(int(getattr(cfg, "plot_label_size", 15))); ax.yaxis.label.set_fontweight(fw)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontsize(int(getattr(cfg, "plot_tick_size", 13))); t.set_fontweight(fw)
        for sp in ax.spines.values(): sp.set_linewidth(float(getattr(cfg, "plot_axis_linewidth", 1.6)))
        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize(int(getattr(cfg, "plot_legend_size", 11))); txt.set_fontweight(fw)
            try: leg.get_frame().set_linewidth(1.2)
            except: pass
    except: pass


def style_figure_publication(fig, cfg):
    apply_publication_plot_style(cfg)
    for ax in fig.axes: style_axes_publication(ax, cfg)
    try:
        for txt in fig.texts:
            txt.set_fontsize(int(getattr(cfg, "plot_suptitle_size", 16)))
            txt.set_fontweight(getattr(cfg, "plot_font_weight", "bold"))
    except: pass


def _safe_panel_name(text, fallback):
    text = (text or fallback).strip()
    text = re.sub(r"^\(([A-Za-z0-9]+)\)\s*", r"\1_", text)
    text = re.sub(r"[^\w\-.]+", "_", text); text = re.sub(r"_+", "_", text).strip("_")
    return text[:90] if text else fallback


def save_individual_panels(fig, axes, out_dir, prefix, cfg, panel_names=None):
    if not bool(getattr(cfg, "save_individual_panels", True)): return
    try:
        style_figure_publication(fig, cfg); fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        root = Path(out_dir) / getattr(cfg, "individual_panel_dir", "individual_panels") / prefix
        root.mkdir(parents=True, exist_ok=True)
        if not isinstance(axes, (list, tuple, np.ndarray)): axes = [axes]
        flat = []
        for a in axes:
            if isinstance(a, (list, tuple, np.ndarray)): flat.extend(list(np.ravel(a)))
            else: flat.append(a)
        fmt = str(getattr(cfg, "individual_panel_format", "png")).lstrip(".")
        dpi = int(getattr(cfg, "individual_panel_dpi", 300))
        for i, ax in enumerate(flat, 1):
            if ax is None: continue
            title = ax.get_title() or f"panel_{i:02d}"
            name = _safe_panel_name(panel_names[i-1] if panel_names and i-1 < len(panel_names) else title, f"panel_{i:02d}")
            bbox = ax.get_tightbbox(renderer)
            if bbox is None: bbox = ax.get_window_extent(renderer)
            bbox = bbox.transformed(fig.dpi_scale_trans.inverted()).expanded(1.08, 1.12)
            fig.savefig(root / f"{prefix}_{i:02d}_{name}.{fmt}", dpi=dpi, bbox_inches=bbox)
        _plot_log(f"[Panels] individual panels → '{root}'", cfg)
    except Exception as exc: _plot_log(f"[Panels][WARN] {exc}", cfg)


def _read_csv_rows(path):
    path = Path(path)
    if not path.exists(): return []
    with open(path, newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def _as_float(v, default=np.nan):
    try: return float(v)
    except: return default


def _as_int(v, default=0):
    try: return int(float(v))
    except: return default


def _tokenize_type_string(s):
    base = re.findall(r"[A-Za-z]+|cn\d+|r\d+(?:\.\d+)?|sim\d+|\d+|[A-Za-z0-9_.:-]+", str(s))
    compact = re.sub(r"\s+", "", str(s))
    grams = [compact[i:i+3] for i in range(max(0, len(compact) - 2))]
    return base + grams


def _hash_type_vectors(types, hash_dim=128):
    X = np.zeros((len(types), int(hash_dim)), dtype=np.float32)
    for i, typ in enumerate(types):
        for tok in _tokenize_type_string(typ):
            h = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16)
            j = h % int(hash_dim); sign = 1.0 if ((h >> 8) & 1) else -1.0; X[i, j] += sign
        norm = np.linalg.norm(X[i])
        if norm > 0: X[i] /= norm
    return X


def _pca_2d_numpy(X):
    if len(X) == 0: return np.zeros((0, 2), dtype=np.float32)
    if len(X) == 1: return np.zeros((1, 2), dtype=np.float32)
    X = np.asarray(X, dtype=np.float64); X = X - X.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False); Y = U[:, :2] * S[:2]
        if Y.shape[1] == 1: Y = np.c_[Y[:, 0], np.zeros(len(Y))]
        return Y.astype(np.float32)
    except: return np.zeros((len(X), 2), dtype=np.float32)


def _sample_types_for_plot(all_types, priority, max_types):
    all_sorted = sorted(all_types); max_types = int(max_types)
    if max_types <= 0 or len(all_sorted) <= max_types: return all_sorted
    pri = sorted(priority & all_types); rest = [x for x in all_sorted if x not in set(pri)]
    if len(pri) >= max_types:
        step = max(1, int(np.ceil(len(pri) / max_types))); return pri[::step][:max_types]
    remain = max_types - len(pri); step = max(1, int(np.ceil(len(rest) / max(1, remain))))
    return pri + rest[::step][:remain]


class PublicationPlotter:

    @staticmethod
    def _plot_root(out_dir, cfg):
        root = Path(out_dir) / getattr(cfg, "publication_plot_dir", "publication_figures")
        root.mkdir(parents=True, exist_ok=True); return root

    @staticmethod
    def save(fig, axes, out_path, cfg, panel_prefix=None, panel_names=None):
        if not (MATPLOTLIB_OK and getattr(cfg, "publication_plot_enabled", True)):
            plt.close(fig); return
        apply_publication_plot_style(cfg); style_figure_publication(fig, cfg)
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
        dpi = int(getattr(cfg, "individual_panel_dpi", 300))
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        try: fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        except: pass
        if panel_prefix: save_individual_panels(fig, axes, out_path.parent, panel_prefix, cfg, panel_names=panel_names)
        plt.close(fig); _plot_log(f"[Figure] {out_path}", cfg)

    @staticmethod
    def plot_universe_summary(universe, out_dir, cfg):
        if not (getattr(cfg, "publication_plot_enabled", True) and getattr(cfg, "plot_universe_summary", True)): return
        root = PublicationPlotter._plot_root(out_dir, cfg); apply_publication_plot_style(cfg)
        s = universe.compress_stats or {}
        raw_counts = np.array(sorted([len(v) for v in universe.fp_structs.values()], reverse=True), dtype=float)
        eff_counts = np.array(sorted([len(v) for v in universe.eff_fp_structs.values()], reverse=True), dtype=float)
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
        ax = axes[0, 0]
        vals = [s.get("n_raw", len(universe.all_fps)), s.get("n_eff", len(universe.eff_universe))]
        ax.bar(["Raw fingerprints", "Effective types"], vals, edgecolor="black", linewidth=1.2)
        for x, y in enumerate(vals): ax.text(x, y, f"{int(y):,}", ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel("Number of types"); ax.set_title("(a) Topology compression"); ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax = axes[0, 1]; reduction = float(s.get("reduction_pct", 0.0))
        ax.bar(["Compression"], [reduction], edgecolor="black", linewidth=1.2)
        ax.set_ylim(0, max(100, reduction * 1.15)); ax.set_ylabel("Reduction (%)"); ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.text(0, reduction, f"{reduction:.1f}%", ha="center", va="bottom", fontweight="bold"); ax.set_title("(b) Compression ratio")
        ax = axes[1, 0]
        if len(raw_counts): ax.loglog(np.arange(1, len(raw_counts)+1), raw_counts, lw=2.2, label="Raw fp")
        if len(eff_counts): ax.loglog(np.arange(1, len(eff_counts)+1), eff_counts, lw=2.2, label="Effective type")
        ax.set_xlabel("Rank"); ax.set_ylabel("Structure frequency"); ax.set_title("(c) Frequency distribution"); ax.legend(frameon=True)
        ax = axes[1, 1]
        names = ["Geom.", "Topo.", "No geom."]
        vals2 = [int(s.get("n_geom_ward_groups", 0)), int(s.get("n_topology_clustered", 0)), int(s.get("n_no_geom", 0))]
        if max(vals2) == 0: names = ["Meta-types", "Clusters", "Singletons"]; vals2 = [int(s.get("n_meta_types", 0)), int(s.get("n_clusters", 0)), int(s.get("n_singletons", 0))]
        ax.bar(names, vals2, edgecolor="black", linewidth=1.2); ax.set_ylabel("Count"); ax.set_title("(d) Clustering composition")
        for x, y in enumerate(vals2): ax.text(x, y, f"{int(y):,}", ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        for ax in axes.ravel(): ax.grid(True, alpha=0.25)
        fig.tight_layout()
        PublicationPlotter.save(fig, axes.ravel(), root / "figure_01_universe_compression_summary.png", cfg,
                                panel_prefix="figure_01_universe_compression_summary",
                                panel_names=["raw_vs_effective", "compression_ratio", "frequency_distribution", "clustering_composition"])

    @staticmethod
    def plot_topology_state(universe, out_dir, cfg, selected_indices, stage, before_eff=None):
        if not (getattr(cfg, "publication_plot_enabled", True) and getattr(cfg, "plot_topology_state_map", True)): return
        if not universe.eff_universe: return
        selected_eff = _covered_by_indices(universe, selected_indices)
        before_eff = set(before_eff or set()); priority = before_eff | selected_eff
        types = _sample_types_for_plot(set(universe.eff_universe), priority, int(getattr(cfg, "plot_topology_max_types", 12000)))
        if not types: return
        X = _hash_type_vectors(types, hash_dim=int(getattr(cfg, "plot_hash_dim", 128)))
        Y = _pca_2d_numpy(X); type_to_xy = {t: Y[i] for i, t in enumerate(types)}
        root = PublicationPlotter._plot_root(out_dir, cfg)
        with open(root / f"{stage}_topology_state_embedding.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["effective_type", "x", "y", "covered_before", "covered_by_selected", "covered_after"])
            after_eff = before_eff | selected_eff
            for t in types:
                x, y = type_to_xy[t]; w.writerow([t, f"{x:.8g}", f"{y:.8g}", int(t in before_eff), int(t in selected_eff), int(t in after_eff)])
        if before_eff:
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
            for title, cb, nw, ax in [("Before supplement", before_eff, set(), axes[0]),
                                       ("After supplement", before_eff, selected_eff - before_eff, axes[1])]:
                arr_un = np.array([type_to_xy[t] for t in types if t not in cb and t not in nw])
                arr_b = np.array([type_to_xy[t] for t in types if t in cb])
                arr_n = np.array([type_to_xy[t] for t in types if t in nw])
                if len(arr_un): ax.scatter(arr_un[:,0],arr_un[:,1],s=8,color=getattr(cfg,"plot_state_color_uncovered","#BDBDBD"),alpha=float(getattr(cfg,"plot_state_alpha_uncovered",0.16)),linewidths=0,rasterized=True,label=f"Uncovered ({len(arr_un):,})")
                if len(arr_b): ax.scatter(arr_b[:,0],arr_b[:,1],s=12*float(getattr(cfg,"plot_state_size_before_scale",0.85)),color=getattr(cfg,"plot_state_color_before","#008837"),alpha=float(getattr(cfg,"plot_state_alpha_before",0.50)),linewidths=0,rasterized=True,label=f"Covered before ({len(arr_b):,})")
                if len(arr_n): ax.scatter(arr_n[:,0],arr_n[:,1],s=14*float(getattr(cfg,"plot_state_size_new_scale",1.45)),color=getattr(cfg,"plot_state_color_new","#D7191C"),alpha=float(getattr(cfg,"plot_state_alpha_new",0.90)),linewidths=0,rasterized=True,label=f"Newly covered ({len(arr_n):,})")
                ax.set_xlabel("Topology-PC1"); ax.set_ylabel("Topology-PC2"); ax.set_title(title); ax.legend(frameon=True); ax.grid(True, alpha=0.20)
            fig.tight_layout()
            PublicationPlotter.save(fig, axes, root / f"figure_{stage}_topology_state_map.png", cfg,
                                    panel_prefix=f"figure_{stage}_topology_state_map",
                                    panel_names=["before_supplement", "after_supplement"])
        else:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.2))
            arr_un = np.array([type_to_xy[t] for t in types if t not in selected_eff])
            arr_s = np.array([type_to_xy[t] for t in types if t in selected_eff])
            if len(arr_un): ax.scatter(arr_un[:,0],arr_un[:,1],s=8,color=getattr(cfg,"plot_state_color_uncovered","#BDBDBD"),alpha=float(getattr(cfg,"plot_state_alpha_uncovered",0.16)),linewidths=0,rasterized=True,label=f"Uncovered ({len(arr_un):,})")
            if len(arr_s): ax.scatter(arr_s[:,0],arr_s[:,1],s=14*float(getattr(cfg,"plot_state_size_new_scale",1.45)),color=getattr(cfg,"plot_state_color_new","#D7191C"),alpha=float(getattr(cfg,"plot_state_alpha_new",0.90)),linewidths=0,rasterized=True,label=f"Covered by selected ({len(arr_s):,})")
            ax.set_xlabel("Topology-PC1"); ax.set_ylabel("Topology-PC2"); ax.set_title(f"{stage.capitalize()} topology coverage map"); ax.legend(frameon=True); ax.grid(True, alpha=0.20)
            fig.tight_layout()
            PublicationPlotter.save(fig, [ax], root / f"figure_{stage}_topology_state_map.png", cfg,
                                    panel_prefix=f"figure_{stage}_topology_state_map", panel_names=["topology_coverage_map"])

    @staticmethod
    def _effective_type_rcut_sets(universe):
        rcut_to_eff: Dict[float, Set[str]] = defaultdict(set)
        for raw_fp, eff_fp in getattr(universe, "fp_to_eff", {}).items():
            r = _rcut_from_fp_or_eff(str(raw_fp), default=np.nan)
            if r is not None and np.isfinite(r): rcut_to_eff[round(float(r), 6)].add(str(eff_fp))
        for eff_fp in getattr(universe, "eff_universe", set()):
            m = re.match(r"^r([0-9]+(?:\.[0-9]+)?):", str(eff_fp))
            if m: rcut_to_eff[round(float(m.group(1)), 6)].add(str(eff_fp))
        return dict(rcut_to_eff)

    @staticmethod
    def _ordered_plot_rcuts(universe, cfg):
        rcut_to_eff = PublicationPlotter._effective_type_rcut_sets(universe)
        available = set(rcut_to_eff)
        configured = [round(float(r), 6) for r in getattr(cfg, "rcut_list", [])]
        ordered = [r for r in configured if r in available]
        ordered += [r for r in sorted(available) if r not in set(ordered)]
        return ordered

    @staticmethod
    def _weighted_env_coverage_percent(universe, covered_eff, target_eff=None):
        target = set(target_eff if target_eff is not None else universe.eff_universe)
        if not target: return 100.0
        weights = {e: max(1, len(universe.eff_fp_structs.get(e, []))) for e in target}
        den = float(sum(weights.values()))
        if den <= 0: return 100.0
        return 100.0 * float(sum(weights[e] for e in (covered_eff & target) if e in weights)) / den

    @staticmethod
    def _select_curve_steps(n_selected, max_points):
        max_points = max(2, int(max_points))
        if n_selected + 1 <= max_points: return set(range(0, n_selected + 1))
        stride = int(np.ceil(n_selected / max(1, max_points - 1)))
        steps = set(range(0, n_selected + 1, stride)); steps.add(n_selected); return steps

    @staticmethod
    def _gap_supplement_coverage_trajectory(universe, selected_indices, before_eff, cfg):
        rcut_to_eff = PublicationPlotter._effective_type_rcut_sets(universe)
        rcuts = PublicationPlotter._ordered_plot_rcuts(universe, cfg)
        selected_indices = [int(i) for i in selected_indices]
        record_steps = PublicationPlotter._select_curve_steps(len(selected_indices),
                                                              int(getattr(cfg, "plot_gap_supplement_max_curve_points", 650)))
        covered = set(before_eff) & set(universe.eff_universe)
        xs = []; curves = {r: [] for r in rcuts}; weighted = []
        def record(step):
            xs.append(int(step))
            for r in rcuts:
                t2 = rcut_to_eff.get(r, set())
                curves[r].append(100.0 * len(covered & t2) / len(t2) if t2 else np.nan)
            weighted.append(PublicationPlotter._weighted_env_coverage_percent(universe, covered))
        record(0)
        for rank, idx in enumerate(selected_indices, start=1):
            covered |= set(universe.eff_struct_fps.get(int(idx), set()))
            if rank in record_steps: record(rank)
        return (np.asarray(xs, dtype=float),
                {r: np.asarray(vals, dtype=float) for r, vals in curves.items()},
                np.asarray(weighted, dtype=float))

    @staticmethod
    def _write_gap_supplement_coverage_csv(csv_path, xs, curves, weighted):
        csv_path = Path(csv_path); csv_path.parent.mkdir(parents=True, exist_ok=True)
        rcuts = list(curves.keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["supplement_structures", "weighted_env_coverage_pct"] + [f"rcut_{r:g}_coverage_pct" for r in rcuts])
            for i, x in enumerate(xs):
                w.writerow([int(x), f"{weighted[i]:.8f}"] + [f"{curves[r][i]:.8f}" if np.isfinite(curves[r][i]) else "" for r in rcuts])

    @staticmethod
    def _draw_gap_supplement_convergence(ax, universe, selected_indices, before_eff, cfg):
        xs, curves, weighted = PublicationPlotter._gap_supplement_coverage_trajectory(universe, selected_indices, before_eff, cfg)
        ax.plot(xs, weighted, "k--", lw=2.2, label="weighted env coverage")
        for r, y in curves.items(): ax.plot(xs, y, lw=2.0, label=f"rcut={r:g} Å")
        ax.axhline(100.0, color="red", ls=":", lw=1.8, label="target")
        ax.set_xlim(left=0); ax.set_ylim(0, 104); ax.set_xlabel("supplement structures"); ax.set_ylabel("coverage (%)")
        ax.set_title("(E) Gap-supplement coverage convergence"); ax.grid(True, alpha=0.22); ax.legend(frameon=True, loc="lower right")
        return xs, curves, weighted

    @staticmethod
    def _draw_after_gap_supplement_map(ax, universe, selected_indices, before_eff, cfg, stage, root=None):
        selected_eff = _covered_by_indices(universe, selected_indices)
        before_eff = set(before_eff or set()) & set(universe.eff_universe)
        newly_eff = (selected_eff - before_eff) & set(universe.eff_universe)
        still_uncovered = set(universe.eff_universe) - before_eff - newly_eff
        types = _sample_types_for_plot(set(universe.eff_universe), before_eff | newly_eff | still_uncovered,
                                       int(getattr(cfg, "plot_topology_max_types", 12000)))
        if not types: ax.set_axis_off(); return
        X = _hash_type_vectors(types, hash_dim=int(getattr(cfg, "plot_hash_dim", 128)))
        Y = _pca_2d_numpy(X); type_to_xy = {t: Y[i] for i, t in enumerate(types)}
        if root is not None:
            with open(Path(root) / f"{stage}_after_gap_supplement_embedding.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["effective_type", "x", "y", "state"])
                for t in types:
                    state = "newly_covered" if t in newly_eff else "covered_before" if t in before_eff else "still_uncovered"
                    x, y = type_to_xy[t]; w.writerow([t, f"{x:.8g}", f"{y:.8g}", state])
        for arr, lbl, col, alpha, sz, mk, ec in [
            (np.array([type_to_xy[t] for t in types if t in still_uncovered]),
             f"still uncovered ({len(still_uncovered):,})", getattr(cfg,"plot_state_color_uncovered","#BDBDBD"),
             float(getattr(cfg,"plot_state_alpha_uncovered",0.16)),10,"o","none"),
            (np.array([type_to_xy[t] for t in types if t in before_eff]),
             f"covered before ({len(before_eff):,})", getattr(cfg,"plot_state_color_before","#008837"),
             float(getattr(cfg,"plot_state_alpha_before",0.50)),13*float(getattr(cfg,"plot_state_size_before_scale",0.85)),"o","none"),
            (np.array([type_to_xy[t] for t in types if t in newly_eff]),
             f"newly covered ({len(newly_eff):,})", getattr(cfg,"plot_state_color_new","#D7191C"),
             float(getattr(cfg,"plot_state_alpha_new",0.90)),23*float(getattr(cfg,"plot_state_size_new_scale",1.45)),"^","black"),
        ]:
            if len(arr): ax.scatter(arr[:,0],arr[:,1],s=sz,color=col,alpha=alpha,linewidths=0.35 if ec!="none" else 0,edgecolors=ec,marker=mk,rasterized=True,label=lbl)
            else: ax.scatter([],[],s=sz,color=col,alpha=alpha,linewidths=0.35 if ec!="none" else 0,edgecolors=ec,marker=mk,label=lbl)
        ax.set_xlabel("Topology-2D-1"); ax.set_ylabel("Topology-2D-2"); ax.set_title("(B) After gap supplement")
        ax.grid(True, alpha=0.20); ax.legend(frameon=True, loc="lower left")

    @staticmethod
    def plot_gap_supplement_panels(universe, out_dir, cfg, selected_indices, before_eff, stage="phase2"):
        if not (MATPLOTLIB_OK and getattr(cfg, "publication_plot_enabled", True) and getattr(cfg, "plot_gap_supplement_panels", True)): return
        if not universe.eff_universe: return
        before_eff = set(before_eff or set()) & set(universe.eff_universe)
        selected_indices = [int(i) for i in selected_indices]
        if not selected_indices and not before_eff: return
        root = PublicationPlotter._plot_root(out_dir, cfg); apply_publication_plot_style(cfg)
        fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
        PublicationPlotter._draw_after_gap_supplement_map(axes[0], universe, selected_indices, before_eff, cfg, stage=stage, root=root)
        xs, curves, weighted = PublicationPlotter._draw_gap_supplement_convergence(axes[1], universe, selected_indices, before_eff, cfg)
        PublicationPlotter._write_gap_supplement_coverage_csv(root / f"{stage}_gap_supplement_coverage_convergence.csv", xs, curves, weighted)
        fig.tight_layout()
        PublicationPlotter.save(fig, axes, root / f"figure_{stage}_gap_supplement_B_E.png", cfg,
                                panel_prefix=f"figure_{stage}_gap_supplement_B_E",
                                panel_names=["after_gap_supplement", "gap_supplement_coverage_convergence"])


def _covered_by_indices(u, indices):
    covered = set()
    for idx in indices: covered |= set(getattr(u, "eff_struct_fps", {}).get(int(idx), set()))
    return covered


def _make_convergence_plot(manifest, out_dir, cfg, csv_name, fig_name, panel_prefix, xlabel, title_a):
    if not (MATPLOTLIB_OK and getattr(cfg, "publication_plot_enabled", True) and getattr(cfg, "plot_selection_convergence", True)) or not manifest: return
    root = PublicationPlotter._plot_root(out_dir, cfg)
    rows = [{
        # Use global sequential rank (rank_global) when available — avoids the
        # per-pass rank reset that caused a double-curve artifact in two-tier mode.
        "step": int(r.get("rank_global", i + 1)),
        "new_gain": int(r.get("new_gain", 0)),
        # Use global cumulative coverage when available (recalculated in _two_tier_cover).
        "covered": int(r.get("covered_global", r.get("covered", 0))),
        "uncovered": int(r.get("uncovered_global", r.get("uncovered", 0))),
        "coverage_pct": float(r.get("coverage_pct_global", r.get("coverage_pct", 0.0))),
        "pass": str(r.get("pass", "")),
    } for i, r in enumerate(manifest)]
    write_csv(root / csv_name, rows)
    ranks = np.array([r["step"] for r in rows], dtype=float)
    gains = np.array([r["new_gain"] for r in rows], dtype=float)
    cov_pct = np.array([r["coverage_pct"] for r in rows], dtype=float)
    covered = np.array([r["covered"] for r in rows], dtype=float)
    passes = [r["pass"] for r in rows]
    pass_colors = ["#2ca02c" if p == "rare" else "#ff7f0e" if p == "common" else "#1f77b4" for p in passes]
    apply_publication_plot_style(cfg); fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    if np.isfinite(cov_pct).any():
        for i in range(len(ranks) - 1): ax.plot(ranks[i:i+2], cov_pct[i:i+2], color=pass_colors[i], lw=2.0)
        ax.set_ylabel("Coverage (%)"); ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    else:
        ax.plot(ranks, covered, lw=2.6); ax.set_ylabel("Covered effective types")
    ax.set_xlabel(xlabel); ax.set_title(title_a); ax.grid(True, alpha=0.25)
    ax = axes[1]; ax.plot(ranks, gains, lw=1.7); ax.set_xlabel(xlabel); ax.set_ylabel("Newly covered types"); ax.set_title("(b) Marginal gain trajectory"); ax.grid(True, alpha=0.25)
    ax = axes[2]
    if len(gains): bins = min(50, max(10, int(np.sqrt(len(gains))))); ax.hist(gains[gains > 0], bins=bins, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Newly covered types per structure"); ax.set_ylabel("Frequency"); ax.set_title("(c) Gain distribution"); ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    PublicationPlotter.save(fig, axes, root / fig_name, cfg, panel_prefix=panel_prefix,
                            panel_names=["coverage_convergence", "marginal_gain", "gain_distribution"])


def _publication_plot_phase2_convergence(manifest, out_dir, cfg):
    _make_convergence_plot(manifest, out_dir, cfg,
                           csv_name="phase2_gap_supplement_coverage_convergence.csv",
                           fig_name="figure_phase2_coverage_convergence.png",
                           panel_prefix="figure_phase2_coverage_convergence",
                           xlabel="Selected Phase2 structures",
                           title_a="(a) Phase2 coverage convergence")


def _publication_plot_joint_summary(summary, out_dir, cfg):
    if not (MATPLOTLIB_OK and getattr(cfg, "publication_plot_enabled", True) and getattr(cfg, "plot_joint_summary", True)) or not summary: return
    root = PublicationPlotter._plot_root(out_dir, cfg)
    labels = ["Probe input", "Phase2 input", "Probe kept", "Phase2 kept"]
    vals = [summary.get("probe_input", 0), summary.get("phase2_input", 0),
            summary.get("kept_probe_structures", 0), summary.get("kept_phase2_structures", 0)]
    apply_publication_plot_style(cfg); fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(labels, vals, edgecolor="black", linewidth=1.2)
    for x, y in enumerate(vals): ax.text(x, y, f"{int(y):,}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Number of structures"); ax.set_title("Probe + Phase2 joint minimum-cover summary")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True)); plt.setp(ax.get_xticklabels(), rotation=20, ha="right"); ax.grid(True, axis="y", alpha=0.25); fig.tight_layout()
    PublicationPlotter.save(fig, [ax], root / "figure_probe_phase2_joint_summary.png", cfg,
                            panel_prefix="figure_probe_phase2_joint_summary", panel_names=["joint_min_cover_summary"])


PublicationPlotter.plot_phase2_convergence = staticmethod(_publication_plot_phase2_convergence)
PublicationPlotter.plot_joint_summary = staticmethod(_publication_plot_joint_summary)


def _publication_plot_initial_training_convergence(manifest, out_dir, cfg):
    _make_convergence_plot(manifest, out_dir, cfg,
                           csv_name="initial_training_coverage_convergence.csv",
                           fig_name="figure_initial_training_coverage_convergence.png",
                           panel_prefix="figure_initial_training_coverage_convergence",
                           xlabel="Selected initial-training structures",
                           title_a="(a) Initial-training coverage convergence")


PublicationPlotter.plot_initial_training_convergence = staticmethod(_publication_plot_initial_training_convergence)


# =============================================================================
# Universe cache IO and main flow
# =============================================================================

def save_universe_cache(path, signature, universe):
    _atomic_pickle_dump({"__topology_universe_cache__": "CompactProbePhase2V1",
                         "signature": signature, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "universe_state": universe.state_dict()}, path)


def load_universe_cache(path, signature):
    data = _atomic_pickle_load(path)
    if not isinstance(data, dict): return None
    if data.get("signature") != signature:
        print(f"[UniverseCache] Ignore stale cache: {path}"); return None
    state = data.get("universe_state") or data.get("state")
    if not isinstance(state, dict): return None
    u = TopologyUniverse(); u.load_state_dict(state); return u


def load_structures(path):
    require_ase(); print(f"[IO] Read structures: {path}")
    return list(read(path, index=":"))


def main(cfg: Config) -> None:
    require_ase(); require_spdkit()
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    structs_large = load_structures(cfg.large_file)
    extractor = FingerprintExtractor(cfg.rcut_list, cfg.center_elements)
    universe_sig = _universe_signature(cfg)
    universe_cache = out_dir / cfg.universe_cache_file

    # ── Root-cause fix: load/save frozen fp_to_eff mapping ───────────────────
    # The path is resolved relative to output_dir.  When re-running on a
    # selected subset, set output_dir to the SAME directory as the original
    # run so that this file is found automatically.
    frozen_path = out_dir / cfg.freeze_fp_to_eff_file
    frozen_fp_to_eff: Optional[Dict[str, str]] = None
    if cfg.freeze_fp_to_eff_enabled:
        frozen_fp_to_eff = load_frozen_fp_to_eff(frozen_path, cfg)

    universe = None
    if not cfg.force_rebuild_universe:
        universe = load_universe_cache(universe_cache, universe_sig)
        if universe is not None: print(f"[UniverseCache] Loaded: {universe_cache}")

    # Always recover the exact BondFilter result before reporting/export.
    # If a complete bond_filter checkpoint exists, this is loaded immediately;
    # otherwise the filter is re-run.  This avoids treating fingerprint-empty
    # structures from a cached universe as "nonphysical" by mistake.
    physical_idx: Set[int]
    if cfg.bond_filter_enabled:
        bf_sig = _make_signature(
            "compact_bond_filter_v1",
            {
                "large_file": _file_fingerprint(cfg.large_file),
                "bond_filter_coeff": cfg.bond_filter_coeff,
            },
        )
        bf_ckpt = out_dir / cfg.checkpoint_dir / "bond_filter.pkl" if cfg.checkpoint_enabled else None
        physical_idx = BondFilter(cfg.bond_filter_coeff).filter_indices(
            structs_large,
            bf_ckpt,
            bf_sig,
            cfg.resume_from_checkpoint,
            cfg.checkpoint_interval,
        )
    else:
        physical_idx = set(range(len(structs_large)))

    write_nonphysical_xyz_after_bond_filter(cfg, structs_large, physical_idx, out_dir)

    if universe is None:
        universe = TopologyUniverse()
        u_ckpt = out_dir / cfg.checkpoint_dir / "universe_build.pkl" if cfg.checkpoint_enabled else None
        universe.build(structs_large, extractor, physical_idx=physical_idx,
                       geom_max_samples=cfg.geom_max_samples, collect_all_examples=cfg.export_cluster_env_cifs,
                       checkpoint_path=u_ckpt, checkpoint_signature=universe_sig,
                       resume=cfg.resume_from_checkpoint, interval=cfg.checkpoint_interval)
        # Pass frozen mapping if available; otherwise run Ward clustering and
        # save the resulting mapping for future re-runs.
        universe.compress(cfg, frozen_fp_to_eff=frozen_fp_to_eff)
        if cfg.freeze_fp_to_eff_enabled and frozen_fp_to_eff is None:
            # First run: save the freshly computed mapping as the reference.
            save_frozen_fp_to_eff(frozen_path, cfg, universe.fp_to_eff)
        save_universe_cache(universe_cache, universe_sig, universe)
        _maybe_remove_checkpoint(u_ckpt, cfg.checkpoint_keep_after_success)
    else:
        # Universe was loaded from cache.  If the frozen mapping file does not
        # exist yet (e.g. cache was created before this feature was added),
        # save the mapping from the cached universe so that subsequent re-runs
        # on subsets can use it.
        if cfg.freeze_fp_to_eff_enabled and not frozen_path.exists():
            save_frozen_fp_to_eff(frozen_path, cfg, universe.fp_to_eff)
    export_cluster_env_cifs(cfg, structs_large, universe, out_dir)
    PublicationPlotter.plot_universe_summary(universe, out_dir, cfg)

    if not cfg.probe_file:
        if not cfg.initial_train_enabled:
            print("[STOP] cfg.probe_file is empty and cfg.initial_train_enabled=False.")
            write_report(cfg, out_dir, len(structs_large), 0, physical_idx, universe, None, [], set(), None)
            return
        initial_sig = _initial_train_signature(cfg, universe_sig)
        initial_indices, initial_manifest, initial_summary = run_initial_training_selection(cfg, universe, initial_sig, out_dir)
        initial_xyz = out_dir / cfg.initial_train_output_xyz
        if initial_indices:
            if cfg.preserve_input_xyz_format: write_selected_xyz_from_source(cfg.large_file, initial_indices, initial_xyz)
            else: write(str(initial_xyz), [structs_large[i] for i in initial_indices])
        write_csv(out_dir / cfg.initial_train_manifest_csv, initial_training_manifest_rows(initial_indices, initial_manifest, universe))
        short = {k: v for k, v in initial_summary.items() if k != "manifest"}
        (out_dir / cfg.initial_train_summary_json).write_text(json.dumps(_jsonable(short), indent=2, ensure_ascii=False), encoding="utf-8")
        write_csv(out_dir / cfg.initial_train_report_csv, [{"key": k, "value": v} for k, v in short.items()], fieldnames=["key", "value"])
        PublicationPlotter.plot_initial_training_convergence(initial_manifest, out_dir, cfg)
        PublicationPlotter.plot_topology_state(universe, out_dir, cfg, initial_indices, stage="initial_training", before_eff=set())
        write_report(cfg, out_dir, len(structs_large), 0, physical_idx, universe, None, [], set(), None, initial_train_summary=initial_summary)
        print("\n[DONE] Single-set initial training outputs:")
        print(f"  Initial training set : {initial_xyz}")
        print(f"  Manifest             : {out_dir / cfg.initial_train_manifest_csv}")
        print(f"  Summary              : {out_dir / cfg.initial_train_summary_json}")
        print(f"  Report               : {out_dir / cfg.report_file}")
        if cfg.publication_plot_enabled: print(f"  Figures              : {out_dir / cfg.publication_plot_dir}")
        return

    structs_probe = load_structures(cfg.probe_file)
    probe_sig = _probe_signature(cfg, universe_sig)
    probe_data = extract_probe_data(cfg, structs_probe, extractor, universe, probe_sig, out_dir)
    phase2_sig = _phase2_signature(cfg, universe_sig, probe_sig)
    phase2_indices, phase2_manifest, covered_gap = run_phase2_gap_supplement(cfg, universe, probe_data, phase2_sig, out_dir)
    phase2_xyz = out_dir / cfg.phase2_output_xyz
    if phase2_indices:
        if cfg.preserve_input_xyz_format: write_selected_xyz_from_source(cfg.large_file, phase2_indices, phase2_xyz)
        else: write(str(phase2_xyz), [structs_large[i] for i in phase2_indices])
    write_csv(out_dir / cfg.phase2_manifest_csv, phase2_manifest_rows(phase2_indices, phase2_manifest, universe))
    PublicationPlotter.plot_phase2_convergence(phase2_manifest, out_dir, cfg)
    PublicationPlotter.plot_topology_state(universe, out_dir, cfg, phase2_indices, stage="phase2", before_eff=probe_data.covered_eff)
    PublicationPlotter.plot_gap_supplement_panels(universe, out_dir, cfg, phase2_indices, before_eff=probe_data.covered_eff, stage="phase2")
    joint_items: List[Tuple[str, int]] = []; joint_summary = None
    if cfg.probe_phase2_joint_min_cover_enabled:
        joint_items, joint_summary = run_probe_phase2_joint_cover(cfg, probe_data, universe, phase2_indices, out_dir)
        write_mixed_xyz(cfg.probe_file, cfg.large_file, joint_items, out_dir / cfg.probe_phase2_joint_output_xyz, preserve_text=cfg.preserve_input_xyz_format)
        write_csv(out_dir / cfg.probe_phase2_joint_manifest_csv, joint_manifest_rows(joint_items, probe_data, universe))
        if joint_summary is not None:
            short2 = {k: v for k, v in joint_summary.items() if k != "manifest"}
            (out_dir / cfg.probe_phase2_joint_summary_json).write_text(json.dumps(_jsonable(short2), indent=2, ensure_ascii=False), encoding="utf-8")
            write_csv(out_dir / cfg.probe_phase2_joint_report_csv, [{"key": k, "value": v} for k, v in short2.items()], fieldnames=["key", "value"])
            PublicationPlotter.plot_joint_summary(joint_summary, out_dir, cfg)
    write_report(cfg, out_dir, len(structs_large), len(structs_probe), physical_idx, universe,
                 probe_data, phase2_indices, covered_gap, joint_summary)
    print("\n[DONE] Main outputs:")
    print(f"  Phase2 supplement        : {out_dir / cfg.phase2_output_xyz}")
    if cfg.probe_phase2_joint_min_cover_enabled: print(f"  Final probe+Phase2 joint : {out_dir / cfg.probe_phase2_joint_output_xyz}")
    print(f"  Report                   : {out_dir / cfg.report_file}")
    if cfg.publication_plot_enabled: print(f"  Figures                  : {out_dir / cfg.publication_plot_dir}")


if __name__ == "__main__":
    cfg = Config(
        large_file="initial_remaining.xyz",
        probe_file="",
        output_dir="topo_sampler_out-full-stack-initial",

        use_geom_cluster=True,
        geom_cluster_backend="ward",
        strict_geom_cluster_backend=True,
        rcut_list=[2.7, 3.0],
        center_elements=("Fe", "C"),
        bond_tol=0.05,
        angle_tol=5.0,
        bond_filter_coeff=0.8,
        write_nonphysical_xyz=True,
        nonphysical_xyz_file="nonphysical_bond_filtered.xyz",
        nonphysical_manifest_csv="nonphysical_bond_filtered_manifest.csv",

        # ── Route A: Jaccard pre-dedup ────────────────────────────────────
        jaccard_dedup_enabled=True,
        jaccard_dedup_threshold=0.85,
        jaccard_dedup_threshold_common=0.70,
        jaccard_dedup_size_band_frac=0.25,

        # ── Route C: Two-tier greedy cover ────────────────────────────────
        two_tier_cover_enabled=True,
        rare_env_freq_threshold=5,
        common_env_coverage_pct=85.0,

        # ── Route D: Post-cover coverage-aware Jaccard dedup ─────────────
        post_cover_jaccard_dedup_enabled=True,
        post_cover_jaccard_threshold=0.85,
        post_cover_jaccard_size_band_frac=0.25,

        # ── Frozen fp_to_eff (ROOT-CAUSE FIX) ────────────────────────────
        # First run:  Ward clustering runs normally; fp_to_eff is saved to
        #             output_dir / freeze_fp_to_eff_file.
        # Re-run on selected subset:  set output_dir to the SAME directory as
        #             the original run, and the frozen mapping is loaded and
        #             applied — Ward re-clustering is skipped entirely.
        #             This makes the effective-type space a fixed reference
        #             frame, eliminating spurious redundancy in re-runs.
        # To force re-clustering (e.g. after changing bond_tol / angle_tol):
        #             delete freeze_fp_to_eff_file or set
        #             freeze_fp_to_eff_enabled=False.
        freeze_fp_to_eff_enabled=True,
        freeze_fp_to_eff_file="fp_to_eff_frozen.pkl",

        initial_train_enabled=True,
        initial_train_budget=0,
        initial_train_target_coverage_pct=100.0,

        export_cluster_env_cifs=False,
        checkpoint_enabled=True,
        resume_from_checkpoint=True,
        publication_plot_enabled=True,
    )
    main(cfg)
