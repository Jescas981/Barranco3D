import shutil
import traceback
from itertools import combinations
from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    reconstruction,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("frames/Barranco3D")
EXPERIMENTS_DIR = Path("experiments/hloc")

PLATFORMS = ["Car", "Drone", "Mapilary", "Pedestrian"]
COMBINATION_SIZE = 3
NUM_RETRIEVALS = 50

GLOBAL_FEATURE = "global-feats-netvlad"
LOCAL_FEATURE = "feats-superpoint-n4096-r1024"


# ============================================================
# EXPERIMENTS
# ============================================================

def get_experiments():
    # All size-3 combinations of platforms...
    experiments = list(combinations(PLATFORMS, COMBINATION_SIZE))

    # ...plus the single combination using all platforms together.
    all_platforms = tuple(PLATFORMS)
    if all_platforms not in experiments:
        experiments.append(all_platforms)

    return experiments


def experiment_name(platforms):
    return "_".join(platforms)


def get_paths(platforms):
    root = EXPERIMENTS_DIR / experiment_name(platforms)

    return {
        "name": experiment_name(platforms),
        "root": root,
        "images": root / "images",
        "features": root / "features",
        "pairs": root / "pairs",
        "sfm": root / "sfm",
    }


# ============================================================
# CREATE EXPERIMENTS
# ============================================================

def create_experiments(experiments):
    for platforms in experiments:
        p = get_paths(platforms)
        p["images"].mkdir(parents=True, exist_ok=True)

        for platform in platforms:
            source = (DATASET_DIR / platform).resolve()
            target = p["images"] / platform

            if not source.exists():
                raise FileNotFoundError(
                    f"Platform directory not found: {source}"
                )

            if not target.exists():
                target.symlink_to(
                    source,
                    target_is_directory=True,
                )

        print(f"[READY] {p['name']}")


# ============================================================
# STEP 1 — NETVLAD
# ============================================================

def extract_global_features(experiments):
    print("\n" + "=" * 70)
    print("STEP 1 — GLOBAL FEATURE EXTRACTION")
    print("=" * 70)

    conf = extract_features.confs["netvlad"]

    for platforms in experiments:
        p = get_paths(platforms)

        p["features"].mkdir(
            parents=True,
            exist_ok=True,
        )

        output = (
            p["features"]
            / f"{GLOBAL_FEATURE}.h5"
        )

        print(f"\n[{p['name']}]")

        if output.exists():
            print("  [SKIP] NetVLAD already exists")
            continue

        print("  [RUN] Extracting NetVLAD...")

        extract_features.main(
            conf,
            p["images"],
            p["features"],
        )

        if output.exists():
            print(f"  [DONE] {output}")
        else:
            print(
                f"  [WARNING] Output not found: {output}"
            )


# ============================================================
# STEP 2 — RETRIEVAL PAIRS
# ============================================================

def generate_retrieval_pairs(experiments):
    print("\n" + "=" * 70)
    print("STEP 2 — IMAGE RETRIEVAL")
    print("=" * 70)

    for platforms in experiments:
        p = get_paths(platforms)

        p["pairs"].mkdir(
            parents=True,
            exist_ok=True,
        )

        retrieval = (
            p["features"]
            / f"{GLOBAL_FEATURE}.h5"
        )

        pairs = (
            p["pairs"]
            / "pairs-netvlad.txt"
        )

        print(f"\n[{p['name']}]")

        if pairs.exists():
            print(
                "  [SKIP] Retrieval pairs already exist"
            )
            continue

        if not retrieval.exists():
            print(
                f"  [ERROR] NetVLAD features not found:\n"
                f"          {retrieval}"
            )
            continue

        print(
            "  [RUN] Generating retrieval pairs..."
        )

        pairs_from_retrieval.main(
            retrieval,
            pairs,
            num_matched=NUM_RETRIEVALS,
        )

        if pairs.exists():
            print(f"  [DONE] {pairs}")
        else:
            print(
                "  [WARNING] Retrieval pairs were not created"
            )


# ============================================================
# STEP 3 — SUPERPOINT
# ============================================================

def extract_local_features(experiments):
    print("\n" + "=" * 70)
    print("STEP 3 — LOCAL FEATURE EXTRACTION")
    print("=" * 70)

    conf = extract_features.confs[
        "superpoint_aachen"
    ]

    for platforms in experiments:
        p = get_paths(platforms)

        p["features"].mkdir(
            parents=True,
            exist_ok=True,
        )

        output = (
            p["features"]
            / f"{LOCAL_FEATURE}.h5"
        )

        print(f"\n[{p['name']}]")

        if output.exists():
            print(
                "  [SKIP] SuperPoint already exists"
            )
            continue

        print(
            "  [RUN] Extracting SuperPoint..."
        )

        extract_features.main(
            conf,
            p["images"],
            p["features"],
        )

        if output.exists():
            print(f"  [DONE] {output}")
        else:
            print(
                f"  [WARNING] Output not found: {output}"
            )


# ============================================================
# STEP 4 — SUPERGLUE
# ============================================================

def match_features_for_experiments(experiments):
    print("\n" + "=" * 70)
    print("STEP 4 — FEATURE MATCHING")
    print("=" * 70)

    conf = match_features.confs[
        "superglue"
    ]

    for platforms in experiments:
        p = get_paths(platforms)

        pairs = (
            p["pairs"]
            / "pairs-netvlad.txt"
        )

        features = (
            p["features"]
            / f"{LOCAL_FEATURE}.h5"
        )

        matches = (
            p["features"]
            / f"{LOCAL_FEATURE}_"
              f"{conf['output']}_"
              f"{pairs.stem}.h5"
        )

        print(f"\n[{p['name']}]")

        if not pairs.exists():
            print(
                f"  [ERROR] Retrieval pairs not found:\n"
                f"          {pairs}"
            )
            continue

        if not features.exists():
            print(
                f"  [ERROR] SuperPoint features not found:\n"
                f"          {features}"
            )
            continue

        if matches.exists():
            print(
                "  [SKIP] SuperGlue matches already exist"
            )
            continue

        print(
            "  [RUN] Matching with SuperGlue..."
        )

        match_features.main(
            conf,
            pairs,
            features,
            matches=matches,
        )

        if matches.exists():
            print(f"  [DONE] {matches}")
        else:
            print(
                f"  [WARNING] Matches not found:\n"
                f"          {matches}"
            )


# ============================================================
# STEP 5 — COLMAP SFM
# ============================================================

def _has_valid_model(sparse_dir):
    """A model is only valid if it actually contains reconstruction files,
    not just an empty directory left behind by a crashed run."""
    if not sparse_dir.exists():
        return False

    # hloc/pycolmap either write directly into sparse/ or into sparse/0/
    candidates = [sparse_dir] + [d for d in sparse_dir.iterdir() if d.is_dir()]
    for d in candidates:
        if (d / "cameras.bin").exists() or (d / "cameras.txt").exists():
            return True
    return False


def _clean_stale_sfm_state(sfm_dir):
    """Remove leftovers from a previously failed/interrupted SfM run.

    hloc's reconstruction.main() creates sfm_dir/database.db and will
    error out (or silently reuse a broken DB) if that file already
    exists from a prior crashed attempt. Since we only reach this point
    when there is no *valid* sparse model yet, it's always safe to wipe
    the partial state and start clean.
    """
    database = sfm_dir / "database.db"
    if database.exists():
        print("  [CLEANUP] Removing stale database.db from a previous failed run")
        database.unlink()

    sparse_dir = sfm_dir / "sparse"
    if sparse_dir.exists():
        print("  [CLEANUP] Removing incomplete sparse/ directory")
        shutil.rmtree(sparse_dir)


def run_sfm_reconstruction(experiments):
    print("\n" + "=" * 70)
    print("STEP 5 — COLMAP SFM RECONSTRUCTION")
    print("=" * 70)

    conf = match_features.confs[
        "superglue"
    ]

    for platforms in experiments:
        p = get_paths(platforms)

        pairs = (
            p["pairs"]
            / "pairs-netvlad.txt"
        )

        features = (
            p["features"]
            / f"{LOCAL_FEATURE}.h5"
        )

        matches = (
            p["features"]
            / f"{LOCAL_FEATURE}_"
              f"{conf['output']}_"
              f"{pairs.stem}.h5"
        )

        sfm_dir = p["sfm"]
        sparse_dir = sfm_dir / "sparse"

        print(f"\n[{p['name']}]")

        if _has_valid_model(sparse_dir):
            print(
                "  [SKIP] SfM reconstruction already exists"
            )
            continue

        if not pairs.exists():
            print(
                f"  [ERROR] Retrieval pairs not found:\n"
                f"          {pairs}"
            )
            continue

        if not features.exists():
            print(
                f"  [ERROR] SuperPoint features not found:\n"
                f"          {features}"
            )
            continue

        if not matches.exists():
            print(
                f"  [ERROR] SuperGlue matches not found:\n"
                f"          {matches}"
            )
            continue

        sfm_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # No valid model exists yet, so any leftover database.db or
        # partial sparse/ folder here is stale wreckage from a run
        # that crashed or was killed mid-way. Clear it before retrying,
        # otherwise reconstruction.main() will fail immediately trying
        # to (re)create sfm_dir/database.db.
        _clean_stale_sfm_state(sfm_dir)

        print(
            "  [RUN] Running COLMAP SfM..."
        )

        try:
            model = reconstruction.main(
                sfm_dir,
                p["images"],
                pairs,
                features,
                matches,
                verbose=True,
            )
        except Exception:
            print(
                "  [ERROR] SfM reconstruction raised an exception:"
            )
            traceback.print_exc()
            # Don't let one bad combination kill the whole pipeline —
            # clean up so the next attempt starts fresh, then move on.
            _clean_stale_sfm_state(sfm_dir)
            continue

        if model is None:
            print(
                "  [WARNING] SfM did not return a model "
                "(likely insufficient matches/overlap for this "
                "combination of platforms)"
            )
            _clean_stale_sfm_state(sfm_dir)
            continue

        print(
            "  [DONE] SfM reconstruction"
        )

        print(
            f"         Output: {sfm_dir}"
        )

        print(
            f"         Registered images: "
            f"{model.num_reg_images()}"
        )

        print(
            f"         3D points: "
            f"{model.num_points3D()}"
        )


# ============================================================
# RUN SELECTED STEP
# ============================================================

def run_step(step, experiments):
    steps = {
        1: extract_global_features,
        2: generate_retrieval_pairs,
        3: extract_local_features,
        4: match_features_for_experiments,
        5: run_sfm_reconstruction,
    }

    if step not in steps:
        raise ValueError(
            f"Invalid step: {step}"
        )

    steps[step](experiments)


# ============================================================
# MAIN
# ============================================================

def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: "
            f"{DATASET_DIR}"
        )

    experiments = get_experiments()

    print("=" * 70)
    print("Barranco3D HLoc Pipeline")
    print("=" * 70)

    print("\nExperiments:")

    for index, platforms in enumerate(
        experiments,
        start=1,
    ):
        print(
            f"  {index}. "
            f"{' + '.join(platforms)}"
        )

    create_experiments(
        experiments
    )

    for step in range(1, 6):
        run_step(
            step,
            experiments,
        )

    print("\n" + "=" * 70)
    print("HLoc pipeline completed")
    print("=" * 70)


if __name__ == "__main__":
    main()