from itertools import combinations
from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("frames/Barranco3D")
EXPERIMENTS_DIR = Path("experiments/hloc")

PLATFORMS = [
    "Car",
    "Drone",
    "Mapilary",
    "Pedestrian",
]

COMBINATION_SIZE = 3
NUM_RETRIEVALS = 20


# ============================================================
# EXPERIMENTS
# ============================================================

def get_experiments():

    return list(
        combinations(
            PLATFORMS,
            COMBINATION_SIZE,
        )
    )


def experiment_name(platforms):

    return "_".join(platforms)


# ============================================================
# CREATE EXPERIMENT
# ============================================================

def create_experiments(experiments):

    for platforms in experiments:

        name = experiment_name(platforms)

        experiment_dir = (
            EXPERIMENTS_DIR / name
        )

        image_dir = (
            experiment_dir / "images"
        )

        image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for platform in platforms:

            source = (
                DATASET_DIR / platform
            ).resolve()

            if not source.exists():

                raise FileNotFoundError(
                    f"Platform directory does not exist: "
                    f"{source}"
                )

            target = (
                image_dir / platform
            )

            if target.exists():
                continue

            target.symlink_to(
                source,
                target_is_directory=True,
            )

        print(
            f"[READY] {name}"
        )


# ============================================================
# STEP 1
# GLOBAL FEATURES — NETVLAD
# ============================================================

def extract_global_features(experiments):

    print()
    print("=" * 70)
    print("STEP 1 — GLOBAL FEATURE EXTRACTION")
    print("=" * 70)

    conf = extract_features.confs["netvlad"]

    for platforms in experiments:

        name = experiment_name(platforms)

        experiment_dir = (
            EXPERIMENTS_DIR / name
        )

        image_dir = (
            experiment_dir / "images"
        )

        feature_dir = (
            experiment_dir / "features"
        )

        feature_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            feature_dir
            / "global-feats-netvlad.h5"
        )

        print()
        print(f"[{name}]")

        if output_file.exists():

            print(
                f"  [SKIP] NetVLAD already exists"
            )

            continue

        print(
            f"  [RUN] Extracting NetVLAD..."
        )

        extract_features.main(
            conf,
            image_dir,
            feature_dir,
        )

        if output_file.exists():

            print(
                f"  [DONE] {output_file}"
            )

        else:

            print(
                f"  [WARNING] Expected output "
                f"not found: {output_file}"
            )


# ============================================================
# STEP 2
# RETRIEVAL PAIRS
# ============================================================

def generate_retrieval_pairs(experiments):

    print()
    print("=" * 70)
    print("STEP 2 — IMAGE RETRIEVAL")
    print("=" * 70)

    for platforms in experiments:

        name = experiment_name(platforms)

        experiment_dir = (
            EXPERIMENTS_DIR / name
        )

        feature_dir = (
            experiment_dir / "features"
        )

        pairs_dir = (
            experiment_dir / "pairs"
        )

        pairs_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        retrieval_path = (
            feature_dir
            / "global-feats-netvlad.h5"
        )

        pairs_path = (
            pairs_dir
            / "pairs-netvlad.txt"
        )

        print()
        print(f"[{name}]")

        if pairs_path.exists():

            print(
                f"  [SKIP] Retrieval pairs already exist"
            )

            continue

        if not retrieval_path.exists():

            print(
                f"  [ERROR] NetVLAD features not found"
            )

            print(
                f"          {retrieval_path}"
            )

            print(
                f"  Run step 1 first."
            )

            continue

        print(
            f"  [RUN] Generating retrieval pairs..."
        )

        pairs_from_retrieval.main(
            retrieval_path,
            pairs_path,
            num_matched=NUM_RETRIEVALS,
        )

        if pairs_path.exists():

            print(
                f"  [DONE] {pairs_path}"
            )


# ============================================================
# STEP 3
# LOCAL FEATURES — SUPERPOINT
# ============================================================

def extract_local_features(experiments):

    print()
    print("=" * 70)
    print("STEP 3 — LOCAL FEATURE EXTRACTION")
    print("=" * 70)

    conf = extract_features.confs[
        "superpoint_aachen"
    ]

    for platforms in experiments:

        name = experiment_name(platforms)

        experiment_dir = (
            EXPERIMENTS_DIR / name
        )

        image_dir = (
            experiment_dir / "images"
        )

        feature_dir = (
            experiment_dir / "features"
        )

        feature_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            feature_dir
            / "feats-superpoint-n4096-r1024.h5"
        )

        print()
        print(f"[{name}]")

        if output_file.exists():

            print(
                f"  [SKIP] SuperPoint already exists"
            )

            continue

        print(
            f"  [RUN] Extracting SuperPoint..."
        )

        extract_features.main(
            conf,
            image_dir,
            feature_dir,
        )

        if output_file.exists():

            print(
                f"  [DONE] {output_file}"
            )

        else:

            print(
                f"  [WARNING] Expected output "
                f"not found: {output_file}"
            )


# ============================================================
# STEP 4
# FEATURE MATCHING — SUPERGLUE
# ============================================================

def match_features_for_experiments(experiments):

    print()
    print("=" * 70)
    print("STEP 4 — FEATURE MATCHING")
    print("=" * 70)

    conf = match_features.confs[
        "superglue"
    ]

    for platforms in experiments:

        name = experiment_name(platforms)

        experiment_dir = (
            EXPERIMENTS_DIR / name
        )

        pairs_dir = (
            experiment_dir / "pairs"
        )

        feature_dir = (
            experiment_dir / "features"
        )

        pairs_path = (
            pairs_dir
            / "pairs-netvlad.txt"
        )

        print()
        print(f"[{name}]")

        if not pairs_path.exists():

            print(
                f"  [ERROR] Retrieval pairs not found"
            )

            print(
                f"          {pairs_path}"
            )

            print(
                f"  Run step 2 first."
            )

            continue

        # ----------------------------------------------------
        # HLoc creates the matches file from the pairs.
        # The exact filename depends on the matcher
        # configuration.
        # ----------------------------------------------------

        expected_matches = (
            experiment_dir
            / "matches"
            / "matches-superglue.h5"
        )

        if expected_matches.exists():

            print(
                f"  [SKIP] SuperGlue matches already exist"
            )

            continue

        print(
            f"  [RUN] Matching with SuperGlue..."
        )

        matches_path = match_features.main(
            conf,
            pairs_path,
            feature_dir,
        )

        print(
            f"  [DONE] {matches_path}"
        )


# ============================================================
# RUN SELECTED STEP
# ============================================================

def run_step(step, experiments):

    if step == 1:

        extract_global_features(
            experiments
        )

    elif step == 2:

        generate_retrieval_pairs(
            experiments
        )

    elif step == 3:

        extract_local_features(
            experiments
        )

    elif step == 4:

        match_features_for_experiments(
            experiments
        )

    else:

        raise ValueError(
            f"Invalid step: {step}"
        )


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

    print()
    print("Experiments:")

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

    # Run all steps sequentially.
    #
    # Each step checks whether its output
    # already exists before running.

    for step in range(1, 5):

        run_step(
            step,
            experiments,
        )

    print()
    print("=" * 70)
    print("HLoc pipeline completed")
    print("=" * 70)


if __name__ == "__main__":
    main()