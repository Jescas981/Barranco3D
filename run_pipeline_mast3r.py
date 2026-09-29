import os
import shutil
import traceback
from itertools import combinations
from pathlib import Path

import h5py

from hloc import (
    extract_features,
    pairs_from_retrieval,
    reconstruction,
    match_dense,
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
NUM_RETRIEVALS = 50

GLOBAL_FEATURE = "global-feats-netvlad"

# MASt3R configuration:
#
# "mast3r"    -> standard MASt3R
# "aerialmd"  -> aerial-finetuned MASt3R
#
MAST3R_WEIGHTS = "aerialmd"

MAST3R_FEATURES = "features-mast3r"
MAST3R_MATCHES = "matches-mast3r"


# ============================================================
# IMAGE / H5 UTILITIES
# ============================================================

IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
}


def _count_images(images_dir):
    count = 0

    for _root, _dirs, files in os.walk(
        images_dir,
        followlinks=True,
    ):
        for name in files:
            if Path(name).suffix.lower() in IMG_EXTENSIONS:
                count += 1

    return count


def _count_pairs(pairs_path):
    with open(pairs_path) as f:
        return sum(
            1
            for line in f
            if line.strip()
        )


def _h5_leaf_count(h5_path, leaf_name):
    count = 0

    def _visit(name, obj):
        nonlocal count

        if (
            isinstance(obj, h5py.Dataset)
            and name.rsplit("/", 1)[-1] == leaf_name
        ):
            count += 1

    try:
        with h5py.File(h5_path, "r") as f:
            f.visititems(_visit)

    except Exception:
        return 0

    return count


def _features_file_is_complete(
    h5_path,
    images_dir,
    leaf_name,
):
    if not h5_path.exists():
        return False

    expected = _count_images(images_dir)
    actual = _h5_leaf_count(
        h5_path,
        leaf_name,
    )

    return (
        expected > 0
        and actual >= expected
    )


def _matches_file_is_complete(
    matches_path,
    pairs_path,
):
    if not matches_path.exists():
        return False

    if not pairs_path.exists():
        return False

    expected = _count_pairs(pairs_path)

    actual = _h5_leaf_count(
        matches_path,
        "matches0",
    )

    return (
        expected > 0
        and actual >= expected
    )


def _dense_matches_are_complete(
    matches_path,
    pairs_path,
):
    """
    match_dense initially creates:

        pair/
            keypoints0
            keypoints1
            scores

    After aggregate_matches(), it creates:

        pair/
            matches0
            matching_scores0

    Reconstruction requires matches0.
    """

    return _matches_file_is_complete(
        matches_path,
        pairs_path,
    )


def _get_image_names_from_features(
    h5_path,
):
    names = []

    def _visit(name, obj):

        if (
            isinstance(obj, h5py.Dataset)
            and name.rsplit("/", 1)[-1]
            == "keypoints"
        ):
            names.append(
                name.rsplit("/", 1)[0]
            )

    with h5py.File(h5_path, "r") as f:
        f.visititems(_visit)

    return sorted(names)


# ============================================================
# EXPERIMENTS
# ============================================================

def get_experiments():

    experiments = list(
        # combinations(
        #     PLATFORMS,
        #     COMBINATION_SIZE,
        # )
    )

    all_platforms = tuple(PLATFORMS)

    if all_platforms not in experiments:
        experiments.append(
            all_platforms
        )

    return experiments


def experiment_name(platforms):
    return "_".join(platforms)


def get_paths(platforms):

    root = (
        EXPERIMENTS_DIR
        / experiment_name(platforms)
    )

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

        p["images"].mkdir(
            parents=True,
            exist_ok=True,
        )

        for platform in platforms:

            source = (
                DATASET_DIR
                / platform
            ).resolve()

            target = (
                p["images"]
                / platform
            )

            if not source.exists():
                raise FileNotFoundError(
                    f"Platform directory not found: "
                    f"{source}"
                )

            if not target.exists():

                target.symlink_to(
                    source,
                    target_is_directory=True,
                )

        print(
            f"[READY] {p['name']}"
        )


# ============================================================
# STEP 1
# NETVLAD
# ============================================================

def extract_global_features(
    experiments,
):

    print("\n" + "=" * 70)
    print(
        "STEP 1 — GLOBAL FEATURE EXTRACTION"
    )
    print("=" * 70)

    conf = extract_features.confs[
        "netvlad"
    ]

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

        print(
            f"\n[{p['name']}]"
        )

        if output.exists():

            if _features_file_is_complete(
                output,
                p["images"],
                "global_descriptor",
            ):

                print(
                    "  [SKIP] NetVLAD already exists"
                )

                continue

            print(
                "  [CLEANUP] Removing incomplete "
                "NetVLAD file"
            )

            output.unlink()

        print(
            "  [RUN] Extracting NetVLAD..."
        )

        extract_features.main(
            conf,
            p["images"],
            p["features"],
        )


# ============================================================
# STEP 2
# RETRIEVAL
# ============================================================

def generate_retrieval_pairs(
    experiments,
):

    print("\n" + "=" * 70)
    print(
        "STEP 2 — IMAGE RETRIEVAL"
    )
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

        print(
            f"\n[{p['name']}]"
        )

        if pairs.exists():

            print(
                "  [SKIP] Retrieval pairs already exist"
            )

            continue

        if not retrieval.exists():

            print(
                f"  [ERROR] NetVLAD not found:\n"
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


# ============================================================
# STEP 3
# MASt3R DENSE MATCHING
# ============================================================

def match_dense_mast3r(
    experiments,
):

    print("\n" + "=" * 70)
    print(
        "STEP 3 — MASt3R DENSE MATCHING"
    )
    print("=" * 70)

    print(
        f"\nMASt3R weights: {MAST3R_WEIGHTS}"
    )

    for platforms in experiments:

        p = get_paths(platforms)

        pairs = (
            p["pairs"]
            / "pairs-netvlad.txt"
        )

        p["features"].mkdir(
            parents=True,
            exist_ok=True,
        )

        features = (
            p["features"]
            / f"{MAST3R_FEATURES}.h5"
        )

        matches = (
            p["features"]
            / f"{MAST3R_MATCHES}.h5"
        )

        print(
            f"\n[{p['name']}]"
        )

        if not pairs.exists():

            print(
                "  [ERROR] Retrieval pairs not found:"
            )
            print(
                f"          {pairs}"
            )

            continue

        # ----------------------------------------------------
        # Check if the final dense matching output is complete
        # ----------------------------------------------------

        if (
            features.exists()
            and matches.exists()
            and _features_file_is_complete(
                features,
                p["images"],
                "keypoints",
            )
            and _dense_matches_are_complete(
                matches,
                pairs,
            )
        ):

            print(
                "  [SKIP] MASt3R matching already complete"
            )

            continue

        # ----------------------------------------------------
        # Remove incomplete files
        # ----------------------------------------------------

        if features.exists():

            print(
                "  [CLEANUP] Removing incomplete "
                "MASt3R features"
            )

            features.unlink()

        if matches.exists():

            print(
                "  [CLEANUP] Removing incomplete "
                "MASt3R matches"
            )

            matches.unlink()

        # ----------------------------------------------------
        # Run MASt3R
        # ----------------------------------------------------

        print(
            "  [RUN] MASt3R dense matching..."
        )

        try:

            match_dense.main(
                match_dense.confs["mast3r"]
                | {
                    "model": {
                        "name": "mast3r",
                        "weights": MAST3R_WEIGHTS,
                    }
                },
                pairs,
                p["images"],
                export_dir=p["features"],
                matches=matches,
                features=features,
            )

        except Exception:

            print(
                "  [ERROR] MASt3R matching failed:"
            )

            traceback.print_exc()

            continue

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if not features.exists():

            print(
                "  [ERROR] MASt3R feature file "
                "was not created"
            )

            continue

        if not matches.exists():

            print(
                "  [ERROR] MASt3R match file "
                "was not created"
            )

            continue

        print(
            "  [DONE] MASt3R matching"
        )

        print(
            f"         Features: {features}"
        )

        print(
            f"         Matches:  {matches}"
        )

        print(
            f"         Keypoint groups: "
            f"{_h5_leaf_count(features, 'keypoints')}"
        )

        print(
            f"         Match groups: "
            f"{_h5_leaf_count(matches, 'matches0')}"
        )


# ============================================================
# STEP 4
# COLMAP SFM
# ============================================================

def _has_valid_model(
    sparse_dir,
):

    if not sparse_dir.exists():
        return False

    candidates = [
        sparse_dir
    ] + [
        d
        for d in sparse_dir.iterdir()
        if d.is_dir()
    ]

    for d in candidates:

        if (
            (d / "cameras.bin").exists()
            or
            (d / "cameras.txt").exists()
        ):

            return True

    return False


def _clean_stale_sfm_state(
    sfm_dir,
):

    database = (
        sfm_dir
        / "database.db"
    )

    if database.exists():

        print(
            "  [CLEANUP] Removing stale database.db"
        )

        database.unlink()

    sparse_dir = (
        sfm_dir
        / "sparse"
    )

    if sparse_dir.exists():

        print(
            "  [CLEANUP] Removing incomplete sparse/"
        )

        shutil.rmtree(
            sparse_dir
        )


def run_sfm_reconstruction(
    experiments,
):

    print("\n" + "=" * 70)
    print(
        "STEP 4 — COLMAP SFM RECONSTRUCTION"
    )
    print("=" * 70)

    for platforms in experiments:

        p = get_paths(platforms)

        pairs = (
            p["pairs"]
            / "pairs-netvlad.txt"
        )

        features = (
            p["features"]
            / f"{MAST3R_FEATURES}.h5"
        )

        matches = (
            p["features"]
            / f"{MAST3R_MATCHES}.h5"
        )

        sfm_dir = p["sfm"]
        sparse_dir = (
            sfm_dir / "sparse"
        )

        print(
            f"\n[{p['name']}]"
        )

        # ----------------------------------------------------
        # Existing model
        # ----------------------------------------------------

        if _has_valid_model(
            sparse_dir
        ):

            print(
                "  [SKIP] SfM reconstruction already exists"
            )

            continue

        # ----------------------------------------------------
        # Validate inputs
        # ----------------------------------------------------

        if not pairs.exists():

            print(
                f"  [ERROR] Pairs not found:\n"
                f"          {pairs}"
            )

            continue

        # if not features.exists():

        #     print(
        #         f"  [ERROR] MASt3R features not found:\n"
        #         f"          {features}"
        #     )

        #     continue

        # if not matches.exists():

        #     print(
        #         f"  [ERROR] MASt3R matches not found:\n"
        #         f"          {matches}"
        #     )

        #     continue

        # if not _matches_file_is_complete(
        #     matches,
        #     pairs,
        # ):

        #     print(
        #         "  [ERROR] MASt3R matches are incomplete"
        #     )

        #     print(
        #         f"          expected pairs: "
        #         f"{_count_pairs(pairs)}"
        #     )

        #     print(
        #         f"          matches0: "
        #         f"{_h5_leaf_count(matches, 'matches0')}"
        #     )

        #     continue

        # ----------------------------------------------------
        # Prepare SfM
        # ----------------------------------------------------

        sfm_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        _clean_stale_sfm_state(
            sfm_dir
        )

        # ----------------------------------------------------
        # Work around symlinked directories
        # ----------------------------------------------------

        image_names = (
            _get_image_names_from_features(
                features
            )
        )

        print(
            f"  [INFO] Images: "
            f"{len(image_names)}"
        )

        # ----------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------

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
                image_list=image_names,
                verbose=True,
            )

        except Exception:

            print(
                "  [ERROR] SfM reconstruction failed:"
            )

            traceback.print_exc()

            _clean_stale_sfm_state(
                sfm_dir
            )

            continue

        if model is None:

            print(
                "  [WARNING] No reconstruction model"
            )

            _clean_stale_sfm_state(
                sfm_dir
            )

            continue

        print(
            "  [DONE] SfM reconstruction"
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
    print(
        "Barranco3D MASt3R HLoc Pipeline"
    )
    print("=" * 70)

    print(
        f"\nMASt3R weights: "
        f"{MAST3R_WEIGHTS}"
    )

    print("\nExperiments:")

    for i, platforms in enumerate(
        experiments,
        start=1,
    ):

        print(
            f"  {i}. "
            f"{' + '.join(platforms)}"
        )

    create_experiments(
        experiments
    )

    # NetVLAD
    extract_global_features(
        experiments
    )

    # Retrieval
    generate_retrieval_pairs(
        experiments
    )

    # MASt3R
    match_dense_mast3r(
        experiments
    )

    # COLMAP
    run_sfm_reconstruction(
        experiments
    )

    print("\n" + "=" * 70)
    print(
        "MASt3R HLoc pipeline completed"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()