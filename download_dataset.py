from pathlib import Path
import argparse

import cv2
from huggingface_hub import snapshot_download


REPO_ID = "Jescas981/Barranco3D"

DATASET_DIR = Path("data/Barranco3D")
OUTPUT_DIR = Path("frames/Barranco3D")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and prepare Barranco3D by extracting frames from videos."
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frame extraction rate in frames per second. Default: 2.",
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Local dataset directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where extracted PNG frames will be stored.",
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Hugging Face download and use the existing dataset.",
    )

    return parser.parse_args()


def download_dataset(dataset_dir: Path):
    """
    Download the Barranco3D dataset from Hugging Face.
    """

    print("=" * 60)
    print("Downloading Barranco3D dataset")
    print("=" * 60)

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Repository : {REPO_ID}")
    print(f"Local path : {dataset_dir.resolve()}")
    print()

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(dataset_dir),
    )

    print()
    print("Download complete.")
    print()


def find_videos(dataset_dir: Path):
    """
    Find all MP4 videos case-insensitively while excluding
    the output/prepared directories.
    """

    videos = []

    for path in dataset_dir.rglob("*"):

        if not path.is_file():
            continue

        # Avoid processing extracted/generated data
        if "prepared" in path.parts:
            continue

        if "frames" in path.parts:
            continue

        # Case-insensitive extension check
        if path.suffix.lower() == ".mp4":
            videos.append(path)

    return sorted(videos)


def get_platform(
    video_path: Path,
    dataset_dir: Path,
):
    """
    The first directory relative to the dataset root
    is considered the platform.

    Example:

        Pedestrian/
        Car/
        Drone/
    """

    relative = video_path.relative_to(dataset_dir)

    if len(relative.parts) == 1:
        return "Unknown"

    return relative.parts[0]


def create_video_prefix(
    video_path: Path,
    dataset_dir: Path,
):
    """
    Generate a prefix that identifies the video.

    Example:

        Car/session_01/video_01.mp4

    becomes:

        session_01_video_01
    """

    relative = video_path.relative_to(dataset_dir)

    parts = list(relative.parts)

    # Remove extension
    parts[-1] = Path(parts[-1]).stem

    # Remove platform
    parts = parts[1:]

    return "_".join(parts)


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float,
    prefix: str,
):
    """
    Extract PNG frames from a video at the requested FPS.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Could not open: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if video_fps <= 0:
        print(f"[ERROR] Invalid FPS: {video_path}")
        cap.release()
        return 0

    duration = total_frames / video_fps

    frame_interval = 1.0 / fps

    current_time = 0.0
    frame_id = 0
    extracted = 0

    while current_time < duration:

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            current_time * 1000,
        )

        success, frame = cap.read()

        if not success:
            break

        output_path = (
            output_dir
            / f"{prefix}_{frame_id:06d}.png"
        )

        cv2.imwrite(
            str(output_path),
            frame,
        )

        extracted += 1
        frame_id += 1

        current_time += frame_interval

    cap.release()

    return extracted


def prepare_dataset(
    dataset_dir: Path,
    output_dir: Path,
    fps: float,
):
    print("=" * 60)
    print("Preparing Barranco3D dataset")
    print("=" * 60)

    print(
        f"Dataset directory : "
        f"{dataset_dir.resolve()}"
    )

    print(
        f"Output directory  : "
        f"{output_dir.resolve()}"
    )

    print(f"Extraction FPS    : {fps}")
    print()

    videos = find_videos(dataset_dir)

    print(f"Videos found: {len(videos)}")
    print()

    if not videos:
        print("No MP4 files found.")
        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_frames = 0

    for video_path in videos:

        platform = get_platform(
            video_path,
            dataset_dir,
        )

        prefix = create_video_prefix(
            video_path,
            dataset_dir,
        )

        relative = video_path.relative_to(
            dataset_dir
        )

        # Preserve the directory structure.
        #
        # Example:
        #
        # Car/session/video.mp4
        #
        # becomes:
        #
        # frames/Barranco3D/Car/session/

        output_subdir = relative.parent

        frame_output_dir = (
            output_dir / output_subdir
        )

        frame_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("-" * 60)
        print(f"Platform : {platform}")
        print(f"Video    : {video_path}")
        print(f"Output   : {frame_output_dir}")
        print(f"Prefix   : {prefix}")

        extracted = extract_frames(
            video_path=video_path,
            output_dir=frame_output_dir,
            fps=fps,
            prefix=prefix,
        )

        total_frames += extracted

        print(f"Frames   : {extracted}")

    print()
    print("=" * 60)
    print("Preparation complete")
    print("=" * 60)

    print(
        f"Videos processed : {len(videos)}"
    )

    print(
        f"Frames extracted : {total_frames}"
    )

    print(
        f"Output directory : "
        f"{output_dir.resolve()}"
    )


def main():

    args = parse_args()

    # --------------------------------------------------
    # 1. Download dataset
    # --------------------------------------------------

    if not args.skip_download:
        download_dataset(
            args.dataset_dir
        )
    else:
        print(
            "Skipping Hugging Face download."
        )

    # --------------------------------------------------
    # 2. Extract frames
    # --------------------------------------------------

    prepare_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
