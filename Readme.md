# Barranco3D

Extract video frames and reconstruct scenes from combinations of camera platforms.
Run matching, SfM, and MVS separately on the hardware each stage needs.

## Main scripts

Run from the repository root, with the required dependencies installed:

```bash
# Optional: discover dataset videos and draw camera crops or masks
python src/edit_regions.py --config config.yaml

# Optional: sample flow distributions and block maps across all platforms
python src/analyze_optical_flow.py --config config.yaml --scene DavidHouse

# Extract frames from all platforms in a scene
python extract_frames.py --config config.yaml

# Stage 1: global/local features, retrieval, and matching
python src/build_bank_matching.py --config config.yaml

# Stage 2: sparse reconstruction (CPU)
python src/build_sfm_sparser.py --config config.yaml

# Stage 3: dense reconstruction (GPU)
python src/build_mvs.py --config config.yaml
```

Configure scenes, experiments, and CPU/GPU resources in [config.yaml](config.yaml).
Set video input paths in `data.datasets_root` and sampling options in
`extraction` (`strategy`, `fps`, `format`, `threads`). The extractor also accepts
`--scenes DavidHouse --fps 2` to override the YAML. Existing frame folders must
be empty or new. The editor discovers videos under `data.datasets_root` (`dataset/` here) and opens
locally in your browser. Car has independent rules for `cam0`–`cam7`; Drone and
Pedestrian each use one rule. Extraction applies
saved crops and exports masks without changing masked image pixels.
Crop rectangles support corner/edge dragging, moving, and numeric coordinates or
width/height. Masks support polygons and rectangles.

For adaptive frame selection using optical flow:

```bash
python extract_frames.py --config config.yaml --strategy optical_flow --mode-threshold-px 20
```

This compares each frame with the last selected frame and saves it when modal
block displacement reaches the threshold in pixels. Configure block matching in
`extraction.optical_flow`; pure `optical_flow` ignores `fps`.
Mean, median, mode, and original frame indices are logged under
`frames/<scene>/_colmap/flow/`. The first frame is always saved; `keep_last: true`
also saves the final frame regardless of threshold.

A third strategy, `hybrid`, combines optical flow with a time-based fallback:

```bash
python extract_frames.py --config config.yaml --strategy hybrid --fps 2 --mode-threshold-px 20
```

Select on motion, or save the next available frame after 0.5 video seconds without
selection. Every saved frame resets the timer and optical-flow reference.
Logs distinguish `threshold` and `fps_fallback` selections. This prevents long
selection gaps; it does not retain every source frame.


The three reconstruction stage scripts accept `--dry-run`, `--scenes DavidHouse`, and
`--experiments sift sp-sg` (experiment names from the YAML).

Supported presets: `sift`, `sp-sg`, `sp-lg`, `mast3r`, and `mast3r-aerialmd`.

## Data and reuse

```text
datasets/<scene>/<platform>/   # Input videos; dataset/ is also supported
frames/<scene>/<platform>/     # Extracted images
outputs/coalitions/<scene>/    # Cached features, matches, and reconstructions
```

Features are shared per scene and extractor configuration. Retrieval is cached
per coalition; matching reuses the required pairs per matcher configuration.
Rerun a stage to reuse completed work. To switch machines, copy the scene's
frames and outputs, then update paths and resources in the YAML.

## Code

- `src/`: command-line entry points.
- `lima3d/`: pipeline modules and shared utilities.
- `hloc/` and `third_party/`: HLoc integration and external dependencies.
- `tests/`: automated tests (`python -m unittest discover -s tests -v`).

See the [pipeline guide](docs/scene_coalitions.md) for dependencies, configuration,
and resume details.

Job durations and outcomes are saved per execution in
`outputs/coalitions/_runs/<run_id>/timings.json`; worker logs remain in `_logs/`.

Use `none`, `crop`, or `mask` under `regions.Car.cameras.<camera>` for Car,
and `regions.Drone` / `regions.Pedestrian` for the single-camera platforms. See [per-platform regions](docs/scene_coalitions.md#per-platform-regions)
for crop coordinates, mask paths, and MVS behavior.

Configure `execution.global`, `execution.local`, `execution.matching`, and
`execution.dense` under `defaults` or per experiment. Each accepts `batch_size`,
`workers`, `max_in_flight`, `loader_workers`, and `prefetch`. SIFT supports
independent CPU/GPU worker processes; neural workers use separate model instances.
See [batching](docs/scene_coalitions.md#extraction-batches-and-prefetch) for examples.


## Image velocity distributions

`flow_distribution.py` now saves Car histograms per camera with shared bins and
camera colors: a 3×3 grid and an overlaid comparison, in px/s and % of image
width/s. Percentages are normalized separately within each camera. Missing
camera samples are identified explicitly. This measures image motion, not road
speed. New CSVs preserve camera IDs; older CSVs recover them from video names.

Regenerate figures from existing measurements without reprocessing videos:

```bash
python flow_distribution.py --plot-only --output flow_distribution
```

Use `--min-speed 0` to include stationary pairs, `--min-valid-frac` to change
the validity filter, and `--hist-bins` to control shared bin counts.
