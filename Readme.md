# Barranco3D

Extract video frames and reconstruct scenes from combinations of camera platforms.
Run matching, SfM, and MVS separately on the hardware each stage needs.

## Main scripts

Run from the repository root, with the required dependencies installed:

```bash
# Optional: discover dataset videos and draw camera crops or masks
python src/edit_regions.py --config config.yaml

# Extract frames from all platforms in a scene
python extract_colmap_frames.py --config config.yaml

# Stage 1: global/local features, retrieval, and matching
python src/build_bank_matching.py --config config.yaml

# Stage 2: sparse reconstruction (CPU)
python src/build_sfm_sparser.py --config config.yaml

# Stage 3: dense reconstruction (GPU)
python src/build_mvs.py --config config.yaml
```

Configure scenes, experiments, and CPU/GPU resources in [config.yaml](config.yaml).
Set video input paths in `data.datasets_root` and sampling options in
`extraction` (`fps`, `format`, `threads`). The extractor also accepts
`--scenes DavidHouse --fps 2` to override the YAML. Existing frame folders must
be empty or new. The editor discovers videos under `data.datasets_root` (`dataset/` here) and opens
locally in your browser. Car has independent rules for `cam0`–`cam7`; Drone and
Pedestrian each use one rule. Extraction applies
saved crops and exports masks without changing masked image pixels.

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
