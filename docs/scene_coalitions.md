# Scene reconstruction pipeline

The pipeline reconstructs every nonempty combination of the configured platforms.
Car, Drone, and Pedestrian produce seven coalitions. Car's camera subfolders
remain separate cameras, while Car counts as one platform. No Shapley values
are calculated.

## Requirements

Use a Python environment with PyTorch, torchvision, NumPy, OpenCV, h5py, SciPy,
Pillow, tqdm, PyYAML, and pycolmap. Frame extraction also requires FFmpeg and
ffprobe. MVS uses pycolmap with CUDA support (`pycolmap-cuda`) and a compatible GPU;
no COLMAP executable is required. Remove the old `mvs.colmap` YAML setting
and `--colmap` CLI option when upgrading. Existing MVS workspaces can resume.

Initialize SuperPoint/SuperGlue when using those presets:

```bash
git submodule update --init --recursive third_party/SuperGluePretrainedNetwork
```

For LightGlue, install it in the same environment:

```bash
python -m pip install git+https://github.com/cvg/LightGlue.git
```

MASt3R presets use `hloc/matchers/mast3r.py` and the corresponding original or
AerialMD checkpoint. Learned extractors and matchers may download weights on
first use. MVS checks `pycolmap.has_cuda` and the dense reconstruction APIs
before starting. Installing a CUDA-enabled COLMAP executable does not enable
CUDA in pycolmap.

## Main stages

Run these commands from the repository root, on the machine assigned to each stage:

```bash
python src/build_bank_matching.py --config config.yaml
python src/build_sfm_sparser.py --config config.yaml
python src/build_mvs.py --config config.yaml
```

1. **Matching bank:** global/local features, retrieval, sequential pairs, and matches.
   Uses `resources.bank.gpus` and `resources.bank.threads`.
2. **SfM:** independent sparse reconstructions per coalition and configuration.
   Uses `resources.sfm.workers` and `resources.sfm.threads` per worker, without
   requiring a GPU. `workers: 0` selects a count based on available CPUs.
3. **MVS:** dense reconstruction of completed SfM models. Uses `resources.mvs`,
   with one GPU per job and separate reconstructions running across GPUs.

All three accept `--dry-run`, `--scenes DavidHouse`, and `--experiments sift sp-sg`.
Experiment filters refer to YAML `name` values. Logs are written under
`<output_root>/_logs/`.

## Configuration

Edit [config.yaml](../config.yaml):

- `data`: frame/output roots, scenes, platforms, and bank device.
- `defaults`: shared extraction, retrieval, and reconstruction settings.
- `experiments`: unique names, presets, and per-experiment overrides.
- `resources`: GPU indices/counts, CPU threads, and SfM worker count.
- `mvs`: enablement, image size, and memory/cache settings.

Relative paths are resolved against the YAML file's directory. `gpus` accepts
an integer count or a list such as `[0, 2]`. Only active stages check GPU
availability. Choose SfM concurrency according to available RAM as well as CPUs.
Set `mvs: false` on an experiment to exclude it from dense reconstruction.

Presets are `sift` (SIFT + nearest-neighbor matching), `sp-sg` (SuperPoint +
SuperGlue), `sp-lg` (SuperPoint + LightGlue), `mast3r`, and `mast3r-aerialmd`.
Global extractors include `netvlad`, `openibl`, `megaloc`, and `dir`, subject to
their dependencies and weights being available.

`camera_mode: PER_FOLDER` shares intrinsics within each image folder. Separate
images from different lenses, zoom settings, or resolutions into different
folders, or use `PER_IMAGE`. The pipeline does not enforce a synchronized rig.

## Retrieval and cache reuse

Global and local features are extracted once per scene and extractor configuration.
SuperGlue and LightGlue share SuperPoint features when extraction settings match.

Retrieval selects each image's top-k neighbors within its coalition, excluding
itself. Filtering the full-scene top-k would miss valid neighbors in smaller
coalitions. Search runs in CPU batches rather than storing the full similarity
matrix. Lists are shared across experiments with the same global extractor and
`top_k`; changing only `sequential_window` reuses retrieval results.

A positive `sequential_window` adds pairs within the same video and camera,
counted in extracted images. It recognizes the extractor's numeric filename
indices and does not connect different clips or cameras. Zero disables it.

Matching processes the deduplicated union of required pairs per matcher
configuration. Each coalition uses only its images and pairs. Geometric
verification and SfM remain separate per coalition.

MASt3R stores coordinates and confidence per pair in `dense_raw/`, then assembles
COLMAP-compatible features and matches per coalition in `dense_assembled/`.
This is a correspondence adapter, not MASt3R's native global 3D optimization.
Its inference resolution is 512 px; `resize_max` controls sparse extractors.
Changing the pair set or keypoint limit can repeat assembly without repeating
existing neural pair inference.

## Resume and move between machines

Copy `frames/<scene>/`, the complete `outputs/coalitions/<scene>/` directory,
and the YAML. Preserve relative image names and contents. Update data paths and
hardware resources in the destination YAML, keeping experiment names and
algorithm settings consistent.

Portable snapshots identify image contents, not their absolute paths or timestamps.
Changing paths, CPU threads, or GPU indices does not invalidate completed work.
Algorithm or dependency changes may require new artifacts; rebuild the bank
before running downstream stages with changed bank settings.

For older banks, run the updated bank stage in its original location before
copying it, so its portable identity can be registered. Older MVS workspaces
may require a new workspace; originals are preserved.

Completed SfM models are skipped. Interrupted reconstructions restart in a new
attempt directory, rather than resuming an optimizer iteration. A `no_model`
result is cached; change the seed or pair settings to try a new reconstruction.
Failed jobs can be retried. A lock prevents concurrent schedulers from writing
to the same output root.

MVS can run later without repeating the bank or SfM. It resumes undistortion,
stereo, and fusion using validated outputs. Interrupted stereo retains valid
maps; interrupted fusion runs again. The result is `workspace/fused.ply`, not
a mesh. Input SfM models are preserved.

## Auxiliary scripts and tests

The main workflow uses the three stage scripts above. Auxiliary entry points are
`src/coalition_pipeline.py` (legacy monolithic workflow) and
`src/coalition_mvs.py` (standalone MVS, including external COLMAP models).
The monolithic workflow retains its legacy inventory behavior; use the staged
workflow for portable snapshots and resource scheduling.

```bash
python src/coalition_mvs.py --model-path /path/to/colmap/model \
  --image-path /path/to/images --output /path/to/mvs
python -m unittest discover -s tests -v
```

The implementation lives in `lima3d/`, with file and path utilities in `utils/`.
Tests cover caching, scheduling, imports, portability, and resume behavior;
MVS pycolmap calls are simulated and do not validate full GPU densification.

## Progress display

Interactive terminals show one job counter per active stage and a live line for
running workers. Counters distinguish `completed`, `cached`, `skipped`, and
`failed`, with queued/running counts. Retrieval jobs shared by experiments count
once. Totals grow as downstream jobs are discovered; resolved jobs include
failures, so a finished counter does not imply every reconstruction succeeded.

Worker lines show the GPU index or CPU worker PID, task name, elapsed time,
current operation, and recent log output. SIFT, sparse matching, and MASt3R
report image/pair counts; learned feature extraction also exposes its own log
progress. SfM and MVS show the current operation without a speculative ETA.
Updates to worker status files are throttled to avoid per-item disk writes.

Fully reused jobs count as cached; jobs that compute any missing work count as
completed. A new SfM attempt producing `no_model` counts as skipped; a previously
cached `no_model` counts as cached. MVS without a valid model is skipped, while
missing or incomplete required SfM output is reported as a failure.

When output is redirected, animated bars are disabled; start/end messages and
final counters remain available. Detailed worker logs are under `_logs/` and
atomic current-status files are under `_tasks/` in the output root.

## Frame extraction configuration

```bash
python extract_frames.py --config config.yaml --dry-run
python extract_frames.py --config config.yaml --scenes DavidHouse
```

The extractor reads `data.datasets_root`, `data.frames_root`, `data.scenes`,
`data.platforms`, and the `extraction` section (`strategy`, `fps`, `format`, `threads`, `optical_flow`).
`scenes: null` selects all scene folders under the video input root. YAML paths
are relative to the configuration file. If `datasets_root` is omitted, it looks
for `datasets/`, falling back to `dataset/`.

CLI options override YAML values: `--fps`, `--format`, `--threads`, `--groups`,
`--scenes`, and `--output` (one scene only). Explicit CLI paths are relative to
the working directory. With no scene argument or `--config`, it reads
`config.yaml`. The original positional invocation remains available without YAML:
`python extract_frames.py DavidHouse --fps 2`.

Fixed-FPS extraction requires FFmpeg/ffprobe and PyYAML when using configuration;
it does not load reconstruction models. Videos run sequentially and nonempty
output directories are rejected. Choose another `data.frames_root` or `--output`
to create a separate extraction without overwriting existing frames.

## Job durations

Each scheduler invocation writes an independent report to
`<output_root>/_runs/<run_id>/timings.json`. Its path is printed at startup.
The report is updated atomically as jobs start and finish; previous runs are
preserved when you resume or repeat an experiment.

Each job records `stage`, `scene`, `experiment`, `coalition` where applicable,
`started_at`, `finished_at`, `elapsed_seconds`, `status`, resource, and log path.
Timestamps use UTC. Duration uses a monotonic clock and measures worker wall time,
including loading and saving, but excluding queue wait. Completion is detected
by the scheduler's polling loop, so measurements can include a small polling delay.

Jobs resolved without launching a worker have `launched: false` and zero duration.
For example, cached SfM does not inherit the original reconstruction's duration.
A worker that only validates cached data still records its actual runtime and
`cached` status. Partial recomputation records the duration of this invocation,
not the cumulative cost of previous attempts. Shared retrieval is recorded once
under its executing experiment; bank work is shared across coalitions.

The top-level `elapsed_seconds` is total invocation wall time; `stage_totals`
contains summed job seconds and outcome counts per stage. Concurrent job times
must not be interpreted as total wall time. Failed and gracefully interrupted
workers retain their elapsed time. After a forced kill or power loss, an unfinished
record remains `running` with no final duration; no duration is invented.

This reporting applies to `src/build_bank_matching.py`,
`src/build_sfm_sparser.py`, and `src/build_mvs.py`. It does not recover durations
from older runs or time the standalone auxiliary scripts.

## Per-platform regions

Choose a default mode per platform and optional overrides per camera. Omitted
platforms use `none`. Camera rules take precedence, including an explicit `none`.
Rules apply to the same camera across selected scenes and experiments:

```yaml
regions:
  Car:
    mode: none
    cameras:
      cam0: {mode: mask, path: masks/Car/cam0/selection.png}
      cam1: {mode: crop, box: [0, 0, 1920, 900]}
      cam2: {mode: none}
  Drone:
    mode: mask
    path: masks/drone.png  # One shared mask, same dimensions as every Drone image
  Pedestrian:
    mode: none
```

- `none`: unchanged images and matching behavior; existing caches remain usable.
- `crop`: physically crops derived images without changing originals. Global/local
  extraction, dense matching, SfM, and MVS all use these cropped images. Coordinates
  and camera dimensions refer to the crop. JPEG crops are saved at quality 100;
  use PNG source frames when lossless derived crops are required.
- `mask`: preserves original pixels and dimensions. Black pixels exclude keypoints;
  nonzero pixels allow them. Sparse features are filtered before matching, while
  MASt3R correspondences are filtered at both endpoints before track assembly.
  Global features and retrieval still see the original content, and MASt3R can
  still use excluded regions as context. This mode does not mask MVS depth maps
  or guarantee exclusion from the dense cloud.

For per-image masks, replace `path` with `root: masks`. The expected filename is
`masks/<scene>/<platform>/<subfolders>/<image_filename>.png`, including the
original image extension, for example `masks/DavidHouse/Car/cam0/frame.jpg.png`.
All masks for a masked platform are required and must match image dimensions.
Paths in the YAML are relative to the YAML directory. Crop coordinates must be
valid for every affected image; each camera selects one mode; different cameras can use different modes.

Run the matching bank after changing regions. Region settings and mask contents
select a separate snapshot, preventing stale features or reconstructions from
being reused. Existing snapshots are preserved; raw inference caches are currently
not shared across region variants. Derived crop inputs live under the snapshot's
`images/` directory. Moving machines still requires the original frames, output
scene directories, and mask files; adjust YAML paths as needed. Mask content hashes
are independent of their absolute location.

Region handling is supported by frame extraction and the three-stage
reconstruction pipeline. The auxiliary monolithic CLI does not implement these rules.

## Video region editor

```bash
python src/edit_regions.py --config config.yaml
python extract_frames.py --config config.yaml --scenes DavidHouse --dry-run
python extract_frames.py --config config.yaml --scenes DavidHouse
```

The editor discovers videos under `data.datasets_root` (`dataset/` by default),
including platforms not currently enabled for reconstruction. No video path is
required. In the browser, choose a scene, platform, camera, and video, then a
timestamp and mode. Car subfolders such as `cam0`–`cam7` are independent cameras;
Drone and Pedestrian each share one camera rule across their videos:

- **Crop:** drag the rectangle to keep.
- **Mask:** click polygon vertices around an area to exclude, then select
  **Finish polygon** or press Enter. Add multiple polygons if needed.
- **None:** disable region filtering for that platform.

Use **Undo** or **Clear** to adjust the selection, then **Save to config.yaml**.
The editor writes only the selected camera's region (or the platform rule for
Drone/Pedestrian), preserves unrelated YAML
settings/comments, and backs up the configuration under `.region_editor_backups/`.
Masks are PNG files under `masks/`, named by their content; accompanying JSON
files retain editable polygons and video provenance. Existing mask files remain
available when a new selection is saved. The interface and exported masks use
original encoded video coordinates, without autorotation.

These are static camera rules, not object tracking. A region drawn on one Car
video applies to videos of that camera only. Saving cam0 does not replace cam1.
Choose representative frames; all videos sharing a rule must have compatible
dimensions. Save or discard pending edits before switching the selected video.
Optional CLI filters are `--scene`, `--platform`, and `--camera`; `--video`
remains available for opening a specific file directly.

### Extraction behavior

The extractor reads the same `regions` section. Crops are applied by FFmpeg
while frames are generated. `_colmap/extraction.json` records those crops;
the staged pipeline recognizes them and does not crop a second time. Changing
or disabling an already applied crop requires extracting into a new frame root.

Mask mode preserves all image pixels and exports matching-sized PNG masks under
`frames/<scene>/_colmap/masks/<image_relative_path>.png`. The generated standalone
COLMAP script uses that mask directory. The staged pipeline uses configured masks;
if their original paths are unavailable after moving the dataset, it can use the
exported copies when the configured source rule matches the extraction manifest.
Mask mode still does not exclude pixels from MVS depth estimation.

Extraction checks shared mask dimensions and crop bounds before writing frames.
With per-image masks (`root`), expected frame masks must already exist and are
validated as frames are exported. Existing frame output directories are never
overwritten. Keep the `_colmap` metadata when transferring frames between machines.

### Remote machines

Use `--no-browser --port 8765` on the remote machine and forward that port:
`ssh -L 8765:127.0.0.1:8765 user@host`. Open the exact URL printed by the editor,
including its session token, in your local browser. The server binds only to
loopback and stops with Ctrl+C. No videos are uploaded to an external service.
Dependencies are FFmpeg/ffprobe, Pillow, and PyYAML; GPU models are not loaded.

## Optical-flow frame selection

Set `extraction.strategy: optical_flow` or pass `--strategy optical_flow`.
`--mode-threshold-px 20` overrides `extraction.optical_flow.mode_threshold_px`.
All source frames are decoded sequentially on CPU; `fps` is ignored in this mode.
RAM usage is bounded by the current frame and reference, not video length.

Each camera/video is processed independently. The first frame becomes the reference.
OpenCV normalized-correlation block matching estimates displacement against the
last selected frame. Select when the modal magnitude is **>= Mth**, then replace
the reference. Default blocks are 180 px, grid stride 270 px, and search radius
100 px in each direction. Coordinates use original pixels after any crop, without
resizing. Mth must be positive and no greater than the configured search radius.

The histogram uses `mode_bin_width_px` (default 1 px). The reported mode is the
median observation in the most populated bin; ties choose the lowest bin. This
keeps stationary flow at zero instead of reporting a bin midpoint. Mean and median
are also recorded but do not control selection. `min_texture_std: 5` and
`min_correlation: 0.7` reject unreliable blocks. With no valid blocks the frame is
skipped. Fast motion outside the search range, weak texture, or large masks can
prevent threshold selection; inspect `frames_without_valid_flow` in the manifest.
This threshold is a selection trigger, not a guarantee of a maximum frame gap or
of successful reconstruction.

`keep_last: true` saves the last frame even below Mth; set it to false for threshold-only
selection after the first frame. Logs under `_colmap/flow/<platform>/<camera>/`
contain source frame indices, timestamps, reference indices, statistics, and selection
reasons. A forced final selection adds a `reason: last` event for that frame.
Image filenames use consecutive selected-frame indices; consult the log for source indices.

Crops are applied before flow estimation and export. Shared camera masks exclude
blocks whose reference or candidate footprint intersects excluded pixels; exported
image pixels remain intact. Per-image mask roots are unsupported for optical-flow
selection; use a shared mask `path` created by the region editor. OpenCV and NumPy
are required in addition to the fixed-FPS extraction dependencies.

The region editor supports crop corner/edge handles, dragging the entire crop,
and manual left/top/right/bottom/width/height fields. Mask drawing offers polygon
and rectangle tools; rectangles are saved as four-vertex polygons.

## Hybrid frame selection

Three extraction strategies are available: `fps`, `optical_flow`, and `hybrid`.
The hybrid strategy uses the same optical-flow parameters, crops, and shared masks:

```yaml
extraction:
  strategy: hybrid
  fps: 2
  optical_flow:
    mode_threshold_px: 20
    keep_last: true
```

Or pass `--strategy hybrid --fps 2 --mode-threshold-px 20` to `extract_frames.py`.
Save the first frame, then select whenever modal displacement reaches Mth **or**
`1 / fps` seconds have elapsed since the most recent selection. Each selection
resets both the reference image and timer. This is an inactivity timeout, not a
union with a fixed FPS sampling grid; motion may produce a higher extraction rate.
If both conditions hold, the logged reason is `threshold`. Timeout selections use
`fps_fallback`, even when no valid flow blocks exist. `keep_last` retains its usual
behavior without duplicating a frame already selected.

The timer uses video timestamps, not processing/wall-clock time. Selection occurs
at the first decodable frame at or after the deadline, so frame cadence or gaps
in the source can make the actual interval longer than `1 / fps`. Invalid or
non-increasing timestamps fall back to the nominal source frame duration, giving
approximate timing; `timestamp_fallback_frames` records this in `extraction.json`.
If neither timestamps nor source FPS can advance time, extraction fails explicitly.
`selection_time_seconds` in the JSONL is the elapsed selection clock;
`timestamp_seconds` remains the original decoder timestamp. The manifest also
records `fallback_fps` and `fallback_selected_frames` per video.
Console output labels the active strategy (`Hybrid` or `Optical flow`), prints
saved filenames with source indices, timestamps, and selection reasons, and reports
counts for flow, FPS fallback, first, and final frames every 100 checked frames and
at video completion. These reason counts are also saved as `selection_counts`.

Both hybrid and fixed-FPS extraction require a positive `fps` no greater than the
reported source FPS. Hybrid still evaluates flow on every decoded frame: the
fallback prevents long gaps in selection, not missing/corrupt source frames or
all possible reconstruction failures. Pure optical-flow behavior is unchanged.

## Extraction batches and prefetch

Execution settings are independent of feature/matcher settings and cache keys.
Set `defaults.execution` and override fields per operation under an experiment:

```yaml
experiments:
  - name: sift
    preset: sift
    sift_device: cuda
    execution:
      local:
        batch_size: 4          # Images processed sequentially by each SIFT worker per job
        workers: 8             # Up to eight independent SIFT processes on the assigned GPU
        max_in_flight: 10       # Up to ten jobs running or waiting; not ten GPU batches at once
        loader_workers: 2      # CPU loading/preprocessing threads
        prefetch: 16            # Prepared input items (images here)
      matching:
        batch_size: 2
        workers: 2
        max_in_flight: 4
  - name: sp-sg
    preset: sp-sg
    execution:
      local: {batch_size: 4, workers: 1}
      matching: {batch_size: 2, workers: 1}
  - name: mast3r
    preset: mast3r
    execution:
      dense: {batch_size: 4, workers: 1, max_in_flight: 2}
```

These are tuning examples, not benchmarked optimal settings. Defaults keep one
inference worker per operation. `max_in_flight` defaults to the worker count;
setting it lower caps effective concurrency. Operations run in pipeline order,
not simultaneously. Runtime settings apply to a bank task on its assigned GPU;
`resources.bank.gpus` still controls scheduling across GPUs/scenes.

- `global`: tensor batches for NetVLAD; other global extractors use batch size 1
  and can use independent workers.
- `local`: tensor batches for SuperPoint. SIFT batches are jobs containing images
  processed serially by one native extractor, with separate spawned processes for
  concurrent jobs. No native multi-image SIFT tensor operation is implied.
- `matching`: tensor batches for nearest-neighbor and SuperGlue when feature counts
  and tensor dimensions agree. LightGlue requires batch size 1, with independent
  workers available. No padding or keypoint truncation is introduced to force batching.
- `dense`: MASt3R and MASt3R+AerialMD batches count image pairs. Neural inference is
  batched; correspondence extraction remains per pair. Each worker holds its own model.

Neural workers use separate CUDA streams and inference contexts. SIFT uses spawn,
never fork, with one native extractor per process. The coordinator is the only
HDF5 writer. Existing complete image/pair records are reused, including after a
failed execution; unfinished batches are recomputed. Results are consumed in order.
Queue and prefetch limits bound host memory, while increasing workers duplicates
models and GPU contexts. For SIFT, `resources.bank.threads` applies per process;
eight workers with four threads each can use up to 32 extraction CPU threads.

`loader_workers: 0` disables CPU prefetch. `prefetch` counts queued input items;
`max_in_flight` counts jobs, each containing up to `batch_size` items, in addition
to the currently consumed result. Variable-shaped inputs split neural batches into
smaller compatible groups within each window, so actual batch size can be lower.
Increasing workers does not guarantee overlapping kernels or greater throughput:
GPU resources, transfer, disk I/O, and startup costs can dominate. Measure completed
images/pairs per second on representative sequences, not GPU utilization alone.
Out-of-memory errors stop the task; reduce batch size/workers and rerun to resume.

Legacy `local_batch_size`, `dense_batch_size`, `loader_workers`, and `prefetch`
settings remain accepted; explicit `execution` fields take precedence. Per-operation
maps merge with defaults, preserving fields not overridden. Changing runtime options
does not invalidate feature/matching caches. Experiments sharing a completed bank
reuse it regardless of execution settings. This update adds no further MASt3R cache
invalidation beyond the earlier adapter batching change. Floating-point inference
may differ slightly between execution configurations.

## Optical-flow distribution experiment

```bash
python src/analyze_optical_flow.py --config config.yaml --scene DavidHouse \
  --samples-per-video 24 --lag-seconds 0.2 --threads 1
```

The experiment samples each original video uniformly across its recorded span,
comparing pairs at a fixed nominal time lag. It discovers all platform folders
independently of reconstruction platform filters; optionally pass `--platforms`.
It reuses optical-flow parameters and camera-specific crops/masks from the YAML.
The scene and existing extraction outputs are never modified.

A timestamped folder under `outputs/flow_experiments/<scene>/` contains a self-contained
`report.html`, PNG/PDF distributions and camera block maps, raw `pairs.jsonl`,
`blocks_summary.json`, `summary.json`, and provenance/duration in `experiment.json`.
Use `--output` for a new or empty custom folder. The output distinguishes missing
flow from zero displacement and reports valid-block coverage, search-limit matches,
modal-bin support, and ties. A block's temporal mode is different from the mode
across blocks in one frame pair; plots label the distinction explicitly.

This is a sampled exploratory analysis, with equal weight per sampled pair and
roughly equal sampling per clip (not duration-weighted or camera-balanced). It does
not replay hybrid selection or estimate physical velocity. Pixel flow is affected
by scene depth, camera optics, motion, and search-range limits. The actual timestamp
interval and source indices are retained for every measurement. Regenerate figures
without decoding videos again with `--report-only <existing-experiment-folder>`.
