# Scene reconstruction pipeline

The pipeline reconstructs every nonempty combination of the configured platforms.
Car, Drone, and Pedestrian produce seven coalitions. Car's camera subfolders
remain separate cameras, while Car counts as one platform. No Shapley values
are calculated.

## Requirements

Use a Python environment with PyTorch, torchvision, NumPy, OpenCV, h5py, SciPy,
Pillow, tqdm, PyYAML, and pycolmap. Frame extraction also requires FFmpeg and
ffprobe. MVS requires a COLMAP executable with CUDA support and a compatible GPU.

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
first use. The COLMAP executable and Python's pycolmap are separate installations;
CUDA support in one does not enable it in the other.

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
- `mvs`: enablement, COLMAP executable, image size, and memory/cache settings.

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
MVS subprocesses are simulated and do not validate full GPU densification.
