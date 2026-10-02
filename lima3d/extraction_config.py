"""Resolve frame-extraction settings without loading reconstruction dependencies."""
import math
from pathlib import Path

EXTRACTION_KEYS = {'fps', 'format', 'threads', 'strategy', 'optical_flow'}


def extraction_jobs(args):
    config_path = args.config
    # Preserve the original positional CLI; otherwise use the shared config.
    if config_path is None and args.scene is None:
        config_path = Path('config.yaml')
    raw = {}
    base = Path.cwd()
    if config_path is not None:
        import yaml
        config_path = Path(config_path).resolve()
        base = config_path.parent
        try:
            raw = yaml.safe_load(config_path.read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f'Invalid YAML: {exc}') from exc
        if not isinstance(raw, dict):
            raise ValueError('Configuration must be a YAML mapping')
    data = raw.get('data', {})
    settings = raw.get('extraction', {})
    if not isinstance(data, dict) or not isinstance(settings, dict):
        raise ValueError('data and extraction must be YAML mappings')
    unknown = set(settings) - EXTRACTION_KEYS
    if unknown:
        raise ValueError(f'Unknown extraction settings: {sorted(unknown)}')
    from .optical_flow import validate_options
    strategy = getattr(args, 'strategy', None) or settings.get('strategy', 'fps')
    if strategy not in ('fps', 'optical_flow', 'hybrid'):
        raise ValueError('extraction.strategy must be fps, optical_flow or hybrid')
    flow = validate_options(settings.get('optical_flow'))
    threshold = getattr(args, 'mode_threshold_px', None)
    if threshold is not None:
        if strategy not in ('optical_flow', 'hybrid'):
            raise ValueError('--mode-threshold-px requires --strategy optical_flow or hybrid')
        flow = validate_options({**flow, 'mode_threshold_px': threshold})
    fps = args.fps if args.fps is not None else settings.get('fps')
    if strategy == 'optical_flow':
        fps = None
    elif isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Set a finite positive extraction.fps or --fps')
    threads = args.threads if args.threads is not None else settings.get('threads', 1)
    if type(threads) is not int or threads < 1:
        raise ValueError('extraction.threads must be a positive integer')
    image_format = args.format if args.format is not None else settings.get('format', 'jpg')
    if image_format not in ('jpg', 'png'):
        raise ValueError('extraction.format must be jpg or png')
    groups = args.groups if args.groups is not None else data.get('platforms')
    if groups is not None and (not isinstance(groups, list) or not groups or
                               any(not isinstance(g, str) or not g for g in groups)):
        raise ValueError('data.platforms must be a nonempty list of names')
    for key in ('datasets_root', 'frames_root'):
        if key in data and (not isinstance(data[key], str) or not data[key]):
            raise ValueError(f'data.{key} must be a nonempty path string')
    datasets = base / data.get('datasets_root', 'datasets')
    if 'datasets_root' not in data and not datasets.exists() and (base / 'dataset').is_dir():
        datasets = base / 'dataset'
    frames = base / data.get('frames_root', 'frames')
    if args.scene is not None:
        scene = args.scene
        if not scene.is_dir() and len(scene.parts) == 1:
            scene = datasets / scene
        scenes = [scene.resolve()]
    else:
        selected = args.scenes if args.scenes is not None else data.get('scenes')
        if selected is None:
            if not datasets.is_dir():
                raise ValueError(f'Dataset root does not exist: {datasets}')
            scenes = sorted(p.resolve() for p in datasets.iterdir() if p.is_dir() and not p.name.startswith('.'))
        else:
            if not isinstance(selected, list) or not selected or any(
                    not isinstance(n, str) or not n or n in ('.', '..') or Path(n).name != n for n in selected):
                raise ValueError('data.scenes / --scenes must contain scene folder names')
            if len(set(selected)) != len(selected):
                raise ValueError('Scene names must be unique')
            scenes = [(datasets / name).resolve() for name in selected]
    if not scenes:
        raise ValueError('No scenes selected')
    if args.output is not None and len(scenes) != 1:
        raise ValueError('--output requires exactly one scene')
    from .regions import parse_regions
    region_settings = raw.get('regions') or {}
    regions = parse_regions(region_settings, base, groups if groups is not None else list(region_settings))
    jobs = []
    for scene in scenes:
        if not scene.is_dir():
            raise ValueError(f'Scene does not exist: {scene}')
        output = (args.output if args.output is not None else frames / scene.name).resolve()
        if output == scene or scene.is_relative_to(output):
            raise ValueError('Output cannot be the input scene or its ancestor')
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f'Output must be new or empty: {output}')
        jobs.append(dict(scene=scene, output=output, fps=float(fps) if fps is not None else None, groups=groups,
                         strategy=strategy, optical_flow=flow,
                         image_format=image_format, threads=threads, dry_run=args.dry_run, regions=regions))
    return jobs
