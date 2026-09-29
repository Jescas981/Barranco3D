"""Lee config.yaml y devuelve un Namespace compatible con el pipeline por experimento."""
import re
from pathlib import Path
from types import SimpleNamespace

from coalition_pipeline import PRESETS, build_parser, validate

TOP = {'data', 'resources', 'mvs', 'defaults', 'experiments'}
DATA = {'frames_root', 'output_root', 'scenes', 'platforms', 'device'}
RES = {'bank', 'sfm', 'mvs'}
GPU_RES = {'gpus', 'threads'}
SFM_RES = {'workers', 'threads'}
MVS = {'enabled', 'colmap', 'max_image_size', 'cache_gb', 'num_sources'}
OPTIONS = {'global_feature', 'top_k', 'sequential_window', 'query_batch', 'database_batch',
           'resize_max', 'max_keypoints', 'seed', 'sift_device', 'camera_mode', 'mvs'}
CHOICES = {'global_feature': ['netvlad', 'openibl', 'megaloc', 'dir'],
           'camera_mode': ['PER_FOLDER', 'PER_IMAGE'],
           'sift_device': ['auto', 'cpu', 'cuda'], 'device': ['auto', 'cpu', 'cuda']}


def _section(value, allowed, where):
    value = value or {}
    if not isinstance(value, dict):
        raise ValueError(f'config: {where} debe ser un mapa')
    extra = set(value) - allowed
    if extra:
        raise ValueError(f'config: claves desconocidas en {where}: {sorted(extra)}')
    return value


def _gpu_ids(value, where):
    if isinstance(value, bool):
        raise ValueError(f'config: {where}.gpus inválido')
    ids = list(range(value)) if isinstance(value, int) else [int(g) for g in value]
    if not ids or len(set(ids)) != len(ids) or min(ids) < 0:
        raise ValueError(f'config: {where}.gpus debe ser un número > 0 o una lista de ids únicos')
    return ids


def load_config(path, only_experiments=None, only_scenes=None, dry_run=False):
    import yaml
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    _section(raw, TOP, 'la raíz')
    data = _section(raw.get('data'), DATA, 'data')

    rsec = _section(raw.get('resources'), RES, 'resources')
    res = {'bank': {'gpus': 1, 'threads': 4, **_section(rsec.get('bank'), GPU_RES, 'resources.bank')},
           'sfm': {'workers': 0, 'threads': 2, **_section(rsec.get('sfm'), SFM_RES, 'resources.sfm')},
           'mvs': {'gpus': 1, 'threads': 4, **_section(rsec.get('mvs'), GPU_RES, 'resources.mvs')}}
    res['bank_ids'] = _gpu_ids(res['bank']['gpus'], 'resources.bank')
    res['mvs_ids'] = _gpu_ids(res['mvs']['gpus'], 'resources.mvs')
    if (res['bank']['threads'] < 1 or res['mvs']['threads'] < 1
            or res['sfm']['threads'] < 1 or res['sfm']['workers'] < 0):
        raise ValueError('config: resources inválido (threads deben ser > 0, workers >= 0)')

    mvs = {'enabled': True, 'colmap': 'colmap', 'max_image_size': 1024, 'cache_gb': 2.0,
           'num_sources': 10, **_section(raw.get('mvs'), MVS, 'mvs')}
    defaults = _section(raw.get('defaults'), OPTIONS, 'defaults')
    entries = raw.get('experiments')
    if not entries:
        raise ValueError('config: falta la lista experiments')
    if data.get('device', 'auto') not in CHOICES['device']:
        raise ValueError('config: data.device debe ser auto, cpu o cuda')

    parser = build_parser()
    experiments, seen, scenes = {}, set(), None
    for entry in entries:
        entry = _section(entry, OPTIONS | {'name', 'preset'}, 'experiments')
        merged = {**defaults, **entry}
        name, preset = merged.get('name'), merged.get('preset')
        if not name or not re.fullmatch(r'[A-Za-z0-9_.-]+', str(name)):
            raise ValueError(f'config: nombre de experimento inválido: {name!r}')
        if name in seen:
            raise ValueError(f'config: experimento repetido: {name}')
        seen.add(name)
        if preset not in PRESETS:
            raise ValueError(f'config: preset desconocido en {name}: {preset} (usa {list(PRESETS)})')
        for key, allowed in CHOICES.items():
            if key in merged and merged[key] not in allowed:
                raise ValueError(f'config: {key} en {name} debe ser uno de {allowed}')
        if only_experiments and name not in only_experiments:
            continue

        ns = parser.parse_args([])                     # valores por defecto del pipeline
        ns.frames_root = Path(data.get('frames_root', 'frames'))
        ns.output_root = Path(data.get('output_root', 'outputs/coalitions'))
        ns.scenes = list(only_scenes) if only_scenes else data.get('scenes')
        ns.platforms = data.get('platforms', ns.platforms)
        ns.device = data.get('device', 'auto')
        ns.configs = [preset]
        for key in OPTIONS - {'mvs'}:
            if key in merged:
                setattr(ns, key, merged[key])
        ns.threads = res['bank']['threads']            # pares, banco y SIFT
        ns.sfm_threads = res['sfm']['threads']         # forma parte del run_id del SfM
        ns.experiment = name
        ns.mvs = bool(merged.get('mvs', mvs['enabled']))
        ns.dry_run = dry_run
        found = validate(ns, parser)
        scenes = scenes or found
        experiments[name] = ns

    missing = set(only_experiments or []) - seen
    if missing:
        raise ValueError(f'config: experimentos inexistentes: {sorted(missing)}')
    if not experiments:
        raise ValueError('config: ningún experimento seleccionado')
    return SimpleNamespace(path=path, scenes=scenes, experiments=experiments, res=res, mvs=mvs,
                           output_root=Path(data.get('output_root', 'outputs/coalitions')).resolve())