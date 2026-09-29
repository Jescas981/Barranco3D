"""Identidad por contenido y rutas portables entre las tres máquinas del pipeline."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .pipeline import inventory
from .utils.io import digest, save_json, scene_lock


def file_hash(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def portable_context(scene, args):
    scene = Path(scene).resolve()
    images, legacy_snapshot = inventory(scene, args.platforms)
    names = sorted(n for group in images.values() for n in group)
    parent = args.output_root.resolve() / scene.name
    with scene_lock(parent / '.inventory.lock', wait=True):
        index_path = parent / '.content-index.json'
        try:
            index = json.loads(index_path.read_text())
        except (OSError, ValueError):
            index = {}
        same_location = index.get('source') == str(scene)
        previous = index.get('files', {}) if same_location else {}
        files, rows = {}, []
        for name in names:
            path = scene / name
            stat = path.stat()
            identity = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
            cached = previous.get(name, {})
            checksum = cached.get('sha256') if cached.get('stat') == identity else None
            checksum = checksum or file_hash(path)
            files[name] = {'stat': identity, 'sha256': checksum}
            rows.append([name, stat.st_size, checksum])
        signature = digest({'images': rows})
        root = parent / signature
        # Adoptar snapshots antiguos sin mover ni duplicar sus HDF5.
        candidates = sorted(parent.glob('*/dataset.json'))
        for candidate in candidates:
            data = json.loads(candidate.read_text())
            if data.get('content_signature') == signature:
                root = candidate.parent
                break
        else:
            legacy = parent / legacy_snapshot
            if (legacy / 'dataset.json').is_file():
                root = legacy
        dataset = {'scene': str(scene), 'snapshot': root.name,
                   'content_signature': signature, 'images': images, 'files': rows}
        save_json(root / 'dataset.json', dataset)
        save_json(index_path, {'source': str(scene), 'files': files})
    return images, root.name, names, root


def artifact_ref(root, path):
    return Path(path).relative_to(root).as_posix()


def artifact_path(root, value):
    """Resolver referencias nuevas relativas y antiguas bajo el mismo snapshot."""
    value = Path(value)
    if value.is_absolute():
        try:
            value = value.relative_to(root)
        except ValueError:
            if root.name not in value.parts:
                raise ValueError(f'No se puede trasladar la referencia antigua {value}')
            value = Path(*value.parts[value.parts.index(root.name) + 1:])
    if '..' in value.parts:
        raise ValueError(f'Referencia fuera del snapshot: {value}')
    return root / value


def request_spec(args):
    """Solo parámetros del experimento; no rutas, GPU ni número de hilos."""
    return {key: getattr(args, key) for key in ('global_feature', 'top_k',
            'sequential_window', 'resize_max', 'max_keypoints', 'seed', 'camera_mode')} | {
                'preset': args.configs[0], 'platforms': sorted(args.platforms)}


def check_bank_request(manifest, args):
    request = manifest.get('request')
    if request is not None:
        if request != request_spec(args):
            raise ValueError(f'El banco de {args.experiment} usa otros parámetros. Ejecuta build_bank_matching.py')
        return
    # Compatibilidad con los manifiestos ya generados por la versión anterior.
    saved = manifest['config']
    expected = request_spec(args)
    for key in ('preset','global_feature','top_k','sequential_window','max_keypoints','seed','camera_mode'):
        actual = saved.get(key)
        if key == 'global_feature':
            actual = saved['global']['model']['name']
        if actual != expected[key]:
            raise ValueError(f'Banco antiguo incompatible ({key}); ejecuta build_bank_matching.py')
    if saved.get('local') and saved['local']['preprocessing']['resize_max'] != args.resize_max:
        raise ValueError('El banco usa otro resize_max; ejecuta build_bank_matching.py')


def resolve_job(root, job):
    job = deepcopy(job)
    for key in ('pairs', 'features', 'matches'):
        if key in job:
            job[key] = str(artifact_path(root, job[key]))
    if 'dense' in job:
        for key in ('raw','folder'):
            job['dense'][key] = str(artifact_path(root, job['dense'][key]))
    return job


def reconstruction_root(root, manifest):
    """El hardware no identifica una reconstrucción. Reutilizar también IDs legacy."""
    preset, base = manifest['preset'], manifest['run_base']
    canonical = root / 'reconstructions' / f'{preset}-{digest(base)}'
    if canonical.exists():
        return canonical
    compatible = []
    for config in sorted((root/'reconstructions').glob(f'{preset}-*/config.json')):
        saved = json.loads(config.read_text())
        if saved.get('run_base') == base:
            return config.parent
        if 'threads' in saved and config.parent.name == f"{preset}-{digest({**base, 'threads': saved['threads']})}":
            compatible.append(config.parent)
    if len(compatible) > 1:
        # Preferir el intento con más coaliciones terminadas; no combinar modelos.
        compatible.sort(key=lambda p: (-sum(json.loads(f.read_text()).get('status') in ('complete','no_model')
                            for f in p.glob('*/result.json')), str(p)))
    return compatible[0] if compatible else canonical
