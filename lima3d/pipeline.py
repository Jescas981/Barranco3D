#!/usr/bin/env python3
"""Features y SfM para todas las coaliciones de plataformas, con cachés compartidas."""
from .utils.progress import report as emit_progress

import argparse
from copy import deepcopy
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path

from .execution import execution_options, validate_execution
from .utils.paths import REPO_ROOT
from .utils.io import digest, save_json, scene_lock
import random
import re
import time

PRESETS = {
    'sift': ('sift', 'NN-ratio'),
    'sp-sg': ('superpoint_aachen', 'superglue'),
    'sp-lg': ('superpoint_aachen', 'superpoint+lightglue'),
    'mast3r': (None, 'mast3r'),
    'mast3r-aerialmd': (None, 'mast3r-aerialmd'),
}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
SCHEMA = 1
# Identidad de la implementación compatible con las cachés existentes.
# No depende del CLI/MVS ni de cambios de documentación. Cambiarla únicamente
# si cambia la semántica de extracción/matching disperso y es incompatible.
CACHE_IMPLEMENTATION = '1064e18ac06f324497fad9ecbe6ba2be7648621ec018577eb2afe3c2b7debbe6'


def cache_environment(torch_version, pycolmap_version):
    return {'schema': SCHEMA, 'code': CACHE_IMPLEMENTATION,
            'torch': torch_version, 'pycolmap': pycolmap_version}


def coalitions(platforms):
    return [combo for size in range(1, len(platforms) + 1)
            for combo in combinations(platforms, size)]


def inventory(scene, platforms):
    images = {}
    snapshot = []
    encoded_names = set()
    for platform in platforms:
        directory = scene / platform
        if not directory.is_dir():
            raise ValueError(f'Falta plataforma {directory}')
        names = sorted(p.relative_to(scene).as_posix() for p in directory.rglob('*')
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not names:
            raise ValueError(f'No hay imágenes en {directory}')
        for name in names:
            if any(c.isspace() for c in name):
                raise ValueError(f'HLoc requiere nombres sin espacios: {name}')
            encoded = name.replace('/', '-')
            if encoded in encoded_names:
                raise ValueError(f'Colisión de nombres para matches HLoc: {name}')
            encoded_names.add(encoded)
            stat = (scene / name).stat()
            snapshot.append((name, stat.st_size, stat.st_mtime_ns))
        images[platform] = names
    return images, digest({'root': str(scene.resolve()), 'files': snapshot})


def resolve_sift_device(requested, cuda_compiled, cuda_available):
    if requested == 'cuda':
        if not cuda_compiled:
            raise RuntimeError('Este pycolmap fue compilado sin CUDA. Usa --sift-device cpu '
                               'para conservar GPU en otras etapas, o instala pycolmap con CUDA.')
        if not cuda_available:
            raise RuntimeError('SIFT CUDA solicitado pero no hay una GPU CUDA disponible')
        return 'cuda'
    if requested == 'auto':
        return 'cuda' if cuda_compiled and cuda_available else 'cpu'
    if requested != 'cpu':
        raise ValueError(f'Dispositivo SIFT desconocido: {requested}')
    return 'cpu'


def extract_sift(conf, scene, names, path, device='cpu', threads=1, batch_size=1, loader_workers=1, prefetch=4, workers=1, max_in_flight=1):
    """SIFT de COLMAP directamente: no necesita Kornia ni pesos descargables."""
    import h5py
    import numpy as np
    import pycolmap
    import torch
    from hloc.extract_features import ImageDataset
    selected_device = resolve_sift_device(device, pycolmap.has_cuda, torch.cuda.is_available())
    print(f'  SIFT device: {selected_device} (pycolmap CUDA: {pycolmap.has_cuda})', flush=True)
    options = {**conf['model'].get('options', {}), 'normalization': pycolmap.Normalization.L2}
    from .utils.batching import prefetched, compatible_batches
    from .sift_worker import extract_batches
    dataset = ImageDataset(scene, conf['preprocessing'], names)
    batches = compatible_batches(enumerate(prefetched(dataset, loader_workers, prefetch)), batch_size,
                                 lambda item: None)
    print(f'  SIFT: workers={workers}, images/job={batch_size}, max_in_flight={max_in_flight}', flush=True)
    results = extract_batches(batches, options, selected_device, threads, workers, max_in_flight)
    completed = 0
    for batch in results:
        for index, keypoints, descriptors, scales, original_size in batch:
            with h5py.File(path, 'a') as features:
                group = features.create_group(names[index])
                points = group.create_dataset('keypoints', data=keypoints.astype(np.float32))
                points.attrs['uncertainty'] = float(scales.mean())
                group.create_dataset('descriptors', data=descriptors.T.astype(np.float16))
                group.create_dataset('scores', data=np.zeros(len(keypoints), dtype=np.float16))
                group.create_dataset('image_size', data=original_size)
            completed += 1
            emit_progress('SIFT extraction', completed, len(dataset), 'images', work=True)


def ensure_features(conf, scene, names, path, global_features=False, device='cpu', threads=1, batch_size=1, loader_workers=1, prefetch=4, workers=1, max_in_flight=1):
    import h5py
    from hloc import extract_features
    required = ('global_descriptor', 'image_size') if global_features else (
        'keypoints', 'descriptors', 'scores', 'image_size')
    missing = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'a') as features:
        for name in names:
            valid = name in features and all(key in features[name] for key in required)
            if valid:
                group = features[name]
                valid = group['image_size'].shape == (2,)
                if global_features:
                    valid = valid and group['global_descriptor'].ndim == 1 and group['global_descriptor'].size > 0
                else:
                    n = len(group['keypoints'])
                    valid = valid and group['keypoints'].shape == (n, 2) and group['descriptors'].ndim == 2
                    valid = valid and group['descriptors'].shape[1] == n and group['scores'].shape == (n,)
            if not valid:
                if name in features:
                    del features[name]
                missing.append(name)
    if missing:
        emit_progress('Global features' if global_features else 'Local features', unit='images', work=True)
        print(f'  Features: {len(missing)} pendientes / {len(names)}', flush=True)
        if conf['model']['name'] == 'dog' and conf['model'].get('descriptor') == 'sift':
            extract_sift(conf, scene, missing, path, device=device, threads=threads, batch_size=batch_size, loader_workers=loader_workers, prefetch=prefetch, workers=workers, max_in_flight=max_in_flight)
        else:
            runtime = dict(batch_size=batch_size, loader_workers=loader_workers, prefetch=prefetch, workers=workers, max_in_flight=max_in_flight)
            extract_features.main(conf, scene, image_list=missing, feature_path=path, **runtime)
    else:
        print(f'  [CACHE] {path.name}', flush=True)
    # HLoc escribe cada imagen por separado; cualquier excepción detiene el pipeline.
    with h5py.File(path, 'r') as features:
        if any(name not in features or not all(k in features[name] for k in required) for name in names):
            raise RuntimeError(f'Features incompletas: {path}')


def retrieval_pairs(features_path, names, top_k, batch_size=32, database_batch=512):
    """Top-k por consulta dentro de la coalición, sin matriz N x N en RAM/GPU."""
    import h5py
    import numpy as np
    import torch
    names = sorted(names)
    k = min(top_k, len(names) - 1)
    if k <= 0:
        return []
    pairs = set()
    with h5py.File(features_path, 'r') as features:
        def read(batch):
            array = np.stack([features[name]['global_descriptor'][...] for name in batch]).astype(np.float32)
            if not np.isfinite(array).all():
                raise ValueError('Descriptor global no finito')
            vectors = torch.from_numpy(array)
            return torch.nn.functional.normalize(vectors, dim=1)
        for q0 in range(0, len(names), batch_size):
            query = read(names[q0:q0 + batch_size])
            best_scores = torch.full((len(query), k), -float('inf'))
            best_indices = torch.full((len(query), k), -1, dtype=torch.long)
            for d0 in range(0, len(names), database_batch):
                database = read(names[d0:d0 + database_batch])
                scores = query @ database.T
                qi = torch.arange(q0, q0 + len(query))[:, None]
                di = torch.arange(d0, d0 + len(database))[None, :]
                scores[qi == di] = -float('inf')
                indices = di.expand(len(query), -1)
                joined_scores = torch.cat([best_scores, scores], dim=1)
                joined_indices = torch.cat([best_indices, indices], dim=1)
                # Estable: en empates, conservar el orden lexicográfico de imágenes.
                order = torch.argsort(joined_scores, dim=1, descending=True, stable=True)[:, :k]
                best_scores = joined_scores.gather(1, order)
                best_indices = joined_indices.gather(1, order)
            for row, candidates in enumerate(best_indices.tolist()):
                for idx in candidates:
                    if idx >= 0:
                        pairs.add(tuple(sorted((names[q0 + row], names[idx]))))
    return sorted(pairs)


def sequential_pairs(names, window):
    """Unir cada muestra con las siguientes N del mismo video/carpeta.

    Acepta los nombres exportados video_mp4__000001.jpg y, en general,
    prefijo + índice numérico final. Ordena por índice, no lexicográficamente.
    """
    if window < 0:
        raise ValueError('La ventana secuencial no puede ser negativa')
    if window == 0:
        return []
    sequences = {}
    for name in sorted(set(names)):
        path = Path(name)
        match = re.fullmatch(r'(.*?)([0-9]+)', path.stem)
        if match is None:
            raise ValueError(f'No se puede inferir la secuencia de {name}; usa nombres con índice numérico final')
        prefix, index = match.groups()
        sequences.setdefault((path.parent.as_posix(), prefix), []).append((int(index), name))
    pairs = set()
    for sequence in sequences.values():
        sequence.sort()
        if len({index for index, _ in sequence}) != len(sequence):
            raise ValueError(f'Índices duplicados en la secuencia {sequence[0][1]}')
        for position, (_, name) in enumerate(sequence):
            for _, other in sequence[position + 1:position + 1 + window]:
                pairs.add(tuple(sorted((name, other))))
    return sorted(pairs)


def ensure_matches(conf, pairs, features_path, matches_path, device, batch_size=1, workers=1,
                   loader_workers=1, prefetch=4, max_in_flight=1):
    """Una inferencia y escritura por vez; recicla pares, incluidos matches vacíos."""
    import h5py
    import numpy as np
    import torch
    from hloc import match_features, matchers
    from hloc.utils.base_model import dynamic_load
    from hloc.utils.parsers import names_to_pair
    missing = []
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(matches_path, 'a') as matches, h5py.File(features_path, 'r') as features:
        for a, b in pairs:
            key = names_to_pair(a, b)
            count = len(features[a]['keypoints'])
            valid = key in matches and all(k in matches[key] for k in ('matches0', 'matching_scores0'))
            if valid:
                valid = all(matches[key][k].shape == (count,) for k in ('matches0', 'matching_scores0'))
            if not valid:
                if key in matches:
                    del matches[key]
                missing.append((a, b))
    if not missing:
        print(f'  [CACHE] matches: {len(pairs)} pares', flush=True)
        return
    emit_progress('Matching', 0, len(missing), 'pairs', work=True)
    print(f'  Matching: {len(missing)} nuevos / {len(pairs)} pares', flush=True)
    from .utils.batching import prefetched, compatible_batches, inference_jobs
    if batch_size > 1 and conf['model']['name'] not in ('nearest_neighbor', 'superglue'):
        raise ValueError('This matcher requires batch_size=1; independent workers are supported')
    dataset = match_features.FeaturePairsDataset(missing, features_path, features_path)
    batches = compatible_batches(enumerate(prefetched(dataset, loader_workers, prefetch)), batch_size,
        lambda item: tuple((k, tuple(v.shape)) for k, v in sorted(item[1].items())))
    import threading
    model_initialization = threading.Lock()
    def infer(model, batch):
        data = torch.utils.data.default_collate([item[1] for item in batch])
        count0, count1 = data['keypoints0'].shape[1], data['keypoints1'].shape[1]
        if min(count0, count1) == 0:
            pred = {'matches0': torch.full((len(batch), count0), -1),
                    'matching_scores0': torch.zeros(len(batch), count0)}
        else:
            data = {k: v if k.startswith('image') else v.to(device) for k, v in data.items()}
            # Lazily build the model: empty pairs do not require weights.
            if not model:
                with model_initialization:
                    model.append(dynamic_load(matchers, conf['model']['name'])(deepcopy(conf['model'])).eval().to(device))
            pred = model[0](data)
        return [(idx, {k: pred[k][row:row+1].detach().cpu()
                       for k in ('matches0', 'matching_scores0')}) for row, (idx, _) in enumerate(batch)]
    jobs = inference_jobs(list, infer, batches, workers, max_in_flight, device)
    print(f'  Matching: batch_size={batch_size}, workers={workers}, max_in_flight={max_in_flight}', flush=True)
    completed = 0
    for results in jobs:
        for index, pred in results:
            match_features.writer_fn((names_to_pair(*missing[index]), pred), matches_path)
            completed += 1
            emit_progress('Matching', completed, len(missing), 'pairs', work=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reconstruct(scene, names, pairs_path, features, matches, folder, threads, seed, camera_mode):
    with scene_lock(folder / '.sfm.lock', wait=True):
        return _reconstruct(scene, names, pairs_path, features, matches, folder, threads, seed, camera_mode)


def _reconstruct(scene, names, pairs_path, features, matches, folder, threads, seed, camera_mode):
    import pycolmap
    from hloc import reconstruction
    report = folder / 'result.json'
    if report.exists():
        result = json.loads(report.read_text())
        if result['status'] == 'no_model':
            return result
        model = folder / result.get('model_path', '')
        if result['status'] == 'complete' and all((model / f).is_file() for f in ('cameras.bin', 'images.bin', 'points3D.bin')):
            # Cargar también valida que el modelo cacheado sea legible.
            pycolmap.Reconstruction(model)
            print(f'  [CACHE] SfM {folder.name}', flush=True)
            return result
    folder.mkdir(parents=True, exist_ok=True)
    # No borrar un modelo previo o restos de un intento fallido.
    emit_progress('Geometric verification / SfM', work=True)
    attempt = folder / f'attempt_{time.time_ns()}'
    result = {'status': 'running', 'input_images': len(names), 'model_path': attempt.name}
    save_json(report, result)
    try:
        model = None
        if len(names) >= 2 and pairs_path.read_text().strip():
            model = reconstruction.main(
                attempt, scene, pairs_path, features, matches, image_list=names,
                camera_mode=getattr(pycolmap.CameraMode, camera_mode),
                mapper_options={'num_threads': threads, 'random_seed': seed},
            )
        result.update(status='no_model' if model is None else 'complete',
                      registered_images=0 if model is None else model.num_reg_images(),
                      points3D=0 if model is None else model.num_points3D())
    except Exception as exc:
        result.update(status='error', error=str(exc))
        save_json(report, result)
        raise
    save_json(report, result)
    return result


def run_scene(scene, args):
    images, snapshot = inventory(scene, args.platforms)
    combos = coalitions(args.platforms)
    print(f'\n{scene.name}: {sum(map(len, images.values()))} imágenes; {len(combos)} coaliciones', flush=True)
    for combo in combos:
        print(f'  {"+".join(combo)}: {sum(len(images[p]) for p in combo)} imágenes')
    print(f'  Configuraciones: {", ".join(args.configs)} | global: {args.global_feature}')
    if args.sequential_window:
        # Validar nombres incluso en dry-run y antes de cargar modelos.
        sequential_count = len(sequential_pairs(
            [name for group in images.values() for name in group], args.sequential_window))
        print(f'  Secuenciales: ventana {args.sequential_window}, {sequential_count} pares en escena completa')
    if args.dry_run:
        return
    import torch
    import pycolmap
    from hloc import extract_features, match_features
    names = sorted(name for group in images.values() for name in group)
    root = args.output_root.resolve() / scene.name / snapshot
    # Configuración + implementación + versiones: no reutilizar resultados incompatibles.
    code_root = REPO_ROOT
    environment = cache_environment(torch.__version__, pycolmap.__version__)
    def config_id(conf):
        return digest({'environment': environment, 'config': conf})
    # Detectar dependencias faltantes antes de iniciar una extracción larga.
    import importlib.util
    if any(preset.startswith('sp-') for preset in args.configs):
        superpoint = code_root / 'third_party/SuperGluePretrainedNetwork/models/superpoint.py'
        if not superpoint.is_file():
            raise RuntimeError('Falta SuperGluePretrainedNetwork. Ejecuta: git submodule update --init --recursive third_party/SuperGluePretrainedNetwork')
    if 'sp-lg' in args.configs and importlib.util.find_spec('lightglue') is None:
        raise RuntimeError('Falta LightGlue; instala el paquete del repositorio cvg/LightGlue para usar sp-lg')
    with scene_lock(root.parent / '.pipeline.lock'):
        root.mkdir(parents=True, exist_ok=True)
        save_json(root / 'dataset.json', {'scene': str(scene), 'snapshot': snapshot, 'images': images})
        global_conf = deepcopy(extract_features.confs[args.global_feature])
        global_id = config_id(global_conf)
        global_path = root / 'features' / f'global-{global_id}.h5'
        save_json(global_path.with_suffix('.json'), global_conf)
        ensure_features(global_conf, scene, names, global_path, True, **execution_options(args, 'global'))
        local_paths = {}
        for preset in args.configs:
            local_key, matcher_key = PRESETS[preset]
            dense = local_key is None
            conf = None
            if dense:
                from .dense import dense_config, ensure_dense_raw, assemble_dense
                matcher_conf = dense_config(preset)
                # Separar checkpoints y versiones del adaptador denso.
                matcher_code = hashlib.sha256((code_root / 'hloc/matchers/mast3r.py').read_bytes()).hexdigest()
                local_id = config_id({'dense': matcher_conf, 'matcher_code': matcher_code})
                path = None
            else:
                conf = deepcopy(extract_features.confs[local_key])
                conf['preprocessing']['resize_max'] = args.resize_max
                if local_key == 'sift':
                    conf['model']['descriptor'] = 'sift'
                    conf['model'].setdefault('options', {})['max_num_features'] = args.max_keypoints
                else:
                    conf['model']['max_keypoints'] = args.max_keypoints
                local_id = config_id(conf)
                path = root / 'features' / f'local-{local_id}.h5'
                if local_id not in local_paths:
                    save_json(path.with_suffix('.json'), conf)
                    sift_device = args.sift_device or getattr(args, 'requested_device', args.device)
                    ensure_features(conf, scene, names, path, device=sift_device, threads=args.threads, **execution_options(args, 'local'))
                    local_paths[local_id] = path
            if args.until == 'features':
                if dense:
                    print('  MASt3R genera keypoints durante matching; --until features solo extrae globales')
                continue
            retrieval_id = digest({'global': global_id, 'top_k': args.top_k,
                                   'sequential_window': args.sequential_window})
            pair_root = root / 'pairs' / retrieval_id
            coalition_pairs = {}
            union = set()
            for combo in combos:
                members = sorted(name for p in combo for name in images[p])
                label = '+'.join(combo)
                pair_path = pair_root / f'{label}.txt'
                if pair_path.exists():
                    pairs = [tuple(line.split()) for line in pair_path.read_text().splitlines() if line]
                    allowed = set(members)
                    if any(len(pair) != 2 or not set(pair) <= allowed for pair in pairs):
                        raise ValueError(f'Caché de pares inválida: {pair_path}')
                else:
                    pairs = retrieval_pairs(global_path, members, args.top_k, args.query_batch, args.database_batch)
                    pairs = sorted(set(pairs) | set(sequential_pairs(members, args.sequential_window)))
                    pair_path.parent.mkdir(parents=True, exist_ok=True)
                    temp = pair_path.with_suffix('.tmp')
                    temp.write_text(''.join(f'{a} {b}\n' for a, b in pairs))
                    temp.replace(pair_path)
                coalition_pairs[combo] = (members, pair_path)
                union.update(pairs)
            if args.until == 'pairs':
                continue
            if dense:
                match_id = local_id
                match_path = root / 'dense_raw' / f'{match_id}.h5'
                save_json(match_path.with_suffix('.json'), matcher_conf)
                ensure_dense_raw(matcher_conf, scene, sorted(union), match_path, args.device, **execution_options(args, 'dense'))
            else:
                matcher_conf = deepcopy(match_features.confs[matcher_key])
                match_id = digest({'local': local_id, 'matcher': config_id(matcher_conf)})
                match_path = root / 'matches' / f'{match_id}.h5'
                save_json(match_path.with_suffix('.json'), {'local': local_id, 'matcher': matcher_conf})
                ensure_matches(matcher_conf, sorted(union), path, match_path, args.device, **execution_options(args, 'matching'))
            if args.until == 'matches' and not dense:
                continue
            run_id = digest({'matches': match_id, 'retrieval': retrieval_id, 'seed': args.seed,
                             'threads': args.threads, 'camera_mode': args.camera_mode,
                             **({'dense_max_keypoints': args.max_keypoints} if dense else {})})
            run_root = root / 'reconstructions' / f'{preset}-{run_id}'
            save_json(run_root / 'config.json', {'preset': preset, 'global': global_conf,
                      'local': conf, 'matcher': matcher_conf, 'top_k': args.top_k,
                      'sequential_window': args.sequential_window, 'max_keypoints': args.max_keypoints,
                      'seed': args.seed, 'threads': args.threads, 'camera_mode': args.camera_mode,
                      'environment': environment})
            summary = {}
            for combo, (members, pair_path) in coalition_pairs.items():
                label = '+'.join(combo)
                coalition_features, coalition_matches = path, match_path
                if dense:
                    pairs = [tuple(line.split()) for line in pair_path.read_text().splitlines() if line]
                    coalition_features, coalition_matches = assemble_dense(
                        match_path, pairs, members, root / 'dense_assembled' / match_id,
                        args.max_keypoints, matcher_conf['cell_size'])
                if args.until == 'matches':
                    continue
                print(f'[{scene.name}/{preset}] SfM {label}', flush=True)
                summary[label] = reconstruct(scene, members, pair_path, coalition_features, coalition_matches,
                                             run_root / label, args.threads, args.seed, args.camera_mode)
                save_json(run_root / 'summary.json', summary)
        print(f'Resultados/caché: {root}', flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames-root', type=Path, default=Path('frames'))
    parser.add_argument('--scenes', nargs='+', help='Por defecto: todas las escenas bajo frames/')
    parser.add_argument('--platforms', nargs='+', default=['Car', 'Drone', 'Pedestrian'])
    parser.add_argument('--configs', nargs='+', choices=PRESETS, default=['sift', 'sp-sg'])
    parser.add_argument('--global-feature', choices=['netvlad', 'openibl', 'megaloc', 'dir'], default='netvlad')
    parser.add_argument('--output-root', type=Path, default=Path('outputs/coalitions'))
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--sequential-window', type=int, default=0,
                        help='Añadir pares con las siguientes N muestras de cada video; 0 desactiva')
    parser.add_argument('--local-batch-size', type=int, default=1)
    parser.add_argument('--dense-batch-size', type=int, default=1)
    parser.add_argument('--loader-workers', type=int, default=1)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--query-batch', type=int, default=32)
    parser.add_argument('--database-batch', type=int, default=512)
    parser.add_argument('--resize-max', type=int, default=1024)
    parser.add_argument('--max-keypoints', type=int, default=4096)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--sift-device', choices=['auto', 'cpu', 'cuda'],
                        help='Dispositivo SIFT; por defecto hereda --device. CUDA requiere pycolmap con CUDA')
    parser.add_argument('--camera-mode', choices=['PER_FOLDER', 'PER_IMAGE'], default='PER_FOLDER')
    parser.add_argument('--until', choices=['features', 'pairs', 'matches', 'sfm'], default='sfm')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def validate(args, parser):
    """Valida argumentos y devuelve la lista de escenas."""
    args.requested_device = args.device
    for key in ('local_batch_size', 'dense_batch_size', 'prefetch', 'loader_workers'):
        value = getattr(args, key)
        if type(value) is not int or value < (0 if key == 'loader_workers' else 1):
            parser.error(f'{key} has an invalid value')
    validate_execution(getattr(args, 'execution', {}))
    if execution_options(args, 'global')['batch_size'] > 1 and args.global_feature != 'netvlad':
        parser.error('This global extractor requires execution.global.batch_size=1; workers may be increased')
    if 'sp-lg' in args.configs and execution_options(args, 'matching')['batch_size'] > 1:
        parser.error('LightGlue requires execution.matching.batch_size=1; workers may be increased')
    for name in ('top_k', 'query_batch', 'database_batch', 'resize_max', 'max_keypoints', 'threads'):
        if getattr(args, name) < 1:
            parser.error(f'{name} debe ser positivo')
    if args.sequential_window < 0:
        parser.error('sequential-window debe ser >= 0')
    if args.max_keypoints > 32767:
        parser.error('max-keypoints debe ser <= 32767 (formato de matches HLoc)')
    if len(set(args.platforms)) != len(args.platforms) or any(Path(p).name != p or p in ('.', '..') for p in args.platforms):
        parser.error('Plataformas deben ser nombres de carpetas únicos')
    args.configs = list(dict.fromkeys(args.configs))
    root = args.frames_root.resolve()
    if not root.is_dir():
        parser.error(f'No existe {root}')
    if args.scenes and any(Path(s).name != s or s in ('.', '..') for s in args.scenes):
        parser.error('scenes debe contener nombres de carpetas')
    scenes = [root / s for s in args.scenes] if args.scenes else sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if not scenes:
        parser.error('No se encontraron escenas')
    return scenes


def setup_runtime(args):
    """Limita hilos/núcleos, resuelve el dispositivo y fija semillas."""
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = str(args.threads)
    if hasattr(os, 'sched_getaffinity'):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available[:args.threads])
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    import cv2
    import numpy as np
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA no disponible')
    args.device = 'cuda' if args.device != 'cpu' and torch.cuda.is_available() else 'cpu'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def main():
    parser = build_parser()
    args = parser.parse_args()
    scenes = validate(args, parser)
    if not args.dry_run:
        setup_runtime(args)
    for scene in scenes:
        run_scene(scene, args)


if __name__ == '__main__':
    main()
