"""Tareas atómicas: pares, banco por experimento y worker de SfM."""
import hashlib
import importlib.util
import json
import os
from copy import deepcopy
from pathlib import Path

from coalition_pipeline import (PRESETS, cache_environment, coalitions, digest,
    ensure_features, ensure_matches, inventory, reconstruct, retrieval_pairs,
    save_json, scene_lock, sequential_pairs)

CODE_ROOT = Path(__file__).resolve().parent


def context(scene, args):
    images, snapshot = inventory(scene, args.platforms)
    names = sorted(n for g in images.values() for n in g)
    root = args.output_root.resolve() / scene.name / snapshot
    return images, snapshot, names, root


def environment():
    import torch
    import pycolmap
    return cache_environment(torch.__version__, pycolmap.__version__)


def config_id(env, conf):
    return digest({'environment': env, 'config': conf})


def read_pairs(path):
    return [tuple(l.split()) for l in path.read_text().splitlines() if l]


def write_pairs(path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(''.join(f'{a} {b}\n' for a, b in pairs))
    tmp.replace(path)


def check_dependencies(presets):
    if any(c.startswith('sp-') for c in presets):
        sp = CODE_ROOT / 'third_party/SuperGluePretrainedNetwork/models/superpoint.py'
        if not sp.is_file():
            raise RuntimeError('Falta SuperGluePretrainedNetwork. Ejecuta: git submodule update '
                               '--init --recursive third_party/SuperGluePretrainedNetwork')
    if 'sp-lg' in presets and importlib.util.find_spec('lightglue') is None:
        raise RuntimeError('Falta LightGlue; instala cvg/LightGlue para usar sp-lg')


def stage_pairs(scene, args):
    """Globales + pares por coalición de UN experimento."""
    from hloc import extract_features
    images, snapshot, names, root = context(scene, args)
    env = environment()
    with scene_lock(root.parent / '.pairs.lock'):
        root.mkdir(parents=True, exist_ok=True)
        save_json(root / 'dataset.json', {'scene': str(scene), 'snapshot': snapshot, 'images': images})
        gconf = deepcopy(extract_features.confs[args.global_feature])
        gid = config_id(env, gconf)
        gpath = root / 'features' / f'global-{gid}.h5'
        save_json(gpath.with_suffix('.json'), gconf)
        ensure_features(gconf, scene, names, gpath, True)

        rid = digest({'global': gid, 'top_k': args.top_k, 'sequential_window': args.sequential_window})
        pair_root = root / 'pairs' / rid
        index = {'global_conf': gconf, 'retrieval_id': rid, 'coalitions': {}}
        union = set()
        for combo in coalitions(args.platforms):
            members = sorted(n for p in combo for n in images[p])
            label = '+'.join(combo)
            pair_path = pair_root / f'{label}.txt'
            if pair_path.exists():
                pairs = read_pairs(pair_path)
                allowed = set(members)
                if any(len(p) != 2 or not set(p) <= allowed for p in pairs):
                    raise ValueError(f'Caché de pares inválida: {pair_path}')
            else:
                pairs = retrieval_pairs(gpath, members, args.top_k, args.query_batch, args.database_batch)
                pairs = sorted(set(pairs) | set(sequential_pairs(members, args.sequential_window)))
                write_pairs(pair_path, pairs)
            index['coalitions'][label] = {'members': members, 'pairs': str(pair_path)}
            union.update(pairs)
        union_path = pair_root / 'union.txt'
        write_pairs(union_path, sorted(union))
        index['union'] = str(union_path)
        save_json(root / f'bank_index-{args.experiment}.json', index)


def bank_preset(scene, args, preset):
    """Locales + matches de UN experimento; escribe jobs/<experimento>.json."""
    from hloc import extract_features, match_features
    _, _, names, root = context(scene, args)
    index_path = root / f'bank_index-{args.experiment}.json'
    if not index_path.is_file():
        raise RuntimeError(f'Falta {index_path}; corre primero la fase de pares')
    index = json.loads(index_path.read_text())
    env = environment()
    local_key, matcher_key = PRESETS[preset]
    dense = local_key is None
    union = read_pairs(Path(index['union']))
    conf, features = None, None

    with scene_lock(root.parent / f'.bank-{preset}.lock'):
        if dense:
            from coalition_dense import dense_config, ensure_dense_raw
            matcher_conf = dense_config(preset)
            code = hashlib.sha256((CODE_ROOT / 'hloc/matchers/mast3r.py').read_bytes()).hexdigest()
            match_id = config_id(env, {'dense': matcher_conf, 'matcher_code': code})
            match_path = root / 'dense_raw' / f'{match_id}.h5'
            save_json(match_path.with_suffix('.json'), matcher_conf)
            ensure_dense_raw(matcher_conf, scene, union, match_path, args.device)
        else:
            conf = deepcopy(extract_features.confs[local_key])
            conf['preprocessing']['resize_max'] = args.resize_max
            if local_key == 'sift':
                conf['model']['descriptor'] = 'sift'
                conf['model'].setdefault('options', {})['max_num_features'] = args.max_keypoints
            else:
                conf['model']['max_keypoints'] = args.max_keypoints
            local_id = config_id(env, conf)
            features = root / 'features' / f'local-{local_id}.h5'
            save_json(features.with_suffix('.json'), conf)
            ensure_features(conf, scene, names, features,
                            device=args.sift_device or args.requested_device, threads=args.threads)
            matcher_conf = deepcopy(match_features.confs[matcher_key])
            match_id = digest({'local': local_id, 'matcher': config_id(env, matcher_conf)})
            match_path = root / 'matches' / f'{match_id}.h5'
            save_json(match_path.with_suffix('.json'), {'local': local_id, 'matcher': matcher_conf})
            ensure_matches(matcher_conf, union, features, match_path, args.device)

        base = {'matches': match_id, 'retrieval': index['retrieval_id'], 'seed': args.seed,
                'camera_mode': args.camera_mode,
                **({'dense_max_keypoints': args.max_keypoints} if dense else {})}
        jobs = []
        for label, c in index['coalitions'].items():
            job = {'label': label, 'members': c['members'], 'pairs': c['pairs']}
            if dense:  # el ensamblado ocurre en la etapa SfM (CPU)
                job['dense'] = {'raw': str(match_path),
                                'folder': str(root / 'dense_assembled' / match_id),
                                'max_keypoints': args.max_keypoints,
                                'cell_size': matcher_conf['cell_size']}
            else:
                job.update(features=str(features), matches=str(match_path))
            jobs.append(job)
        save_json(root / 'jobs' / f'{args.experiment}.json', {
            'scene': str(scene), 'preset': preset, 'experiment': args.experiment,
            'run_base': base, 'jobs': jobs,
            'config': {'preset': preset, 'experiment': args.experiment,
                       'global': index['global_conf'], 'local': conf, 'matcher': matcher_conf,
                       'top_k': args.top_k, 'sequential_window': args.sequential_window,
                       'max_keypoints': args.max_keypoints, 'seed': args.seed,
                       'camera_mode': args.camera_mode, 'environment': env}})


def sfm_worker(job, threads):
    try:
        if 'dense' in job:
            from coalition_dense import assemble_dense
            d = job['dense']
            features, matches = assemble_dense(
                Path(d['raw']), read_pairs(Path(job['pairs'])), job['members'],
                Path(d['folder']), d['max_keypoints'], d['cell_size'])
        else:
            features, matches = Path(job['features']), Path(job['matches'])
        res = reconstruct(Path(job['scene']), job['members'], Path(job['pairs']),
                          features, matches, Path(job['run_root']) / job['label'],
                          threads, job['seed'], job['camera_mode'])
    except Exception as exc:
        res = {'status': 'error', 'error': str(exc)}
    return job['run_root'], job['label'], res