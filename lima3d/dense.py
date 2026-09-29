"""Adaptador MASt3R: inferencias por par compartidas, tracks aislados por coalición."""
from .utils.progress import report as emit_progress

from collections import defaultdict
from pathlib import Path

DENSE_VERSION = 1


def dense_config(preset):
    weights = {'mast3r': 'mast3r', 'mast3r-aerialmd': 'aerialmd'}[preset]
    return {'model': {'name': 'mast3r', 'weights': weights},
            'preprocessing': {'grayscale': False, 'resize_max': 512, 'dfactor': 16},
            'input_range': [-1, 1], 'cell_size': 1.0, 'adapter_version': DENSE_VERSION}


def raw_valid(group):
    if not all(k in group for k in ('keypoints0', 'keypoints1', 'scores')):
        return False
    n = group['scores'].shape
    return len(n) == 1 and group['keypoints0'].shape == group['keypoints1'].shape == (n[0], 2)


def ensure_dense_raw(conf, scene, pairs, path, device):
    import h5py
    import numpy as np
    import torch
    from hloc import matchers
    from hloc.match_dense import ImagePairDataset
    from hloc.utils.base_model import dynamic_load
    from hloc.utils.parsers import names_to_pair
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = []
    with h5py.File(path, 'a') as handle:
        for pair in sorted(set(pairs)):
            key = names_to_pair(*pair)
            if key not in handle or not raw_valid(handle[key]):
                if key in handle:
                    del handle[key]
                pending.append(pair)
    if not pending:
        print(f'  [CACHE] MASt3R raw: {len(pairs)} pares', flush=True)
        return
    emit_progress('MASt3R inference', 0, len(pending), 'pairs', work=True)
    print(f'  MASt3R {conf["model"]["weights"]}: {len(pending)} pares nuevos', flush=True)
    model = dynamic_load(matchers, 'mast3r')(conf['model']).eval().to(device)
    dataset = ImagePairDataset(scene, conf['preprocessing'], pending)
    with torch.inference_mode():
        for index in range(len(dataset)):
            image0, image1, scale0, scale1, name0, name1 = dataset[index]
            # HLoc entrega [0,1]; la implementación MASt3R original usa [-1,1].
            pred = model({'image0': (image0[None].to(device) * 2 - 1),
                          'image1': (image1[None].to(device) * 2 - 1)})
            k0 = (pred['keypoints0'].detach().cpu().numpy() + 0.5) * scale0 - 0.5
            k1 = (pred['keypoints1'].detach().cpu().numpy() + 0.5) * scale1 - 0.5
            scores = pred['scores'].detach().cpu().numpy()
            valid = np.isfinite(k0).all(1) & np.isfinite(k1).all(1) & np.isfinite(scores)
            with h5py.File(path, 'a') as handle:
                group = handle.create_group(names_to_pair(name0, name1))
                group.create_dataset('keypoints0', data=k0[valid].astype(np.float32))
                group.create_dataset('keypoints1', data=k1[valid].astype(np.float32))
                group.create_dataset('scores', data=scores[valid].astype(np.float32))
                handle.flush()
            del pred, image0, image1
            if (index + 1) % 10 == 0 or index + 1 == len(dataset):
                print(f'  MASt3R {index + 1}/{len(dataset)}', flush=True)
            emit_progress('MASt3R inference', index + 1, len(dataset), 'pairs', work=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def assemble_dense(raw_path, pairs, names, folder, max_keypoints, cell_size=1.0):
    from .utils.io import digest, scene_lock
    identifier = digest({'pairs': sorted(set(pairs)), 'names': sorted(set(names)),
                         'max_keypoints': max_keypoints, 'cell_size': cell_size,
                         'adapter': DENSE_VERSION})
    with scene_lock(folder / identifier / '.assembly.lock', wait=True):
        return _assemble_dense(raw_path, pairs, names, folder, max_keypoints, cell_size)


def _assemble_dense(raw_path, pairs, names, folder, max_keypoints, cell_size=1.0):
    """Cuantiza correspondencias SOLO de esta coalición; nunca usa pares externos.

    Una imagen por vez en memoria. Cada celda de 1 px es un keypoint; se
    prioriza la suma de confianza si hay más puntos que max_keypoints.
    """
    import h5py
    import numpy as np
    from hloc.utils.parsers import names_to_pair
    from .utils.io import digest, save_json
    if max_keypoints < 1 or cell_size <= 0:
        raise ValueError('max_keypoints y cell_size deben ser positivos')
    pairs = sorted(set(pairs))
    names = sorted(set(names))
    allowed = set(names)
    if any(not set(pair) <= allowed for pair in pairs):
        raise ValueError('Pares fuera de la coalición')
    specification = {'pairs': pairs, 'names': names, 'max_keypoints': max_keypoints,
                     'cell_size': cell_size, 'adapter': DENSE_VERSION}
    identifier = digest(specification)
    folder = folder / identifier
    folder.mkdir(parents=True, exist_ok=True)
    features, matches = folder / 'features.h5', folder / 'matches.h5'
    marker = folder / 'complete.json'
    if marker.exists() and features.exists() and matches.exists():
        with h5py.File(features) as f, h5py.File(matches) as m:
            valid = all(name in f and 'keypoints' in f[name] for name in names)
            valid = valid and all(names_to_pair(*p) in m and
                    all(k in m[names_to_pair(*p)] for k in ('matches0', 'matching_scores0')) for p in pairs)
        if valid:
            print(f'  [CACHE] MASt3R ensamblado: {identifier}', flush=True)
            return features, matches
    emit_progress('Dense track assembly', work=True)
    adjacency = defaultdict(list)
    for a, b in pairs:
        adjacency[a].append((names_to_pair(a, b), 'keypoints0'))
        adjacency[b].append((names_to_pair(a, b), 'keypoints1'))
    feature_tmp, match_tmp = folder / 'features.tmp.h5', folder / 'matches.tmp.h5'
    with h5py.File(raw_path) as raw, h5py.File(feature_tmp, 'w') as f:
        for name in names:
            totals = defaultdict(float)
            for pair_key, side in adjacency[name]:
                group = raw[pair_key]
                cells = np.rint(group[side][...] / cell_size).astype(np.int64)
                for cell, score in zip(cells, group['scores'][...]):
                    totals[tuple(cell)] += float(score)
            chosen = sorted(totals, key=lambda cell: (-totals[cell], cell))[:max_keypoints]
            chosen.sort()
            points = np.asarray(chosen, dtype=np.float32).reshape(-1, 2) * cell_size
            group = f.create_group(name)
            keypoints = group.create_dataset('keypoints', data=points)
            keypoints.attrs['uncertainty'] = cell_size
            group.create_dataset('scores', data=np.asarray([totals[cell] for cell in chosen], dtype=np.float32))
    with h5py.File(raw_path) as raw, h5py.File(feature_tmp) as f, h5py.File(match_tmp, 'w') as m:
        for a, b in pairs:
            def lookup(name):
                cells = np.rint(f[name]['keypoints'][...] / cell_size).astype(np.int64)
                return {tuple(cell): i for i, cell in enumerate(cells)}
            ids0, ids1 = lookup(a), lookup(b)
            group = raw[names_to_pair(a, b)]
            cells0 = np.rint(group['keypoints0'][...] / cell_size).astype(np.int64)
            cells1 = np.rint(group['keypoints1'][...] / cell_size).astype(np.int64)
            scores = group['scores'][...]
            matches0 = np.full(len(ids0), -1, dtype=np.int32)
            confidence = np.zeros(len(ids0), dtype=np.float32)
            used1 = set()
            for row in np.argsort(-scores, kind='stable'):
                i, j = ids0.get(tuple(cells0[row])), ids1.get(tuple(cells1[row]))
                if i is not None and j is not None and matches0[i] < 0 and j not in used1:
                    matches0[i], confidence[i] = j, scores[row]
                    used1.add(j)
            output = m.create_group(names_to_pair(a, b))
            output.create_dataset('matches0', data=matches0)
            output.create_dataset('matching_scores0', data=confidence)
    feature_tmp.replace(features)
    match_tmp.replace(matches)
    save_json(marker, {'id': identifier, 'num_images': len(names), 'num_pairs': len(pairs)})
    return features, matches
