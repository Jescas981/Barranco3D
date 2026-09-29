#!/usr/bin/env python3
"""MVS de COLMAP reanudable, sobre SfM existentes de cualquier matcher."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from run_scene_coalitions import digest, save_json, scene_lock


def model_files(model):
    for extension in ('bin', 'txt'):
        files = [model / (name + '.' + extension) for name in ('cameras', 'images', 'points3D')]
        if all(p.is_file() for p in files):
            return files + [model / (name + '.' + extension) for name in ('rigs', 'frames')
                            if (model / (name + '.' + extension)).is_file()]
    raise ValueError(f'No es un modelo COLMAP completo: {model}')


def model_signature(model):
    hasher = hashlib.sha256()
    for path in model_files(model):
        hasher.update(path.name.encode())
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                hasher.update(chunk)
    return hasher.hexdigest()


def read_names(model):
    import pycolmap
    reconstruction = pycolmap.Reconstruction(model)
    names = sorted(image.name for image in reconstruction.images.values() if image.has_pose)
    if len(names) < 2:
        raise ValueError(f'MVS requiere al menos dos imágenes registradas: {model}')
    return names


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def map_valid(path):
    """Comprobar cabecera y tamaño exacto del mapa float32 COLMAP."""
    if not path.is_file():
        return False
    try:
        with path.open('rb') as f:
            header = bytearray()
            while header.count(b'&') < 3 and len(header) < 128:
                byte = f.read(1)
                if not byte:
                    return False
                header.extend(byte)
        w, h, channels = map(int, header.decode().strip('&').split('&'))
        return min(w, h, channels) > 0 and path.stat().st_size == len(header) + w*h*channels*4
    except (ValueError, UnicodeError):
        return False


def ply_valid(path):
    """Validar cabecera y longitud de la nube PLY de stereo_fusion."""
    if not nonempty(path):
        return False
    sizes = {'float': 4, 'float32': 4, 'double': 8, 'float64': 8,
             'uchar': 1, 'uint8': 1, 'char': 1, 'int8': 1,
             'short': 2, 'ushort': 2, 'int': 4, 'uint': 4}
    try:
        with path.open('rb') as f:
            if f.readline().strip() != b'ply':
                return False
            count = None
            stride = 0
            encoding = None
            for _ in range(100):
                line = f.readline().decode('ascii').strip()
                fields = line.split()
                if not fields:
                    return False
                if fields[0] == 'format':
                    encoding = fields[1]
                elif fields[0] == 'element':
                    if fields[1] != 'vertex':
                        return False
                    count = int(fields[2])
                elif fields[0] == 'property':
                    stride += sizes[fields[1]]
                elif line == 'end_header':
                    if count is None or count < 0 or stride == 0:
                        return False
                    if encoding in ('binary_little_endian', 'binary_big_endian'):
                        return path.stat().st_size == f.tell() + count * stride
                    if encoding == 'ascii':
                        return sum(1 for row in f if row.strip()) == count
                    return False
    except (UnicodeError, ValueError, KeyError, IndexError):
        return False
    return False


def run_mvs(model, images, output=None, *, colmap='colmap', max_image_size=1024,
            cache_gb=2.0, threads=2, gpu_index='0', num_sources=10, dry_run=False):
    model, images = Path(model).resolve(), Path(images).resolve()
    if max_image_size < 1 or threads < 1 or num_sources < 1 or not math.isfinite(cache_gb) or cache_gb <= 0:
        raise ValueError('Tamaños, threads, fuentes y caché deben ser positivos')
    binary = shutil.which(colmap)
    if not binary:
        raise ValueError(f'No se encuentra COLMAP: {colmap}')
    names = read_names(model)
    image_state = []
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Nombre de imagen no relativo: {name}')
        # Los pipelines anteriores usan enlaces de plataformas hacia frames/.
        image = (images / relative).resolve()
        stat = image.stat()
        image_state.append((name, stat.st_size, stat.st_mtime_ns))
    binary_stat = Path(binary).stat()
    config = {'version': 1, 'model': str(model), 'model_hash': model_signature(model),
              'images': str(images), 'image_state': digest(image_state),
              'max_image_size': max_image_size, 'cache_gb': cache_gb, 'threads': threads,
              'gpu_index': gpu_index, 'num_sources': num_sources,
              'colmap': [binary, binary_stat.st_size, binary_stat.st_mtime_ns]}
    output = Path(output).resolve() if output else model.parent / 'mvs' / f'{model.name}-{digest(config)}'
    if output == model or output == images or model.is_relative_to(output) or images.is_relative_to(output):
        raise ValueError('La salida MVS no puede contener ni reemplazar la entrada')
    print(f'MVS: {model} -> {output}', flush=True)
    if dry_run:
        return output
    output.mkdir(parents=True, exist_ok=True)
    with scene_lock(output / '.lock'):
        state_path = output / 'mvs_state.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['config'] != config:
                raise ValueError('La salida tiene otra configuración. Usa otra carpeta MVS')
        else:
            if any(p.name != '.lock' for p in output.iterdir()):
                raise ValueError('Salida MVS existente sin estado: usa una carpeta nueva')
            state = {'config': config, 'steps': {}, 'status': 'running'}
            save_json(state_path, state)
        workspace = output / 'workspace'
        fused = workspace / 'fused.ply'
        undistort = [binary, 'image_undistorter', '--image_path', str(images),
                     '--input_path', str(model), '--output_path', str(workspace),
                     '--output_type', 'COLMAP', '--max_image_size', str(max_image_size),
                     '--num_patch_match_src_images', str(num_sources)]
        stereo = [binary, 'patch_match_stereo', '--workspace_path', str(workspace),
                  '--workspace_format', 'COLMAP', '--PatchMatchStereo.gpu_index', str(gpu_index),
                  '--PatchMatchStereo.max_image_size', str(max_image_size),
                  '--PatchMatchStereo.geom_consistency', '1',
                  '--PatchMatchStereo.cache_size', str(cache_gb)]
        fuse = [binary, 'stereo_fusion', '--workspace_path', str(workspace),
                '--workspace_format', 'COLMAP', '--input_type', 'geometric',
                '--output_path', str(fused), '--StereoFusion.num_threads', str(threads),
                '--StereoFusion.max_image_size', str(max_image_size),
                '--StereoFusion.use_cache', '1', '--StereoFusion.cache_size', str(cache_gb)]
        def undistorted():
            return all(nonempty(workspace / 'images' / name) for name in names) and all(
                nonempty(workspace / 'sparse' / f'{part}.bin') for part in ('cameras','images','points3D')) and nonempty(workspace/'stereo/patch-match.cfg')
        def maps():
            return all(map_valid(workspace / 'stereo' / kind / (name + '.geometric.bin'))
                       for name in names for kind in ('depth_maps', 'normal_maps'))
        validators = {'undistort': undistorted, 'stereo': maps, 'fusion': lambda: ply_valid(fused)}
        steps = [('undistort', undistort), ('stereo', stereo), ('fusion', fuse)]
        # Ejecutar con cuota de CPU local al subproceso, sin cambiar la afinidad del padre.
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
        affinity = sorted(os.sched_getaffinity(0))[:threads] if hasattr(os, 'sched_getaffinity') else None
        def limit_cpu():
            if affinity:
                os.sched_setaffinity(0, affinity)
        dirty = False
        try:
            for step, command in steps:
                valid = validators[step]()
                if not dirty and state['steps'].get(step) == 'complete' and valid:
                    print(f'  [CACHE] MVS {step}', flush=True)
                    continue
                if dirty and step == 'stereo':
                    # Entradas de undistortion cambiaron: no aceptar mapas anteriores.
                    for kind in ('depth_maps', 'normal_maps', 'consistency_graphs'):
                        for path in (workspace/'stereo'/kind).rglob('*.bin'):
                            path.rename(path.with_name(path.name + f'.stale-{time.time_ns()}'))
                elif step == 'stereo':
                    # Un mapa de profundidad aislado no debe hacer que COLMAP
                    # omita regenerar su normal correspondiente, o viceversa.
                    for name in names:
                        for mode in ('photometric', 'geometric'):
                            pair = [workspace/'stereo'/kind/(name + f'.{mode}.bin')
                                    for kind in ('depth_maps', 'normal_maps')]
                            if not all(map_valid(path) for path in pair):
                                for path in pair:
                                    if path.exists():
                                        path.rename(path.with_name(path.name + f'.incomplete-{time.time_ns()}'))
                state['status'] = 'running'
                state['steps'][step] = 'running'
                for later, _ in steps[steps.index((step, command)) + 1:]:
                    state['steps'].pop(later, None)
                save_json(state_path, state)
                print(f'  [RUN] MVS {step}', flush=True)
                with (output / f'{step}.log').open('a') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, preexec_fn=limit_cpu if affinity else None)
                if not validators[step]():
                    raise RuntimeError(f'MVS {step} terminó sin salidas completas; revisa {output / (step + ".log")}')
                state['steps'][step] = 'complete'
                save_json(state_path, state)
                dirty = True
            state['status'] = 'complete'
            state.pop('error', None)
        except BaseException as exc:
            state['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error'
            state['error'] = str(exc)
            raise
        finally:
            save_json(state_path, state)
    print(f'Nube densa: {fused}', flush=True)
    return output


def discover_models(root, presets=None, coalitions=None):
    """Solo resultados terminados; no escanear modelos parciales de intentos fallidos."""
    jobs = []
    for report in sorted(root.rglob('result.json')):
        if 'reconstructions' not in report.parts:
            continue
        result = json.loads(report.read_text())
        if result.get('status') != 'complete':
            continue
        config_path = report.parent.parent / 'config.json'
        if not config_path.exists():
            continue
        conf = json.loads(config_path.read_text())
        if presets and conf['preset'] not in presets:
            continue
        if coalitions and report.parent.name not in coalitions:
            continue
        dataset_path = next((p / 'dataset.json' for p in report.parents if (p/'dataset.json').is_file()), None)
        if dataset_path is None:
            raise ValueError(f'No se encuentra dataset.json para {report}')
        images = Path(json.loads(dataset_path.read_text())['scene'])
        model = report.parent / result['model_path']
        jobs.append((model, images))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--sfm-root', type=Path, help='Buscar reconstrucciones terminadas bajo esta carpeta')
    source.add_argument('--model-path', type=Path, help='Modelo COLMAP explícito (también MASt3R/AerialMD)')
    parser.add_argument('--image-path', type=Path)
    parser.add_argument('--output', type=Path, help='Solo con --model-path; por defecto carpeta MVS independiente')
    parser.add_argument('--configs', nargs='+')
    parser.add_argument('--coalitions', nargs='+')
    parser.add_argument('--colmap', default='colmap')
    parser.add_argument('--max-image-size', type=int, default=1024)
    parser.add_argument('--cache-gb', type=float, default=2.0)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--gpu-index', default='0')
    parser.add_argument('--num-sources', type=int, default=10)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.model_path:
        if not args.image_path:
            parser.error('--model-path requiere --image-path')
        jobs = [(args.model_path, args.image_path)]
    else:
        if args.output or args.image_path:
            parser.error('--output/--image-path requieren --model-path')
        if not args.sfm_root.is_dir():
            parser.error('No existe sfm-root')
        jobs = discover_models(args.sfm_root, args.configs, args.coalitions)
    if not jobs:
        print('No hay reconstrucciones SfM terminadas para MVS')
    for model, images in jobs:
        run_mvs(model, images, args.output, colmap=args.colmap, max_image_size=args.max_image_size,
                cache_gb=args.cache_gb, threads=args.threads, gpu_index=args.gpu_index,
                num_sources=args.num_sources, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
