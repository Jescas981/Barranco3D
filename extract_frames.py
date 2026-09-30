#!/usr/bin/env python3
"""Extrae una escena de videos a un proyecto COLMAP, usando FFmpeg en serie."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.m4v', '.webm', '.mts'}


def probe_video(path):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,avg_frame_rate:format=duration',
         '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    info = json.loads(result.stdout)
    if not info.get('streams'):
        raise ValueError(f'Sin pista de video: {path}')
    stream = info['streams'][0]
    try:
        fps = float(Fraction(stream['avg_frame_rate']))
    except (KeyError, ValueError, ZeroDivisionError):
        fps = None
    return {'width': stream['width'], 'height': stream['height'],
            'source_fps': fps, 'duration_seconds': info.get('format', {}).get('duration')}


def plan_scene(scene, groups, fps):
    videos = sorted(p for p in scene.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    plan = []
    for video in videos:
        relative = video.relative_to(scene)
        if groups and relative.parts[0] not in groups:
            continue
        info = probe_video(video)
        if info['source_fps'] and fps > info['source_fps'] + 1e-6:
            raise ValueError(f'{relative}: FPS solicitados ({fps}) mayores que los originales ({info["source_fps"]:.3f})')
        # Reflejar exactamente las carpetas de plataformas/cámaras de entrada.
        folder = relative.parent
        clip_metadata = None
        manifest = video.parent / 'manifest.json'
        if manifest.is_file():
            data = json.loads(manifest.read_text())
            clip_metadata = next((c for c in data.get('clips', []) if c.get('name') == video.name), None)
        plan.append({'video': str(video), 'relative_video': relative.as_posix(),
                     'image_folder': folder.as_posix(),
                     'prefix': video.name.replace('.', '_'),
                     'clip_metadata': clip_metadata, **info})
    if not plan:
        raise ValueError(f'No hay videos para extraer en {scene}')
    dimensions = {}
    for entry in plan:
        folder = entry['image_folder']
        size = (entry['width'], entry['height'])
        if folder in dimensions and dimensions[folder] != size:
            raise ValueError(f'{folder}: videos con resoluciones distintas; sepáralos en subcarpetas para COLMAP')
        dimensions[folder] = size
    destinations = [(p['image_folder'], p['prefix']) for p in plan]
    if len(destinations) != len(set(destinations)):
        raise ValueError('Hay nombres de videos que generarían imágenes con el mismo nombre')
    return plan


def write_instructions(output, has_masks=False):
    # Rutas relativas al script para que el proyecto pueda moverse.
    commands = '''#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
colmap feature_extractor --database_path database.db --image_path .. --ImageReader.single_camera_per_folder 1
colmap exhaustive_matcher --database_path database.db
colmap mapper --database_path database.db --image_path .. --output_path sparse
'''
    if has_masks:
        commands = commands.replace('--ImageReader.single_camera_per_folder 1',
                                    '--ImageReader.single_camera_per_folder 1 --ImageReader.mask_path masks')
    (output / 'run_colmap.sh').write_text(commands)
    (output / 'README.md').write_text("""# COLMAP project

Images are in the parent folder, grouped by platform and camera.
Run `sh run_colmap.sh` to reconstruct; extraction does not run COLMAP.
The script creates a database and writes sparse models under `sparse/`.

Intrinsics are shared per folder. Keep lens, zoom, and resolution consistent
within each folder. No synchronized camera rig is assumed.

`extraction.json` records source videos, FPS, frame counts, and available clip
metadata. `image_list.txt` contains paths relative to the scene's image root.
Names include the source video and a zero-based sample index, not the original
frame index. Each clip is sampled independently using timestamps; variable-rate
video can produce repeated images. Clip metadata does not imply synchronization.

Encoded dimensions are preserved without automatic rotation or resizing.
JPEG uses high quality; PNG avoids additional compression loss.
Exhaustive matching can be expensive for large image collections.
""")


def extract_scene(scene, output, fps, groups=None, image_format='jpg', threads=1, dry_run=False, regions=None):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    regions = regions or {}
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('FPS debe ser finito y mayor que cero')
    if threads < 1:
        raise ValueError('threads debe ser >= 1')
    if image_format not in ('jpg', 'png'):
        raise ValueError('Formato esperado: jpg o png')
    if not scene.is_dir():
        raise ValueError(f'No existe la escena: {scene}')
    if output == scene or scene.is_relative_to(output):
        raise ValueError('La salida no puede ser la escena ni un directorio antecesor')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'La salida debe ser una carpeta nueva o vacía: {output}')
    for executable in ('ffmpeg', 'ffprobe'):
        if not shutil.which(executable):
            raise ValueError(f'Falta instalar {executable}')
    plan = plan_scene(scene, groups, fps)
    for entry in plan:
        platform = Path(entry['relative_video']).parts[0]
        from lima3d.regions import region_for_name
        region = region_for_name(regions, entry['relative_video'])
        entry['region'] = region
        if region['mode'] == 'crop':
            left, top, right, bottom = region['box']
            if not (0 <= left < right <= entry['width'] and 0 <= top < bottom <= entry['height']):
                raise ValueError(f'Crop exceeds video dimensions: {entry["relative_video"]}')
            entry['output_width'], entry['output_height'] = right-left, bottom-top
        elif region['mode'] == 'mask' and 'path' in region:
            from PIL import Image
            with Image.open(region['path']) as mask:
                if mask.size != (entry['width'], entry['height']):
                    raise ValueError(f'Mask dimensions differ from video: {entry["relative_video"]}')
    if any(Path(p['image_folder']).parts and Path(p['image_folder']).parts[0] == '_colmap' for p in plan):
        raise ValueError('El nombre _colmap está reservado para metadatos de salida')
    print(f'Escena: {scene.name} | videos: {len(plan)} | FPS: {fps:g}')
    for entry in plan:
        print(f'  {entry["relative_video"]} -> {output / entry["image_folder"]}')
    if dry_run:
        print(f'Salida prevista: {output} (sin escribir archivos)')
        return plan
    output.mkdir(parents=True, exist_ok=True)
    project = output / '_colmap'
    project.mkdir()
    (project / 'sparse').mkdir()
    metadata = {'scene': str(scene), 'requested_fps': fps, 'format': image_format,
                'status': 'in_progress', 'videos': plan, 'source_regions': regions,
                'applied_regions': {entry['image_folder']: entry['region'] for entry in plan
                                    if entry['region']['mode'] == 'crop'}}
    manifest = project / 'extraction.json'
    manifest.write_text(json.dumps(metadata, indent=2) + '\n')
    try:
        with (project / 'image_list.txt').open('w') as listing:
            for index, entry in enumerate(plan, 1):
                folder = output / entry['image_folder']
                folder.mkdir(parents=True, exist_ok=True)
                # image2 interpreta % como patrón; escapar los % del nombre/ruta.
                pattern = str(folder / entry['prefix']).replace('%', '%%') + f'__%06d.{image_format}'
                filters = f'fps=fps={fps:.12g}:start_time=0:round=near'
                if entry['region']['mode'] == 'crop':
                    left, top, right, bottom = entry['region']['box']
                    filters += f',crop=w={right-left}:h={bottom-top}:x={left}:y={top}:exact=1'
                command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                           '-threads', str(threads), '-noautorotate', '-i', entry['video'],
                           '-map', '0:v:0', '-an', '-sn', '-dn', '-filter_threads', str(threads),
                           '-vf', filters,
                           '-threads', str(threads), '-start_number', '0']
                command += ['-q:v', '2'] if image_format == 'jpg' else ['-compression_level', '1']
                command += [pattern]
                print(f'[{index}/{len(plan)}] Extrayendo {entry["relative_video"]}', flush=True)
                subprocess.run(command, check=True)
                images = sorted(p for p in folder.iterdir()
                                if p.name.startswith(entry['prefix'] + '__') and p.suffix == '.' + image_format)
                if not images:
                    raise RuntimeError(f'No se extrajeron imágenes de {entry["relative_video"]}')
                for image in images:
                    relative = image.relative_to(output).as_posix()
                    listing.write(relative + '\n')
                    if entry['region']['mode'] == 'mask':
                        from PIL import Image
                        from lima3d.regions import mask_path
                        source = mask_path(regions, scene, relative)
                        with Image.open(source) as mask:
                            if mask.size != (entry['width'], entry['height']):
                                raise ValueError(f'Mask dimensions differ from image: {relative}')
                            target = project / 'masks' / (relative + '.png')
                            target.parent.mkdir(parents=True, exist_ok=True)
                            mask.convert('L').save(target)
                entry['extracted_frames'] = len(images)
                manifest.write_text(json.dumps(metadata, indent=2) + '\n')
        write_instructions(project, has_masks=(project / 'masks').is_dir())
        metadata['status'] = 'complete'
        metadata['total_frames'] = sum(p['extracted_frames'] for p in plan)
        print(f'Listo: {metadata["total_frames"]} imágenes en {output}')
    except BaseException as exc:
        metadata['status'] = 'failed'
        metadata['error'] = str(exc)
        raise
    finally:
        manifest.write_text(json.dumps(metadata, indent=2) + '\n')
    return plan


def main(argv=None):
    from lima3d.extraction_config import extraction_jobs
    parser = argparse.ArgumentParser(description='Extract scene frames using FFmpeg and config.yaml.')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('scene', nargs='?', type=Path, help='Scene name or input path (legacy CLI)')
    selection.add_argument('--scenes', nargs='+', help='Override data.scenes')
    parser.add_argument('--config', type=Path, help='Shared YAML; defaults to config.yaml when no scene is given')
    parser.add_argument('--fps', type=float, help='Override extraction.fps')
    parser.add_argument('--output', type=Path, help='Output directory for a single scene')
    parser.add_argument('--groups', nargs='+', help='Override data.platforms')
    parser.add_argument('--format', choices=['jpg', 'png'], help='Override extraction.format')
    parser.add_argument('--threads', type=int, help='Override extraction.threads')
    parser.add_argument('--dry-run', action='store_true', help='Inspect videos without writing files')
    args = parser.parse_args(argv)
    try:
        jobs = extraction_jobs(args)
        for job in jobs:
            extract_scene(**job)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
