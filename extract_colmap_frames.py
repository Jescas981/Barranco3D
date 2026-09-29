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


def write_instructions(output):
    # Rutas relativas al script para que el proyecto pueda moverse.
    commands = '''#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
colmap feature_extractor --database_path database.db --image_path .. --ImageReader.single_camera_per_folder 1
colmap exhaustive_matcher --database_path database.db
colmap mapper --database_path database.db --image_path .. --output_path sparse
'''
    (output / 'run_colmap.sh').write_text(commands)
    (output / 'README.md').write_text('''# Proyecto COLMAP

Imágenes en la carpeta superior, organizadas por plataforma y cámara; `sparse/` queda
preparado para la reconstrucción. COLMAP creará `database.db` al extraer
características. Ejecuta `sh run_colmap.sh` cuando quieras reconstruir.
El script de extracción no ejecuta COLMAP.

Se comparten intrínsecos por carpeta: Car/cam0, Car/cam1, Drone, etc.
Esto supone la misma lente y zoom entre videos de una carpeta. Si cambian,
separa los videos en subcarpetas antes de extraer. Las resoluciones deben
coincidir dentro de cada carpeta.
No se configura un rig sincronizado: los recortes no necesariamente coinciden
en tiempo y no se dispone de extrínsecos calibrados.

`extraction.json` registra videos, FPS, cantidades y metadatos de recortes
cuando están disponibles. `image_list.txt` contiene rutas relativas a la carpeta de la escena en frames/.
Los nombres contienen el video de origen y un índice de muestra desde cero;
ese índice no es el índice del frame original. FFmpeg usa timestamps para
muestrear a FPS constantes; con video variable puede repetir una imagen para
cubrir un hueco temporal. Cada clip empieza su propia rejilla temporal en cero.
Los metadatos de recortes se conservan como procedencia, no como sincronización.

Se preservan las dimensiones codificadas, sin auto-rotación ni redimensionado.
JPEG usa calidad alta; PNG evita pérdida adicional. El matcher exhaustivo
compara todos los pares y puede ser costoso con muchos miles de imágenes.
''')


def extract_scene(scene, output, fps, groups=None, image_format='jpg', threads=1, dry_run=False):
    scene, output = Path(scene).resolve(), Path(output).resolve()
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
                'status': 'in_progress', 'videos': plan}
    manifest = project / 'extraction.json'
    manifest.write_text(json.dumps(metadata, indent=2) + '\n')
    try:
        with (project / 'image_list.txt').open('w') as listing:
            for index, entry in enumerate(plan, 1):
                folder = output / entry['image_folder']
                folder.mkdir(parents=True, exist_ok=True)
                # image2 interpreta % como patrón; escapar los % del nombre/ruta.
                pattern = str(folder / entry['prefix']).replace('%', '%%') + f'__%06d.{image_format}'
                command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                           '-threads', str(threads), '-noautorotate', '-i', entry['video'],
                           '-map', '0:v:0', '-an', '-sn', '-dn', '-filter_threads', str(threads),
                           '-vf', f'fps=fps={fps:.12g}:start_time=0:round=near',
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
                    listing.write(image.relative_to(output).as_posix() + '\n')
                entry['extracted_frames'] = len(images)
                manifest.write_text(json.dumps(metadata, indent=2) + '\n')
        write_instructions(project)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path, help='Nombre de escena (DavidHouse) o ruta (datasets/DavidHouse)')
    parser.add_argument('--fps', type=float, required=True, help='Imágenes por segundo; admite 0.5, 2, etc.')
    parser.add_argument('--output', type=Path, help='Por defecto: frames/<escena>')
    parser.add_argument('--groups', nargs='+', help='Solo estas carpetas: Car Drone Pedestrian')
    parser.add_argument('--format', choices=['jpg', 'png'], default='jpg')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    scene = args.scene
    if len(scene.parts) == 1 and not scene.is_dir():
        scene = Path('datasets') / args.scene
        if not scene.is_dir() and (Path('dataset') / args.scene).is_dir():
            scene = Path('dataset') / args.scene
    output = args.output or Path('frames') / scene.name
    try:
        extract_scene(scene, output, args.fps, args.groups, args.format, args.threads, args.dry_run)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
