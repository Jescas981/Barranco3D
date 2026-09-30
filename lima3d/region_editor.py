"""Local camera-region editor with automatic dataset video discovery."""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import math
from pathlib import Path
import secrets
import subprocess
from urllib.parse import parse_qs, urlparse
import webbrowser

import yaml
from PIL import Image, ImageDraw

from .regions import parse_regions
from .utils.io import save_json, scene_lock


def video_info(video):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height:format=duration', '-of', 'json', str(video)],
        check=True, capture_output=True, text=True, timeout=30)
    data = json.loads(result.stdout)
    if not data.get('streams'):
        raise ValueError('The selected file has no video stream')
    return dict(width=data['streams'][0]['width'], height=data['streams'][0]['height'],
                duration=float(data.get('format', {}).get('duration', 0)))


def frame_png(video, seconds):
    return subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1',
        '-ss', str(seconds), '-noautorotate', '-i', str(video), '-map', '0:v:0',
        '-frames:v', '1', '-vf', 'scale=1280:1280:force_original_aspect_ratio=decrease',
        '-filter_threads', '1', '-threads', '1', '-f', 'image2pipe', '-c:v', 'png', 'pipe:1'],
        check=True, capture_output=True, timeout=60).stdout


def validate_selection(selection, size):
    width, height = size
    mode = selection.get('mode')
    if mode == 'none':
        return {'mode': 'none'}
    if mode == 'crop':
        entry = {'mode': mode, 'box': selection.get('box')}
        parse_regions({'platform': entry}, Path('.'), ['platform'])
        if entry['box'][2] > width or entry['box'][3] > height:
            raise ValueError('Crop is outside the video frame')
        return entry
    if mode != 'mask':
        raise ValueError('Choose none, crop, or mask')
    polygons = selection.get('polygons')
    if not isinstance(polygons, list) or not polygons:
        raise ValueError('Draw at least one excluded polygon')
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError('Every polygon needs at least three vertices')
        for point in polygon:
            if (not isinstance(point, list) or len(point) != 2 or
                    any(type(v) is not int for v in point) or
                    not (0 <= point[0] <= width and 0 <= point[1] <= height)):
                raise ValueError('Polygon vertices must be integer coordinates inside the frame')
        area = sum(polygon[i][0]*polygon[(i+1)%len(polygon)][1] -
                   polygon[(i+1)%len(polygon)][0]*polygon[i][1] for i in range(len(polygon)))
        if not area:
            raise ValueError('Polygon has zero area')
    return {'mode': mode, 'polygons': polygons}


def replace_regions(text, regions):
    """Rewrite only the regions block, preserving unrelated comments/settings."""
    tree = yaml.compose(text)
    replacement = yaml.safe_dump({'regions': regions}, sort_keys=False)
    for key, value in tree.value:
        if key.value == 'regions':
            return text[:key.start_mark.index] + replacement + text[value.end_mark.index:]
    return text.rstrip() + '\n\n' + replacement


def save_selection(config, platform, video, info, selection, camera=None):
    selected = validate_selection(selection, (info['width'], info['height']))
    if not platform or Path(platform).name != platform or platform in ('.', '..'):
        raise ValueError('Platform must be a single folder name')
    if camera is not None and (not camera or Path(camera).name != camera or camera in ('.', '..')):
        raise ValueError('Camera must be a single folder name')
    with scene_lock(config.with_name('.region-editor.lock'), wait=True):
        original = config.read_text()
        settings = yaml.safe_load(original)
        if not isinstance(settings, dict):
            raise ValueError('Config must be a YAML mapping')
        entry = dict(selected)
        if selected['mode'] == 'mask':
            image = Image.new('L', (info['width'], info['height']), 255)
            draw = ImageDraw.Draw(image)
            for polygon in selected['polygons']:
                draw.polygon([tuple(point) for point in polygon], fill=0)
            stream = io.BytesIO(); image.save(stream, format='PNG')
            content = stream.getvalue()
            identifier = hashlib.sha256(content).hexdigest()[:16]
            relative = Path('masks') / platform / (camera or 'default') / f'{identifier}.png'
            path = config.parent / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                temporary = path.with_suffix('.tmp'); temporary.write_bytes(content); temporary.replace(path)
            save_json(path.with_suffix('.json'), {**selected, 'video': str(video), 'platform': platform, 'camera': camera, **info})
            entry = {'mode': 'mask', 'path': relative.as_posix()}
        regions = settings.get('regions') or {}
        if not isinstance(regions, dict):
            raise ValueError('regions must be a mapping')
        if camera is None:
            regions[platform] = entry
        else:
            parent = regions.setdefault(platform, {'mode': 'none'})
            if not isinstance(parent, dict) or not isinstance(parent.get('cameras', {}), dict):
                raise ValueError('Camera regions must be mappings')
            parent.setdefault('cameras', {})[camera] = entry
        updated = replace_regions(original, regions)
        # Validate before replacing the user's file.
        parsed = yaml.safe_load(updated)
        if parsed.get('regions') != regions:
            raise ValueError('Could not update regions safely')
        backup_dir = config.parent / '.region_editor_backups'
        backup_dir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        backup = backup_dir / f'{config.name}.{stamp}.bak'
        backup.write_text(original)
        temporary = config.with_name(config.name + '.regions.tmp')
        temporary.write_text(updated); temporary.replace(config)
    return {'region': entry, 'config': str(config), 'backup': str(backup)}


VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.m4v', '.webm', '.mts'}


def discover_videos(config):
    settings = yaml.safe_load(config.read_text())
    if not isinstance(settings, dict):
        raise ValueError('Config must be a YAML mapping')
    data = settings.get('data') or {}
    root = (config.parent / data.get('datasets_root', 'dataset')).resolve()
    if not root.is_dir():
        raise ValueError(f'Dataset folder does not exist: {root}')
    records = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        relative = path.relative_to(root)
        if len(relative.parts) < 3:
            continue
        scene, platform = relative.parts[:2]
        camera = relative.parts[2] if platform.lower() == 'car' and len(relative.parts) >= 4 else None
        records.append(dict(id=str(len(records)), path=path, scene=scene, platform=platform,
                            camera=camera, video=path.name, relative=relative.as_posix()))
    return records


def initial_selection(config, record, info):
    settings = yaml.safe_load(config.read_text())
    parent = (settings.get('regions') or {}).get(record['platform'], {'mode': 'none'})
    initial = {k:v for k,v in parent.items() if k != 'cameras'} or {'mode':'none'}
    if record['camera'] is not None:
        initial = parent.get('cameras', {}).get(record['camera'], initial)
    if initial.get('mode') == 'mask' and 'path' in initial:
        sidecar = (config.parent / initial['path']).with_suffix('.json')
        if sidecar.exists():
            stored = json.loads(sidecar.read_text())
            if (stored.get('width'), stored.get('height')) == (info['width'], info['height']):
                initial = stored
    return initial


def serve(config, platform=None, video=None, port=8765, open_browser=True, camera=None, scene=None):
    records = discover_videos(config) if video is None else []
    if video is not None:
        # Keep the explicit-video CLI available, inferring its scope when possible.
        try:
            records = [r for r in discover_videos(config) if r['path'].resolve() == video.resolve()]
        except ValueError:
            pass
        if not records:
            if not platform:
                raise ValueError('--platform is required only for videos outside dataset')
            records = [dict(id='0', path=video, scene=scene or '', platform=platform,
                            camera=camera, video=video.name, relative=video.name)]
    else:
        records = [r for r in records if (platform is None or r['platform'] == platform)
                   and (camera is None or r['camera'] == camera)
                   and (scene is None or r['scene'] == scene)]
    if not records:
        raise ValueError('No videos found for the selected dataset filters')
    catalog = {r['id']:r for r in records}
    default_id = records[0]['id']
    token = secrets.token_urlsafe(24)
    metadata = {}
    html = Path(__file__).with_name('ui').joinpath('region_editor.html').read_bytes()

    def select(identifier):
        if not isinstance(identifier, str) or identifier not in catalog:
            raise ValueError('Unknown video; select one from the dataset list')
        record = catalog[identifier]
        if identifier not in metadata:
            metadata[identifier] = video_info(record['path'])
        return record, metadata[identifier]

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, data, content_type='application/json'):
            body = json.dumps(data).encode() if content_type == 'application/json' else data
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers(); self.wfile.write(body)

        def authorized(self, query):
            return secrets.compare_digest(query.get('token', [''])[0], token)

        def do_GET(self):
            parsed = urlparse(self.path); query = parse_qs(parsed.query)
            if not self.authorized(query):
                return self.respond(403, {'error': 'Open the URL printed by the editor'})
            try:
                if parsed.path == '/':
                    return self.respond(200, html.replace(b'__SESSION_TOKEN__', token.encode()), 'text/html; charset=utf-8')
                if parsed.path == '/region_geometry.js':
                    script = Path(__file__).with_name('ui').joinpath('region_geometry.js').read_bytes()
                    return self.respond(200, script, 'text/javascript; charset=utf-8')
                if parsed.path == '/catalog':
                    return self.respond(200, {'videos': [{k:v for k,v in r.items() if k != 'path'} for r in records]})
                record, info = select(query.get('video', [default_id])[0])
                if parsed.path == '/info':
                    return self.respond(200, {**info, **{k:v for k,v in record.items() if k != 'path'},
                        'config': str(config), 'initial': initial_selection(config, record, info)})
                if parsed.path == '/frame':
                    seconds = float(query.get('t', ['0'])[0])
                    if not math.isfinite(seconds) or seconds < 0 or (info['duration'] and seconds >= info['duration']):
                        raise ValueError('Timestamp must be within the video')
                    data = frame_png(record['path'], seconds)
                    if not data:
                        raise ValueError('No frame at this timestamp; select an earlier time')
                    return self.respond(200, data, 'image/png')
                self.respond(404, {'error': 'Not found'})
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                self.respond(400, {'error': str(exc)})

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != '/save' or not self.authorized(parse_qs(parsed.query)):
                return self.respond(403, {'error': 'Invalid request'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1024*1024:
                    raise ValueError('Invalid selection size')
                selection = json.loads(self.rfile.read(length))
                if not isinstance(selection, dict):
                    raise ValueError('Selection must be an object')
                record, info = select(selection.get('video_id', default_id))
                self.respond(200, save_selection(config, record['platform'], record['path'], info,
                                                 selection, camera=record['camera']))
            except (ValueError, OSError, yaml.YAMLError) as exc:
                self.respond(400, {'error': str(exc)})

        def log_message(self, format, *args):
            pass  # Do not put the access token in logs.

    server = HTTPServer(('127.0.0.1', port), Handler)
    url = f'http://127.0.0.1:{server.server_port}/?token={token}'
    print(f'Region editor: {url}\nDiscovered videos: {len(records)}\nPress Ctrl+C to stop.', flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description='Discover dataset videos and draw camera crops or masks.')
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--video', type=Path, help='Optional explicit video; default: discover dataset')
    parser.add_argument('--platform', help='Optional platform filter')
    parser.add_argument('--camera', help='Optional camera filter, e.g. cam0')
    parser.add_argument('--scene', help='Optional scene filter')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    try:
        serve(args.config.resolve(), args.platform, args.video.resolve() if args.video else None,
              args.port, not args.no_browser, args.camera, args.scene)
    except (ValueError, OSError, subprocess.SubprocessError, yaml.YAMLError) as exc:
        parser.exit(1, f'Error: {exc}\n')
