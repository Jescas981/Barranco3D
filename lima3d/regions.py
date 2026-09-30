"""Per-platform image regions: unchanged inputs, physical crops, or point masks."""
from pathlib import Path
import hashlib
import json


def _parse_rule(entry, base, scope):
    if not isinstance(entry, dict):
        raise ValueError(f'regions.{scope} must be a mapping')
    mode = entry.get('mode', 'none')
    allowed = {'none': {'mode'}, 'crop': {'mode', 'box'}, 'mask': {'mode', 'path', 'root'}}
    if mode not in allowed or set(entry) - allowed[mode]:
        raise ValueError(f'Invalid region settings for {scope}')
    item = dict(entry, mode=mode)
    if mode == 'crop':
        box = item.get('box')
        if (not isinstance(box, list) or len(box) != 4 or
                any(type(x) is not int for x in box) or
                not (0 <= box[0] < box[2] and 0 <= box[1] < box[3])):
            raise ValueError('crop.box must be [left, top, right, bottom] in original pixels')
    elif mode == 'mask':
        keys = [k for k in ('path', 'root') if k in item]
        if len(keys) != 1 or not isinstance(item[keys[0]], str) or not item[keys[0]]:
            raise ValueError('mask requires exactly one path (shared) or root (per image)')
        item[keys[0]] = str((base / item[keys[0]]).resolve())
    return item


def parse_regions(value, base, platforms):
    """Normalize YAML platform defaults and camera overrides to path prefixes."""
    if value is None:
        return {}
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError('regions must map platform names to region settings')
    result = {}
    for platform, entry in value.items():
        if not isinstance(entry, dict):
            raise ValueError(f'regions.{platform} must be a mapping')
        default = _parse_rule({k:v for k,v in entry.items() if k != 'cameras'}, base, platform)
        if platform in platforms and default['mode'] != 'none':
            result[platform] = default
        cameras = entry.get('cameras', {})
        if not isinstance(cameras, dict):
            raise ValueError(f'regions.{platform}.cameras must be a mapping')
        for camera, rule in cameras.items():
            if not isinstance(camera, str) or not camera or camera in ('.', '..') or Path(camera).name != camera:
                raise ValueError('Camera names must be single folder names')
            parsed = _parse_rule(rule, base, platform + '/' + camera)
            if platform in platforms:
                result[platform + '/' + camera] = parsed
    return result


def region_for_name(regions, name):
    """The most specific camera rule wins, including an explicit none override."""
    parts = Path(name).parts
    for length in range(len(parts)-1, 0, -1):
        key = '/'.join(parts[:length])
        if key in regions:
            return regions[key]
    return {'mode': 'none'}


def mask_path(regions, scene, name):
    region = region_for_name(regions, name)
    if region.get('mode') != 'mask':
        return None
    if '_mask_dir' in region:
        return Path(region['_mask_dir']) / (name + '.png')
    if 'path' in region:
        return Path(region['path'])
    return Path(region['root']) / scene.name / (name + '.png')


def effective_regions(scene, regions, platforms=None):
    """Do not apply a crop twice to frames already processed during extraction."""
    manifest = Path(scene) / '_colmap/extraction.json'
    if not manifest.is_file():
        return regions
    metadata = json.loads(manifest.read_text())
    result = dict(regions)
    for platform, applied in metadata.get('applied_regions', {}).items():
        if platforms is not None and platform.split('/')[0] not in platforms:
            continue
        if applied.get('mode') == 'crop':
            requested = region_for_name(regions, platform + '/_frame')
            if requested != applied:
                raise ValueError(f'{platform}: frames already cropped with {applied["box"]}; '
                                 'restore the same region or extract original videos into a new frames directory')
            result.pop(platform, None)
            if region_for_name(result, platform + '/_frame').get('mode') != 'none':
                result[platform] = {'mode': 'none'}
    for platform, region in list(result.items()):
        if region.get('mode') != 'mask':
            continue
        source = Path(region.get('path', region.get('root', '')))
        if (not source.exists() and region == metadata.get('source_regions', {}).get(platform)
                and (Path(scene) / '_colmap/masks').is_dir()):
            result[platform] = {'mode': 'mask', '_mask_dir': str(Path(scene) / '_colmap/masks')}
    return result


def region_identity(scene, names, regions):
    if not regions:
        return None
    from PIL import Image
    rows = []
    hashes = {}
    for name in names:
        region = region_for_name(regions, name)
        mode = region.get('mode', 'none')
        if mode == 'none':
            continue
        with Image.open(scene / name) as image:
            width, height = image.size
        if mode == 'crop':
            box = region['box']
            if box[2] > width or box[3] > height:
                raise ValueError(f'Crop exceeds image dimensions: {name}')
            rows.append([name, mode, box])
        else:
            path = mask_path(regions, scene, name)
            if path not in hashes:
                with Image.open(path) as mask:
                    hashes[path] = (mask.size, hashlib.sha256(path.read_bytes()).hexdigest())
            if hashes[path][0] != (width, height):
                raise ValueError(f'Mask dimensions differ from image: {name}')
            rows.append([name, mode, hashes[path][1]])
    return {'version': 1, 'images': rows} if rows else None


def prepared_scene(scene, args, root, names):
    regions = effective_regions(scene, getattr(args, 'regions', {}), args.platforms)
    if not any(r['mode'] == 'crop' for r in regions.values()):
        return scene
    from PIL import Image
    from .utils.io import scene_lock
    output = root / 'images'
    with scene_lock(root / '.crop.lock', wait=True):
        for name in names:
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            region = region_for_name(regions, name)
            if region.get('mode') == 'crop':
                if not target.exists():
                    temporary = target.with_name(target.stem + '.tmp' + target.suffix)
                    with Image.open(scene / name) as image:
                        cropped = image.crop(region['box'])
                        if target.suffix.lower() in ('.jpg', '.jpeg'):
                            cropped.convert('RGB').save(temporary, quality=100, subsampling=0)
                        else:
                            cropped.save(temporary)
                    temporary.replace(target)
            else:
                source = (scene / name).resolve()
                if target.is_symlink():
                    if target.resolve() == source:
                        continue
                    temporary = target.with_name(target.name + '.link.tmp')
                    temporary.unlink(missing_ok=True)
                    temporary.symlink_to(source)
                    temporary.replace(target)
                elif not target.exists():
                    target.symlink_to(source)
    return output


def valid_points(points, path):
    import numpy as np
    from PIL import Image
    if path is None:
        return np.ones(len(points), dtype=bool)
    with Image.open(path) as image:
        mask = np.asarray(image.convert('L'))
    finite = np.isfinite(points).all(axis=1)
    xy = np.floor(np.where(np.isfinite(points), points, -1)).astype(np.int64)
    valid = finite & (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] < mask.shape[0])
    indices = np.flatnonzero(valid)
    valid[indices] &= mask[xy[indices, 1], xy[indices, 0]] > 0
    return valid


def filter_features(source, target, scene, names, regions):
    import h5py
    import numpy as np
    temporary = target.with_name(target.name + '.tmp')
    if target.exists():
        return target
    with h5py.File(source) as src, h5py.File(temporary, 'w') as dst:
        for name in names:
            src.copy(name, dst.require_group(str(Path(name).parent)) if '/' in name else dst,
                     name=Path(name).name)
            group = dst[name]
            valid = valid_points(group['keypoints'][...], mask_path(regions, scene, name))
            for key in ('keypoints', 'descriptors', 'scores'):
                data = group[key][...]
                attrs = dict(group[key].attrs)
                data = data[:, valid] if key == 'descriptors' else data[valid]
                del group[key]
                group.create_dataset(key, data=data)
                group[key].attrs.update(attrs)
    temporary.replace(target)
    return target


def filter_dense(source, target, scene, pairs, regions):
    import h5py
    from hloc.utils.parsers import names_to_pair
    temporary = target.with_name(target.name + '.tmp')
    # Pair sets can grow: rebuild the inexpensive filtered view atomically.
    with h5py.File(source) as src, h5py.File(temporary, 'w') as dst:
        for a, b in pairs:
            key = names_to_pair(a, b)
            raw = src[key]
            valid = valid_points(raw['keypoints0'][...], mask_path(regions, scene, a))
            valid &= valid_points(raw['keypoints1'][...], mask_path(regions, scene, b))
            group = dst.create_group(key)
            for field in ('keypoints0', 'keypoints1', 'scores'):
                group.create_dataset(field, data=raw[field][...][valid])
    temporary.replace(target)
    return target
