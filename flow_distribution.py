#!/usr/bin/env python3
"""Distribution of image motion per platform and suggested Mth (no frames written).

Reads the same config.yaml as the extraction script (scenes, platforms, block_size,
regions, config Mth). For every video it samples frames every --lag-seconds and
measures the modal block displacement between consecutive samples:
  dense flow -> median per block -> drop low-texture/masked blocks
  -> modal displacement (most populated bin of the (dx, dy) block histogram).
Then converts it to image velocity (px/s, and % of image width per s) and derives

    Mth ~= typical velocity (px/s) x target keyframe interval (s)

Outputs in --output: samples.csv, summary.json, speed_<platform>.png
and per-camera Car histograms. --plot-only reuses samples.csv without reading videos.
"""
import argparse
import csv
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.m4v', '.webm', '.mts'}
PERCENTILES = [10, 25, 50, 75, 90, 95]


def block_stats(flow, gray, bs, tex_thr, keep_mask):
    h, w = gray.shape
    ny, nx = h // bs, w // bs
    f = flow[:ny * bs, :nx * bs].reshape(ny, bs, nx, bs, 2)
    g = gray[:ny * bs, :nx * bs].reshape(ny, bs, nx, bs).astype(np.float32)
    block_flow = np.median(f, axis=(1, 3)).reshape(-1, 2)
    valid = (g.std(axis=(1, 3)) > tex_thr).reshape(-1)
    if keep_mask is not None:
        k = keep_mask[:ny * bs, :nx * bs].reshape(ny, bs, nx, bs).mean(axis=(1, 3)).reshape(-1)
        valid &= k > 0.9
    return block_flow, valid


def modal_displacement(vectors, bin_px):
    keys = np.floor(vectors / bin_px).astype(int)
    uniq, counts = np.unique(keys, axis=0, return_counts=True)
    best = uniq[np.argmax(counts)]
    near = np.all(np.abs(keys - best) <= 1, axis=1)
    dx, dy = vectors[near].mean(axis=0)
    return float(dx), float(dy)


def to_gray(frame, crop):
    if crop:
        left, top, right, bottom = crop
        frame = frame[top:bottom, left:right]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def pair_flow(a, b, scale):
    if scale != 1:
        a_s = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        b_s = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        a_s, b_s = a, b
    flow = cv2.calcOpticalFlowFarneback(a_s, b_s, None, 0.5, 4, 21, 3, 5, 1.1, 0)
    if scale != 1:
        flow = cv2.resize(flow, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR) / scale
    return flow


def process_video(path, ctx, p, writer):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print(f'  skip (no fps): {path}')
        return
    step_frames = p.lag_frames or max(1, round(p.lag_seconds * fps))
    lag_s = step_frames / fps
    prev = None
    index = sample = 0
    while cap.grab():
        if index % step_frames == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            gray = to_gray(frame, p.crop)
            if prev is not None:
                vectors, valid = block_stats(pair_flow(prev, gray, p.scale), prev,
                                             p.block_size, p.texture_thr, p.keep_mask)
                if valid.sum() >= p.min_valid_blocks:
                    dx, dy = modal_displacement(vectors[valid], p.bin_px)
                    mode_px = float(np.hypot(dx, dy))
                else:
                    dx = dy = mode_px = float('nan')
                width = gray.shape[1]
                writer.writerow([ctx['scene'], ctx['platform'], ctx['video'], f'{index / fps:.3f}',
                                 f'{lag_s:.3f}', f'{dx:.3f}', f'{dy:.3f}', f'{mode_px:.3f}',
                                 f'{mode_px / lag_s:.3f}', f'{100 * mode_px / lag_s / width:.3f}',
                                 f'{valid.mean():.3f}', width, ctx['mth'], ctx['camera']])
            prev = gray
            sample += 1
            if p.max_samples and sample >= p.max_samples:
                break
        index += 1
    cap.release()


def pcts(values):
    return {f'P{q}': float(np.percentile(values, q)) for q in PERCENTILES}



def camera_name(row):
    """Preserve camera folders; recover identifiers in older basename-only CSVs."""
    if row.get('camera'):
        return row['camera']
    video = Path(row['video'])
    if video.parent.name not in ('', '.', row['platform']):
        return video.parent.name
    match = re.search(r'(?:^|[_/])cam(\d+)(?=[_/.]|$)', row['video'], re.I)
    return f'cam{int(match.group(1))}' if match else 'unknown'


def usable_rows(rows, min_valid_frac, min_speed):
    return [r for r in rows
            if all(np.isfinite(float(r[key])) for key in
                   ('mode_px', 'speed_px_s', 'speed_pct_width_s', 'valid_block_frac'))
            and float(r['valid_block_frac']) >= min_valid_frac
            and float(r['speed_px_s']) >= min_speed]


def plot_distributions(rows, output, min_valid_frac=0.2, min_speed=1.0, bins=50):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    by_platform = {}
    for row in rows:
        by_platform.setdefault(row['platform'], []).append(row)
    for platform, samples in sorted(by_platform.items()):
        valid = usable_rows(samples, min_valid_frac, min_speed)
        if not valid:
            continue
        speed = np.array([float(r['speed_px_s']) for r in valid])
        lags = np.array([float(r['lag_s']) for r in valid])
        lag = (f'{lags[0]:g} s' if np.allclose(lags, lags[0])
               else f'{lags.min():g}–{lags.max():g} s')
        scenes = ', '.join(sorted({r['scene'] for r in samples}))
        caption = (f'Source: samples.csv · {scenes} · actual lag: {lag}\n'
                   f'Valid blocks ≥ {min_valid_frac:g}; image speed ≥ {min_speed:g} px/s')
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.hist(speed, bins=bins, color='#4878a8', alpha=0.8)
        for q, ls in ((25, ':'), (50, '--'), (75, ':')):
            ax.axvline(np.percentile(speed, q), color='k', ls=ls, label=f'P{q}')
        ax.set(xlabel='Image velocity (px/s)', ylabel='Number of frame pairs',
               title=f'{platform}: image velocity distribution')
        ax.legend()
        fig.text(0.5, 0.015, caption, ha='center', fontsize=8)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(output / f'speed_{platform}.png', dpi=160)
        plt.close(fig)
        if platform.lower() != 'car':
            continue
        camera_rows = {}
        for row in samples:
            camera_rows.setdefault(camera_name(row), []).append(row)
        cameras = sorted(set(camera_rows) | {f'cam{i}' for i in range(9)},
                         key=lambda name: (int(name[3:]) if re.fullmatch(r'cam\d+', name) else 999, name))
        colors = plt.get_cmap('tab10')
        report = {}
        for camera in cameras:
            selected = usable_rows(camera_rows.get(camera, []), min_valid_frac, min_speed)
            report[camera] = {'pairs_total': len(camera_rows.get(camera, [])),
                              'pairs_used': len(selected)}
            if selected:
                report[camera].update({key: pcts(np.array([float(r[key]) for r in selected]))
                                      for key in ('speed_px_s', 'speed_pct_width_s')})
        (output / 'camera_summary_Car.json').write_text(json.dumps(report, indent=2))
        for metric, label, suffix in (
                ('speed_px_s', 'Image velocity (px/s)', ''),
                ('speed_pct_width_s', 'Image velocity (% of image width/s)', '_pct_width')):
            values = {camera: np.array([float(r[metric]) for r in
                      usable_rows(camera_rows.get(camera, []), min_valid_frac, min_speed)])
                      for camera in cameras}
            all_values = np.concatenate([v for v in values.values() if len(v)])
            edges = np.histogram_bin_edges(all_values, bins=bins, range=(0, max(float(all_values.max()), 1e-6)))
            fig, axes = plt.subplots(int(np.ceil(len(cameras)/3)), 3,
                                     figsize=(14, 10), sharex=True, sharey=True, squeeze=False)
            overlay, ax_overlay = plt.subplots(figsize=(11, 6))
            for index, camera in enumerate(cameras):
                ax = axes.flat[index]
                v = values[camera]
                color = colors(index % 10)
                ax.set_title(f'{camera} · {len(v)} usable / {report[camera]["pairs_total"]} pairs', color=color)
                ax.set(xlabel=label, ylabel='Frame pairs (%)')
                ax.grid(axis='y', alpha=0.2)
                if not len(v):
                    message = 'No samples' if not report[camera]['pairs_total'] else 'No pairs pass filters'
                    ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes)
                    print(f'Car {camera}: {message}')
                    continue
                weights = np.full(len(v), 100.0 / len(v))
                ax.hist(v, bins=edges, weights=weights, color=color, alpha=0.8,
                        label=camera)
                median = np.median(v)
                ax.axvline(median, color='black', ls='--', lw=1,
                           label=f'Median: {median:.2f}')
                ax.legend(fontsize=8)
                ax_overlay.hist(v, bins=edges, weights=weights, histtype='step',
                                linewidth=1.7, color=color, label=f'{camera} (n={len(v)})')
            for ax in list(axes.flat)[len(cameras):]:
                ax.set_visible(False)
            fig.suptitle('Car: image velocity by camera · dashed lines show medians', fontsize=15)
            fig.text(0.5, 0.015, caption + '\nShared bins; percentages normalized within each camera.',
                     ha='center', fontsize=9)
            fig.tight_layout(rect=(0, 0.085, 1, 0.96))
            fig.savefig(output / f'speed_Car_cameras{suffix}.png', dpi=160)
            plt.close(fig)
            ax_overlay.set(xlabel=label, ylabel='Frame pairs (%)',
                           title='Car: comparison of camera velocity distributions')
            ax_overlay.legend(ncol=3, fontsize=9)
            ax_overlay.grid(axis='y', alpha=0.2)
            overlay.text(0.5, 0.015, caption + '\nShared bins; percentages normalized within each camera.',
                         ha='center', fontsize=8)
            overlay.tight_layout(rect=(0, 0.11, 1, 1))
            overlay.savefig(output / f'speed_Car_cameras_overlay{suffix}.png', dpi=160)
            plt.close(overlay)

def main():
    from lima3d.extraction_config import extraction_jobs
    from lima3d.optical_flow import validate_options
    from lima3d.regions import region_for_name

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument('scene', nargs='?', type=Path)
    sel.add_argument('--scenes', nargs='+', help='Override data.scenes')
    ap.add_argument('--config', type=Path)
    ap.add_argument('--groups', nargs='+', help='Override data.platforms')
    ap.add_argument('--output', type=Path, default=Path('flow_distribution'))
    ap.add_argument('--lag-seconds', type=float, default=0.5,
                    help='Time between compared frames (0.25-0.5 s recommended; consecutive video frames are too noisy)')
    ap.add_argument('--lag-frames', type=int,
                    help='Compare frames this many video frames apart (1 = frame by frame). Overrides --lag-seconds')
    ap.add_argument('--target-interval', type=float, nargs='+', default=[0.25, 0.5, 1.0],
                    help='Desired keyframe interval(s) in s at typical motion')
    ap.add_argument('--speed-percentile', type=float, default=50,
                    help='Percentile of image velocity regarded as "typical"')
    ap.add_argument('--min-valid-frac', type=float, default=0.2,
                    help='Discard pairs whose fraction of textured blocks is below this')
    ap.add_argument('--min-speed', type=float, default=1.0,
                    help='Discard pairs slower than this (px/s): vehicle stopped / no motion')
    ap.add_argument('--scale', type=float, default=0.5, help='Downscale before flow; results rescaled to full-res px')
    ap.add_argument('--texture-thr', type=float, default=4.0)
    ap.add_argument('--min-valid-blocks', type=int, default=10)
    ap.add_argument('--bin-px', type=float, default=0.5)
    ap.add_argument('--plot-only', action='store_true', help='Regenerate figures from output/samples.csv without processing videos')
    ap.add_argument('--hist-bins', type=int, default=50, help='Shared histogram bins for camera comparisons')
    ap.add_argument('--max-samples', type=int, default=0, help='Per video (0 = all)')
    a = ap.parse_args()
    if a.hist_bins < 1:
        ap.error('--hist-bins must be positive')
    if a.plot_only:
        with (a.output / 'samples.csv').open(newline='') as source:
            rows = list(csv.DictReader(source))
        plot_distributions(rows, a.output, a.min_valid_frac, a.min_speed, a.hist_bins)
        print(f'Saved figures to {a.output}')
        return

    job_args = SimpleNamespace(scene=a.scene, scenes=a.scenes, config=a.config, strategy='optical_flow',
                               mode_threshold_px=None, fps=None, output=None, groups=a.groups,
                               format=None, threads=None, dry_run=True)
    # extraction_jobs requires empty output folders; nothing is written here, so
    # folders without videos look empty during that call.
    real_iterdir = Path.iterdir

    def iterdir_no_output_check(self):
        has_video = any(Path(f).suffix.lower() in VIDEO_EXTENSIONS
                        for _, _, files in os.walk(self) for f in files)
        return real_iterdir(self) if has_video else iter(())

    with mock.patch.object(Path, 'iterdir', iterdir_no_output_check):
        jobs = extraction_jobs(job_args)

    a.output.mkdir(parents=True, exist_ok=True)
    csv_path = a.output / 'samples.csv'
    with csv_path.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['scene', 'platform', 'video', 't_s', 'lag_s', 'dx_px', 'dy_px', 'mode_px',
                         'speed_px_s', 'speed_pct_width_s', 'valid_block_frac', 'width_px', 'config_mth_px', 'camera'])
        for job in jobs:
            scene = Path(job['scene']).resolve()
            of = validate_options(job.get('optical_flow'))
            groups = job.get('groups')
            regions = job.get('regions') or {}
            videos = sorted(v for v in scene.rglob('*') if v.is_file() and v.suffix.lower() in VIDEO_EXTENSIONS)
            print(f'\nScene {scene.name} | block_size={of["block_size"]} | '
                  f'config Mth={of["mode_threshold_px"]:g} px | lag={a.lag_seconds:g}s')
            for v in videos:
                rel = v.relative_to(scene)
                if groups and rel.parts[0] not in groups:
                    continue
                region = region_for_name(regions, rel.as_posix())
                crop, keep = None, None
                if region['mode'] == 'crop':
                    crop = tuple(region['box'])
                elif region['mode'] == 'mask' and 'path' in region:
                    keep = cv2.imread(str(region['path']), cv2.IMREAD_GRAYSCALE) > 0  # black = excluded
                p = SimpleNamespace(lag_seconds=a.lag_seconds, lag_frames=a.lag_frames, scale=a.scale, block_size=of['block_size'],
                                    texture_thr=a.texture_thr, min_valid_blocks=a.min_valid_blocks,
                                    bin_px=a.bin_px, crop=crop, keep_mask=keep, max_samples=a.max_samples)
                print(f'  {rel}', flush=True)
                process_video(v, {'scene': scene.name, 'platform': rel.parts[0],
                                  'mth': of['mode_threshold_px'], 'video': rel.as_posix(),
                                  'camera': camera_name({'platform': rel.parts[0], 'video': rel.as_posix()})}, p, writer)

    # ---- summary per platform (all scenes pooled) and per scene/platform
    rows = list(csv.DictReader(csv_path.open()))
    out = {'settings': {'lag_seconds': a.lag_seconds, 'speed_percentile': a.speed_percentile,
                        'min_valid_frac': a.min_valid_frac, 'min_speed_px_s': a.min_speed},
           'platforms': {}, 'by_scene': {}}

    def usable(rs):
        return [r for r in rs if r['mode_px'] != 'nan'
                and float(r['valid_block_frac']) >= a.min_valid_frac
                and float(r['speed_px_s']) >= a.min_speed]

    by_platform, by_scene = {}, {}
    for r in rows:
        by_platform.setdefault(r['platform'], []).append(r)
        by_scene.setdefault((r['scene'], r['platform']), []).append(r)

    for plat, rs in sorted(by_platform.items()):
        ok = usable(rs)
        e = {'pairs_total': len(rs), 'pairs_used': len(ok),
             'mean_valid_block_frac': float(np.mean([float(r['valid_block_frac']) for r in rs])),
             'config_mth_px': float(rs[0]['config_mth_px'])}
        print(f'\n=== {plat} === pairs: {len(rs)} total, {len(ok)} used '
              f'(valid blocks >= {a.min_valid_frac:g}, speed >= {a.min_speed:g} px/s)')
        if not ok:
            print('  no usable pairs')
            out['platforms'][plat] = e
            continue
        speed = np.array([float(r['speed_px_s']) for r in ok])
        pct = np.array([float(r['speed_pct_width_s']) for r in ok])
        width = float(np.median([float(r['width_px']) for r in ok]))
        typ = float(np.percentile(speed, a.speed_percentile))
        typ_pct = float(np.percentile(pct, a.speed_percentile))
        e.update({'image_width_px': width, 'speed_px_s': pcts(speed), 'speed_pct_width_s': pcts(pct),
                  'typical_speed_px_s': typ, 'typical_speed_pct_width_s': typ_pct, 'suggested_mth': {}})
        print('  image velocity (px/s):      ' + '  '.join(f'P{q}={np.percentile(speed, q):.1f}' for q in PERCENTILES))
        print('  image velocity (% width/s): ' + '  '.join(f'P{q}={np.percentile(pct, q):.1f}' for q in PERCENTILES))
        print(f'  typical (P{a.speed_percentile:g}): {typ:.1f} px/s = {typ_pct:.1f} % width/s  (width ~{width:.0f} px)')
        print(f'  {"target interval":>16} {"Mth (px)":>9} {"% width":>8}')
        for T in a.target_interval:
            mth = typ * T
            e['suggested_mth'][f'{T:g}s'] = {'mth_px': mth, 'pct_width': 100 * mth / width}
            print(f'  {T:>14g} s {mth:>9.1f} {100 * mth / width:>8.1f}')
        cfg = e['config_mth_px']
        e['config_mth_implied_interval_s'] = cfg / typ
        print(f'  config Mth={cfg:g} px -> one keyframe every ~{cfg / typ:.2f} s at typical motion')
        out['platforms'][plat] = e

    for (scene, plat), rs in sorted(by_scene.items()):
        ok = usable(rs)
        if ok:
            out['by_scene'][f'{scene}|{plat}'] = {
                'pairs_used': len(ok),
                'speed_px_s': pcts(np.array([float(r['speed_px_s']) for r in ok]))}
    (a.output / 'summary.json').write_text(json.dumps(out, indent=2))

    plot_distributions(rows, a.output, a.min_valid_frac, a.min_speed, a.hist_bins)
    print(f'\nSaved to {a.output}')


if __name__ == '__main__':
    main()