#!/usr/bin/env python3
"""Motor de colas: pares -> banco -> SfM -> MVS, según las etapas activadas.

También es el punto de entrada de los subprocesos (--worker)."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .utils.paths import REPO_ROOT

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .config import load_config
from .pipeline import (PRESETS, coalitions, run_scene,
                                setup_runtime)
from .utils.io import digest, save_json, scene_lock
from .stages import (bank_preset, check_dependencies, context, sfm_worker,
                              stage_pairs)

from .artifacts import check_bank_request, resolve_job, reconstruction_root

HEAVY = {'mast3r': 3, 'mast3r-aerialmd': 3, 'sp-sg': 2, 'sp-lg': 2, 'sift': 1}
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
ALL_STAGES = ('pairs', 'bank', 'sfm', 'mvs')


# ------------------------------------------------------------------ workers
def run_worker(a):
    cfg = load_config(a.config, [a.experiment], [a.scene])
    args, scene = cfg.experiments[a.experiment], cfg.scenes[0]
    try:
        if a.worker == 'pairs':
            setup_runtime(args)
            stage_pairs(scene, args)
        elif a.worker == 'bank':
            setup_runtime(args)
            bank_preset(scene, args, args.configs[0])
        elif a.worker == 'sfm':
            job = json.loads(Path(a.task_json).read_text())
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            for v in THREAD_VARS:
                os.environ[v] = str(args.sfm_threads)
            import torch
            torch.set_num_threads(args.sfm_threads)
            save_json(Path(job['result_file']), sfm_worker(job, args.sfm_threads)[2])
        elif a.worker == 'mvs':
            from .mvs import run_mvs
            job = json.loads(Path(a.task_json).read_text())
            m = cfg.mvs  # CUDA_VISIBLE_DEVICES expone una sola GPU: para COLMAP es la 0
            run_mvs(Path(job['model']), Path(job['images']), None, colmap=m['colmap'],
                    max_image_size=m['max_image_size'], cache_gb=m['cache_gb'],
                    threads=cfg.res['mvs']['threads'], gpu_index='0',
                    num_sources=m['num_sources'])
    except Exception:
        traceback.print_exc()
        sys.exit(1)


# ------------------------------------------------------------------ utilidades
class Task:
    def __init__(self, kind, name, resource, priority, cmd, done, group=None, mode='w'):
        self.kind, self.name, self.resource = kind, name, resource
        self.priority, self.cmd, self.done = priority, cmd, done
        self.group, self.mode = group, mode  # 'w' exclusivo, 'r' compartido


def cached_result(folder):
    report = folder / 'result.json'
    if not report.is_file():
        return None
    try:
        result = json.loads(report.read_text())
    except ValueError:
        return None
    if result.get('status') == 'no_model':
        return result
    if result.get('status') == 'complete':
        model = folder / result.get('model_path', '')
        if all((model / f).is_file() and (model / f).stat().st_size > 0
               for f in ('cameras.bin', 'images.bin', 'points3D.bin')):
            return result
    return None


def make_slots(res, stages):
    """Devuelve (slots GPU, slots CPU para SfM, GPU permitidas por etapa)."""
    cores = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else []
    bank_ids = res['bank_ids'] if stages & {'pairs', 'bank'} else []
    mvs_ids = res['mvs_ids'] if 'mvs' in stages else []
    allowed = {'bank': set(bank_ids), 'mvs': set(mvs_ids)}
    ids = sorted(set(bank_ids) | set(mvs_ids))
    # Una GPU compartida reserva el máximo de núcleos de sus dos usos.
    per = {g: max(res['bank']['threads'] if g in bank_ids else 0,
                  res['mvs']['threads'] if g in mvs_ids else 0) for g in ids}
    st = res['sfm']['threads']
    want_cpu = 'sfm' in stages
    if not cores:
        return ([(g, None) for g in ids],
                [None] * (res['sfm']['workers'] or 4) if want_cpu else [], allowed)
    need = sum(per.values()) + (st if want_cpu else 0)
    if len(cores) < need:
        raise SystemExit(f'Se necesitan al menos {need} núcleos; hay {len(cores)}')
    gpu_slots, i = [], 0
    for g in ids:
        gpu_slots.append((g, cores[i:i + per[g]]))
        i += per[g]
    rest = cores[i:]
    capacity = len(rest) // st if want_cpu else 0
    workers = min(res['sfm']['workers'] or capacity, capacity)
    if res['sfm']['workers'] > capacity:
        print(f'AVISO: {res["sfm"]["workers"]} workers no caben; uso {capacity}', flush=True)
    return gpu_slots, [rest[j * st:(j + 1) * st] for j in range(workers)], allowed


# ------------------------------------------------------------------ orquestador
def orchestrate(cfg, stages):
    out = cfg.output_root
    log_dir, task_dir = out / '_logs', out / '_tasks'
    for d in (log_dir, task_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Prefer the MVS-enabled alias when identical reconstructions are deduplicated.
    exps = dict(sorted(cfg.experiments.items(), key=lambda item: not item[1].mvs))
    gpu_free, cpu_free, allowed = make_slots(cfg.res, stages)
    print(f'Etapas: {", ".join(s for s in ALL_STAGES if s in stages)} | '
          f'banco: GPU {sorted(allowed["bank"])} | mvs: GPU {sorted(allowed["mvs"])} | '
          f'SfM: {len(cpu_free)} workers x {cfg.res["sfm"]["threads"]} threads', flush=True)

    first = next(iter(exps.values()))
    info = {}
    for scene in cfg.scenes:
        _, _, names, root = context(scene, first)
        info[scene.name] = {'root': root, 'n': len(names), 'scene': scene}

    ready = {'gpu': [], 'cpu': []}
    running, busy, runs, failures = [], {}, {}, []
    stats = dict.fromkeys(ALL_STAGES, 0)
    waiting_bank = {}
    expanded_runs = set()
    t0 = time.time()

    # ---- panel de progreso
    ncoal = len(coalitions(first.platforms))
    n_jobs = len(info) * len(exps)
    totals = {'pairs': n_jobs, 'bank': n_jobs, 'sfm': n_jobs * ncoal,
              'mvs': len(info) * sum(bool(e.mvs) for e in exps.values()) * ncoal}  # cota superior
    active = [s for s in ALL_STAGES if s in stages]
    bars, live = {}, []
    if tqdm:
        for i, k in enumerate(active):
            bars[k] = tqdm(total=totals[k], desc=f'{k:5}', position=i, dynamic_ncols=True)
        for j in range(len(gpu_free) + 1):  # una línea por GPU + una para SfM
            live.append(tqdm(total=0, position=len(active) + j, bar_format='{desc}',
                             dynamic_ncols=True))

    def say(msg):
        (tqdm.write if tqdm else print)(msg)

    def tail(task):
        try:
            with (log_dir / f'{task.kind}-{task.name}.log').open('rb') as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 2000))
                text = f.read().decode(errors='ignore')
        except OSError:
            return ''
        parts = [s.strip() for s in text.replace('\r', '\n').split('\n') if s.strip()]
        return parts[-1][:80] if parts else ''

    def refresh():
        for k, bar in bars.items():
            bar.n = stats[k]
            bar.set_postfix(fail=len(failures))
            bar.refresh()
        if not live:
            return
        gpu_tasks = [t for _, t, _ in running if t.resource == 'gpu']
        for i, line in enumerate(live[:-1]):
            if i < len(gpu_tasks):
                t = gpu_tasks[i]
                line.set_description_str(f'GPU task: {t.kind} {t.name} | {tail(t)}')
            else:
                line.set_description_str('GPU idle')
        n_sfm = sum(t.resource == 'cpu' for _, t, _ in running)
        live[-1].set_description_str(f'SfM workers running: {n_sfm}')

    def close_bars():
        refresh()
        for bar in [*bars.values(), *live]:
            bar.close()

    def stamp():
        s = int(time.time() - t0)
        return f'[{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}]'

    def cmd_for(kind, scene, exp, task_json=None):
        cmd = [sys.executable, '-m', 'lima3d.scheduler', '--config', str(cfg.path),
               '--worker', kind, '--experiment', exp, '--scene', scene]
        return cmd + (['--task-json', str(task_json)] if task_json else [])

    def fail(label, reason):
        failures.append(f'{label}: {reason}')
        say(f'{stamp()} FALLO {label}: {reason}')

    def key_of(exp):  # sp-sg y sp-lg comparten .h5 local; los densos, el raw por preset
        preset = exps[exp].configs[0]
        return PRESETS[preset][0] or preset

    # ---- SfM y MVS
    def expand(sname, exp):
        args = exps[exp]
        root = info[sname]['root']
        path = root / 'jobs' / f'{exp}.json'
        if not path.is_file():
            fail(f'{sname}/{exp}', f'falta {path}; corre build_bank_matching.py')
            return
        man = json.loads(path.read_text())
        check_bank_request(man, args)
        run_root = reconstruction_root(root, man)
        if str(run_root) in expanded_runs:
            say(f'[CACHE] Experimento equivalente: {exp} -> {run_root.name}')
            return
        expanded_runs.add(str(run_root))
        if not (run_root / 'config.json').exists():
            save_json(run_root / 'config.json',
                      {**man['config'], 'experiment': exp, 'threads': args.sfm_threads,
                       'run_base': man['run_base']})
        runs[str(run_root)] = {'left': len(man['jobs']), 'summary': {}}
        for j in man['jobs']:
            job = {**resolve_job(root, j), 'scene': str(info[sname]['scene']), 'run_root': str(run_root),
                   'seed': man['run_base']['seed'], 'camera_mode': man['run_base']['camera_mode']}
            if 'sfm' in stages:
                submit_sfm(sname, exp, job)
            else:  # solo MVS: usa los SfM ya terminados
                res = cached_result(run_root / job['label'])
                if res is not None:
                    queue_mvs(sname, exp, job, res)
                elif args.mvs:
                    fail(f'mvs/{sname}/{exp}/{job["label"]}', 'SfM pendiente o incompleto; ejecuta build_sfm_sparser.py')

    def submit_sfm(sname, exp, job):
        cached = cached_result(Path(job['run_root']) / job['label'])
        if cached is not None:
            sfm_finished(sname, exp, job, cached)
            return
        tid = f'{sname}-{exp}-{job["label"]}'
        job['result_file'] = str(task_dir / f'sfm-{tid}.result.json')
        Path(job['result_file']).unlink(missing_ok=True)
        path = task_dir / f'sfm-{tid}.json'
        save_json(path, job)

        def done(ok):
            try:
                if not ok:
                    raise ValueError('worker SfM falló')
                res = json.loads(Path(job['result_file']).read_text())
            except (OSError, ValueError):
                res = {'status': 'error', 'error': 'el proceso terminó sin resultado'}
            sfm_finished(sname, exp, job, res)

        ready['cpu'].append(Task('sfm', tid, 'cpu', (-len(job['members']),),
                                 cmd_for('sfm', sname, exp, path), done,
                                 group=(sname, key_of(exp)), mode='r'))

    def sfm_finished(sname, exp, job, res):
        stats['sfm'] += 1
        run = runs.get(job['run_root'])
        if run is not None:
            run['summary'][job['label']] = res
            run['left'] -= 1
            if run['left'] == 0:
                save_json(Path(job['run_root']) / 'summary.json', run['summary'])
        if res['status'] == 'error':
            fail(f'sfm/{sname}/{exp}/{job["label"]}', res.get('error', ''))
            return
        queue_mvs(sname, exp, job, res)

    def queue_mvs(sname, exp, job, res):
        if 'mvs' not in stages or not exps[exp].mvs or res['status'] != 'complete':
            return
        model = Path(job['run_root']) / job['label'] / res['model_path']
        try:
            size = (model / 'images.bin').stat().st_size
        except OSError:
            size = 0
        tid = f'{sname}-{exp}-{job["label"]}'
        path = task_dir / f'mvs-{tid}.json'
        save_json(path, {'model': str(model), 'images': job['scene']})

        def done(ok):
            stats['mvs'] += ok
            if not ok:
                fail(f'mvs/{tid}', 'ver log')

        ready['gpu'].append(Task('mvs', tid, 'gpu', (2, -size),
                                 cmd_for('mvs', sname, exp, path), done))

    # ---- pares y banco
    def on_pairs(sname, exp, ok):
        bank = waiting_bank.pop((sname, exp), None)
        if not ok:
            fail(f'pairs/{sname}/{exp}', 'ver log')
            if bank:
                failures.append(f'bank/{sname}/{exp}: omitido (fallaron los pares)')
            return
        stats['pairs'] += 1
        if bank:
            ready['gpu'].append(bank)

    def on_bank(sname, exp, ok):
        if not ok:
            fail(f'bank/{sname}/{exp}', 'ver log')
            return
        stats['bank'] += 1
        if 'sfm' in stages:
            expand(sname, exp)

    # ---- semillas según las etapas activadas
    pair_groups = {}
    for sname, meta in info.items():
        for exp, args in exps.items():
            preset = args.configs[0]
            bank = Task('bank', f'{sname}-{exp}', 'gpu', (1, -meta['n'] * HEAVY[preset]),
                        cmd_for('bank', sname, exp),
                        (lambda ok, s=sname, e=exp: on_bank(s, e, ok)),
                        group=(sname, key_of(exp)), mode='w')
            if 'pairs' in stages:
                if 'bank' in stages:
                    waiting_bank[(sname, exp)] = bank
                pair_key = (sname, args.global_feature, args.top_k, args.sequential_window)
                pair_groups.setdefault(pair_key, []).append(exp)
            elif 'bank' in stages:
                ready['gpu'].append(bank)
            else:
                expand(sname, exp)

    for (sname, _, _, _), members in pair_groups.items():
        leader = members[0]
        def pairs_done(ok, sname=sname, members=members, leader=leader):
            if ok:
                root = info[sname]['root']
                index = json.loads((root / f'bank_index-{leader}.json').read_text())
                for exp in members[1:]:
                    save_json(root / f'bank_index-{exp}.json', index)
            for exp in members:
                on_pairs(sname, exp, ok)
        ready['gpu'].append(Task('pairs', f'{sname}-{leader}', 'gpu', (0, -info[sname]['n']),
                                 cmd_for('pairs', sname, leader), pairs_done,
                                 group=(sname, 'pairs'), mode='w'))

    # ---- bucle principal
    def runnable(task):
        if task.group is None:
            return True
        state = busy.get(task.group, 0)
        return state == 0 if task.mode == 'w' else state >= 0

    def pick(queue, ok=lambda t: True):
        queue.sort(key=lambda t: t.priority)
        for i, task in enumerate(queue):
            if runnable(task) and ok(task):
                return queue.pop(i)
        return None

    def launch(task, slot):
        gpu, cores = slot if task.resource == 'gpu' else (None, slot)
        if cores and task.resource == 'gpu':  # solo los núcleos que le tocan a su etapa
            cores = cores[:cfg.res['mvs' if task.kind == 'mvs' else 'bank']['threads']]
        env = os.environ.copy()
        # Workers must find both the src package and the repository's HLoc checkout.
        env['PYTHONPATH'] = os.pathsep.join(
            [str(REPO_ROOT), env.get('PYTHONPATH', '')])
        if gpu is not None:
            visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            ids = visible.split(',') if visible is not None else None
            env['CUDA_VISIBLE_DEVICES'] = ids[gpu] if ids is not None else str(gpu)
        pin = (lambda: os.sched_setaffinity(0, cores)) if cores else None
        with (log_dir / f'{task.kind}-{task.name}.log').open('a') as log:
            proc = subprocess.Popen(task.cmd, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, preexec_fn=pin, start_new_session=True)
        if task.group:
            busy[task.group] = -1 if task.mode == 'w' else busy.get(task.group, 0) + 1
        running.append((proc, task, slot))
        where = f' GPU {gpu}' if gpu is not None else ''
        say(f'{stamp()} START {task.kind} {task.name}{where}')

    def release(task):
        if not task.group:
            return
        if task.mode == 'w':
            busy.pop(task.group, None)
        else:
            busy[task.group] -= 1
            if busy[task.group] == 0:
                busy.pop(task.group)

    try:
        while any(ready.values()) or running:
            for slot in list(gpu_free):  # cada tarea solo usa las GPU de su etapa
                task = pick(ready['gpu'],
                            lambda t, g=slot[0]: g in allowed['mvs' if t.kind == 'mvs' else 'bank'])
                if task is not None:
                    gpu_free.remove(slot)
                    launch(task, slot)
            while cpu_free:
                task = pick(ready['cpu'])
                if task is None:
                    break
                launch(task, cpu_free.pop(0))
            for item in running[:]:
                proc, task, slot = item
                code = proc.poll()
                if code is None:
                    continue
                running.remove(item)
                (gpu_free if task.resource == 'gpu' else cpu_free).append(slot)
                release(task)
                if task.kind != 'sfm' or code != 0:
                    say(f'{stamp()} END   {task.kind} {task.name} (código {code})')
                task.done(code == 0)
                if task.kind == 'sfm':
                    say(f'{stamp()} sfm hechos: {stats["sfm"]}  mvs hechos: {stats["mvs"]}')
            refresh()
            time.sleep(0.5)
    except BaseException:
        # Cancelar también hijos (COLMAP/DataLoader) y esperar antes de soltar locks.
        for proc, _, _ in running:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        for proc, _, _ in running:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        close_bars()
        raise

    close_bars()
    print(f'\nPares {stats["pairs"]} | bancos {stats["bank"]} | SfM {stats["sfm"]} | '
          f'MVS {stats["mvs"]} | tiempo {stamp()}', flush=True)
    if failures:
        print('Fallos:\n  ' + '\n  '.join(failures), flush=True)
    return len(failures)


# ------------------------------------------------------------------ entradas
def describe(cfg, stages):
    print(f'Config: {cfg.path}\nEtapas: {", ".join(s for s in ALL_STAGES if s in stages)}')
    print(f'Recursos: {cfg.res}')
    for name, a in cfg.experiments.items():
        print(f'  - {name}: preset={a.configs[0]} global={a.global_feature} top_k={a.top_k} '
              f'seq={a.sequential_window} kp={a.max_keypoints} mvs={a.mvs}')
    first = next(iter(cfg.experiments.values()))
    for scene in cfg.scenes:
        run_scene(scene, first)


def validate_active_gpus(cfg, stages):
    required = set()
    if stages & {'pairs', 'bank'} and any(a.device != 'cpu' for a in cfg.experiments.values()):
        required.update(cfg.res['bank_ids'])
    if 'mvs' in stages and any(a.mvs for a in cfg.experiments.values()):
        required.update(cfg.res['mvs_ids'])
    if not required:
        return
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                                check=True, capture_output=True, text=True)
        available = len(result.stdout.splitlines())
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError('No se detectan GPU NVIDIA para la etapa seleccionada') from exc
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        available = 0 if visible in ('', '-1') else min(available, len(visible.split(',')))
    if any(g >= available for g in required):
        raise ValueError(f'GPU configuradas {sorted(required)}; hay {available} GPU visibles. Ajusta resources de esta etapa')


def main(stages, description=None):
    stages = frozenset(stages)
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', type=Path, default=Path('config.yaml'))
    p.add_argument('--experiments', nargs='+', help='Solo estos experimentos del config')
    p.add_argument('--scenes', nargs='+', help='Solo estas escenas (sobrescribe data.scenes)')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    cfg = load_config(a.config, a.experiments, a.scenes, a.dry_run)
    if a.dry_run:
        describe(cfg, stages)
        return
    if stages & {'pairs', 'bank'}:
        check_dependencies([e.configs[0] for e in cfg.experiments.values()])
    validate_active_gpus(cfg, stages)
    with scene_lock(cfg.output_root / '.scheduler.lock'):
        failures = orchestrate(cfg, stages)
    sys.exit(1 if failures else 0)


if __name__ == '__main__':  # modo worker
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--worker', required=True, choices=ALL_STAGES)
    p.add_argument('--experiment', required=True)
    p.add_argument('--scene', required=True)
    p.add_argument('--task-json')
    run_worker(p.parse_args())
