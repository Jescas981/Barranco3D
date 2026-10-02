"""Bounded CPU prefetch and shape-compatible batches without image padding."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque


def prefetched(dataset, workers=1, prefetch=4):
    if type(workers) is not int or workers < 0 or type(prefetch) is not int or prefetch < 1:
        raise ValueError('workers must be nonnegative and prefetch must be positive integers')
    if workers == 0:
        for index in range(len(dataset)):
            yield dataset[index]
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        indices = iter(range(len(dataset)))
        for _ in range(min(prefetch, len(dataset))):
            pending.append(pool.submit(dataset.__getitem__, next(indices)))
        while pending:
            value = pending.popleft().result()
            index = next(indices, None)
            if index is not None:
                pending.append(pool.submit(dataset.__getitem__, index))
            yield value


def compatible_batches(items, batch_size, shape):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    # Only one bounded window is retained, even with many distinct shapes.
    window = []
    for item in items:
        window.append(item)
        if len(window) == batch_size:
            yield from _group(window, shape)
            window = []
    yield from _group(window, shape)


def _group(window, shape):
    groups = {}
    for item in window:
        groups.setdefault(shape(item), []).append(item)
    yield from groups.values()


def bounded_map(executor, function, items, max_in_flight):
    """Ordered results with at most max_in_flight submitted jobs, including running jobs."""
    if type(max_in_flight) is not int or max_in_flight < 1:
        raise ValueError('max_in_flight must be positive')
    items = iter(items)
    pending = deque()
    try:
        for _ in range(max_in_flight):
            item = next(items, None)
            if item is None:
                break
            pending.append(executor.submit(function, item))
        while pending:
            yield pending.popleft().result()
            item = next(items, None)
            if item is not None:
                pending.append(executor.submit(function, item))
    finally:
        for future in pending:
            future.cancel()


def inference_jobs(factory, function, batches, workers=1, max_in_flight=1, device='cpu'):
    """One model and CUDA stream per worker; results must be materialized on CPU."""
    import threading
    import torch
    from contextlib import nullcontext
    local = threading.local()
    initialization_lock = threading.Lock()

    def run(batch):
        if not hasattr(local, 'model'):
            local.stream = torch.cuda.Stream(device=device) if str(device).startswith('cuda') else None
            with torch.cuda.stream(local.stream) if local.stream is not None else nullcontext():
                with initialization_lock:
                    local.model = factory()
        with torch.inference_mode(), (torch.cuda.stream(local.stream) if local.stream is not None else nullcontext()):
            result = function(local.model, batch)
            if local.stream is not None:
                local.stream.synchronize()
            return result

    if workers == 1:
        for batch in batches:
            yield run(batch)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            yield from bounded_map(pool, run, batches, max_in_flight)
