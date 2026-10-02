"""Spawn-safe SIFT workers: independent native extractors, no HDF5 writes."""
_extractor = None


def initialize(options, device, threads):
    import pycolmap
    global _extractor
    _extractor = pycolmap.FeatureExtractor.create(
        options=pycolmap.FeatureExtractionOptions(num_threads=threads,
            use_gpu=device == 'cuda', sift=pycolmap.SiftExtractionOptions(options)),
        device=getattr(pycolmap.Device, device))


def extract_batch(batch):
    import numpy as np
    import pycolmap
    results = []
    for index, data in batch:
        gray = data['image'][0]
        bitmap = pycolmap.Bitmap.from_array(np.rint(gray * 255).clip(0, 255).astype(np.uint8))
        raw_keypoints, raw_descriptors = _extractor.extract(bitmap)
        keypoints = np.asarray([(p.x, p.y) for p in raw_keypoints], dtype=np.float32).reshape(-1, 2)
        descriptors = np.asarray(raw_descriptors.to_float().data, dtype=np.float32)
        scales = data['original_size'] / np.array(gray.shape[::-1])
        keypoints = (keypoints + 0.5) * scales[None] - 0.5
        descriptors = descriptors / np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-8)
        results.append((index, keypoints, descriptors, scales, data['original_size']))
    return results


def extract_batches(batches, options, device, threads, workers, max_in_flight):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from .utils.batching import bounded_map
    if workers == 1:
        initialize(options, device, threads)
        try:
            for batch in batches:
                yield extract_batch(batch)
        finally:
            global _extractor
            _extractor = None
    else:
        # Spawn is required: forking a process that initialized CUDA is unsafe.
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                                 initializer=initialize, initargs=(options, device, threads)) as pool:
            yield from bounded_map(pool, extract_batch, batches, max_in_flight)
