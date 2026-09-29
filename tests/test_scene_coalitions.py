import contextlib
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('pipeline', Path(__file__).resolve().parents[1] / 'run_scene_coalitions.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
HAS_DEPS = all(importlib.util.find_spec(p) for p in ('torch', 'h5py', 'cv2', 'pycolmap'))


class PlanningTests(unittest.TestCase):
    def test_sift_device_selection(self):
        self.assertEqual(m.resolve_sift_device('auto', False, True), 'cpu')
        self.assertEqual(m.resolve_sift_device('auto', True, True), 'cuda')
        self.assertEqual(m.resolve_sift_device('cpu', True, True), 'cpu')
        self.assertEqual(m.resolve_sift_device('cuda', True, True), 'cuda')
        with self.assertRaisesRegex(RuntimeError, 'compilado sin CUDA'):
            m.resolve_sift_device('cuda', False, True)
        with self.assertRaisesRegex(RuntimeError, 'disponible'):
            m.resolve_sift_device('cuda', True, False)

    def test_sequential_pairs_numeric_order_and_boundaries(self):
        names = ['Car/cam0/a_mp4__10.jpg', 'Car/cam0/a_mp4__2.jpg',
                 'Car/cam0/a_mp4__1.jpg', 'Car/cam0/b_mp4__1.jpg',
                 'Car/cam1/a_mp4__1.jpg', 'Drone/a_mp4__1.jpg']
        expected = {tuple(sorted((names[2], names[1]))),
                    tuple(sorted((names[1], names[0])))}
        self.assertEqual(set(m.sequential_pairs(names, 1)), expected)
        self.assertEqual(len(m.sequential_pairs(names, 2)), 3)
        self.assertEqual(m.sequential_pairs(names, 0), [])
        # La unión con retrieval no duplica pares ya presentes.
        self.assertEqual(set(m.sequential_pairs(names, 1)) | expected, expected)
        with self.assertRaises(ValueError):
            m.sequential_pairs(['Drone/photo.jpg'], 1)
        with self.assertRaises(ValueError):
            m.sequential_pairs(['Drone/f_01.jpg', 'Drone/f_1.png'], 1)

    def test_coalitions_and_inventory(self):
        self.assertEqual(len(m.coalitions(['Car', 'Drone', 'Pedestrian'])), 7)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for p in ('Car', 'Drone', 'Pedestrian'):
                (root / p).mkdir()
                (root / p / 'x.jpg').write_bytes(b'image')
            images, first = m.inventory(root, ['Car', 'Drone', 'Pedestrian'])
            (root / 'Car' / 'x.jpg').write_bytes(b'changed image')
            self.assertNotEqual(first, m.inventory(root, list(images))[1])


@unittest.skipUnless(HAS_DEPS, 'Requiere entorno HLoc')
class PipelineTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)

    def test_retrieval_is_restricted_and_batched(self):
        import h5py
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'global.h5'
            names = ['Car/a.jpg', 'Car/b.jpg', 'Drone/c.jpg', 'Pedestrian/d.jpg']
            descriptors = np.random.default_rng(3).normal(size=(4, 8))
            with h5py.File(path, 'w') as f:
                for name, desc in zip(names, descriptors):
                    f.create_dataset(name + '/global_descriptor', data=desc)
            for subset in (names, names[:2], names[:1]):
                pairs = m.retrieval_pairs(path, subset, 2, 1, 1)
                ref = set()
                d = descriptors[:len(subset)]
                d = d / np.linalg.norm(d, axis=1, keepdims=True)
                sim = d @ d.T
                np.fill_diagonal(sim, -np.inf)
                for i, name in enumerate(subset):
                    for j in np.argsort(-sim[i])[:min(2, len(subset)-1)]:
                        ref.add(tuple(sorted((name, subset[j]))))
                self.assertEqual(pairs, sorted(ref))

    def test_sift_matches_cache_and_repair(self):
        import cv2
        import h5py
        import numpy as np
        from hloc import extract_features, match_features
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ['a.png', 'b.png', 'c.png']
            texture = np.random.default_rng(7).integers(0, 256, (160, 160), dtype=np.uint8)
            for i, name in enumerate(names):
                cv2.imwrite(str(root / name), np.roll(texture, i * 3, axis=1))
            conf = deepcopy(extract_features.confs['sift'])
            conf['model']['descriptor'] = 'sift'
            conf['model']['options'] = {'max_num_features': 128}
            path = root / 'local.h5'
            with contextlib.redirect_stdout(io.StringIO()):
                m.ensure_features(conf, root, names, path)
                with patch.object(m, 'extract_sift', side_effect=AssertionError('Cache miss')):
                    m.ensure_features(conf, root, names, path)
                matches = root / 'matches.h5'
                mc = match_features.confs['NN-ratio']
                m.ensure_matches(mc, [('a.png', 'b.png')], path, matches, 'cpu')
                with h5py.File(matches, 'r') as f:
                    original = f['a.png/b.png/matches0'][...]
                    self.assertGreater((original >= 0).sum(), 0)
                with patch('hloc.utils.base_model.dynamic_load', side_effect=AssertionError('Cache miss')):
                    m.ensure_matches(mc, [('a.png', 'b.png')], path, matches, 'cpu')
                m.ensure_matches(mc, [('a.png', 'b.png'), ('a.png', 'c.png')], path, matches, 'cpu')
                with h5py.File(matches, 'r') as f:
                    np.testing.assert_array_equal(original, f['a.png/b.png/matches0'][...])
                with h5py.File(matches, 'a') as f:
                    del f['a.png/c.png/matching_scores0']
                m.ensure_matches(mc, [('a.png', 'c.png')], path, matches, 'cpu')
                with h5py.File(matches, 'r') as f:
                    self.assertIn('matching_scores0', f['a.png/c.png'])

    def test_reconstruction_uses_only_members_and_distinguishes_errors(self):
        from hloc import reconstruction
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = root / 'pairs.txt'
            pairs.write_text('Car/a.jpg Drone/b.jpg\n')
            names = ['Car/a.jpg', 'Drone/b.jpg']
            with patch.object(reconstruction, 'main', return_value=None) as call:
                result = m.reconstruct(root, names, pairs, root/'f.h5', root/'m.h5', root/'sfm', 1, 0, 'PER_FOLDER')
                self.assertEqual(call.call_args.kwargs['image_list'], names)
                self.assertEqual(result['status'], 'no_model')
            with patch.object(reconstruction, 'main', side_effect=RuntimeError('failure')):
                with self.assertRaises(RuntimeError):
                    m.reconstruct(root, names, pairs, root/'f.h5', root/'m.h5', root/'failed', 1, 0, 'PER_FOLDER')
            self.assertEqual(json.loads((root/'failed/result.json').read_text())['status'], 'error')


if __name__ == '__main__':
    unittest.main()
