import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lima3d.regions import (parse_regions, prepared_scene, region_identity,
                           valid_points, filter_features, filter_dense)
from lima3d.artifacts import portable_context
from hloc.utils.parsers import names_to_pair


class RegionTests(unittest.TestCase):
    def make_inputs(self, root):
        scene = root / 'frames/Scene'
        for platform in ('Car', 'Drone'):
            folder = scene / platform
            folder.mkdir(parents=True)
            Image.fromarray(np.full((6, 8, 3), 150, np.uint8)).save(folder / 'frame.png')
        mask = root / 'mask.png'
        pixels = np.full((6, 8), 255, np.uint8)
        pixels[:, :3] = 0
        Image.fromarray(pixels).save(mask)
        return scene, mask

    def test_none_preserves_snapshot_crop_is_separate_and_original_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); scene, mask = self.make_inputs(root)
            args = SimpleNamespace(output_root=root/'outputs', platforms=['Car','Drone'])
            original = (scene/'Car/frame.png').read_bytes()
            baseline = portable_context(scene, args)
            args.regions = parse_regions({'Car': {'mode': 'none'}}, root, args.platforms)
            self.assertEqual(portable_context(scene, args)[1], baseline[1])
            args.regions = parse_regions({'Car': {'mode': 'crop', 'box': [1, 1, 7, 5]}}, root, args.platforms)
            _, snapshot, names, folder = portable_context(scene, args)
            self.assertNotEqual(snapshot, baseline[1])
            prepared = prepared_scene(scene, args, folder, names)
            with Image.open(prepared/'Car/frame.png') as image:
                self.assertEqual(image.size, (6, 4))
            with Image.open(prepared/'Drone/frame.png') as image:
                self.assertEqual(image.size, (8, 6))
            self.assertEqual((scene/'Car/frame.png').read_bytes(), original)
            prepared_scene(scene, args, folder, names)
            moved = root/'relocated'
            shutil.copytree(scene, moved)
            prepared_scene(moved, args, folder, names)
            self.assertEqual((prepared/'Drone/frame.png').resolve(), (moved/'Drone/frame.png').resolve())
            args.regions['Car']['box'] = [0, 0, 99, 99]
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                portable_context(scene, args)

    def test_masks_invalidate_cache_by_content_not_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); scene, mask = self.make_inputs(root)
            args = SimpleNamespace(output_root=root/'outputs', platforms=['Car','Drone'],
                                   regions={'Car': {'mode':'mask', 'path':str(mask)}})
            first = portable_context(scene, args)[1]
            other = root/'copy.png'; shutil.copy(mask, other)
            args.regions['Car']['path'] = str(other)
            self.assertEqual(portable_context(scene, args)[1], first)
            Image.new('L', (8, 6), 255).save(other)
            self.assertNotEqual(portable_context(scene, args)[1], first)
            Image.new('L', (4, 4), 255).save(other)
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                portable_context(scene, args)

    def test_sparse_and_dense_filtering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); scene, mask = self.make_inputs(root)
            regions = {'Car': {'mode':'mask', 'path':str(mask)}}
            names = ['Car/frame.png', 'Drone/frame.png']
            points = np.array([[1., 1.], [4., 2.], [-1., 0.], [float('nan'), 0.]])
            np.testing.assert_array_equal(valid_points(points, mask), [False, True, False, False])
            source = root/'features.h5'
            with h5py.File(source, 'w') as f:
                for name in names:
                    g=f.create_group(name)
                    g.create_dataset('keypoints', data=points[:2]).attrs['uncertainty']=2.
                    g.create_dataset('descriptors', data=np.arange(6).reshape(3, 2))
                    g.create_dataset('scores', data=[.2, .8])
                    g.create_dataset('image_size', data=[8, 6])
            target = filter_features(source, root/'filtered.h5', scene, names, regions)
            with h5py.File(target) as f, h5py.File(source) as original:
                self.assertEqual(f[names[0]]['descriptors'].shape, (3, 1))
                self.assertEqual(len(f[names[1]]['keypoints']), 2)
                self.assertEqual(len(original[names[0]]['keypoints']), 2)
                self.assertEqual(f[names[0]]['keypoints'].attrs['uncertainty'], 2.)
            raw=root/'raw.h5'; key=names_to_pair(*names)
            with h5py.File(raw, 'w') as f:
                g=f.create_group(key)
                g.create_dataset('keypoints0', data=points[:2])
                g.create_dataset('keypoints1', data=points[:2])
                g.create_dataset('scores', data=[.2, .8])
            dense=filter_dense(raw, root/'dense.h5', scene, [tuple(names)], regions)
            with h5py.File(dense) as f:
                self.assertEqual(len(f[key]['scores']), 1)

    def test_invalid_modes_and_per_image_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); scene, mask=self.make_inputs(root)
            for entry in ({'mode':'invalid'}, {'mode':'mask'}, {'mode':'crop','box':[1,2,0,0]}):
                with self.assertRaises(ValueError):
                    parse_regions({'Car':entry}, root, ['Car'])
            path=root/'masks/Scene/Car/frame.png.png'; path.parent.mkdir(parents=True)
            shutil.copy(mask,path)
            self.assertEqual(parse_regions({'Drone': {'mode':'crop', 'box':[0,0,2,2]}}, root, ['Car']), {})
            regions=parse_regions({'Car':{'mode':'mask','root':'masks'}},root,['Car'])
            self.assertIsNotNone(region_identity(scene,['Car/frame.png'],regions))
