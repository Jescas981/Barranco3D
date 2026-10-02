import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lima3d import dense as dense
from lima3d import pipeline as pipeline


class DenseTests(unittest.TestCase):
    def test_preserve_legacy_identity_and_separate_checkpoints(self):
        self.assertEqual(pipeline.CACHE_IMPLEMENTATION, '1064e18ac06f324497fad9ecbe6ba2be7648621ec018577eb2afe3c2b7debbe6')
        self.assertNotEqual(dense.dense_config('mast3r'), dense.dense_config('mast3r-aerialmd'))

    def test_dense_presets_reach_all_coalitions(self):
        import h5py
        import numpy as np
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root/'Scene'
            for platform in ('Car','Drone','Pedestrian'):
                (scene/platform).mkdir(parents=True)
                (scene/platform/'frame_000001.jpg').write_bytes(b'image')
            args = SimpleNamespace(platforms=['Car','Drone','Pedestrian'],
                configs=['mast3r','mast3r-aerialmd'],global_feature='netvlad',
                sequential_window=0,dry_run=False,output_root=root/'outputs',
                resize_max=1024,max_keypoints=100,until='sfm',top_k=2,
                query_batch=2,database_batch=2,device='cpu',threads=1,
                seed=0,camera_mode='PER_FOLDER')
            def features(conf, scene, names, path, global_features, **kwargs):
                self.assertTrue(global_features)
                path.parent.mkdir(parents=True,exist_ok=True)
                with h5py.File(path,'w') as f:
                    for name in names: f.create_dataset(name+'/global_descriptor',data=np.ones(4))
            def raw(conf,scene,pairs,path,device,**kwargs):
                path.parent.mkdir(parents=True,exist_ok=True)
                with h5py.File(path,'w') as f:
                    from hloc.utils.parsers import names_to_pair
                    for pair in pairs:
                        g=f.create_group(names_to_pair(*pair))
                        g.create_dataset('keypoints0',data=np.array([[3.,4.]]))
                        g.create_dataset('keypoints1',data=np.array([[4.,4.]]))
                        g.create_dataset('scores',data=np.array([.8]))
            with contextlib.redirect_stdout(io.StringIO()), patch.object(pipeline,'ensure_features',side_effect=features) as extract, patch.object(dense,'ensure_dense_raw',side_effect=raw) as infer, patch.object(pipeline,'reconstruct',return_value={'status':'no_model'}) as sfm:
                pipeline.run_scene(scene,args)
            self.assertEqual(extract.call_count,1)
            self.assertEqual(infer.call_count,2)
            self.assertEqual(sfm.call_count,14)
            self.assertNotEqual(infer.call_args_list[0].args[3],infer.call_args_list[1].args[3])
            for call in sfm.call_args_list:
                self.assertTrue(call.args[3].is_file())
                self.assertTrue(call.args[4].is_file())

    def test_raw_resume_and_coalition_isolation(self):
        import cv2
        import h5py
        import numpy as np
        import torch
        from hloc.utils.parsers import names_to_pair
        torch.set_num_threads(1)
        calls = []
        class FakeMatcher(torch.nn.Module):
            def __init__(self, conf):
                super().__init__()
            def forward(self, data):
                calls.append(1)
                assert float(data['image0'].min()) == -1.0
                return {'keypoints0': torch.tensor([[4.,4.], [8.,8.]]),
                        'keypoints1': torch.tensor([[5.,4.], [9.,8.]]),
                        'scores': torch.tensor([.8,.9])}
            def forward_batch(self, data):
                return [self.forward({'image0': image[None]}) for image in data['image0']]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ['Car/a.jpg', 'Drone/b.jpg', 'Pedestrian/c.jpg']
            for name in names:
                (root/name).parent.mkdir()
                cv2.imwrite(str(root/name), np.zeros((32,48,3),np.uint8))
            raw = root/'raw.h5'
            pairs = [(names[0],names[1]), (names[0],names[2])]
            with contextlib.redirect_stdout(io.StringIO()), patch('hloc.utils.base_model.dynamic_load', return_value=FakeMatcher):
                dense.ensure_dense_raw(dense.dense_config('mast3r'),root,pairs[:1],raw,'cpu')
                self.assertEqual(len(calls),1)
                dense.ensure_dense_raw(dense.dense_config('mast3r'),root,pairs,raw,'cpu')
                self.assertEqual(len(calls),2)
                dense.ensure_dense_raw(dense.dense_config('mast3r'),root,pairs,raw,'cpu')
                self.assertEqual(len(calls),2)
                with h5py.File(raw,'a') as f:
                    del f[names_to_pair(*pairs[1])]['scores']
                dense.ensure_dense_raw(dense.dense_config('mast3r'),root,pairs,raw,'cpu')
                self.assertEqual(len(calls),3)
                batched = root/'batched.h5'
                dense.ensure_dense_raw(dense.dense_config('mast3r'),root,pairs,batched,'cpu',
                                       batch_size=2,loader_workers=2,prefetch=2)
                with h5py.File(raw) as single, h5py.File(batched) as batch:
                    for pair in pairs:
                        key = names_to_pair(*pair)
                        for field in ('keypoints0','keypoints1','scores'):
                            np.testing.assert_array_equal(single[key][field][...],batch[key][field][...])
                before = len(calls)
                dense.ensure_dense_raw(dense.dense_config('mast3r'),root,pairs,batched,'cpu',batch_size=4)
                self.assertEqual(len(calls),before)
                # Un par externo con coordenadas distintas no debe influir en Car+Drone.
                with h5py.File(raw,'a') as f:
                    f[names_to_pair(*pairs[1])]['keypoints0'][...] = [[20,20],[24,24]]
                features,matches=dense.assemble_dense(raw,pairs[:1],names[:2],root/'assembled',100)
                with h5py.File(features) as f:
                    self.assertEqual(sorted(f.keys()),['Car','Drone'])
                    self.assertLess(f[names[0]]['keypoints'][...].max(),10)
                with h5py.File(matches) as f:
                    np.testing.assert_array_equal(f[names_to_pair(*pairs[0])]['matches0'][...],[0,1])
                stat=features.stat().st_mtime_ns
                dense.assemble_dense(raw,pairs[:1],names[:2],root/'assembled',100)
                self.assertEqual(features.stat().st_mtime_ns,stat)
                f_empty,_=dense.assemble_dense(raw,[],[names[0]],root/'assembled',100)
                with h5py.File(f_empty) as f:
                    self.assertEqual(f[names[0]]['keypoints'].shape,(0,2))


if __name__=='__main__':
    unittest.main()
