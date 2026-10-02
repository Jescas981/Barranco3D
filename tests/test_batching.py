"""Check bounded loading and variable-shape batch export without pretrained weights."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import h5py
from PIL import Image
from lima3d.utils.batching import prefetched, compatible_batches

class BatchTests(unittest.TestCase):
    def test_order_shapes_and_errors(self):
        self.assertEqual(list(prefetched(list(range(19)), 3, 4)), list(range(19)))
        self.assertEqual(list(compatible_batches([1,2,3,4,5],4,lambda x:x%2)),[[1,3],[2,4],[5]])
        class Broken:
            def __len__(self): return 8
            def __getitem__(self,index):
                if index==2: raise ValueError('decode error')
                return index
        with self.assertRaisesRegex(ValueError,'decode error'):
            list(prefetched(Broken(),2,3))

    def test_local_export_variable_shapes_and_keypoint_counts(self):
        from hloc import extract_features
        calls=[]
        class Model(torch.nn.Module):
            def __init__(self,conf): super().__init__()
            def forward(self,data):
                images=data['image']; calls.append(len(images))
                counts=[1 if x.mean()<.5 else 2 for x in images]
                return {'keypoints':[torch.ones(n,2) for n in counts],
                        'scores':[torch.ones(n) for n in counts],
                        'descriptors':[torch.ones(8,n) for n in counts]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);names=[]
            for i,(shape,value) in enumerate([((32,40),0),((32,40),255),((40,32),0),((32,40),0)]):
                name=f'{i}.png';names.append(name);Image.fromarray(np.full(shape,value,dtype=np.uint8)).save(root/name)
            paths=[]
            for size in (1,4):
                path=root/f'{size}.h5';paths.append(path)
                with patch.object(extract_features,'dynamic_load',return_value=Model), patch('torch.cuda.is_available',return_value=False):
                    extract_features.main({'model':{'name':'superpoint'},'preprocessing':{'grayscale':True}},root,
                        image_list=names,feature_path=path,batch_size=size,loader_workers=2,prefetch=3,
                        workers=2 if size==4 else 1,max_in_flight=3)
            self.assertIn(3,calls)
            with h5py.File(paths[0]) as single,h5py.File(paths[1]) as batch:
                for name in names:
                    for key in single[name]:
                        np.testing.assert_array_equal(single[name][key][...],batch[name][key][...])

    def test_execution_validation_and_merge(self):
        from types import SimpleNamespace
        from lima3d.execution import execution_options, validate_execution
        args=SimpleNamespace(execution={'local':{'workers':8,'max_in_flight':10,'batch_size':3}})
        self.assertEqual(execution_options(args,'local')['workers'],8)
        self.assertEqual(execution_options(args,'matching')['workers'],1)
        for value in ({'wrong':{}},{'local':{'workers':0}},{'dense':{'batch_size':True}}):
            with self.assertRaises(ValueError):validate_execution(value)

    def test_bounded_concurrent_inference(self):
        import threading
        from lima3d.utils.batching import inference_jobs
        barrier=threading.Barrier(2)
        lock=threading.Lock();models=[]
        def factory():
            model=object()
            with lock:models.append(model)
            return model
        def infer(model,value):
            self.assertFalse(torch.is_grad_enabled())
            if value<2:barrier.wait(timeout=5)
            return value*2
        self.assertEqual(list(inference_jobs(factory,infer,range(6),workers=2,max_in_flight=3)),[0,2,4,6,8,10])
        self.assertEqual(len(models),2)

    def test_matching_workers_batches_resume(self):
        from lima3d.pipeline import ensure_matches
        from hloc.utils.parsers import names_to_pair
        conf={'model':{'name':'nearest_neighbor','ratio_threshold':.8,'do_mutual_check':True}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);features=root/'features.h5'
            with h5py.File(features,'w') as f:
                for name in ('a','b','c','empty'):
                    n=0 if name=='empty' else 4
                    f.create_dataset(name+'/keypoints',data=np.zeros((n,2),np.float32))
                    f.create_dataset(name+'/descriptors',data=np.eye(4,dtype=np.float32)[:,:n])
                    f.create_dataset(name+'/scores',data=np.ones(n,np.float32))
                    f.create_dataset(name+'/image_size',data=[32,32])
            pairs=[('a','b'),('a','c'),('b','c'),('a','empty')]
            for size,workers in ((1,1),(2,2)):
                ensure_matches(conf,pairs,features,root/f'{size}.h5','cpu',batch_size=size,
                               workers=workers,max_in_flight=3,loader_workers=2)
            with h5py.File(root/'1.h5') as a,h5py.File(root/'2.h5') as b:
                for pair in pairs:
                    for key in ('matches0','matching_scores0'):
                        np.testing.assert_array_equal(a[names_to_pair(*pair)][key][...],b[names_to_pair(*pair)][key][...])
            before=(root/'2.h5').stat().st_mtime_ns
            with patch('hloc.utils.base_model.dynamic_load',side_effect=AssertionError('Should reuse cache')):
                ensure_matches(conf,pairs,features,root/'2.h5','cpu',workers=3)

    def test_nested_yaml_overrides(self):
        from lima3d.config import load_config
        from lima3d.execution import execution_options
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'frames/Scene/Car').mkdir(parents=True)
            config=root/'config.yaml'
            config.write_text('''data:
  frames_root: frames
  platforms: [Car]
defaults:
  execution:
    local: {workers: 2, prefetch: 7}
    matching: {workers: 3}
experiments:
  - name: sift
    preset: sift
    execution:
      local: {workers: 8, max_in_flight: 10, batch_size: 4}
''')
            args=load_config(config,dry_run=True).experiments['sift']
            self.assertEqual(execution_options(args,'local'),dict(batch_size=4,workers=8,
                prefetch=7,loader_workers=1,max_in_flight=10))
            self.assertEqual(execution_options(args,'matching')['workers'],3)

    def test_in_flight_is_bounded(self):
        from lima3d.utils.batching import bounded_map
        class Pool:
            outstanding=0
            peak=0
            def submit(self,function,item):
                self.outstanding+=1;self.peak=max(self.peak,self.outstanding)
                owner=self
                class Future:
                    def result(self):
                        owner.outstanding-=1
                        return function(item)
                    def cancel(self):pass
                return Future()
        pool=Pool()
        self.assertEqual(list(bounded_map(pool,lambda x:x,range(30),4)),list(range(30)))
        self.assertEqual(pool.peak,4)
