import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lima3d import artifacts as artifacts
from lima3d import config as configuration
from lima3d import pipeline as pipeline
from lima3d import scheduler as scheduler
from lima3d import stages as stages


def arguments(root, experiment='a'):
    return SimpleNamespace(output_root=root/'outputs', platforms=['Car','Drone','Pedestrian'],
        global_feature='netvlad', top_k=1, sequential_window=0, max_keypoints=128,
        resize_max=128, seed=0, camera_mode='PER_FOLDER', configs=['sift'],
        experiment=experiment, query_batch=2, database_batch=2, threads=1,
        sfm_threads=1, device='cpu', requested_device='cpu', sift_device='cpu', mvs=True)


def make_scene(root):
    scene=root/'frames'/'Scene'
    for platform in ('Car','Drone','Pedestrian'):
        (scene/platform).mkdir(parents=True)
        (scene/platform/'clip__000001.jpg').write_bytes(platform.encode())
        (scene/platform/'clip__000002.jpg').write_bytes((platform+'2').encode())
    return scene


class PortabilityTests(unittest.TestCase):
    def test_snapshot_survives_move_and_mtime_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            machine1=Path(tmp)/'first'; scene=make_scene(machine1)
            args=arguments(machine1)
            _,snapshot,_,root=stages.context(scene,args)
            (root/'sentinel').write_text('reuse')
            machine2=Path(tmp)/'second'
            shutil.copytree(machine1,machine2)
            scene2=machine2/'frames/Scene'
            for p in scene2.rglob('*.jpg'): os.utime(p,(1,1))
            _,new_snapshot,_,new_root=stages.context(scene2,arguments(machine2))
            self.assertEqual(snapshot,new_snapshot)
            self.assertEqual((new_root/'sentinel').read_text(),'reuse')
            (scene2/'Car/clip__000001.jpg').write_bytes(b'changed')
            self.assertNotEqual(stages.context(scene2,arguments(machine2))[1],snapshot)

    def test_adopt_legacy_and_rebase_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); scene=make_scene(root); args=arguments(root)
            images,legacy=pipeline.inventory(scene,args.platforms)
            old=args.output_root/'Scene'/legacy
            old.mkdir(parents=True)
            pipeline.save_json(old/'dataset.json',{'scene':str(scene),'images':images})
            _,_,_,adopted=stages.context(scene,args)
            self.assertEqual(old,adopted)
            self.assertEqual(artifacts.artifact_path(Path('/new/outputs/Scene')/legacy, str(old/'features/f.h5')),
                             Path('/new/outputs/Scene')/legacy/'features/f.h5')

    def test_run_identity_does_not_depend_on_threads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            man={'preset':'sift','run_base':{'matches':'a','retrieval':'b','seed':0,'camera_mode':'PER_FOLDER'}}
            previous=root/'reconstructions'/('sift-'+pipeline.digest({**man['run_base'],'threads':6}))
            previous.mkdir(parents=True)
            pipeline.save_json(previous/'config.json',{'threads':6})
            self.assertEqual(artifacts.reconstruction_root(root,man),previous)
            self.assertEqual(artifacts.reconstruction_root(root,man),previous)

    def test_config_paths_relative_to_yaml_and_inactive_gpu(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); make_scene(root)
            config={'data':{'frames_root':'frames','output_root':'outputs'},
                    'resources':{'sfm':{'threads':1},'mvs':{'gpus':4}},
                    'experiments':[{'name':'sift','preset':'sift'}]}
            path=root/'config.yaml'; path.write_text(yaml.safe_dump(config))
            cfg=configuration.load_config(path)
            self.assertEqual(cfg.scenes,[root/'frames/Scene'])
            self.assertEqual(cfg.output_root,root/'outputs')
            with patch.object(scheduler.subprocess,'run',side_effect=AssertionError('No GPU query for CPU stage')):
                scheduler.validate_active_gpus(cfg,{'sfm'})
            with patch.object(scheduler.os,'sched_getaffinity',return_value=set(range(16))):
                gpu,cpu,allowed=scheduler.make_slots(cfg.res,{'sfm'})
                self.assertEqual(gpu,[])
                self.assertEqual(len(cpu),16)

    def test_retrieval_cached_per_settings_then_portable_bank(self):
        import h5py
        import numpy as np
        from hloc import extract_features
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'machine1'; scene=make_scene(root)
            args=arguments(root)
            calls=[]
            def fake_global(conf, scene, image_list, feature_path, **kwargs):
                calls.append(list(image_list))
                with h5py.File(feature_path,'a') as f:
                    for i,name in enumerate(image_list):
                        f.create_dataset(name+'/global_descriptor',data=np.array([1.,i,1.]))
                        f.create_dataset(name+'/image_size',data=[64,48])
            with patch.object(extract_features,'main',side_effect=fake_global), contextlib.redirect_stdout(io.StringIO()), patch.object(stages,'retrieval_pairs',wraps=stages.retrieval_pairs) as retrieval:
                stages.stage_pairs(scene,args)
                args2=deepcopy(args); args2.experiment='b'
                stages.stage_pairs(scene,args2)
                self.assertEqual(len(calls),1)
                self.assertEqual(retrieval.call_count,7)
                seq=deepcopy(args);seq.experiment='seq';seq.sequential_window=1
                stages.stage_pairs(scene,seq)
                self.assertEqual(retrieval.call_count,7)
                self.assertEqual(len(calls),1)
                args3=deepcopy(args);args3.experiment='c';args3.top_k=2
                stages.stage_pairs(scene,args3)
                self.assertEqual(len(calls),1)
                self.assertEqual(retrieval.call_count,14)
            def fake_local(conf,scene,names,path,**kwargs):
                path.parent.mkdir(parents=True,exist_ok=True)
                with h5py.File(path,'w') as f:
                    for name in names:
                        f.create_dataset(name+'/keypoints',data=np.zeros((0,2)))
                        f.create_dataset(name+'/descriptors',data=np.zeros((128,0)))
                        f.create_dataset(name+'/scores',data=np.zeros(0))
                        f.create_dataset(name+'/image_size',data=[64,48])
            with patch.object(stages,'ensure_features',side_effect=fake_local), contextlib.redirect_stdout(io.StringIO()):
                stages.bank_preset(scene,args,'sift')
            _,_,_,bankroot=stages.context(scene,args)
            manifest=json.loads((bankroot/'jobs/a.json').read_text())
            self.assertTrue(all(not Path(j['pairs']).is_absolute() for j in manifest['jobs']))
            other=Path(tmp)/'machine2';shutil.copytree(root,other)
            copied_args=arguments(other);copied_args.threads=8;copied_args.sfm_threads=16
            _,_,_,copyroot=stages.context(other/'frames/Scene',copied_args)
            artifacts.check_bank_request(manifest,copied_args)
            for j in manifest['jobs']:
                resolved=artifacts.resolve_job(copyroot,j)
                self.assertTrue(all(Path(resolved[k]).is_file() for k in ('pairs','features','matches')))
                job={**resolved,'scene':str(other/'frames/Scene'),'run_root':str(copyroot/'reconstructions/test'),
                     'seed':0,'camera_mode':'PER_FOLDER'}
                with patch.object(stages,'reconstruct',return_value={'status':'no_model'}) as recon:
                    stages.sfm_worker(job,16)
                    self.assertEqual(recon.call_args.args[0],other/'frames/Scene')
            copied_args.top_k=99
            with self.assertRaisesRegex(ValueError,'otros parámetros'):
                artifacts.check_bank_request(manifest,copied_args)


if __name__=='__main__': unittest.main()
