import contextlib
import io
import json
from pathlib import Path
import struct
import shutil
import os
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lima3d import mvs as m


def write_file(path,data=b'x'):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)


class MVSTests(unittest.TestCase):
    def test_resume_after_stereo_failure_and_skip_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); model=root/'sfm'; images=root/'images'; out=root/'dense'
            for name in ('cameras','images','points3D'): write_file(model/f'{name}.bin')
            names=['Car/a.jpg','Drone/b.jpg']
            for name in names: write_file(images/name)
            # Compatibilidad con las carpetas enlazadas del pipeline MASt3R previo.
            external = root/'external'
            external.mkdir()
            (images/'Drone/b.jpg').replace(external/'b.jpg')
            (images/'Drone').rmdir()
            (images/'Drone').symlink_to(external, target_is_directory=True)
            executed=[]; fail=[True]; calls=[]
            def fake_run(stage, **kwargs):
                executed.append(stage)
                calls.append((stage, kwargs))
                workspace=out/'workspace'
                if stage=='image_undistorter':
                    for name in names: write_file(workspace/'images'/name)
                    for name in ('cameras','images','points3D'): write_file(workspace/'sparse'/f'{name}.bin')
                    write_file(workspace/'stereo/patch-match.cfg')
                elif stage=='patch_match_stereo':
                    if fail[0]:
                        fail[0]=False
                        raise RuntimeError('stereo failed')
                    for name in names:
                        for kind,channels in [('depth_maps',1),('normal_maps',3)]:
                            write_file(workspace/'stereo'/kind/(name+'.geometric.bin'),
                                       f'1&1&{channels}&'.encode()+b'\x00'*(4*channels))
                else:
                    write_file(workspace/'fused.ply',b'ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n'+struct.pack('<fff',1,2,3))
            api = SimpleNamespace(
                has_cuda=True, __version__='4.0.0',
                UndistortCameraOptions=SimpleNamespace,
                PatchMatchOptions=SimpleNamespace, StereoFusionOptions=SimpleNamespace,
                undistort_images=lambda **kw: fake_run('image_undistorter', **kw),
                patch_match_stereo=lambda **kw: fake_run('patch_match_stereo', **kw),
                stereo_fusion=lambda **kw: fake_run('stereo_fusion', **kw))
            api.stereo_fusion.__doc__ = 'output_type: str'
            with patch.object(m,'read_names',return_value=names), patch.object(m,'load_pycolmap',return_value=api), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'stereo failed'): m.run_mvs(model,images,out)
                self.assertEqual(json.loads((out/'mvs_state.json').read_text())['steps']['undistort'],'complete')
                self.assertIn('stereo failed', (out/'stereo.log').read_text())
                m.run_mvs(model,images,out)
                self.assertEqual(executed,['image_undistorter','patch_match_stereo','patch_match_stereo','stereo_fusion'])
                m.run_mvs(model,images,out)
                self.assertEqual(calls[0][1]['num_patch_match_src_images'], 10)
                self.assertEqual(calls[0][1]['undistort_options'].max_image_size, 1024)
                stereo_options = calls[2][1]['options']
                self.assertEqual(stereo_options.gpu_index, '0')
                self.assertTrue(stereo_options.geom_consistency)
                self.assertEqual(stereo_options.cache_size, 2.0)
                self.assertEqual(calls[3][1]['output_type'], 'ply')
                self.assertEqual(calls[3][1]['input_type'], 'geometric')
                self.assertEqual(calls[3][1]['options'].num_threads, 2)
                self.assertTrue(calls[3][1]['options'].use_cache)
                self.assertEqual(json.loads((out/'mvs_state.json').read_text())['runtime']['backend'], 'pycolmap')
                self.assertEqual(len(executed),4)
                m.run_mvs(model,images,out,threads=8,gpu_index='3')
                self.assertEqual(len(executed),4)
                moved=root/'copy'
                shutil.copytree(model,moved/'sfm')
                shutil.copytree(images,moved/'images')
                shutil.copytree(out,moved/'dense')
                m.run_mvs(moved/'sfm',moved/'images',moved/'dense',threads=4,gpu_index='1')
                self.assertEqual(len(executed),4)
                with self.assertRaises(ValueError): m.run_mvs(model,images,out,max_image_size=800)
                # Una salida fusionada truncada se detecta y se rehace solo fusion.
                (out/'workspace/fused.ply').write_bytes(b'ply\n')
                m.run_mvs(model,images,out)
                self.assertEqual(executed[-1],'stereo_fusion')
                self.assertEqual(len(executed),5)

    def test_cuda_and_dense_api_required(self):
        for api in (SimpleNamespace(has_cuda=False), SimpleNamespace(has_cuda=True)):
            with self.subTest(api=api), patch.dict(sys.modules, {'pycolmap': api}):
                with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                    m.load_pycolmap()

    def test_runtime_restores_resources_and_captures_native_logs(self):
        before = {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
        affinity = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else None
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp)/'stage.log'
            with self.assertRaisesRegex(RuntimeError, 'failure'):
                with m.stage_runtime(log, 1):
                    self.assertEqual(os.environ['OMP_NUM_THREADS'], '1')
                    if affinity:
                        self.assertEqual(len(os.sched_getaffinity(0)), 1)
                    os.write(1, b'native stdout\n')
                    os.write(2, b'native stderr\n')
                    raise RuntimeError('failure')
            self.assertIn('native stdout', log.read_text())
            self.assertIn('native stderr', log.read_text())
            self.assertIn('failure', log.read_text())
        self.assertEqual(before, {key: os.environ.get(key) for key in before})
        if affinity:
            self.assertEqual(os.sched_getaffinity(0), affinity)

    def test_discovery_only_completed_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); snapshot=root/'Scene/snapshot'
            (snapshot/'reconstructions/mast3r-123/Car').mkdir(parents=True)
            (snapshot/'dataset.json').write_text(json.dumps({'scene':'/images/Scene'}))
            run=snapshot/'reconstructions/mast3r-123'
            (run/'config.json').write_text(json.dumps({'preset':'mast3r'}))
            report=run/'Car/result.json'
            report.write_text(json.dumps({'status':'running','model_path':'attempt1'}))
            self.assertEqual(m.discover_models(root),[])
            report.write_text(json.dumps({'status':'complete','model_path':'attempt1'}))
            self.assertEqual(m.discover_models(root),[(run/'Car/attempt1',Path('/images/Scene'))])
            self.assertEqual(m.discover_models(root,presets=['sift']),[])


if __name__=='__main__': unittest.main()
