import contextlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_mvs as m


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
            executed=[]; fail=[True]
            def fake_run(command,**kwargs):
                stage=command[1]; executed.append(stage)
                workspace=out/'workspace'
                if stage=='image_undistorter':
                    for name in names: write_file(workspace/'images'/name)
                    for name in ('cameras','images','points3D'): write_file(workspace/'sparse'/f'{name}.bin')
                    write_file(workspace/'stereo/patch-match.cfg')
                elif stage=='patch_match_stereo':
                    if fail[0]:
                        fail[0]=False
                        raise subprocess.CalledProcessError(1,command)
                    for name in names:
                        for kind,channels in [('depth_maps',1),('normal_maps',3)]:
                            write_file(workspace/'stereo'/kind/(name+'.geometric.bin'),
                                       f'1&1&{channels}&'.encode()+b'\x00'*(4*channels))
                else:
                    write_file(workspace/'fused.ply',b'ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n'+struct.pack('<fff',1,2,3))
                return subprocess.CompletedProcess(command,0)
            with patch.object(m,'read_names',return_value=names), patch.object(m.shutil,'which',return_value=sys.executable), patch.object(m.subprocess,'run',side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError): m.run_mvs(model,images,out)
                self.assertEqual(json.loads((out/'mvs_state.json').read_text())['steps']['undistort'],'complete')
                m.run_mvs(model,images,out)
                self.assertEqual(executed,['image_undistorter','patch_match_stereo','patch_match_stereo','stereo_fusion'])
                m.run_mvs(model,images,out)
                self.assertEqual(len(executed),4)
                with self.assertRaises(ValueError): m.run_mvs(model,images,out,max_image_size=800)
                # Una salida fusionada truncada se detecta y se rehace solo fusion.
                (out/'workspace/fused.ply').write_bytes(b'ply\n')
                m.run_mvs(model,images,out)
                self.assertEqual(executed[-1],'stereo_fusion')
                self.assertEqual(len(executed),5)

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
