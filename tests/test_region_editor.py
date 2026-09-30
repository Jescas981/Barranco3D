import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lima3d.region_editor import save_selection, validate_selection, video_info, frame_png
from lima3d.regions import parse_regions, effective_regions, prepared_scene, mask_path, region_identity
from lima3d.artifacts import portable_context
spec = importlib.util.spec_from_file_location('frame_extractor_for_editor', ROOT/'extract_colmap_frames.py')
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)


class EditorTests(unittest.TestCase):
    def test_mask_export_yaml_backup_and_other_settings_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = root/'config.yaml'
            before = '# Hardware stays unchanged\nresources: {bank: {threads: 4}}\n# Keep this experiment\nexperiments: [{name: sift, preset: sift}]\nregions:\n  Car: {mode: none}\n  Drone: {mode: none}\n'
            config.write_text(before)
            info = dict(width=64, height=48, duration=2.)
            result = save_selection(config, 'Car', root/'video.mp4', info,
                                    {'mode':'mask','polygons':[[[0,0],[20,0],[20,48],[0,48]]]})
            self.assertEqual(Path(result['backup']).read_text(), before)
            after=config.read_text()
            self.assertTrue(after.startswith(before[:before.index('regions:')]))
            data=yaml.safe_load(after)
            self.assertEqual(data['regions']['Drone'], {'mode':'none'})
            mask=root/data['regions']['Car']['path']
            with Image.open(mask) as image:
                pixels=np.asarray(image)
                self.assertEqual(image.size, (64,48))
                self.assertEqual(pixels[10,10], 0)
                self.assertEqual(pixels[10,40], 255)
            content=mask.read_bytes()
            save_selection(config,'Car',root/'video.mp4',info,{'mode':'crop','box':[2,4,62,46]})
            self.assertEqual(mask.read_bytes(),content)
            self.assertEqual(yaml.safe_load(config.read_text())['regions']['Car']['box'],[2,4,62,46])
            save_selection(config,'Car',root/'video.mp4',info,{'mode':'none'})
            self.assertEqual(yaml.safe_load(config.read_text())['regions']['Car'],{'mode':'none'})

    def test_rejects_invalid_selections(self):
        for selection in ({'mode':'crop','box':[0,0,100,10]}, {'mode':'mask','polygons':[]},
                          {'mode':'mask','polygons':[[[1,1],[2,2],[3,3]]]},
                          {'mode':'mask','polygons':[[[-1,0],[2,0],[2,3]]]}):
            with self.assertRaises(ValueError):
                validate_selection(selection,(64,48))

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_video_editor_to_extraction_to_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scene=root/'dataset/Scene'; video=scene/'Car/clip.mp4'
            video.parent.mkdir(parents=True)
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
                'testsrc2=size=64x48:rate=10:duration=1','-threads','1','-c:v','mpeg4',str(video)],check=True)
            config=root/'config.yaml'
            config.write_text('data:\n  datasets_root: dataset\n  frames_root: frames\n  platforms: [Car]\n  scenes: [Scene]\nextraction: {fps: 2, format: png, threads: 1}\n')
            info=video_info(video)
            with Image.open(io.BytesIO(frame_png(video,.2))) as preview:
                self.assertEqual(preview.width/preview.height,64/48)
            save_selection(config,'Car',video,info,{'mode':'crop','box':[3,5,60,46]})
            with contextlib.redirect_stdout(io.StringIO()):
                extractor.main(['--config',str(config)])
            frames=root/'frames/Scene'; images=list((frames/'Car').glob('*.png'))
            self.assertEqual(len(images),2)
            with Image.open(images[0]) as image:
                self.assertEqual(image.size,(57,41))
            regions=parse_regions(yaml.safe_load(config.read_text())['regions'],root,['Car'])
            self.assertEqual(effective_regions(frames,regions),{})
            args=SimpleNamespace(regions=regions,platforms=['Car'],output_root=root/'outputs')
            _,_,names,cache=portable_context(frames,args)
            self.assertEqual(prepared_scene(frames,args,cache,names),frames)
            with self.assertRaisesRegex(ValueError,'already cropped'):
                effective_regions(frames,{'Car':{'mode':'crop','box':[0,0,30,30]}})
            self.assertEqual(effective_regions(frames,{},['Drone']),{})
            # A fresh extraction exports masks but preserves full-size pixels.
            save_selection(config,'Car',video,info,{'mode':'mask','polygons':[[[0,0],[20,0],[20,48],[0,48]]]})
            output=root/'masked/Scene'
            with contextlib.redirect_stdout(io.StringIO()):
                extractor.main(['--config',str(config),'--output',str(output)])
            for image in (output/'Car').glob('*.png'):
                with Image.open(image) as frame:
                    self.assertEqual(frame.size,(64,48))
                self.assertTrue((output/'_colmap/masks'/('Car/'+image.name+'.png')).is_file())
            self.assertIn('--ImageReader.mask_path masks',(output/'_colmap/run_colmap.sh').read_text())
            regions=parse_regions(yaml.safe_load(config.read_text())['regions'],root,['Car'])
            Path(regions['Car']['path']).unlink()
            portable=effective_regions(output,regions)
            mask=mask_path(portable,output,'Car/clip_mp4__000000.png')
            self.assertTrue(mask.is_file())
            self.assertIsNotNone(region_identity(output,['Car/clip_mp4__000000.png'],portable))


class CameraEditorTests(unittest.TestCase):
    def test_discovery_includes_all_cameras_and_single_camera_platforms(self):
        from lima3d.region_editor import discover_videos
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'config.yaml'
            config.write_text('data: {datasets_root: dataset, platforms: [Car]}\n')
            paths=['Car/cam0/a.mp4','Car/cam0/b.mp4','Car/cam1/a.mp4','Drone/d.MP4','Pedestrian/p.mp4']
            for relative in paths:
                path=root/'dataset/Scene'/relative
                path.parent.mkdir(parents=True,exist_ok=True);path.touch()
            catalog=discover_videos(config)
            self.assertEqual(len(catalog),5)
            self.assertEqual({(r['platform'],r['camera']) for r in catalog},
                             {('Car','cam0'),('Car','cam1'),('Drone',None),('Pedestrian',None)})

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_camera_masks_are_saved_and_extracted_independently(self):
        from lima3d.regions import region_for_name, valid_points
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'config.yaml';scene=root/'dataset/Scene'
            config.write_text('data: {datasets_root: dataset, frames_root: frames, platforms: [Car], scenes: [Scene]}\n'
                              'extraction: {fps: 1, format: png, threads: 1}\nregions:\n  Car: {mode: none}\n')
            for camera in ('cam0','cam1','cam2'):
                video=scene/'Car'/camera/'clip.mp4';video.parent.mkdir(parents=True)
                subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
                    'testsrc2=size=64x48:rate=10:duration=1','-threads','1','-c:v','mpeg4',str(video)],check=True)
                selection=({'mode':'crop','box':[4,6,60,46]} if camera=='cam2' else
                           {'mode':'mask','polygons':[[[0,0],[20,0],[20,48],[0,48]]]}
                           if camera=='cam0' else
                           {'mode':'mask','polygons':[[[40,0],[64,0],[64,48],[40,48]]]})
                save_selection(config,'Car',video,video_info(video),selection,camera=camera)
            raw=yaml.safe_load(config.read_text())['regions']
            self.assertEqual(set(raw['Car']['cameras']),{'cam0','cam1','cam2'})
            self.assertNotEqual(raw['Car']['cameras']['cam0']['path'],raw['Car']['cameras']['cam1']['path'])
            regions=parse_regions(raw,root,['Car'])
            with contextlib.redirect_stdout(io.StringIO()):
                extractor.main(['--config',str(config)])
            frames=root/'frames/Scene'
            for camera, expected in [('cam0',[False,True]),('cam1',[True,False])]:
                path=frames/'_colmap/masks/Car'/camera/'clip_mp4__000000.png.png'
                np.testing.assert_array_equal(valid_points(np.array([[10,10],[50,10]]),path),expected)
            with Image.open(frames/'Car/cam2/clip_mp4__000000.png') as image:
                self.assertEqual(image.size,(56,40))
            effective=effective_regions(frames,regions,['Car'])
            self.assertEqual(region_for_name(effective,'Car/cam2/frame.png')['mode'],'none')
            self.assertEqual(region_for_name(effective,'Car/cam0/frame.png')['mode'],'mask')
            # An explicit camera-level none overrides a platform default.
            override=parse_regions({'Car':{'mode':'crop','box':[0,0,30,30],
                                            'cameras':{'cam0':{'mode':'none'}}}},root,['Car'])
            self.assertEqual(region_for_name(override,'Car/cam0/frame.png')['mode'],'none')
            self.assertEqual(region_for_name(override,'Car/cam1/frame.png')['mode'],'crop')
