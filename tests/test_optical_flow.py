import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from lima3d.optical_flow import validate_options, magnitude_statistics, block_motion
from extract_frames import extract_scene

class OpticalFlowTests(unittest.TestCase):
    def options(self):
        return validate_options(dict(block_size=16, block_stride=16, search_radius=8,
                                     mode_threshold_px=3, keep_last=False))

    def test_statistics(self):
        self.assertEqual(magnitude_statistics([0, 0, 2], .5)['mode_px'], 0)
        self.assertEqual(magnitude_statistics([2, 3], 1)['mode_px'], 2)
        self.assertIsNone(magnitude_statistics([], 1)['mode_px'])
        for options in ({'mode_threshold_px': 0}, {'mode_bin_width_px': 0}, {'keep_last': 1}):
            with self.assertRaises(ValueError): validate_options(options)

    def test_motion_and_mask(self):
        rng = np.random.default_rng(4)
        reference = rng.integers(0, 256, (64, 96), dtype=np.uint8)
        current = cv2.warpAffine(reference, np.float32([[1,0,3],[0,1,0]]), (96,64))
        self.assertEqual(block_motion(reference, current, self.options())['mode_px'], 3)
        self.assertEqual(block_motion(reference, reference, self.options())['mode_px'], 0)
        self.assertEqual(block_motion(reference, current, self.options(), np.zeros_like(reference))['valid_blocks'], 0)
        self.assertEqual(block_motion(np.zeros_like(reference), np.zeros_like(reference), self.options())['valid_blocks'], 0)

    def test_video_selection_and_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); scene = root/'Scene'; camera = scene/'Car/cam0'; camera.mkdir(parents=True)
            video = camera/'clip.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'FFV1'), 10, (96,64))
            self.assertTrue(writer.isOpened())
            base = np.random.default_rng(10).integers(0,256,(64,96,3),dtype=np.uint8)
            for dx in range(11):
                writer.write(cv2.warpAffine(base,np.float32([[1,0,dx],[0,1,0]]),(96,64)))
            writer.release()
            for keep_last, expected in [(False,[0,3,6,9]), (True,[0,3,6,9,10])]:
                output = root/str(keep_last)
                extract_scene(scene,output,None,strategy='optical_flow',image_format='png',
                              optical_flow={**self.options(),'keep_last':keep_last},
                              regions={'Car/cam0': {'mode':'crop','box':[0,0,80,64]}})
                log = output/'_colmap/flow/Car/cam0/clip_avi.jsonl'
                events = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual([e['source_frame_index'] for e in events if e['selected']],expected)
                self.assertEqual(cv2.imread(str(next((output/'Car/cam0').glob('*.png')))).shape[:2],(64,80))
                manifest = json.loads((output/'_colmap/extraction.json').read_text())
                self.assertEqual(manifest['total_frames'],len(expected))
                self.assertEqual(manifest['status'],'complete')

    def test_hybrid_timeout_and_reference_reset(self):
        for moving in (False, True):
            with self.subTest(moving=moving), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);scene=root/'Scene';camera=scene/'Car/cam0';camera.mkdir(parents=True)
                writer=cv2.VideoWriter(str(camera/'clip.avi'),cv2.VideoWriter_fourcc(*'FFV1'),10,(96,64))
                self.assertTrue(writer.isOpened())
                base=np.random.default_rng(20).integers(0,256,(64,96,3),dtype=np.uint8) if moving else np.zeros((64,96,3),np.uint8)
                for index in range(11):
                    dx=3 if moving and index>=2 else 0
                    writer.write(cv2.warpAffine(base,np.float32([[1,0,dx],[0,1,0]]),(96,64)))
                writer.release()
                output=root/'output'
                extract_scene(scene,output,2,strategy='hybrid',image_format='png',optical_flow=self.options())
                events=[json.loads(line) for line in (output/'_colmap/flow/Car/cam0/clip_avi.jsonl').read_text().splitlines()]
                selected=[event for event in events if event['selected']]
                self.assertEqual([event['source_frame_index'] for event in selected],[0,2,7] if moving else [0,5,10])
                self.assertEqual([event['reason'] for event in selected],['first','threshold','fps_fallback'] if moving else ['first','fps_fallback','fps_fallback'])
                self.assertEqual(events[8]['reference_frame_index'],7 if moving else 5)
                metadata=json.loads((output/'_colmap/extraction.json').read_text())
                self.assertEqual(metadata['strategy'],'hybrid')
                self.assertEqual(metadata['requested_fps'],2)
                self.assertEqual(metadata['videos'][0]['flow_selection']['fallback_selected_frames'],1 if moving else 2)
                with self.assertRaises(ValueError):
                    extract_scene(scene,root/'invalid',0,strategy='hybrid',optical_flow=self.options())
                self.assertFalse((root/'invalid').exists())

    def test_video_clock_variable_and_missing_timestamps(self):
        from lima3d.optical_flow import VideoClock
        clock=VideoClock(10)
        np.testing.assert_allclose([clock.update(t) for t in [2,2.1,2.4,2.9]],[0,.1,.4,.9])
        clock=VideoClock(10)
        np.testing.assert_allclose([clock.update(t) for t in [0,0,None,0]],[0,.1,.2,.3])
        self.assertEqual(clock.fallback_frames,3)
        clock=VideoClock(float('nan'));clock.update(0)
        with self.assertRaises(ValueError):clock.update(0)

    def test_hybrid_cli(self):
        from extract_frames import main
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scene=root/'Scene';scene.mkdir()
            with patch('extract_frames.extract_scene') as extract:
                main([str(scene),'--strategy','hybrid','--fps','2','--mode-threshold-px','15',
                      '--output',str(root/'output'),'--dry-run'])
                self.assertEqual(extract.call_args.kwargs['strategy'],'hybrid')
                self.assertEqual(extract.call_args.kwargs['fps'],2)
                self.assertEqual(extract.call_args.kwargs['optical_flow']['mode_threshold_px'],15)
