import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('extractor', Path(__file__).resolve().parents[1] / 'extract_colmap_frames.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'Requiere FFmpeg')
class ExtractionTests(unittest.TestCase):
    def test_scene_sampling_and_camera_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / 'Scene'
            for relative in ('Car/cam0/clip1.mp4', 'Car/cam0/clip2.mp4', 'Car/cam7/clip1.mp4', 'Drone/a.mp4', 'Pedestrian/walk.mp4'):
                target = scene / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                                'testsrc2=size=64x48:rate=10:duration=2', '-threads', '1',
                                '-c:v', 'mpeg4', str(target)], check=True)
            for fps, fmt, expected in ((2, 'jpg', 4), (0.5, 'png', 1)):
                output = root / f'output_{fps}'
                with contextlib.redirect_stdout(io.StringIO()):
                    plan = m.extract_scene(scene, output, fps, None, fmt)
                self.assertEqual(len(plan), 5)
                self.assertEqual({p['image_folder'] for p in plan}, {'Car/cam0', 'Car/cam7', 'Drone', 'Pedestrian'})
                self.assertFalse((output / 'images').exists())
                self.assertTrue(all(p['extracted_frames'] == expected for p in plan))
                self.assertEqual(plan[0]['image_folder'], plan[1]['image_folder'])
                self.assertNotEqual(plan[0]['image_folder'], plan[2]['image_folder'])
                paths = (output / '_colmap' / 'image_list.txt').read_text().splitlines()
                self.assertEqual(len(paths), 5 * expected)
                self.assertEqual(len(paths), len(set(paths)))
                self.assertTrue(all((output / p).is_file() for p in paths))
                data = json.loads((output / '_colmap' / 'extraction.json').read_text())
                self.assertEqual(data['status'], 'complete')
                self.assertTrue((output / '_colmap' / 'sparse').is_dir())
                self.assertIn('single_camera_per_folder 1', (output / '_colmap' / 'run_colmap.sh').read_text())
                with self.assertRaises(ValueError):
                    m.extract_scene(scene, output, fps)
            dry = root / 'dry'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(len(m.extract_scene(scene, dry, 2, dry_run=True)), 5)
            self.assertFalse(dry.exists())
            for fps in (0, -1, float('nan'), 20):
                with self.assertRaises(ValueError):
                    m.extract_scene(scene, root / 'invalid', fps)


if __name__ == '__main__':
    unittest.main()
