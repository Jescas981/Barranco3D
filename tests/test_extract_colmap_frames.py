import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('extractor', Path(__file__).resolve().parents[1] / 'extract_frames.py')
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


class ExtractionConfigTests(unittest.TestCase):
    def make_config(self, root, extra=''):
        for scene in ('A', 'B'):
            (root / 'videos' / scene).mkdir(parents=True)
        config = root / 'config.yaml'
        config.write_text('data:\n  datasets_root: videos\n  frames_root: images\n'
                          '  scenes: null\n  platforms: [Car, Drone]\n'
                          'extraction:\n  fps: 0.5\n  format: png\n  threads: 2\n' + extra)
        return config

    def test_yaml_paths_and_all_scenes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = self.make_config(root)
            with patch.object(m, 'extract_scene') as extract:
                m.main(['--config', str(config), '--dry-run'])
            self.assertEqual(extract.call_count, 2)
            job = extract.call_args_list[0].kwargs
            self.assertEqual(job['scene'], root / 'videos/A')
            self.assertEqual(job['output'], root / 'images/A')
            self.assertEqual((job['fps'], job['image_format'], job['threads']), (0.5, 'png', 2))
            self.assertEqual(job['groups'], ['Car', 'Drone'])
            self.assertTrue(job['dry_run'])
            self.assertFalse((root / 'images').exists())

    def test_cli_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = self.make_config(root)
            with patch.object(m, 'extract_scene') as extract:
                m.main(['--config', str(config), '--scenes', 'B', '--fps', '3',
                        '--format', 'jpg', '--threads', '4', '--groups', 'Drone',
                        '--output', str(root / 'custom')])
            job = extract.call_args.kwargs
            self.assertEqual(job['scene'].name, 'B')
            self.assertEqual((job['fps'], job['image_format'], job['threads']), (3, 'jpg', 4))
            self.assertEqual(job['groups'], ['Drone'])
            self.assertEqual(job['output'], root / 'custom')

    def test_rejects_nonempty_output_before_any_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = self.make_config(root)
            (root / 'images/B').mkdir(parents=True)
            (root / 'images/B/existing.jpg').write_bytes(b'existing')
            with patch.object(m, 'extract_scene') as extract, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    m.main(['--config', str(config)])
                extract.assert_not_called()

    def test_legacy_cli_and_invalid_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = self.make_config(root)
            with patch.object(m, 'extract_scene') as extract:
                m.main([str(root / 'videos/A'), '--fps', '2', '--output', str(root / 'legacy')])
            self.assertEqual(extract.call_args.kwargs['fps'], 2)
            for options in (['--fps', '0'], ['--threads', '0'], ['--output', str(root / 'ambiguous')]):
                with patch.object(m, 'extract_scene') as extract, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        m.main(['--config', str(config), *options])
                    extract.assert_not_called()


if __name__ == '__main__':
    unittest.main()
