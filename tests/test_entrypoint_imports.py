"""Entradas movidas a src y workers importables sin depender del cwd."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lima3d.utils.paths import REPO_ROOT


class EntrypointImportsTests(unittest.TestCase):
    def test_repository_root(self):
        self.assertEqual(REPO_ROOT, ROOT)
        self.assertTrue((REPO_ROOT / 'hloc').is_dir())

    def test_scripts_and_worker_from_another_directory(self):
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        with tempfile.TemporaryDirectory() as cwd:
            for script in sorted((ROOT / 'src').glob('*.py')):
                with self.subTest(script=script.name):
                    result = subprocess.run([sys.executable, str(script), '--help'],
                                            cwd=cwd, env=env, capture_output=True,
                                            text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
            env['PYTHONPATH'] = str(ROOT)
            result = subprocess.run([sys.executable, '-m', 'lima3d.scheduler', '--help'],
                                    cwd=cwd, env=env, capture_output=True,
                                    text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
