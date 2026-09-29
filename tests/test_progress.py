import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lima3d.utils.progress import StageCounts, report, read_status


class ProgressTests(unittest.TestCase):
    def test_partial_work_is_not_reported_as_fully_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            with patch.dict(os.environ, {'LIMA3D_PROGRESS_FILE': str(path)}):
                report('Checking cache')
                self.assertFalse(read_status(path)['worked'])
                report('Matching', 3, 10, 'pairs', work=True)
                self.assertEqual(read_status(path)['current'], 3)
                report('Cached remaining features')
                self.assertTrue(read_status(path)['worked'])

    def test_failures_and_skips_resolve_jobs(self):
        counts = StageCounts()
        for outcome in ('completed', 'cached', 'skipped', 'failed'):
            counts.add()
            counts.finish(outcome)
        self.assertEqual(counts.resolved, counts.total)
        self.assertEqual(counts.counts['failed'], 1)

    def test_missing_or_incomplete_status_is_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            self.assertEqual(read_status(path), {})
            path.write_text('{')
            self.assertEqual(read_status(path), {})
