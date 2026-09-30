import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lima3d.utils.timing import RunTimings


class TimingTests(unittest.TestCase):
    def test_duration_uses_monotonic_clock_and_preserves_history(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('lima3d.utils.timing.time.monotonic', side_effect=[10, 12, 17, 20]):
                report = RunTimings(directory, ['bank'], 'config.yaml')
                index = report.start('bank', 'Scene-sift', scene='Scene')
                report.finish(index, 'completed', exit_code=0)
                report.close('complete')
            saved = json.loads(report.path.read_text())
            self.assertEqual(saved['elapsed_seconds'], 10)
            self.assertEqual(saved['jobs'][0]['elapsed_seconds'], 5)
            self.assertEqual(saved['stage_totals']['bank']['job_seconds'], 5)
            again = RunTimings(directory, ['bank'], 'config.yaml')
            again.skipped('bank', 'Scene-sift', status='cached')
            again.close('complete')
            self.assertNotEqual(report.path, again.path)
            self.assertEqual(json.loads(report.path.read_text()), saved)
            self.assertEqual(again.data['jobs'][0]['elapsed_seconds'], 0)
            self.assertFalse(again.data['jobs'][0]['launched'])

    def test_interrupted_and_failed_work_keeps_measured_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            report = RunTimings(directory, ['sfm'], 'config.yaml')
            index = report.start('sfm', 'Scene-sift-Car')
            report.finish(index, 'failed', exit_code=1)
            report.start('sfm', 'Scene-sift-Drone')
            report.close('interrupted')
            saved = json.loads(report.path.read_text())
            self.assertEqual([j['status'] for j in saved['jobs']], ['failed', 'interrupted'])
            self.assertTrue(all(j['finished_at'] and j['elapsed_seconds'] >= 0 for j in saved['jobs']))
