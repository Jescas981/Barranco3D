"""Persistent wall-clock timings, isolated by scheduler invocation."""
from datetime import datetime, timezone
from pathlib import Path
import socket
import time
from uuid import uuid4

from .io import save_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class RunTimings:
    def __init__(self, output_root, stages, config):
        self.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8]
        self.path = Path(output_root) / '_runs' / self.run_id / 'timings.json'
        self.started = time.monotonic()
        self.clocks = {}
        self.data = dict(run_id=self.run_id, hostname=socket.gethostname(), config=str(config),
                         stages=sorted(stages), started_at=utc_now(), finished_at=None,
                         elapsed_seconds=None, status='running', jobs=[])
        self.flush()

    def flush(self):
        save_json(self.path, self.data)

    def start(self, stage, name, **metadata):
        index = len(self.data['jobs'])
        self.clocks[index] = time.monotonic()
        self.data['jobs'].append(dict(stage=stage, name=name, started_at=utc_now(),
                                     finished_at=None, elapsed_seconds=None,
                                     status='running', **metadata))
        self.flush()
        return index

    def finish(self, index, status, **metadata):
        job = self.data['jobs'][index]
        if job['status'] != 'running':
            return
        job.update(status=status, finished_at=utc_now(),
                   elapsed_seconds=max(0., time.monotonic() - self.clocks.pop(index)), **metadata)
        self.flush()

    def skipped(self, stage, name, status='skipped', **metadata):
        # No worker was launched. Do not pretend to measure original cached computation.
        now = utc_now()
        self.data['jobs'].append(dict(stage=stage, name=name, status=status,
                                     started_at=now, finished_at=now, elapsed_seconds=0.,
                                     launched=False, **metadata))
        self.flush()

    def close(self, status):
        for index in list(self.clocks):
            self.finish(index, 'interrupted' if status == 'interrupted' else 'failed')
        self.data['stage_totals'] = {}
        for job in self.data['jobs']:
            stage = self.data['stage_totals'].setdefault(job['stage'],
                        {'jobs': 0, 'job_seconds': 0., 'outcomes': {}})
            stage['jobs'] += 1
            stage['job_seconds'] += job['elapsed_seconds'] or 0.
            stage['outcomes'][job['status']] = stage['outcomes'].get(job['status'], 0) + 1
        self.data.update(status=status, finished_at=utc_now(),
                         elapsed_seconds=max(0., time.monotonic() - self.started))
        self.flush()
