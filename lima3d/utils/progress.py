"""Small atomic worker status files, independent of terminal rendering."""
import json
import os
from pathlib import Path
import time

from .io import save_json


_last_write = {}

def report(operation, current=None, total=None, unit=None, work=False):
    target = os.environ.get('LIMA3D_PROGRESS_FILE')
    if not target:
        return
    now = time.monotonic()
    key = (target, operation)
    if (current is not None and total is not None and current != total
            and now - _last_write.get(key, -1e9) < 0.5):
        return
    _last_write[key] = now
    path = Path(target)
    try:
        previous = json.loads(path.read_text())
    except (OSError, ValueError):
        previous = {}
    save_json(path, dict(operation=operation, current=current, total=total,
                         unit=unit, worked=work or previous.get('worked', False),
                         updated=time.time()))


def read_status(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return {}


class StageCounts:
    """Count resolved jobs; cached and skipped jobs are not new computation."""
    def __init__(self):
        self.total = 0
        self.counts = dict(completed=0, cached=0, skipped=0, failed=0)

    def add(self):
        self.total += 1

    def finish(self, outcome):
        self.counts[outcome] += 1

    @property
    def resolved(self):
        return sum(self.counts.values())
